"""PDF 解析：页面 → 有序元素流（文本 + 图）。

三种图的来源，本语料全部存在，必须都处理：
  1. raster  —— 嵌入位图，page.get_image_info() 拿 bbox，渲染该区域
  2. vector  —— 矢量插图（石头/大疆Mic2 全篇是这种，get_images() 拿不到）
                把 get_drawings() 的路径矩形聚类成区域再渲染
  3. page-scan —— 整页扫描件（宇树G1/美的，0 字符），整页渲染 + OCR

小图标（22x22 的行内 UI 图标）必须过滤掉，否则一份手册会抽出几百张垃圾图。
"""
from __future__ import annotations

import os
import re
from collections import Counter

import pymupdf

from .layout import Rect, order_elements
from .scan_layout import split_scan_page
from .schema import Element, Figure, PageReport
from .search_text import is_boilerplate, vector_text

# --- 阈值（全部可在 configs 里覆盖）---
MIN_TEXT_CHARS = 15        # 低于此字符数视为无文本层
ICON_MAX_SIDE = 90         # 任一边小于此值 → 行内图标
ICON_MAX_AREA = 7000       # 面积小于此值 → 行内图标
FIG_MIN_AREA = 12000       # 矢量聚类最小面积
TEXT_COVER_MAX = 0.15      # 区域被文字覆盖比例超过此值 → 不是插图（是表格框/文字框）
RENDER_DPI = 200
MERGE_PAD = 6.0            # 矢量路径聚类合并时的容差


def _union(a: Rect, b: Rect) -> Rect:
    return Rect(min(a.x0, b.x0), min(a.y0, b.y0), max(a.x1, b.x1), max(a.y1, b.y1))


def _merge_rects(rects: list[Rect], pad: float = MERGE_PAD) -> list[Rect]:
    """把相邻/重叠的矩形合并成簇。用于把一堆矢量路径聚成一张插图。"""
    merged: list[Rect] = []
    for r in sorted(rects, key=lambda x: x.area, reverse=True):
        target = None
        for m in merged:
            if m.expand(pad).overlap_x(r) > 0 and m.expand(pad).overlap_y(r) > 0:
                target = m
                break
        if target is None:
            merged.append(Rect(r.x0, r.y0, r.x1, r.y1))
        else:
            u = _union(target, r)
            target.x0, target.y0, target.x1, target.y1 = u.x0, u.y0, u.x1, u.y1
    # 合并后再跑一轮，处理链式相邻
    changed = True
    while changed:
        changed = False
        out: list[Rect] = []
        for r in merged:
            hit = None
            for o in out:
                if o.expand(pad).overlap_x(r) > 0 and o.expand(pad).overlap_y(r) > 0:
                    hit = o
                    break
            if hit is None:
                out.append(r)
            else:
                u = _union(hit, r)
                hit.x0, hit.y0, hit.x1, hit.y1 = u.x0, u.y0, u.x1, u.y1
                changed = True
        merged = out
    return merged


def _text_coverage(region: Rect, text_rects: list[Rect]) -> float:
    if region.area <= 0:
        return 1.0
    covered = 0.0
    for t in text_rects:
        covered += region.overlap_x(t) * region.overlap_y(t)
    return min(1.0, covered / region.area)


# 正文里指向插图的引用词。本语料几乎没有「图3-2」式图注，但这类引用很常见
# （实测大疆 Mini4Pro 12 处、联想 12 处、问界 11 处、石头 7 处）。
REF_PAT = re.compile(r"如图|如下图|上图|下图|见图|参照图|参见图|图中|示意图|右图|左图")

ASSOC_MAX_GAP_RATIO = 0.05   # 图文垂直间距上限（占页高）
ASSOC_MIN_XOVER = 0.30       # 图文水平重叠下限（占较窄者宽度）


def _nearby_text(fig: Rect, text_blocks: list[tuple[Rect, str, float]],
                 page_h: float,
                 max_gap_ratio: float = ASSOC_MAX_GAP_RATIO,
                 min_xover: float = ASSOC_MIN_XOVER) -> str:
    """给图找说明文字：取图正上方、正下方最近且横向重叠足够的文本块。

    为什么不用图注正则：实测 5 份正式手册 `^图\\s*\\d+-\\d+` 命中 0 次，
    消费级手册基本不写图注。空间邻近才是可靠信号，引用词作加权。
    """
    max_gap = max(page_h * max_gap_ratio, 12.0)
    above: list[tuple[float, str]] = []
    below: list[tuple[float, str]] = []

    for r, txt, _sz in text_blocks:
        if not txt.strip():
            continue
        # 免责套话块紧贴图片，不排除的话会挤掉真正的说明文字
        if is_boilerplate(txt):
            continue
        xov = fig.overlap_x(r) / max(min(fig.width, r.width), 1.0)
        if xov < min_xover:
            continue
        if r.y1 <= fig.y0 + 4:
            gap = fig.y0 - r.y1
            if gap <= max_gap:
                above.append((gap, txt))
        elif r.y0 >= fig.y1 - 4:
            gap = r.y0 - fig.y1
            if gap <= max_gap:
                below.append((gap, txt))

    above.sort(key=lambda t: t[0])
    below.sort(key=lambda t: t[0])

    picks: list[tuple[int, float, str]] = []
    if above:
        picks.append((0 if REF_PAT.search(above[0][1]) else 1, above[0][0], above[0][1]))
    if below:
        picks.append((0 if REF_PAT.search(below[0][1]) else 1, below[0][0], below[0][1]))

    picks.sort(key=lambda t: (t[0], t[1]))
    return " ".join(t[2] for t in picks[:2]).strip()


class PdfParser:
    def __init__(self, doc_id: str, doc_title: str, product: str,
                 out_dir: str, ocr=None, scan_cfg: dict | None = None):
        self.doc_id = doc_id
        self.doc_title = doc_title
        self.product = product
        self.out_dir = out_dir          # data/extracted/<doc_id>
        self.img_dir = os.path.join(out_dir, "images")
        os.makedirs(self.img_dir, exist_ok=True)
        self.ocr = ocr
        self.scan_cfg = scan_cfg or {}

    # ---------- 图 ----------
    def _save_region(self, page, rect: Rect, fid: str) -> tuple[str, int, int]:
        return self._save_region_pdf(page, (rect.x0, rect.y0, rect.x1, rect.y1), fid)

    def _save_region_pdf(self, page, bbox, fid: str) -> tuple[str, int, int]:
        clip = pymupdf.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
        pix = page.get_pixmap(clip=clip, dpi=RENDER_DPI)
        fn = fid + ".png"
        pix.save(os.path.join(self.img_dir, fn))
        return os.path.join("images", fn), pix.width, pix.height

    def _collect_figures(self, page, pno: int, text_rects: list[Rect],
                         report: PageReport,
                         text_blocks: list[tuple[Rect, str, float]] | None = None,
                         ) -> tuple[list[tuple[Rect, str]], list[Figure]]:
        """返回 (参与阅读顺序的图区域, Figure 列表)。"""
        text_blocks = text_blocks or []
        cand: list[tuple[Rect, str]] = []

        # 1) 位图
        for info in page.get_image_info():
            r = Rect(*info["bbox"])
            if r.width <= 1 or r.height <= 1:
                continue
            if min(r.width, r.height) < ICON_MAX_SIDE or r.area < ICON_MAX_AREA:
                report.n_inline_icons += 1
                continue
            cand.append((r, "raster"))

        # 2) 矢量插图
        draw_rects = []
        for d in page.get_drawings():
            r = Rect(*d["rect"])
            if r.width > 12 and r.height > 12 and r.area > 600:
                draw_rects.append(r)
        for cl in _merge_rects(draw_rects):
            if cl.area < FIG_MIN_AREA:
                continue
            if _text_coverage(cl, text_rects) > TEXT_COVER_MAX:
                continue
            # 与已确认的位图区域大量重叠则跳过（同一张图重复计）
            if any(cl.overlap_x(r) * cl.overlap_y(r) > cl.area * 0.6 for r, _ in cand):
                continue
            cand.append((cl, "vector"))

        # 去掉候选之间的重复（保留面积大的）
        cand.sort(key=lambda t: t[0].area, reverse=True)
        kept: list[tuple[Rect, str]] = []
        for r, src in cand:
            if any(r.overlap_x(k) * r.overlap_y(k) > r.area * 0.5 for k, _ in kept):
                continue
            kept.append((r, src))

        figures: list[Figure] = []
        page_h = page.rect.height
        for i, (r, src) in enumerate(kept, 1):
            fid = "%s_p%03d_f%02d" % (self.doc_id, pno, i)
            rel, w, h = self._save_region(page, r, fid)
            # 图内文字：位图/扫描靠 OCR（这里不做，交给扫描分支），矢量图靠 clip 取字
            inner = ""
            if src == "vector":
                inner = vector_text(page, (r.x0, r.y0, r.x1, r.y1))
            figures.append(Figure(
                figure_id=fid, doc_id=self.doc_id, page=pno,
                bbox=(round(r.x0, 1), round(r.y0, 1), round(r.x1, 1), round(r.y1, 1)),
                path=rel, width=w, height=h, source=src, doc_title=self.doc_title,
                ocr_text=inner,
                nearby_text=_nearby_text(r, text_blocks, page_h),
            ))
        return kept, figures

    # ---------- 主流程 ----------
    def parse(self, pdf_path: str) -> tuple[list[Element], list[Figure], list[PageReport]]:
        doc = pymupdf.open(pdf_path)
        elements: list[Element] = []
        figures: list[Figure] = []
        reports: list[PageReport] = []

        body_size = self._body_font_size(doc)

        for pno, page in enumerate(doc, 1):
            report = PageReport(page=pno)
            raw = page.get_text("text")
            reports.append(report)

            # --- 扫描页：版面切分 + OCR ---
            if len(raw.strip()) < MIN_TEXT_CHARS:
                if self._visual_ratio(page) < 0.30:
                    report.note = "空白/装饰页，跳过"
                    continue
                report.ocr_used = True
                page_els, page_figs, info = split_scan_page(
                    page, self.ocr, self.doc_id, self.doc_title, pno,
                    lambda bbox, fid, _p=page: self._save_region_pdf(_p, bbox, fid),
                    self.scan_cfg)
                report.n_text_blocks = info.get("texts", 0)
                report.n_figures = len(page_figs)
                report.chars = sum(len(e.text) for e in page_els if e.kind == "text")
                report.note = "scan: regions=%d figs=%d toc=%d borders=%d%s" % (
                    info.get("regions", 0), info.get("figures", 0),
                    info.get("toc", 0), info.get("borders", 0),
                    " FALLBACK" if info.get("fallback") else "")
                elements.extend(page_els)
                figures.extend(page_figs)
                continue

            # --- 正常页 ---
            d = page.get_text("dict")
            text_blocks: list[tuple[Rect, str, float]] = []
            for blk in d["blocks"]:
                if blk["type"] != 0:
                    continue
                txt = "".join(
                    s["text"] for line in blk.get("lines", []) for s in line.get("spans", [])
                ).strip()
                if not txt:
                    continue
                # 丢页码 / 纯符号块（如 "02"、"• •"），它们会污染切分
                if len(txt) <= 8 and re.fullmatch(r"[\d\s\-–—/•·.]+", txt):
                    continue
                sizes = [s["size"] for line in blk.get("lines", []) for s in line.get("spans", [])]
                size = Counter([round(s, 1) for s in sizes]).most_common(1)[0][0] if sizes else body_size
                text_blocks.append((Rect(*blk["bbox"]), txt, size))

            report.n_text_blocks = len(text_blocks)
            report.chars = sum(len(t) for _, t, _ in text_blocks)
            text_rects = [r for r, _, _ in text_blocks]

            fig_regions, page_figs = self._collect_figures(
                page, pno, text_rects, report, text_blocks)
            report.n_figures = len(page_figs)
            figures.extend(page_figs)

            # 组装元素并按阅读顺序排序
            items: list[tuple[Rect, Element]] = []
            for r, txt, size in text_blocks:
                items.append((r, Element(
                    kind="text", page=pno, bbox=(r.x0, r.y0, r.x1, r.y1),
                    text=re.sub(r"[ \t]+", " ", txt), font_size=size,
                    is_heading=size >= body_size * 1.18 and len(txt) <= 40,
                )))
            for (r, _src), fig in zip(fig_regions, page_figs):
                items.append((r, Element(
                    kind="figure", page=pno, bbox=(r.x0, r.y0, r.x1, r.y1),
                    figure_id=fig.figure_id,
                )))

            order, n_cols = order_elements(
                [r for r, _ in items], page.rect.width, page.rect.height)
            report.n_columns = n_cols
            for oi, idx in enumerate(order):
                el = items[idx][1]
                el.order = float(oi)
                elements.append(el)

        doc.close()

        # 给图补章节：按阅读顺序找它前面最近的标题（图片检索时需要章节语境）
        fig_map = {f.figure_id: f for f in figures}
        cur_sec = ""
        for el in elements:
            if el.kind == "text" and el.is_heading and el.text.strip():
                cur_sec = el.text.strip()
            elif el.kind == "figure" and el.figure_id in fig_map:
                fig = fig_map[el.figure_id]
                if not fig.section:
                    fig.section = cur_sec or "正文"

        return elements, figures, reports

    @staticmethod
    def _visual_ratio(page) -> float:
        """页面被图像/矢量图形覆盖的面积比例。用于区分「扫描页」和「空白页」。"""
        page_area = max(page.rect.width * page.rect.height, 1.0)
        area = 0.0
        for info in page.get_image_info():
            r = Rect(*info["bbox"])
            area += r.area
        if area < page_area * 0.3:
            varea = 0.0
            for d in page.get_drawings():
                varea += Rect(*d["rect"]).area
            area = max(area, varea)
        return min(1.0, area / page_area)

    @staticmethod
    def _body_font_size(doc) -> float:
        hist: Counter = Counter()
        for page in doc:
            for blk in page.get_text("dict")["blocks"]:
                if blk["type"] != 0:
                    continue
                for line in blk.get("lines", []):
                    for s in line.get("spans", []):
                        hist[round(s["size"], 1)] += len(s["text"])
        return hist.most_common(1)[0][0] if hist else 10.0
