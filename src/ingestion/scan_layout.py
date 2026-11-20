"""扫描页版面切分：把整页扫描图切成「文字区 / 图区 / 目录 / 页眉页脚」。

为什么需要：宇树 G1（27 页）和美的电饭煲（17 页）没有文本层，整页只有一张图。
如果整页当一张图存，二次检索召回的图在界面上既看不清、也没法指认「这是第几页的哪个部位」。
用户明确要求「不要整页一个图」，要「方框式一片属于一个」。

为什么不用 PP-Structure / 版面分析模型：
  1. 依赖重——paddlepaddle ~500MB + 模型下载，CPU 现场部署风险高；
  2. 黑盒——验收必问「凭什么这么切」，形态学 + 投影每一步都能画图复核；
  3. 只需处理 2 份 44 页，规整中文手册，OpenCV 够用。
留了 split_scan_page() 接口，将来可替换为模型实现。

算法流程：
  渲染 → 灰度 → Otsu 二值化(墨迹=255)
    → OCR 取文本框（判断文字区、算图文关联都靠它）
    → 形态学开运算提取横/竖长线 → 轮廓近似为四边形 → 矩形线框
    → 线框区域涂白，剩余部分做投影递归切分(XY-cut)
    → 合并邻近同类碎片（线框块不参与合并，保证「一片一个」）
    → 分类 text/figure/toc/footer
    → 图区与文字区做邻近关联
"""
from __future__ import annotations

import re

import numpy as np

from .schema import Element, Figure, Region, TextBox
from .search_text import clean, is_boilerplate

REF_PAT = re.compile(r"如图|如下图|上图|下图|见图|参照图|参见图|图中|示意图|右图|左图")

DEFAULT_CFG = {
    "render_dpi": 150,
    # 线框检测
    "h_kernel_div": 30,          # 水平核长 = W // 该值
    "v_kernel_div": 40,          # 垂直核长 = H // 该值
    "border_min_w_ratio": 0.10,  # 线框最小宽度占页宽
    "border_min_h_ratio": 0.04,  # 线框最小高度占页高
    "border_min_area_ratio": 0.015,
    "border_rect_fill": 0.75,    # 轮廓面积 / 外接矩形面积，低于此不算矩形
    # 投影切分
    "proj_ink_ratio": 0.004,     # 一行/列至少有该比例墨迹才算有效
    "min_band_h_ratio": 0.006,   # 太矮的带视为噪声
    "row_close_ratio": 0.009,    # 纵向闭运算核长（占页高），用于把同段落的行连起来
    "col_close_ratio": 0.018,    # 横向闭运算核长（占页宽），用于把同一行的汉字连起来
    # 合并
    "merge_gap_ratio": 0.35,     # 垂直间距 < 该比例 × 较小高度 → 合并
    "merge_xover": 0.55,         # 水平重叠比例下限
    "min_region_area_ratio": 0.004,
    # 分类
    "min_ocr_chars": 10,
    "text_area_min": 0.10,       # OCR 框面积占比 ≥ 此 → 文字区
    "fig_area_max": 0.06,        # 低于此 → 图区
    "header_ratio": 0.055,       # 页眉带
    "footer_ratio": 0.945,       # 页脚带
    # 图文关联
    "assoc_max_gap_ratio": 0.05,
    "assoc_min_xover": 0.28,
    # 安全阀
    "max_regions": 60,           # 切太碎就回退整页
}


# ---------------- 基础 ----------------

def render_page(page, dpi: int = 150) -> np.ndarray:
    """渲染整页为 RGB numpy 数组。"""
    pix = page.get_pixmap(dpi=dpi)
    arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n >= 3:
        return arr[:, :, :3].copy()
    return np.repeat(arr[:, :, :1], 3, axis=2)


def binarize(rgb: np.ndarray, adaptive: bool = False) -> np.ndarray:
    """二值化，墨迹=255。扫描件光照不均时可用 adaptive。"""
    import cv2
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if adaptive:
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 10)
    else:
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw


def _runs(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """一维布尔数组中连续 True 的区间。"""
    out = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= min_len:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


# ---------------- 线框检测 ----------------

def detect_border_boxes(bw: np.ndarray, cfg: dict) -> list[Region]:
    """提取版面中的矩形线框。这是「方框式一片属于一个」的落地方式。"""
    import cv2

    H, W = bw.shape
    h_len = max(20, W // cfg["h_kernel_div"])
    v_len = max(20, H // cfg["v_kernel_div"])

    horiz = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1)))
    vert = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len)))
    lines = cv2.bitwise_or(horiz, vert)
    # 闭合线框缺口，否则断开的边无法组成闭合轮廓
    lines = cv2.dilate(lines, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(lines, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[Region] = []
    for c in contours:
        peri = cv2.arcLength(c, True)
        if peri <= 0:
            continue
        poly = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(poly) != 4 or not cv2.isContourConvex(poly):
            continue
        x, y, w, h = cv2.boundingRect(poly)
        if w < cfg["border_min_w_ratio"] * W or h < cfg["border_min_h_ratio"] * H:
            continue
        rect_area = w * h
        if rect_area < cfg["border_min_area_ratio"] * W * H:
            continue
        # 必须"够方"：轮廓面积接近外接矩形，否则是斜线/装饰
        if cv2.contourArea(c) / max(rect_area, 1) < cfg["border_rect_fill"]:
            continue
        boxes.append(Region(x, y, x + w, y + h, is_border=True))

    # 去嵌套：小框被大框 80% 包含则丢弃，保证一片只算一次
    boxes.sort(key=lambda r: r.area, reverse=True)
    kept: list[Region] = []
    for b in boxes:
        if any(_contain_ratio(b, k) > 0.8 for k in kept):
            continue
        kept.append(b)
    return kept


def _contain_ratio(inner: Region, outer: Region) -> float:
    ox = max(0.0, min(inner.x1, outer.x1) - max(inner.x0, outer.x0))
    oy = max(0.0, min(inner.y1, outer.y1) - max(inner.y0, outer.y0))
    if inner.area <= 0:
        return 0.0
    return (ox * oy) / inner.area


# ---------------- 投影递归切分 ----------------

def projection_split(bw: np.ndarray, cfg: dict) -> list[Region]:
    """经典 white-space / XY-cut 的像素版。

    不用纯连通域的原因：文字行会被切成上千个连通域，
    投影先把"行"聚起来更符合版面语义，对中文断行也更鲁棒。

    关键：投影前必须做**形态学闭运算**。
    否则汉字之间的 1~3px 缝隙会被当成栏缝，每个字都变成独立区块
    （实测一页切出 816 个 32×32 的碎片，全部被后续过滤掉，文字彻底丢失）。
      - 纵向闭运算：把同一段落的行连成一条带
      - 横向闭运算：把同一行的汉字连成一段
    """
    import cv2

    H, W = bw.shape
    row_k = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(5, int(H * cfg["row_close_ratio"]))))
    bw_rows = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, row_k)

    row_ink = (bw_rows > 0).sum(axis=1)
    row_thr = max(2, int(cfg["proj_ink_ratio"] * W))
    min_band = max(3, int(cfg["min_band_h_ratio"] * H))

    col_k = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(8, int(W * cfg["col_close_ratio"])), 1))

    out: list[Region] = []
    for y0, y1 in _runs(row_ink > row_thr, min_band):
        sub = bw[y0:y1, :]
        sub_c = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, col_k)
        col_ink = (sub_c > 0).sum(axis=0)
        col_thr = max(2, int(cfg["proj_ink_ratio"] * (y1 - y0)))
        col_runs = _runs(col_ink > col_thr, 2)
        if not col_runs:
            continue
        for x0, x1 in col_runs:
            out.append(Region(x0, y0, x1, y1))
    return out


# ---------------- 合并 ----------------

def _v_gap(a: Region, b: Region) -> float:
    if b.y0 >= a.y1:
        return b.y0 - a.y1
    if a.y0 >= b.y1:
        return a.y0 - b.y1
    return 0.0


def _x_overlap_ratio(a: Region, b: Region) -> float:
    ox = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    return ox / max(min(a.width, b.width), 1.0)


def merge_regions(regions: list[Region], cfg: dict) -> list[Region]:
    """合并被过度切碎的碎片。

    线框块（is_border）不参与合并——它本身就是「一片一个」的原子单元。
    """
    free = [r for r in regions if not r.is_border]
    borders = [r for r in regions if r.is_border]

    changed = True
    while changed:
        changed = False
        for i in range(len(free)):
            for j in range(i + 1, len(free)):
                a, b = free[i], free[j]
                gap = _v_gap(a, b)
                if gap > cfg["merge_gap_ratio"] * min(a.height, b.height):
                    continue
                if _x_overlap_ratio(a, b) < cfg["merge_xover"]:
                    continue
                # 合并
                a.x0 = min(a.x0, b.x0)
                a.y0 = min(a.y0, b.y0)
                a.x1 = max(a.x1, b.x1)
                a.y1 = max(a.y1, b.y1)
                free.pop(j)
                changed = True
                break
            if changed:
                break

    return borders + free


def drop_tiny(regions: list[Region], cfg: dict, page_area: float) -> list[Region]:
    min_area = cfg["min_region_area_ratio"] * page_area
    return [r for r in regions if r.area >= min_area]


# ---------------- 分类 ----------------

def region_stats(r: Region, gray: np.ndarray, boxes: list[TextBox]) -> dict:
    """计算一个区域的特征。"""
    H, W = gray.shape
    x0, y0 = max(0, int(r.x0)), max(0, int(r.y0))
    x1, y1 = min(W, int(r.x1)), min(H, int(r.y1))
    patch = gray[y0:y1, x0:x1] if x1 > x0 and y1 > y0 else np.zeros((1, 1), np.uint8)

    inside = [b for b in boxes if _center_in(b, r)]
    chars = sum(len(b.text) for b in inside)
    box_area = sum(b.area for b in inside)

    ink = float((patch < 128).mean()) if patch.size else 0.0
    std = float(patch.std()) if patch.size else 0.0

    return {
        "ocr_chars": chars,
        "text_area_ratio": min(1.0, box_area / max(r.area, 1.0)),
        "ink_ratio": ink,
        "gray_std": std,
        "lines": [b.text for b in inside],
        "y0": r.y0,
        "y1": r.y1,
    }


def _center_in(b: TextBox, r: Region) -> bool:
    return r.x0 <= b.xc <= r.x1 and r.y0 <= b.yc <= r.y1


def classify_region(r: Region, st: dict, H: int, cfg: dict) -> str:
    """判定 text / figure / toc / footer。"""
    # 页眉页脚 / 页码
    if (r.y0 < cfg["header_ratio"] * H or r.y0 > cfg["footer_ratio"] * H) \
            and st["ocr_chars"] <= 12:
        return "footer"

    lines = [l for l in st["lines"] if l]
    # 目录：多行且行尾多为页码
    if len(lines) >= 8:
        digit_tail = sum(1 for l in lines if re.search(r"[.·…\s]*\d{1,3}$", l))
        if digit_tail / len(lines) > 0.5:
            return "toc"

    if st["ocr_chars"] == 0:
        return "figure"
    if st["text_area_ratio"] >= cfg["text_area_min"] and \
            st["ocr_chars"] >= cfg["min_ocr_chars"]:
        return "text"
    if st["text_area_ratio"] < cfg["fig_area_max"]:
        return "figure"
    return "text" if st["ocr_chars"] >= 15 else "figure"


# ---------------- 图文关联 ----------------

def associate_text(fig: Region, text_regions: list[Region], cfg: dict) -> str:
    """给图区找说明文字：取正上方、正下方最近且横向重叠足够的文字区。"""
    page_h = max((t.y1 for t in text_regions), default=1.0) or 1.0
    max_gap = max(page_h * cfg["assoc_max_gap_ratio"], 12.0)

    above: list[tuple[float, str]] = []
    below: list[tuple[float, str]] = []
    for t in text_regions:
        if not t.ocr_text.strip() or is_boilerplate(t.ocr_text):
            continue
        if _x_overlap_ratio(fig, t) < cfg["assoc_min_xover"]:
            continue
        if t.y1 <= fig.y0 + 4:
            gap = fig.y0 - t.y1
            if gap <= max_gap:
                above.append((gap, t.ocr_text))
        elif t.y0 >= fig.y1 - 4:
            gap = t.y0 - fig.y1
            if gap <= max_gap:
                below.append((gap, t.ocr_text))

    above.sort(key=lambda x: x[0])
    below.sort(key=lambda x: x[0])
    picks = []
    if above:
        picks.append((0 if REF_PAT.search(above[0][1]) else 1, above[0][0], above[0][1]))
    if below:
        picks.append((0 if REF_PAT.search(below[0][1]) else 1, below[0][0], below[0][1]))
    picks.sort(key=lambda x: (x[0], x[1]))
    return clean(" ".join(p[2] for p in picks[:2]))[:300]


# ---------------- 主入口 ----------------

def split_scan_page(page, ocr, doc_id: str, doc_title: str, pno: int,
                    save_figure, cfg: dict | None = None,
                    ) -> tuple[list[Element], list[Figure], dict]:
    """把一页扫描页切成元素流。

    save_figure(pdf_rect, fid) -> (rel_path, w, h)：由调用方负责落盘。
    返回 (elements, figures, info)
    """
    import cv2

    cfg = {**DEFAULT_CFG, **(cfg or {})}
    dpi = cfg["render_dpi"]
    rgb = render_page(page, dpi)
    H, W = rgb.shape[:2]
    scale = 72.0 / dpi                      # 像素 → PDF 点
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    bw = binarize(rgb)

    info = {"regions": 0, "figures": 0, "texts": 0, "toc": 0, "footer": 0,
            "fallback": False, "borders": 0}

    # OCR：直接喂 numpy，避免落盘
    boxes: list[TextBox] = ocr.ocr_boxes(rgb) if ocr is not None else []

    borders = detect_border_boxes(bw, cfg)
    info["borders"] = len(borders)

    # 线框区域涂白，保证切缝不会落进框内
    bw_masked = bw.copy()
    for b in borders:
        cv2.rectangle(bw_masked,
                      (int(b.x0), int(b.y0)), (int(b.x1), int(b.y1)), 0, -1)

    free = projection_split(bw_masked, cfg)
    regions = merge_regions(free + borders, cfg)
    regions = drop_tiny(regions, cfg, W * H)

    # 安全阀：切得太碎说明版面判断失效，回退整页
    if len(regions) > cfg["max_regions"]:
        info["fallback"] = True
        return _fallback_whole_page(page, ocr, doc_id, doc_title, pno, save_figure, cfg, info)

    # 分类
    for i, r in enumerate(regions, 1):
        st = region_stats(r, gray, boxes)
        r.kind = classify_region(r, st, H, cfg)
        r.region_no = i
        r.ocr_text = clean(" ".join(st["lines"])) if st["lines"] else ""

    texts = [r for r in regions if r.kind == "text"]
    figs = [r for r in regions if r.kind == "figure"]
    info["regions"] = len(regions)
    info["texts"] = len(texts)
    info["toc"] = sum(1 for r in regions if r.kind == "toc")
    info["footer"] = sum(1 for r in regions if r.kind == "footer")

    # 图文关联
    for f in figs:
        f.nearby_text = associate_text(f, texts, cfg)

    # 组装元素
    elements: list[Element] = []
    figures: list[Figure] = []
    for r in regions:
        if r.kind in ("footer",):
            continue
        bbox_pdf = (round(r.x0 * scale, 1), round(r.y0 * scale, 1),
                    round(r.x1 * scale, 1), round(r.y1 * scale, 1))
        if r.kind in ("text", "toc"):
            if r.ocr_text:
                elements.append(Element(
                    kind="text", page=pno, bbox=bbox_pdf, text=r.ocr_text,
                    font_size=8.0, order=float(r.region_no)))
            continue

        fid = "%s_p%03d_r%02d" % (doc_id, pno, r.region_no)
        rel, w, h = save_figure(
            (r.x0 * scale, r.y0 * scale, r.x1 * scale, r.y1 * scale), fid)
        figures.append(Figure(
            figure_id=fid, doc_id=doc_id, page=pno, bbox=bbox_pdf,
            path=rel, width=w, height=h, source="page-scan",
            region_kind="figure", region_no=r.region_no,
            ocr_text=r.ocr_text, nearby_text=r.nearby_text,
            doc_title=doc_title,
        ))
        elements.append(Element(
            kind="figure", page=pno, bbox=bbox_pdf,
            figure_id=fid, order=float(r.region_no)))

    info["figures"] = len(figures)
    return elements, figures, info


def _fallback_whole_page(page, ocr, doc_id, doc_title, pno, save_figure,
                         cfg, info) -> tuple[list[Element], list[Figure], dict]:
    """切分失效时的兜底：整页 OCR + 整页一张图，保证不丢数据。"""
    rect = (0.0, 0.0, page.rect.width, page.rect.height)
    fid = "%s_p%03d_scan" % (doc_id, pno)
    rel, w, h = save_figure(rect, fid)

    text = ""
    if ocr is not None:
        rgb = render_page(page, cfg["render_dpi"])
        boxes = ocr.ocr_boxes(rgb)
        text = clean(" ".join(b.text for b in boxes))

    elements: list[Element] = []
    figures: list[Figure] = []
    if text.strip():
        elements.append(Element(kind="text", page=pno, bbox=rect,
                                text=text.strip(), font_size=8.0, order=0.0))
    figures.append(Figure(
        figure_id=fid, doc_id=doc_id, page=pno,
        bbox=(0.0, 0.0, round(page.rect.width, 1), round(page.rect.height, 1)),
        path=rel, width=w, height=h, source="page-scan",
        region_kind="figure", ocr_text=text, doc_title=doc_title,
    ))
    elements.append(Element(kind="figure", page=pno, bbox=rect,
                            figure_id=fid, order=1.0))
    info["figures"] = 1
    return elements, figures, info
