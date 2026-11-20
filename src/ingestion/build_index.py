"""任务一编排：解析 → 切分 → 落盘（嵌入与向量库在下一步）。

用法：
    python -m src.ingestion.build_index                # 全部 16 份
    python -m src.ingestion.build_index --only roborock_g10s_pro huawei_pura70_ultra
    python -m src.ingestion.build_index --no-ocr       # 跳过 OCR（快，但扫描件无文字）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from .chunker import chunk_units, build_units
from .ocr import OcrEngine
from .parser import PdfParser
from .search_text import build_figure_search_text, clean, strip_fig_marks

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PDF_DIR = os.path.join(ROOT, "data", "raw_pdfs")
OUT_ROOT = os.path.join(ROOT, "data", "extracted")
CFG = os.path.join(ROOT, "configs", "manuals.json")

# 课程要求 300~600 字符、重叠 50~100。
# 但 bge-small-zh-v1.5 的输入窗口是 **512 token**，中文约 1 字 1 token，
# 600 字符会超窗被截断（尾部信息丢失，直接影响召回）。
# 因此把上界收到 480，仍落在课程要求的 300~600 区间内。
CHUNK_CFG = dict(target=400, min_size=300, max_size=480, overlap=80)


def load_manuals() -> dict:
    with open(CFG, encoding="utf-8") as f:
        return json.load(f)


def _load_existing_captions(out_dir: str) -> dict[str, str]:
    """读回已有的 vlm_caption，避免重跑时把逐批做的标注冲掉。"""
    p = os.path.join(out_dir, "figures.jsonl")
    if not os.path.exists(p):
        return {}
    caps: dict[str, str] = {}
    try:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("vlm_caption"):
                    caps[row["figure_id"]] = row["vlm_caption"]
    except Exception:
        return {}
    return caps


def run_one(fn: str, meta: dict, ocr: OcrEngine | None, verbose: bool = True,
            fig_collector: list | None = None) -> dict:
    pdf_path = os.path.join(PDF_DIR, fn)
    doc_id = meta["doc_id"]
    out_dir = os.path.join(OUT_ROOT, doc_id)
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    parser = PdfParser(
        doc_id=doc_id,
        doc_title=meta["product"],
        product=meta["product"],
        out_dir=out_dir,
        ocr=ocr if (ocr and ocr.available) else None,
    )
    elements, figures, reports = parser.parse(pdf_path)

    units = build_units(elements)
    chunks = chunk_units(units, doc_id, meta["product"], meta["product"], **CHUNK_CFG)

    # 邻近文本兜底：空间邻近没找到说明文字的图，用它所属 chunk 的文本回填。
    # 这段文本本来就是该图所在段落的上下文，比留空强得多。
    chunk_by_fig: dict[str, str] = {}
    for c in chunks:
        for fid in c.figure_ids:
            chunk_by_fig.setdefault(fid, c.text)
    for fig in figures:
        if not fig.nearby_text and fig.figure_id in chunk_by_fig:
            fig.nearby_text = clean(strip_fig_marks(chunk_by_fig[fig.figure_id]))[:220]

    # 保留已有的人工/VLM 标注：重跑不能把标注冲掉（标注是逐批做的，很贵）
    old_caps = _load_existing_captions(out_dir)
    for fig in figures:
        if fig.figure_id in old_caps:
            fig.vlm_caption = old_caps[fig.figure_id]

    # 构造图片的可检索文本（送嵌入用），必须在 nearby_text 补齐之后
    for fig in figures:
        fig.search_text = build_figure_search_text(fig)

    # 反向连接：哪些 chunk 引用了这张图
    fig_map = {f.figure_id: f for f in figures}
    for fig in figures:
        fig.embedding_ref = "manual_figures:%s" % fig.figure_id
    for c in chunks:
        for fid in c.figure_ids:
            fig = fig_map.get(fid)
            if fig is not None and c.chunk_id not in fig.chunk_ids:
                fig.chunk_ids.append(c.chunk_id)

    elapsed = time.time() - t0

    # 落盘
    with open(os.path.join(out_dir, "figures.jsonl"), "w", encoding="utf-8") as f:
        for fig in figures:
            f.write(json.dumps(fig.to_dict(), ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "chunks.jsonl"), "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    if fig_collector is not None:
        fig_collector.extend(figures)

    ocr_pages = [r.page for r in reports if r.ocr_used]
    stats = {
        "doc_id": doc_id,
        "file": fn,
        "product": meta["product"],
        "pages": len(reports),
        "chars": sum(r.chars for r in reports),
        "text_blocks": sum(r.n_text_blocks for r in reports),
        "figures": len(figures),
        "figure_sources": {
            s: sum(1 for x in figures if x.source == s)
            for s in ("raster", "vector", "page-scan")
        },
        "inline_icons_dropped": sum(r.n_inline_icons for r in reports),
        "multi_column_pages": sum(1 for r in reports if r.n_columns > 1),
        "ocr_pages": len(ocr_pages),
        "figures_with_nearby": sum(1 for x in figures if x.nearby_text),
        "figures_with_inner_text": sum(1 for x in figures if x.ocr_text),
        "figures_with_search_text": sum(1 for x in figures if x.search_text),
        "chunks": len(chunks),
        "chunks_with_figures": sum(1 for c in chunks if c.figure_ids),
        "avg_chunk_chars": round(
            sum(c.char_count for c in chunks) / max(len(chunks), 1), 1),
        "chunk_chars_min": min((c.char_count for c in chunks), default=0),
        "chunk_chars_max": max((c.char_count for c in chunks), default=0),
        "elapsed_sec": round(elapsed, 1),
    }
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "pages": [r.__dict__ for r in reports]},
                  f, ensure_ascii=False, indent=2)

    if verbose:
        print("  pages=%-4d chars=%-7d figs=%-4d (r=%-3d v=%-3d scan=%-2d) "
              "icons_dropped=%-4d 2col=%-3d chunks=%-4d  avg=%.0f  %.1fs"
              % (stats["pages"], stats["chars"], stats["figures"],
                 stats["figure_sources"]["raster"], stats["figure_sources"]["vector"],
                 stats["figure_sources"]["page-scan"], stats["inline_icons_dropped"],
                 stats["multi_column_pages"], stats["chunks"],
                 stats["avg_chunk_chars"], elapsed))
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="任务一：手册解析与切分")
    ap.add_argument("--only", nargs="*", help="只跑指定 doc_id")
    ap.add_argument("--no-ocr", action="store_true", help="跳过 OCR")
    args = ap.parse_args(argv)

    manuals = load_manuals()
    ocr = None if args.no_ocr else OcrEngine()
    if ocr is not None:
        print("OCR backend:", ocr.backend)
        if not ocr.available:
            print("!! OCR 不可用，扫描件将无文字（宇树G1 / 美的电饭煲）")

    targets = [(fn, m) for fn, m in manuals.items()
               if not args.only or m["doc_id"] in args.only]
    missing = [fn for fn, _ in targets if not os.path.exists(os.path.join(PDF_DIR, fn))]
    if missing:
        print("缺失文件:", *missing, sep="\n  ")
        return 1

    print("待处理 %d 份\n" % len(targets))
    all_stats = []
    all_figs: list = []
    for i, (fn, meta) in enumerate(targets, 1):
        print("[%2d/%d] %s" % (i, len(targets), meta["doc_id"]))
        try:
            all_stats.append(run_one(fn, meta, ocr, fig_collector=all_figs))
        except Exception as e:
            print("   FAILED:", type(e).__name__, e)
            import traceback
            traceback.print_exc()

    # 全局图片索引清单（供图片向量库构建使用）
    with open(os.path.join(OUT_ROOT, "figure_index.jsonl"), "w", encoding="utf-8") as f:
        for fig in all_figs:
            f.write(json.dumps(fig.to_dict(), ensure_ascii=False) + "\n")

    # 汇总
    total = {
        "manuals": len(all_stats),
        "pages": sum(s["pages"] for s in all_stats),
        "figures": sum(s["figures"] for s in all_stats),
        "chunks": sum(s["chunks"] for s in all_stats),
        "chunks_with_figures": sum(s["chunks_with_figures"] for s in all_stats),
        "ocr_pages": sum(s["ocr_pages"] for s in all_stats),
    }
    with open(os.path.join(OUT_ROOT, "index_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"total": total, "per_manual": all_stats}, f,
                  ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("合计: %(manuals)d 份 / %(pages)d 页 / %(figures)d 图 / %(chunks)d chunks"
          " (含图 %(chunks_with_figures)d) / OCR %(ocr_pages)d 页" % total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
