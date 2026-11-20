"""从 data/extracted 汇总产物，建立双索引向量库。

用法：
    python -m src.retrieval.build
    python -m src.retrieval.build --persist vectorstore --reset
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .embed import BgeEmbedder
from .vectorstore import DualIndex

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXTRACTED = os.path.join(ROOT, "data", "extracted")


def load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def collect() -> tuple[list[dict], list[dict]]:
    """汇总所有 doc 的 chunks 与 figures。"""
    chunks: list[dict] = []
    figures: list[dict] = []
    if not os.path.isdir(EXTRACTED):
        return chunks, figures
    for name in sorted(os.listdir(EXTRACTED)):
        d = os.path.join(EXTRACTED, name)
        if not os.path.isdir(d) or name.startswith("_"):
            continue
        chunks.extend(load_jsonl(os.path.join(d, "chunks.jsonl")))
        figures.extend(load_jsonl(os.path.join(d, "figures.jsonl")))
    return chunks, figures


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="建立文本 + 图片双索引")
    ap.add_argument("--persist", default=os.path.join(ROOT, "vectorstore"))
    ap.add_argument("--model", default=None, help="覆盖默认 BGE 模型路径")
    ap.add_argument("--no-reset", action="store_true", help="增量追加，不清空旧库")
    args = ap.parse_args(argv)

    chunks, figures = collect()
    print("chunks: %d   figures: %d" % (len(chunks), len(figures)))
    if not chunks and not figures:
        print("没有可建库的数据，先跑 python -m src.ingestion.build_index")
        return 1

    with_cap = sum(1 for f in figures if f.get("vlm_caption"))
    print("已标注图片: %d / %d (%.0f%%)"
          % (with_cap, len(figures), 100.0 * with_cap / max(len(figures), 1)))

    embedder = BgeEmbedder(args.model) if args.model else BgeEmbedder()
    print("加载嵌入模型:", embedder.model_name)
    idx = DualIndex(args.persist, embedder)
    stat = idx.build(chunks, figures, reset=not args.no_reset)
    print("入库完成:", stat)
    print("库内统计:", idx.stats())
    return 0


if __name__ == "__main__":
    sys.exit(main())
