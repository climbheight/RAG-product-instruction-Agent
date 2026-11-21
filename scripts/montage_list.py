# -*- coding: utf-8 -*-
"""按给定的 figure_id 列表顺序拼图，用于逐条复核图注。

与 annotate_batch.py 的区别：那个按「文本弱度」排序，这个按你给的顺序，
方便配合 figure_context.py 打印的上下文列表一一对应。

用法：
    python scripts/montage_list.py --doc aito_m9_cockpit --risky 4 --offset 0
    python scripts/montage_list.py --doc aito_m9_cockpit --ids p040_f04,p042_f02
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.ingestion.annotate import make_montage  # noqa: E402

EXTRACTED = os.path.join(ROOT, "data", "extracted")
OUT = os.path.join(EXTRACTED, "_annot")


def norm(s: str) -> str:
    return re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9]", "", s or "")


def bigrams(s: str) -> set[str]:
    s = norm(s)
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) > 1 else set()


def overlap(a: str, b: str) -> float:
    A, B = bigrams(a), bigrams(b)
    if not A or not B:
        return 0.0
    return len(A & B) / min(len(A), len(B))


def risky_figures(doc: str) -> list[dict]:
    fp = os.path.join(EXTRACTED, doc, "figures.jsonl")
    figs = [json.loads(l) for l in open(fp, encoding="utf-8") if l.strip()]
    out = []
    for f in figs:
        cap = f.get("vlm_caption") or ""
        if not cap or cap.startswith("无效图块"):
            continue
        if (f.get("duplicate_of") or "").strip():
            continue
        if f.get("review_ok"):
            continue
        if overlap(cap, f.get("section")) < 0.05 and \
                overlap(cap, f.get("nearby_text")) < 0.05:
            out.append(f)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", required=True)
    ap.add_argument("--ids", default=None, help="逗号分隔的 figure_id 或短号")
    ap.add_argument("--risky", type=int, default=4)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--cell", type=int, default=620)
    args = ap.parse_args()

    if args.ids:
        short = [s.strip() for s in args.ids.split(",") if s.strip()]
        figs = []
        fp = os.path.join(EXTRACTED, args.doc, "figures.jsonl")
        allf = {json.loads(l)["figure_id"]: json.loads(l)
                for l in open(fp, encoding="utf-8") if l.strip()}
        for s in short:
            fid = s if s.startswith(args.doc) else args.doc + "_" + s
            if fid in allf:
                figs.append(allf[fid])
            else:
                print("未找到 %s" % fid)
    else:
        figs = risky_figures(args.doc)[args.offset:args.offset + args.risky]

    if not figs:
        print("没有图")
        return 1

    items = []
    for i, f in enumerate(figs, 1):
        p = os.path.join(EXTRACTED, args.doc, f["path"])
        items.append((i, p))
        print("%d %-34s [%s] %s" % (
            i, f["figure_id"].replace(args.doc + "_", ""),
            (f.get("section") or "-")[:12], (f.get("vlm_caption") or "")[:44]))

    out = os.path.join(OUT, "review_%s_%03d.png" % (args.doc, args.offset))
    make_montage(items, out, cols=2 if len(items) <= 4 else 3, cell=args.cell)
    print("\nmontage:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
