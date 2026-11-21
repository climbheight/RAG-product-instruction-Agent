# -*- coding: utf-8 -*-
"""回查单张图的完整上下文，用于校正图注。

九宫格只给 section + nearby 前 60 字，不够。本工具把
section / 完整 nearby_text / ocr_text / 关联 chunk 的完整正文 全部打出来。

用法：
    python scripts/figure_context.py aito_m9_cockpit_p015_f01
    python scripts/figure_context.py --doc aito_m9_cockpit --risky 20
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRACTED = os.path.join(ROOT, "data", "extracted")


def load_doc(doc: str):
    fp = os.path.join(EXTRACTED, doc, "figures.jsonl")
    cp = os.path.join(EXTRACTED, doc, "chunks.jsonl")
    figs = [json.loads(l) for l in open(fp, encoding="utf-8") if l.strip()]
    chunks = {}
    if os.path.exists(cp):
        for l in open(cp, encoding="utf-8"):
            if l.strip():
                c = json.loads(l)
                chunks[c.get("chunk_id")] = c
    return figs, chunks


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


def show(doc: str, fig: dict, chunks: dict) -> None:
    print("=" * 78)
    print("figure_id : %s" % fig["figure_id"])
    print("page      : %s   source: %s" % (fig.get("page"), fig.get("source")))
    print("section   : %s" % (fig.get("section") or "-"))
    print("-" * 78)
    print("【当前图注】%s" % (fig.get("vlm_caption") or "（空）"))
    print("-" * 78)
    print("【完整 nearby_text】")
    print((fig.get("nearby_text") or "（空）").strip())
    print("-" * 78)
    print("【ocr_text】")
    print((fig.get("ocr_text") or "（空）").strip()[:600])
    print("-" * 78)
    ids = fig.get("chunk_ids") or []
    print("【关联 chunk 正文】(%d 条)" % len(ids))
    for cid in ids:
        c = chunks.get(cid)
        if not c:
            print("  [%s] 未找到" % cid)
            continue
        txt = (c.get("text") or "").strip()
        print("  ── %s （第 %s 页，%d 字）" % (
            cid, c.get("page_start"), len(txt)))
        print("  " + txt.replace("\n", "\n  "))
    print()


def risky_list(doc: str, figs: list[dict], limit: int) -> list[dict]:
    ann = [f for f in figs
           if not (f.get("duplicate_of") or "").strip()
           and (f.get("vlm_caption") or "").strip()]
    out = []
    for f in ann:
        cap = f.get("vlm_caption") or ""
        if overlap(cap, f.get("section")) < 0.05 and \
                overlap(cap, f.get("nearby_text")) < 0.05:
            out.append(f)
    return out[:limit]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("figure_id", nargs="?", help="要回查的 figure_id")
    ap.add_argument("--doc", help="按文档列出高风险图注")
    ap.add_argument("--risky", type=int, default=20, help="列出前 N 条")
    args = ap.parse_args()

    if args.doc:
        figs, chunks = load_doc(args.doc)
        for f in risky_list(args.doc, figs, args.risky):
            print("%-36s [%s] %s" % (
                f["figure_id"], (f.get("section") or "-")[:14],
                (f.get("vlm_caption") or "")[:50]))
        return 0

    if not args.figure_id:
        ap.print_help()
        return 1

    fid = args.figure_id
    doc = fid.rsplit("_p", 1)[0]
    figs, chunks = load_doc(doc)
    hit = [f for f in figs if f["figure_id"] == fid]
    if not hit:
        print("未找到 %s" % fid)
        return 1
    show(doc, hit[0], chunks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
