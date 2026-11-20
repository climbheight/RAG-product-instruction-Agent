# -*- coding: utf-8 -*-
"""批量标注辅助：按「文本最弱」排序取一批图，拼成九宫格，输出编号→figure_id 映射。

用法：
    python scripts/annotate_batch.py --limit 9 --offset 0
    python scripts/annotate_batch.py --doc aito_m9_cockpit --limit 9
    python scripts/annotate_batch.py --status          # 只看进度

标注流程：
    1. 本脚本生成 montage + 打印映射表
    2. 用视觉看 montage，按编号写描述
    3. python scripts/apply_captions.py --map captions.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.ingestion.annotate import load_figures, make_montage  # noqa: E402

EXTRACTED = os.path.join(ROOT, "data", "extracted")
OUT = os.path.join(EXTRACTED, "_annot")


def collect(doc: str | None = None) -> list[dict]:
    rows = []
    if doc:
        dirs = [doc]
    else:
        dirs = [d for d in sorted(os.listdir(EXTRACTED))
                if os.path.isdir(os.path.join(EXTRACTED, d)) and not d.startswith("_")]
    for d in dirs:
        for r in load_figures(EXTRACTED, d):
            r["_doc"] = d
            rows.append(r)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default=None)
    ap.add_argument("--limit", type=int, default=9)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--all", action="store_true", help="不按弱度排序，按原顺序")
    args = ap.parse_args()

    rows = collect(args.doc)
    total = len(rows)
    done = sum(1 for r in rows if (r.get("vlm_caption") or "").strip())

    if args.status:
        print("总图数: %d   已标注: %d   剩余: %d (%.0f%%)"
              % (total, done, total - done, 100.0 * done / max(total, 1)))
        by_doc: dict[str, list[int]] = {}
        for r in rows:
            if (r.get("duplicate_of") or "").strip():
                continue                      # 重复图不计入分母
            a, t = by_doc.setdefault(r["_doc"], [0, 0])
            by_doc[r["_doc"]] = [a + (1 if (r.get("vlm_caption") or "").strip() else 0), t + 1]
        for d, (a, t) in sorted(by_doc.items()):
            print("  %-28s %3d/%-3d" % (d, a, t))
        return 0

    # 跳过重复图：duplicate_of 已在 dedup_figures.py 里判定，只读代表图
    todo = [r for r in rows
            if not (r.get("vlm_caption") or "").strip()
            and not (r.get("duplicate_of") or "").strip()]
    if not args.all:
        todo.sort(key=lambda r: len(r.get("search_text") or ""))
    batch = todo[args.offset: args.offset + args.limit]
    if not batch:
        print("没有待标注的图了")
        return 0

    items = []
    mapping = {}
    for i, r in enumerate(batch, 1):
        p = os.path.join(EXTRACTED, r["_doc"], r["path"])
        if not os.path.exists(p):
            continue
        items.append((i, p))
        mapping[str(i)] = {
            "figure_id": r["figure_id"],
            "doc": r["_doc"],
            "section": r.get("section", ""),
            "nearby": (r.get("nearby_text") or "")[:60],
        }

    out = os.path.join(OUT, "batch_%s_%03d.png"
                       % (args.doc or "all", args.offset))
    make_montage(items, out, cols=3, cell=340)
    print("montage:", out)
    print("映射:")
    print(json.dumps(mapping, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
