# -*- coding: utf-8 -*-
"""把人工看图写好的描述写回 figures.jsonl，并重建 search_text。

用法：
    python scripts/apply_captions.py --doc dji_mini4pro --map captions.json
    python scripts/apply_captions.py --doc dji_mini4pro --map captions.json --dry-run

captions.json 格式：{"figure_id": "描述", ...}
只更新 map 里出现的 figure_id，其余行原样保留。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.ingestion.search_text import build_figure_search_text  # noqa: E402

EXTRACTED = os.path.join(ROOT, "data", "extracted")


class _Fig:
    pass


def rebuild(row: dict, caption: str) -> str:
    f = _Fig()
    for k in ("doc_title", "section", "page", "nearby_text", "caption",
              "ocr_text", "vlm_caption"):
        setattr(f, k, row.get(k, "") or (0 if k == "page" else ""))
    f.vlm_caption = caption
    return build_figure_search_text(f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(args.map, encoding="utf-8") as f:
        captions: dict[str, str] = json.load(f)

    p = os.path.join(EXTRACTED, args.doc, "figures.jsonl")
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

    hit, miss = 0, []
    for r in rows:
        cap = (captions.get(r["figure_id"]) or "").strip()
        if not cap:
            continue
        r["vlm_caption"] = cap
        r["search_text"] = rebuild(r, cap)
        hit += 1
    for fid in captions:
        if not any(r["figure_id"] == fid for r in rows):
            miss.append(fid)

    print("更新 %d 条，映射中未命中 %d 条" % (hit, len(miss)))
    for m in miss:
        print("  未找到:", m)
    if args.dry_run:
        print("dry-run，未写盘")
        return 0 if hit else 1
    if not hit:
        return 1

    with open(p, "w", encoding="utf-8", newline="") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("已写回:", p)
    return 0 if not miss else 1


if __name__ == "__main__":
    sys.exit(main())
