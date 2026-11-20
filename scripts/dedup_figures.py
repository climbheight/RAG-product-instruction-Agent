# -*- coding: utf-8 -*-
"""图片去重：找出重复/近似重复的插图，只保留一张作为代表。

手册里同一张图经常在多个章节重复出现（如 App 下载页、华为分享示意、
配件清单）。每张都读一遍是浪费，标注时只读代表图，其余继承描述。

判据：dHash + aHash 双哈希，汉明距离 ≤ 阈值即判为重复。
不用深度学习，纯 PIL + numpy，快且够用。

用法：
    python scripts/dedup_figures.py --report          # 只统计，不改文件
    python scripts/dedup_figures.py --apply           # 写回 duplicate_of 字段
    python scripts/dedup_figures.py --apply --prune   # 同时删除重复的 png
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRACTED = os.path.join(ROOT, "data", "extracted")


def dhash(path: str, size: int = 8) -> np.ndarray | None:
    """差分哈希：相邻像素亮度比较，对缩放/亮度变化不敏感。"""
    try:
        im = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    except Exception:
        return None
    a = np.asarray(im, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def ahash(path: str, size: int = 8) -> np.ndarray | None:
    try:
        im = Image.open(path).convert("L").resize((size, size), Image.LANCZOS)
    except Exception:
        return None
    a = np.asarray(im, dtype=np.float32)
    return (a > a.mean()).flatten()


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def pixel_mismatch(path_a: str, path_b: str, size: int = 64,
                   tol: int = 25) -> float:
    """缩放后逐像素比较，返回差异像素占比。

    为什么哈希之外还要这一道：哈希只看结构，改个蓝箭头、换块高亮区域，
    哈希几乎不变，但图的意思完全不同。
    实测「侧气帘（整片蓝色高亮）」和「侧气囊（蓝色箭头指座椅侧面）」
    哈希距离很小，但绝不是同一张图——把描述从前者继承给后者就是错的。
    """
    try:
        a = Image.open(path_a).convert("L").resize((size, size), Image.LANCZOS)
        b = Image.open(path_b).convert("L").resize((size, size), Image.LANCZOS)
    except Exception:
        return 1.0
    aa = np.asarray(a, dtype=np.int16)
    bb = np.asarray(b, dtype=np.int16)
    return float((np.abs(aa - bb) > tol).mean())


def load_all() -> list[dict]:
    rows = []
    for d in sorted(os.listdir(EXTRACTED)):
        p = os.path.join(EXTRACTED, d, "figures.jsonl")
        if not os.path.isdir(os.path.join(EXTRACTED, d)) or not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            r["_doc"] = d
            rows.append(r)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="写回 duplicate_of")
    ap.add_argument("--prune", action="store_true", help="删除重复 png（需配合 --apply）")
    ap.add_argument("--dhash-max", type=int, default=3)
    ap.add_argument("--ahash-max", type=int, default=2)
    ap.add_argument("--mismatch-max", type=float, default=0.002,
                    help="缩放后差异像素占比上限，默认 0.2%%")
    # 阈值标定依据（实测）：
    #   ETC 设备照 真重复         mismatch=0.0000  → 收
    #   安全带 箭头一上一下       mismatch=0.0044  → 拒（语义不同）
    #   侧气帘 vs 侧气囊          mismatch=0.0403  → 拒（完全不同的气囊）
    args = ap.parse_args()

    rows = load_all()
    print("总图数:", len(rows))

    hashes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for r in rows:
        p = os.path.join(EXTRACTED, r["_doc"], r["path"])
        dh = dhash(p)
        ah = ahash(p)
        if dh is not None and ah is not None:
            hashes[r["figure_id"]] = (dh, ah)

    print("可计算哈希:", len(hashes))

    # 并查集聚类
    parent = {fid: fid for fid in hashes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    by_id_all = {r["figure_id"]: r for r in rows}
    ids = list(hashes)
    for i in range(len(ids)):
        a = ids[i]
        for j in range(i + 1, len(ids)):
            b = ids[j]
            if hamming(hashes[a][0], hashes[b][0]) > args.dhash_max:
                continue
            if hamming(hashes[a][1], hashes[b][1]) > args.ahash_max:
                continue
            # 最后一道：像素级差异。哈希过了但画面变了（改箭头/换高亮）的不算重复
            pa = os.path.join(EXTRACTED, by_id_all[a]["_doc"], by_id_all[a]["path"])
            pb = os.path.join(EXTRACTED, by_id_all[b]["_doc"], by_id_all[b]["path"])
            if pixel_mismatch(pa, pb) > args.mismatch_max:
                continue
            union(a, b)

    groups: dict[str, list[str]] = {}
    for fid in ids:
        groups.setdefault(find(fid), []).append(fid)

    dup_groups = {k: v for k, v in groups.items() if len(v) > 1}
    n_dup = sum(len(v) - 1 for v in dup_groups.values())
    print("重复组数: %d   可省去的重复图: %d   去重后需标注: %d"
          % (len(dup_groups), n_dup, len(hashes) - n_dup))

    by_id = {r["figure_id"]: r for r in rows}
    # 每组选页码最小的作为代表
    dup_map: dict[str, str] = {}
    for members in dup_groups.values():
        members.sort(key=lambda f: (by_id[f]["page"], f))
        canon = members[0]
        for m in members[1:]:
            dup_map[m] = canon

    print("\n重复组示例（最多 12 组）:")
    shown = 0
    for members in dup_groups.values():
        if shown >= 12:
            break
        members.sort(key=lambda f: (by_id[f]["page"], f))
        canon = members[0]
        cr = by_id[canon]
        print("  代表 %-34s p%-3d %s" % (canon, cr["page"], cr["section"]))
        for m in members[1:]:
            print("       重复 %-34s p%-3d" % (m, by_id[m]["page"]))
        shown += 1

    if not args.apply:
        print("\n（仅统计，未改文件。加 --apply 写回）")
        return 0

    # 写回 duplicate_of，并给重复图打上继承的标注
    for r in rows:
        fid = r["figure_id"]
        if fid in dup_map:
            r["duplicate_of"] = dup_map[fid]
            canon = by_id[dup_map[fid]]
            cap = (canon.get("vlm_caption") or "").strip()
            if cap and not cap.startswith("无效图块"):
                r["vlm_caption"] = "%s（与第 %d 页重复）" % (cap, canon["page"])
            elif not cap:
                r["vlm_caption"] = ""

    for doc in sorted({r["_doc"] for r in rows}):
        p = os.path.join(EXTRACTED, doc, "figures.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                if r["_doc"] != doc:
                    continue
                d = {k: v for k, v in r.items() if k != "_doc"}
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print("\n已写回 duplicate_of: %d 条" % len(dup_map))

    if args.prune:
        freed = 0
        for fid in dup_map:
            r = by_id[fid]
            p = os.path.join(EXTRACTED, r["_doc"], r["path"])
            if os.path.exists(p):
                freed += os.path.getsize(p)
                os.remove(p)
        print("已删除重复 png: %d 个，释放 %.1f MB" % (len(dup_map), freed / 1e6))

    return 0


if __name__ == "__main__":
    sys.exit(main())
