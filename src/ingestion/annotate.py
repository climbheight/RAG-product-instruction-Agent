"""自动化标注：把多张图拼成带编号的九宫格，供 VLM 一次看多张。

为什么这么干：16 份手册预计 400~600 张图，一张一张看成本太高。
拼成 3×3 的接触印相（contact sheet），每格左上角画 figure 序号，
VLM 一次读一张图就能给出 9 条描述，调用次数降到 1/9。

流程：
    1. make_montage()   → 生成 montage_XX.png
    2. VLM 读图 → 输出 {"1": "描述", "2": "描述", ...}
    3. apply_captions() → 写回 figures.jsonl 的 vlm_caption 字段
"""
from __future__ import annotations

import json
import os
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from .search_text import build_figure_search_text

CELL = 360          # 每格边长（像素）
PAD = 6
LABEL_H = 22
BG = (255, 255, 255)
FG = (0, 0, 0)


def _font(size: int = 16):
    for name in ("msyh.ttc", "simhei.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def make_montage(items: list[tuple[int, str]], out_path: str,
                 cols: int = 3, cell: int = CELL) -> str:
    """items: [(序号, 图片路径), ...]。返回生成的拼图路径。"""
    if not items:
        raise ValueError("no items")
    rows = (len(items) + cols - 1) // cols
    W = cols * (cell + PAD) + PAD
    H = rows * (cell + PAD + LABEL_H) + PAD
    sheet = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(sheet)
    font = _font(18)

    for idx, (no, path) in enumerate(items):
        r, c = divmod(idx, cols)
        x = PAD + c * (cell + PAD)
        y = PAD + r * (cell + PAD + LABEL_H)

        # 序号标签
        draw.rectangle([x, y, x + 46, y + LABEL_H - 2], fill=(230, 230, 230))
        draw.text((x + 6, y + 2), str(no), fill=FG, font=font)

        try:
            im = Image.open(path).convert("RGB")
        except Exception:
            draw.text((x + 60, y + 2), "(无法打开)", fill=(200, 0, 0), font=font)
            continue

        # 等比缩放进格子，留白居中
        im.thumbnail((cell, cell), Image.LANCZOS)
        ox = x + (cell - im.width) // 2
        oy = y + LABEL_H + (cell - im.height) // 2
        sheet.paste(im, (ox, oy))
        # 格子边框，便于分辨边界
        draw.rectangle([x, y + LABEL_H, x + cell, y + LABEL_H + cell],
                       outline=(200, 200, 200))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sheet.save(out_path)
    return out_path


def load_figures(extracted_root: str, doc_id: str) -> list[dict]:
    p = os.path.join(extracted_root, doc_id, "figures.jsonl")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def apply_captions(extracted_root: str, doc_id: str, captions: dict[str, str]) -> int:
    """把 VLM 描述写回 figures.jsonl。返回更新条数。"""
    p = os.path.join(extracted_root, doc_id, "figures.jsonl")
    rows = load_figures(extracted_root, doc_id)
    n = 0
    for r in rows:
        cap = captions.get(r["figure_id"])
        if cap:
            r["vlm_caption"] = cap
            # 描述变了，可检索文本必须重算，否则嵌入的还是旧文本
            r["search_text"] = _rebuild_search_text(r)
            n += 1
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return n


def _rebuild_search_text(row: dict) -> str:
    """按 figures.jsonl 的一行重算 search_text（含 vlm_caption）。"""
    class _F:
        pass
    f = _F()
    for k in ("doc_title", "section", "page", "nearby_text", "caption",
              "ocr_text", "vlm_caption"):
        setattr(f, k, row.get(k, "") or (0 if k == "page" else ""))
    return build_figure_search_text(f)
