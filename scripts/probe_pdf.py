# -*- coding: utf-8 -*-
"""Probe PDF structure: text layer, image blocks, captions, font sizes."""
import sys, os, re, json
import pymupdf

def probe(path):
    doc = pymupdf.open(path)
    print("=" * 78)
    print("FILE:", os.path.basename(path))
    print("pages:", doc.page_count)
    p0 = doc[0]
    print("page size: %.0f x %.0f pt" % (p0.rect.width, p0.rect.height))
    print("metadata:", {k: v for k, v in doc.metadata.items() if v})

    total_chars = 0
    empty_pages = []
    total_imgs = 0
    small_imgs = 0
    captions = []
    font_hist = {}

    for i, page in enumerate(doc):
        txt = page.get_text("text")
        n = len(txt.strip())
        total_chars += n
        if n < 50:
            empty_pages.append((i + 1, n))
        d = page.get_text("dict")
        for blk in d["blocks"]:
            if blk["type"] == 1:  # image block
                total_imgs += 1
                w = blk["bbox"][2] - blk["bbox"][0]
                h = blk["bbox"][3] - blk["bbox"][1]
                if w < 100 or h < 100:
                    small_imgs += 1
            else:
                for line in blk.get("lines", []):
                    for span in line.get("spans", []):
                        sz = round(span["size"], 1)
                        font_hist[sz] = font_hist.get(sz, 0) + len(span["text"])
                        t = span["text"].strip()
                        if re.match(r"^图\s*[\d一二三四五六七八九十]+[-–—.]?\d*", t):
                            if len(captions) < 12:
                                captions.append((i + 1, t[:60]))

    print("\ntotal text chars:", total_chars)
    print("avg chars/page: %.0f" % (total_chars / max(doc.page_count, 1)))
    print("image blocks: %d (small<100pt: %d)" % (total_imgs, small_imgs))
    print("pages with <50 chars:", len(empty_pages), empty_pages[:15])

    print("\nfont size histogram (top 8 by chars):")
    for sz, c in sorted(font_hist.items(), key=lambda x: -x[1])[:8]:
        print("   %5.1f pt : %6d chars" % (sz, c))

    print("\ncaption samples:")
    for pg, t in captions:
        print("   p%-4d %s" % (pg, t))

    # one page detail sample: blocks with bbox
    if doc.page_count > 2:
        page = doc[2]
        print("\n--- page 3 block layout ---")
        d = page.get_text("dict")
        blocks = sorted(d["blocks"], key=lambda b: (b["bbox"][1], b["bbox"][0]))
        for b in blocks[:14]:
            x0, y0, x1, y1 = [round(v) for v in b["bbox"]]
            if b["type"] == 1:
                print("   [IMG ] y=%4d-%4d x=%4d-%4d  w=%d h=%d" % (y0, y1, x0, x1, x1-x0, y1-y0))
            else:
                txt = " ".join(s["text"] for l in b.get("lines", []) for s in l.get("spans", []))
                print("   [TEXT] y=%4d-%4d  %s" % (y0, y1, txt[:70].replace("\n", " ")))
    doc.close()

for p in sys.argv[1:]:
    probe(p)
