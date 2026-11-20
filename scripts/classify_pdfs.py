# -*- coding: utf-8 -*-
"""Batch classify all 16 manuals: text layer / raster images / vector drawings."""
import os, sys
import pymupdf

PDF_DIR = r"E:\code\软件课设智能体\data\raw_pdfs"

rows = []
for fn in sorted(os.listdir(PDF_DIR)):
    if not fn.lower().endswith(".pdf"):
        continue
    path = os.path.join(PDF_DIR, fn)
    try:
        doc = pymupdf.open(path)
    except Exception as e:
        rows.append((fn, "OPEN-ERR", str(e), "", "", ""))
        continue

    n_pages = doc.page_count
    total_chars = 0
    raster_imgs = 0
    vector_pages = 0
    drawing_total = 0
    scanned_pages = 0
    page_sizes = set()

    for page in doc:
        txt = page.get_text("text").strip()
        total_chars += len(txt)
        if len(txt) < 50:
            scanned_pages += 1
        ni = len(page.get_images(full=True))
        raster_imgs += ni
        nd = len(page.get_drawings())
        drawing_total += nd
        if nd > 50:
            vector_pages += 1
        page_sizes.add((round(page.rect.width), round(page.rect.height)))

    doc.close()
    kind = []
    if total_chars < 100:
        kind.append("SCAN")
    if raster_imgs > 0:
        kind.append("RASTER")
    if vector_pages > n_pages * 0.3:
        kind.append("VECTOR")
    if not kind:
        kind.append("TEXT-ONLY")

    rows.append((fn, "+".join(kind), n_pages, total_chars,
                 raster_imgs, vector_pages, scanned_pages,
                 ",".join("%dx%d" % s for s in sorted(page_sizes))))

print("%-46s %-18s %5s %8s %7s %7s %7s %s" % (
    "FILE", "KIND", "PAGES", "CHARS", "RASTER", "VECPG", "SCANPG", "SIZE"))
print("-" * 130)
for r in rows:
    if len(r) == 8:
        print("%-46s %-18s %5s %8s %7s %7s %7s %s" % r)
    else:
        print(r)
