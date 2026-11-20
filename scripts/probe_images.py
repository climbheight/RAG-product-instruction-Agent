# -*- coding: utf-8 -*-
"""Deep probe: how are images embedded? xref images vs vector drawings vs inline."""
import sys, os
import pymupdf

path = sys.argv[1]
doc = pymupdf.open(path)

print("FILE:", os.path.basename(path), "| pages:", doc.page_count)

for i in range(min(doc.page_count, 6)):
    page = doc[i]
    imgs = page.get_images(full=True)           # xref-based embedded images
    infos = page.get_image_info()               # placed images with bbox
    drawings = page.get_drawings()              # vector paths
    blocks_img = [b for b in page.get_text("dict")["blocks"] if b["type"] == 1]

    print("\n--- page %d ---" % (i + 1))
    print("  get_images(full=True): %d" % len(imgs))
    for im in imgs[:6]:
        # im = (xref, smask, width, height, bpc, colorspace, alt_cs, name, filter, referencer)
        print("     xref=%s %dx%d bpc=%s cs=%s filter=%s" % (im[0], im[2], im[3], im[4], im[5], im[8]))
    print("  get_image_info(): %d" % len(infos))
    for inf in infos[:6]:
        bb = [round(v) for v in inf["bbox"]]
        print("     bbox=%s %dx%d" % (bb, inf["width"], inf["height"]))
    print("  blocks type==1: %d" % len(blocks_img))
    print("  get_drawings(): %d paths" % len(drawings))
    # vector drawing bounding area summary
    if drawings:
        big = 0
        for d in drawings:
            r = d["rect"]
            if r.width * r.height > 5000:
                big += 1
        print("     paths with area>5000pt2: %d" % big)

doc.close()
