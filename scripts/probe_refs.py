# -*- coding: utf-8 -*-
"""Probe: (a) in-text figure references, (b) raster manual layout interleaving."""
import os, re
import pymupdf

PDF_DIR = r"E:\code\软件课设智能体\data\raw_pdfs"

REF_PAT = re.compile(r"(如图|如下图|上图|下图|见图|参照图|图中|图示|示意图|右图|左图)")

print("### IN-TEXT FIGURE REFERENCES ###")
for fn in sorted(os.listdir(PDF_DIR)):
    if not fn.endswith(".pdf"):
        continue
    doc = pymupdf.open(os.path.join(PDF_DIR, fn))
    cnt = 0
    samples = []
    for pno, page in enumerate(doc, 1):
        t = page.get_text("text")
        for m in REF_PAT.finditer(t):
            cnt += 1
            if len(samples) < 2:
                s = t[max(0, m.start() - 25): m.start() + 30].replace("\n", " ")
                samples.append("p%d %s" % (pno, s))
    doc.close()
    print("%-44s refs=%3d  %s" % (fn[:44], cnt, " | ".join(samples[:1])))

print("\n\n### RASTER MANUAL LAYOUT (华为 Pura70, page 30-31) ###")
doc = pymupdf.open(os.path.join(PDF_DIR, "华为_Pura70_Ultra_HarmonyOS5_中文用户指南.pdf"))
for pno in [30, 31]:
    page = doc[pno - 1]
    print("\n--- page %d  (%.0fx%.0f) ---" % (pno, page.rect.width, page.rect.height))
    blocks = page.get_text("dict")["blocks"]
    items = []
    for b in blocks:
        x0, y0, x1, y1 = b["bbox"]
        if b["type"] == 1:
            items.append((y0, x0, "IMG ", "w=%d h=%d" % (x1 - x0, y1 - y0)))
        else:
            txt = " ".join(s["text"] for l in b.get("lines", []) for s in l.get("spans", []))
            if txt.strip():
                items.append((y0, x0, "TEXT", txt[:64].replace("\n", " ")))
    items.sort(key=lambda t: (t[0], t[1]))
    for y, x, kind, txt in items:
        print("  y=%4d x=%4d [%s] %s" % (y, x, kind, txt))
doc.close()
