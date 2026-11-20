# -*- coding: utf-8 -*-
"""Experiment: caption patterns across manuals + vector figure region rendering."""
import os, re, sys
import pymupdf

PDF_DIR = r"E:\code\软件课设智能体\data\raw_pdfs"
OUT = r"E:\code\软件课设智能体\data\extracted\_probe"
os.makedirs(OUT, exist_ok=True)

CAP_PATTERNS = [
    (r"^图\s*\d+([-–—.]\d+)*", "图N-M"),
    (r"^图\s*[一二三四五六七八九十]+", "图一"),
    (r"^Figure\s*\d+", "Figure N"),
    (r"^\[\s*图\s*\d+", "[图N"),
    (r"^\d+[-–.]\d+\s*\S", "N-M 标题"),
]

def scan_captions(fn, limit=8):
    path = os.path.join(PDF_DIR, fn)
    doc = pymupdf.open(path)
    hits = {}
    samples = []
    for pno, page in enumerate(doc, 1):
        d = page.get_text("dict")
        for blk in d["blocks"]:
            if blk["type"] != 0:
                continue
            for line in blk.get("lines", []):
                t = "".join(s["text"] for s in line.get("spans", [])).strip()
                if not t or len(t) > 80:
                    continue
                for pat, name in CAP_PATTERNS:
                    if re.match(pat, t):
                        hits[name] = hits.get(name, 0) + 1
                        if len(samples) < limit:
                            samples.append((pno, name, t[:60]))
                        break
    doc.close()
    return hits, samples

print("### CAPTION PATTERN SCAN ###")
for fn in ["问界_M9_智能座舱与车辆操作_官方手册章节摘录.pdf",
           "华为_Pura70_Ultra_HarmonyOS5_中文用户指南.pdf",
           "TP-LINK_WiFi6_无线路由器.pdf",
           "大疆_Mini_4_Pro.pdf",
           "石头_G10S_Pro.pdf"]:
    hits, samples = scan_captions(fn)
    print("\n-- %s" % fn)
    print("   hits:", hits if hits else "NONE")
    for pno, name, t in samples:
        print("   p%-4d [%s] %s" % (pno, name, t))

# ---- vector region render test ----
print("\n\n### VECTOR REGION RENDER TEST ###")
path = os.path.join(PDF_DIR, "石头_G10S_Pro.pdf")
doc = pymupdf.open(path)
for pno in [5, 6]:
    page = doc[pno - 1]
    text_rects = []
    for blk in page.get_text("dict")["blocks"]:
        if blk["type"] == 0:
            text_rects.append(pymupdf.Rect(blk["bbox"]))
    # union of vector drawings, keep sizeable ones
    rects = []
    for d in page.get_drawings():
        r = pymupdf.Rect(d["rect"])
        if r.width > 20 and r.height > 20 and r.width * r.height > 1500:
            rects.append(r)
    # merge overlapping rects into clusters
    merged = []
    for r in rects:
        placed = False
        for m in merged:
            if m.intersects(r + (-5, -5, 5, 5)):
                m |= r
                placed = True
                break
        if not placed:
            merged.append(pymupdf.Rect(r))
    merged = [m for m in merged if m.width * m.height > 8000]
    print("\npage %d: text_blocks=%d  big_vector_clusters=%d" % (pno, len(text_rects), len(merged)))
    for i, m in enumerate(merged[:4]):
        # skip if mostly text-covered
        cov = sum((m & tr).get_area() for tr in text_rects)
        ratio = cov / max(m.get_area(), 1)
        print("   cluster%d %s  area=%.0f textCover=%.2f" % (
            i, [round(v) for v in m], m.get_area(), ratio))
        if ratio < 0.15:
            pix = page.get_pixmap(clip=m, dpi=200)
            fp = os.path.join(OUT, "shitou_p%d_c%d.png" % (pno, i))
            pix.save(fp)
            print("      -> saved %s (%dx%d)" % (os.path.basename(fp), pix.width, pix.height))
doc.close()
