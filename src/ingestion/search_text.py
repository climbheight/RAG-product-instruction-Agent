"""构造图片的可检索文本（search_text）与文本清洗工具。

为什么需要：BGE 只吃文本，图片本身不能嵌入。要让「用答案检索图片」成立，
每张图必须有一段能代表它的文本。本语料几乎没有图注，所以主力是**邻近说明文字**。

拼接模板（优先级从高到低）：
    【产品名】章节 第N页插图。图注：… 说明：<邻近文本> 图中文字：<OCR/矢量取字>
"""
from __future__ import annotations

import re

# 手册里大量出现的免责套话。不清掉会让所有图的向量高度相似，检索直接失效。
BOILERPLATE = (
    "配图仅供参考",
    "请以产品实际为准",
    "图片仅供参考",
    "画面仅供参考",
    "示意图仅供参考",
    "以实际产品为准",
)

_FIG_MARK_RE = re.compile(r"\[\[FIG:[^\]]+\]\]")
_WS_RE = re.compile(r"[ \t\u3000]+")
_SENT_RE = re.compile(r"(?<=[。！？；])")


def strip_fig_marks(text: str) -> str:
    """剥掉切分阶段插入的图片占位符。送嵌入前必须调用。"""
    return _FIG_MARK_RE.sub("", text)


def is_boilerplate(text: str) -> bool:
    """判断一段文字是不是「只剩免责套话」。

    为什么要这个：手册里「配图仅供参考，请以产品实际为准」常常紧贴图片，
    在空间邻近判据下必然胜出，结果所有图的 nearby_text 都是同一句话，
    图片检索直接失效。必须在候选阶段就把这类块排除。
    """
    t = (text or "").strip()
    if not t:
        return True
    core = re.sub(r"[，。、；：！？\s（）()\[\]【】]", "", t)
    if not core:
        return True
    for bp in BOILERPLATE:
        core = core.replace(bp, "")
    return len(core) <= 2


def clean(text: str) -> str:
    """压空白、去换行、剔除免责套话句。"""
    if not text:
        return ""
    t = text.replace("\r", "\n")
    t = _WS_RE.sub(" ", t)
    # 按句切，丢掉含套话的句子
    out = []
    for seg in _SENT_RE.split(t):
        s = seg.strip()
        if not s:
            continue
        if any(bp in s for bp in BOILERPLATE):
            continue
        out.append(s)
    return " ".join(out).strip()


def vector_text(page, bbox, pad: float = 0.18) -> str:
    """从矢量插图区域里取文字。

    关键补丁：石头 G10S Pro、大疆 DJI Mic2 全篇是矢量插图，
    page.get_images() 返回 0，OCR 也无从下手（图是画出来的，不是位图）。
    但矢量图里的标签文字仍在文本层上，用 clip 就能取到。

    必须**外扩一圈**再取：这些标签（"充电触片""沿墙传感器"）是独立文本块，
    位置画在图形 bbox 之外，不外扩会取到空串。
    """
    try:
        import pymupdf
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        dx = max(w * pad, 8.0)
        dy = max(h * pad, 8.0)
        clip = pymupdf.Rect(bbox[0] - dx, bbox[1] - dy, bbox[2] + dx, bbox[3] + dy)
        return clean(page.get_text("text", clip=clip))
    except Exception:
        return ""


def build_figure_search_text(fig, max_chars: int = 350) -> str:
    """拼出送嵌入的图片文本。

    优先级：VLM 画面描述 > 邻近说明 > 图注 > 图内文字。
    VLM 描述最贴「这张图到底画了什么」，邻近说明贴「这张图用在哪一步操作」，
    两者互补；超长时从尾部（OCR）开始牺牲。
    """
    head = "【%s】%s 第%d页插图。" % (
        fig.doc_title or "", fig.section or "正文", fig.page or 0)

    parts: list[tuple[str, str]] = []   # (标签, 内容)，按优先级从高到低
    vlm = getattr(fig, "vlm_caption", "") or ""
    if vlm:
        parts.append(("画面", clean(vlm)))
    if fig.nearby_text:
        parts.append(("说明", clean(fig.nearby_text)))
    if fig.caption:
        parts.append(("图注", clean(fig.caption)))
    if fig.ocr_text:
        parts.append(("图中文字", clean(fig.ocr_text)))

    body = ""
    for label, content in parts:
        if not content:
            continue
        candidate = body + ("%s：%s。" % (label, content))
        if len(head) + len(candidate) > max_chars and body:
            break
        body = candidate

    if not body:
        body = "%s示意图。" % (fig.section or "产品")

    text = head + body
    return text[:max_chars]
