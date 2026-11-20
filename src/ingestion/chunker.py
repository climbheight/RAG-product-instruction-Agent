"""切分：结构化 + 原子图组绑定。

核心问题（老板问的）：图片上下连接的文字怎么切，才不会出现「图注在 A chunk、
图片在 B chunk」或者「图在 A、说明在 B」。

工业成熟做法是 **element-based chunking**（Unstructured.io / LlamaParse / Azure DI 的路子），
不是按字符硬切。具体到这里：

1. 先把元素流拆成**句子**；
2. 每个图元素挂到它**后面那一句**上（图是给下面这段做说明的），句尾无内容则挂前一句；
3. 「图 + 那一句」构成一个**原子单元**，切分时绝不从中间断开；
4. chunk 由若干个原子单元打包而成，因此图永远和它的说明同框；
5. overlap 以**句子边界**为锚（不是硬字符位置），且图片占位符不重复计入 overlap。

这样既满足 300~600 字符 + 50~100 重合的要求，又不会把图文拆散。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .schema import Chunk, Element

FIG_MARK = "[[FIG:%s]]"
FIG_RE = re.compile(r"\[\[FIG:([^\]]+)\]\]")

# 中文断句：句末标点后切；英文句点后必须跟空格，避免切开 4.5V / 1.2.3
SENT_SPLIT = re.compile(r"(?<=[。！？；…])|(?<=[!?;])\s+|(?<=\.)\s+(?=[A-Z\u4e00-\u9fff])")


@dataclass
class Unit:
    """原子单元：若干图 + 一句文本。切分的最小粒度。"""
    text: str
    figures: list[str] = field(default_factory=list)
    page: int = 0
    section: str = ""

    @property
    def size(self) -> int:
        # 只算文本长度：图片占位符在送嵌入前会被剥掉，不该占用字符预算
        return len(self.text)

    @property
    def advance(self) -> int:
        # 打包时的实际消耗：文本 + 1 个 "\n" 拼接符
        return len(self.text) + 1


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in SENT_SPLIT.split(text) if p and p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def build_units(elements: list[Element]) -> list[Unit]:
    """把元素流变成原子单元序列。"""
    units: list[Unit] = []
    section = ""
    pending_figs: list[str] = []
    last_page = 0

    for el in elements:
        last_page = el.page or last_page
        if el.kind == "figure":
            if el.figure_id:
                pending_figs.append(el.figure_id)
            continue

        if el.is_heading and el.text:
            # 标题单独成单元，并更新章节
            section = el.text.strip()
            if pending_figs:
                units.append(Unit(text="", figures=pending_figs, page=el.page, section=section))
                pending_figs = []
            units.append(Unit(text=el.text.strip(), page=el.page, section=section))
            continue

        for s in split_sentences(el.text):
            units.append(Unit(
                text=s,
                figures=pending_figs,
                page=el.page,
                section=section or "正文",
            ))
            pending_figs = []

    # 结尾悬空的图，挂到最后一句上
    if pending_figs:
        if units:
            units[-1].figures.extend(pending_figs)
        else:
            units.append(Unit(text="", figures=pending_figs, page=last_page, section="正文"))
    return units


def _render(u: Unit) -> str:
    marks = "".join(FIG_MARK % f for f in u.figures)
    return (marks + u.text) if u.text else marks


def chunk_units(units: list[Unit], doc_id: str, doc_title: str, product: str,
                target: int = 450, min_size: int = 300, max_size: int = 600,
                overlap: int = 80) -> list[Chunk]:
    """把原子单元打包成 chunk。参数按课程要求：300~600 字符，overlap 50~100。"""
    chunks: list[Chunk] = []
    buf: list[Unit] = []
    buf_len = 0

    def flush() -> list[Unit]:
        nonlocal buf, buf_len
        if not buf:
            return []
        text_parts = [_render(u) for u in buf]
        body = "\n".join(p for p in text_parts if p.strip())
        fig_ids: list[str] = []
        for u in buf:
            for f in u.figures:
                if f not in fig_ids:
                    fig_ids.append(f)
        pages = [u.page for u in buf if u.page]
        sec = next((u.section for u in buf if u.section), "正文")
        body_chars = len(FIG_RE.sub("", body))
        chunks.append(Chunk(
            chunk_id="%s_c%04d" % (doc_id, len(chunks) + 1),
            doc_id=doc_id, doc_title=doc_title, product=product,
            section=sec,
            page_start=min(pages) if pages else 0,
            page_end=max(pages) if pages else 0,
            text=body,
            figure_ids=fig_ids,
            char_count=body_chars,
            est_tokens=body_chars,   # 中文约 1 字 1 token，嵌入时再精确计数
        ))
        # 计算 overlap 回退单元：从尾部往前累加，不超过 overlap 预算
        carry: list[Unit] = []
        acc = 0
        for u in reversed(buf):
            if acc + u.advance > overlap and carry:
                break
            carry.insert(0, u)
            acc += u.advance
        buf = []
        buf_len = 0
        return carry

    for u in units:
        # 单个单元超长（长段落），先强行切开
        if u.size > max_size:
            for sub in _split_long_unit(u, max_size):
                if buf and buf_len + sub.advance > max_size:
                    buf = flush()
                    buf_len = sum(x.advance for x in buf)
                buf.append(sub)
                buf_len += sub.advance
            continue

        # 硬上限：超了就先落盘，不看 min_size（否则会冲过 max_size）
        if buf and buf_len + u.advance > max_size:
            buf = flush()
            buf_len = sum(x.advance for x in buf)
        buf.append(u)
        buf_len += u.advance

    # 收尾
    if buf:
        tail_chars = sum(u.advance for u in buf)
        if chunks and buf_len < min_size and chunks[-1].char_count + tail_chars <= max_size:
            # 尾巴太短，并入上一块（前提是不超过 max_size）
            tail = "\n".join(_render(u) for u in buf)
            prev = chunks[-1]
            extra_figs = [f for u in buf for f in u.figures if f not in prev.figure_ids]
            prev.text = prev.text + "\n" + tail
            prev.figure_ids.extend(extra_figs)
            prev.char_count = len(FIG_RE.sub("", prev.text))
            prev.est_tokens = prev.char_count
            prev.page_end = max(prev.page_end, max((u.page for u in buf), default=prev.page_end))
        else:
            flush()

    return _repair_short_chunks(chunks, doc_id, min_size, max_size)


def _repair_short_chunks(chunks: list[Chunk], doc_id: str,
                         min_size: int, max_size: int) -> list[Chunk]:
    """修短块：小于 min_size 的块尽量并入前一块（前提是不超 max_size），并重编号。"""
    out: list[Chunk] = []
    for c in chunks:
        if out and c.char_count < min_size and \
                out[-1].char_count + c.char_count + 1 <= max_size:
            p = out[-1]
            p.text = p.text + "\n" + c.text
            for f in c.figure_ids:
                if f not in p.figure_ids:
                    p.figure_ids.append(f)
            p.char_count = len(FIG_RE.sub("", p.text))
            p.est_tokens = p.char_count
            p.page_end = max(p.page_end, c.page_end)
        else:
            out.append(c)
    for i, c in enumerate(out, 1):
        c.chunk_id = "%s_c%04d" % (doc_id, i)
    return out


def _split_long_unit(u: Unit, max_size: int) -> list[Unit]:
    """把一个超长句子按逗号/分号二次切开，图只挂第一段。"""
    if u.size <= max_size:
        return [u]
    pieces = re.split(r"(?<=[，,、])", u.text)
    out: list[Unit] = []
    cur = ""
    first = True
    for p in pieces:
        if len(cur) + len(p) > max_size and cur:
            out.append(Unit(text=cur, figures=u.figures if first else [],
                            page=u.page, section=u.section))
            first = False
            cur = p
        else:
            cur += p
    if cur.strip():
        out.append(Unit(text=cur, figures=u.figures if first else [],
                        page=u.page, section=u.section))
    return out
