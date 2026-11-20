"""数据模型：页面元素、图、切分块。

设计原则：文本与图片**分开存储**，靠 id 关联。
BGE 只能吃文本，图片永远是独立文件；chunk 通过 figure_ids 把图带出来。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Element:
    """页面上的一个元素，按阅读顺序排列。"""
    kind: str                 # "text" | "figure"
    page: int                 # 1-based
    bbox: tuple[float, float, float, float]
    text: str = ""            # kind=="text" 时有效
    font_size: float = 0.0    # 正文字号，用于推断标题
    is_heading: bool = False
    figure_id: str | None = None   # kind=="figure" 时有效
    order: float = 0.0        # 阅读顺序键（越小越靠前）

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Figure:
    """一张从手册里取出来的图（位图原图 / 矢量渲染 / 扫描页区块）。"""
    figure_id: str
    doc_id: str
    page: int
    bbox: tuple[float, float, float, float]
    path: str                 # 相对 data/extracted 的 png 路径
    width: int = 0
    height: int = 0
    source: str = "raster"    # "raster" | "vector" | "page-scan"
    caption: str = ""         # 图注（本语料里多为空）
    section: str = ""         # 所在章节
    ocr_text: str = ""        # 图内文字：扫描图走 OCR，矢量图走 get_text(clip)
    doc_title: str = ""
    # --- 图片检索所需 ---
    nearby_text: str = ""     # 邻近说明文字（本语料的主要检索依据）
    region_kind: str = ""     # 扫描页版面分类：figure / text / toc / footer
    region_no: int = 0        # 扫描页内区块序号
    search_text: str = ""     # 送嵌入的可检索文本（由 search_text.py 构造）
    # --- 多模态增强 ---
    vlm_caption: str = ""     # VLM 看图写的描述（比 OCR 更贴近"这张图画的是什么"）
    boilerplate_adjacent: bool = False  # 紧邻"配图仅供参考"套话 → 可确认是真插图
    # --- 关联与连接（payload 结构）---
    chunk_ids: list[str] = field(default_factory=list)  # 反向连接：哪些 chunk 引用了它
    embedding_ref: str = ""   # 嵌入连接，形如 "manual_figures:<figure_id>"
    content_hash: str = ""    # 图片内容哈希，用于去重

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TextBox:
    """OCR 得到的文本框（像素坐标，与渲染图同坐标系）。"""
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    score: float = 0.0

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def xc(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def yc(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass
class Region:
    """扫描页版面切分出的一个区块（像素坐标）。"""
    x0: float
    y0: float
    x1: float
    y1: float
    kind: str = ""            # "text" | "figure" | "toc" | "footer"
    ocr_text: str = ""
    is_border: bool = False   # 是否由矩形线框检出（"方框式一片属于一个"）
    nearby_text: str = ""
    region_no: int = 0

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def xc(self) -> float:
        return (self.x0 + self.x1) / 2

    def bbox(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass
class Chunk:
    """送入嵌入模型的文本单元。"""
    chunk_id: str
    doc_id: str
    doc_title: str
    product: str
    section: str
    page_start: int
    page_end: int
    text: str
    figure_ids: list[str] = field(default_factory=list)
    char_count: int = 0
    est_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PageReport:
    """单页解析报告，用于质量核查。"""
    page: int
    n_text_blocks: int = 0
    n_figures: int = 0
    n_inline_icons: int = 0
    n_columns: int = 1
    ocr_used: bool = False
    chars: int = 0
    note: str = ""
