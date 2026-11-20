"""阅读顺序：多栏检测 + 栏内排序。

为什么需要：本语料里石头 G10S Pro(516x369)、美的(595x411) 是**横向双栏**版式。
如果直接按 y 排序，左栏末尾会和右栏开头交错拼在一起，切出来的 chunk 语义是乱的。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def xc(self) -> float:
        return (self.x0 + self.x1) / 2

    def overlap_x(self, other: "Rect") -> float:
        return max(0.0, min(self.x1, other.x1) - max(self.x0, other.x0))

    def overlap_y(self, other: "Rect") -> float:
        return max(0.0, min(self.y1, other.y1) - max(self.y0, other.y0))

    def expand(self, d: float) -> "Rect":
        return Rect(self.x0 - d, self.y0 - d, self.x1 + d, self.y1 + d)


def detect_columns(rects: list[Rect], page_width: float,
                   page_height: float = 0.0,
                   min_gap: float = 8.0, min_col_ratio: float = 0.15,
                   max_cross_ratio: float = 0.25) -> list[tuple[float, float]]:
    """用 x 投影的空隙找分栏。返回 [(x_start, x_end), ...]，按从左到右。

    判断标准不是「空隙有多宽」，而是「这个空隙有没有被横跨它的块挡住」：
    - 真实栏缝：只有标题会横跨，高度占比很小
    - 假空隙（比如一行里两个词的间距）：几乎没有块横跨，但也没有块在两侧成栏

    本语料里石头 G10S Pro 的栏缝只有 ~10pt，固定阈值 24pt 会直接漏掉，
    所以必须用「跨栏块高度占比」这个更本质的判据。
    """
    if not rects:
        return [(0.0, page_width)]

    if page_height <= 0:
        page_height = max((r.y1 for r in rects), default=1.0) or 1.0

    step = 1.0
    n = int(page_width / step) + 2
    covered = [0] * n
    for r in rects:
        a = max(0, int(r.x0 / step))
        b = min(n - 1, int(r.x1 / step))
        for i in range(a, b + 1):
            covered[i] += 1

    gaps: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if covered[i] == 0:
            j = i
            while j < n and covered[j] == 0:
                j += 1
            if (j - i) * step >= min_gap:
                gaps.append((i, j))
            i = j
        else:
            i += 1

    total = len(rects)
    cuts: list[float] = []
    for a, b in gaps:
        gx0, gx1 = a * step, b * step
        # 页边距不算栏缝
        if gx0 <= 2 or gx1 >= page_width - 2:
            continue
        left = sum(1 for r in rects if r.x1 <= gx0 + 2)
        right = sum(1 for r in rects if r.x0 >= gx1 - 2)
        if left < total * min_col_ratio or right < total * min_col_ratio:
            continue
        # 跨过这条缝的块，总高度占页面比例
        cross_h = sum(r.height for r in rects
                      if r.x0 < gx1 - 2 and r.x1 > gx0 + 2)
        if cross_h / page_height > max_cross_ratio:
            continue
        cuts.append((gx0 + gx1) / 2)

    if not cuts:
        return [(0.0, page_width)]

    bounds = [0.0] + sorted(cuts) + [page_width]
    return [(bounds[k], bounds[k + 1]) for k in range(len(bounds) - 1)]



def order_elements(rects: list[Rect], page_width: float,
                   page_height: float = 0.0,
                   full_width_ratio: float = 0.72) -> tuple[list[int], int]:
    """返回阅读顺序的索引序列，以及检测到的栏数。

    规则：
    - 通栏块（宽度 > page_width * full_width_ratio）当作**分段边界**：
      它上方的栏内内容先读完，再输出通栏块，然后继续下方的栏内内容；
    - 每个分段内部：先按栏序、再按 y 排序（保证左栏读完再读右栏）；
    - 这样无论通栏标题在页首、页中还是页尾，顺序都不会错。
    """
    if not rects:
        return [], 1

    cols = detect_columns(rects, page_width, page_height)
    n_cols = len(cols)

    def col_of(r: Rect) -> int:
        for ci, (a, b) in enumerate(cols):
            if r.xc <= b:
                return ci
        return n_cols - 1

    spans = sorted(
        [(r.y0, r.y1, i) for i, r in enumerate(rects)
         if r.width >= page_width * full_width_ratio],
        key=lambda t: t[0],
    )
    others = [(i, r) for i, r in enumerate(rects)
              if r.width < page_width * full_width_ratio]

    def order_band(band: list[tuple[int, Rect]]) -> list[int]:
        band.sort(key=lambda t: (col_of(t[1]), t[1].y0, t[1].x0))
        return [i for i, _ in band]

    result: list[int] = []
    prev_y = float("-inf")
    for sy0, sy1, si in spans:
        band = [(i, r) for i, r in others if prev_y <= r.y0 < sy0]
        result.extend(order_band(band))
        result.append(si)
        prev_y = sy1
    tail = [(i, r) for i, r in others if r.y0 >= prev_y]
    result.extend(order_band(tail))

    # 兜底：确保每个索引恰好出现一次
    seen = set(result)
    for i in range(len(rects)):
        if i not in seen:
            result.append(i)
    return result, n_cols

