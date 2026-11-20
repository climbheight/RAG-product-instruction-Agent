"""OCR 兜底：只处理**没有文本层**的页面。

引擎选 RapidOCR（PP-OCRv4 的 onnxruntime 封装）：
- 中文识别质量与 PaddleOCR 同级，但不需要装 paddlepaddle（省 ~500MB）
- 纯 CPU 可跑，笔记本友好
- 装不上时自动降级为「跳过 OCR」，并在报告里明确标注，不会静默失败

注意：OCR 只做兜底，不做主力。本语料 1242 页里只有 2 份（宇树G1、美的电饭煲）
共 44 页真正需要 OCR，其余页面直接走 PyMuPDF 文本层。
"""
from __future__ import annotations

import warnings

from .schema import TextBox


class OcrEngine:
    def __init__(self, lang: str = "ch"):
        self.lang = lang
        self._engine = None
        self.backend = "none"
        self._load()

    def _load(self) -> None:
        try:
            from rapidocr_onnxruntime import RapidOCR  # 旧包名
            self._engine = RapidOCR()
            self.backend = "rapidocr_onnxruntime"
            return
        except Exception:
            pass
        try:
            from rapidocr import RapidOCR  # 新包名
            self._engine = RapidOCR()
            self.backend = "rapidocr"
            return
        except Exception:
            pass
        warnings.warn(
            "未找到 RapidOCR，扫描页将无法提取文字。"
            "安装：pip install rapidocr-onnxruntime",
            RuntimeWarning,
        )

    @property
    def available(self) -> bool:
        return self._engine is not None

    def ocr_boxes(self, img) -> list[TextBox]:
        """返回带坐标的文本框。img 可以是图片路径，也可以是 numpy 数组。

        版面切分必须拿到框坐标：既要判断「哪块是文字」，也要算图文关联。
        RapidOCR 原始 result 本来就是 [[四点坐标, 文本, 置信度], ...]，
        之前 __call__ 把坐标丢掉了，这里补回来。
        """
        if self._engine is None:
            return []
        try:
            result, _ = self._engine(img)
        except Exception:
            return []
        out: list[TextBox] = []
        for item in (result or []):
            try:
                box, text, score = item[0], item[1], item[2]
            except (IndexError, TypeError):
                continue
            if not text:
                continue
            try:
                xs = [float(p[0]) for p in box]
                ys = [float(p[1]) for p in box]
            except (TypeError, IndexError, ValueError):
                continue
            out.append(TextBox(
                x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys),
                text=str(text).strip(), score=float(score),
            ))
        return out

    def __call__(self, img_path: str) -> str:
        if self._engine is None:
            return ""
        try:
            result, _ = self._engine(img_path)
        except Exception:
            return ""
        if not result:
            return ""
        # result: [[box, text, score], ...]
        lines = []
        for item in result:
            if len(item) >= 2 and item[1]:
                lines.append(str(item[1]).strip())
        return "\n".join(l for l in lines if l)
