"""BGE 嵌入封装。

课程指定 bge-small-zh-v1.5，文档与查询必须用同一模型。
两个容易踩的点：
  1. **查询侧要加指令前缀**（BGE 官方推荐），文档侧不加。这个差别对召回影响明显。
  2. bge-small-zh-v1.5 的输入窗口是 **512 token**，中文约 1 字 1 token。
     所以 chunk 上界收到 480、图片 search_text 限 350，都是为了避免被静默截断。
"""
from __future__ import annotations

import os

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
# 本地模型目录（离线可用）。存在就优先用它，避免每次联网下载。
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOCAL_MODEL = os.path.join(ROOT, "models", "bge-small-zh-v1.5")
# BGE 中文模型的官方检索指令
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
MAX_SEQ_LEN = 512


def resolve_model(model_name_or_path: str | None = None) -> str:
    """优先本地目录，其次传入路径，最后回落到 HF 仓库名。"""
    if model_name_or_path:
        return model_name_or_path
    if os.path.isfile(os.path.join(LOCAL_MODEL, "config.json")):
        return LOCAL_MODEL
    return DEFAULT_MODEL


class BgeEmbedder:
    def __init__(self, model_name_or_path: str | None = None, device: str | None = None):
        self.model_name = resolve_model(model_name_or_path)
        self._model = None
        self._device = device

    def _ensure(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        # 优先本地目录（离线可用）；没有才联网。实测本机 huggingface.co 比镜像快，
        # 所以不强制设 HF_ENDPOINT。现场若访问不了 HF，可 export HF_ENDPOINT=https://hf-mirror.com
        self._model = SentenceTransformer(
            self.model_name, device=self._device, trust_remote_code=False)
        self._model.max_seq_length = MAX_SEQ_LEN

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        self._ensure()
        if not texts:
            return np.zeros((0, 512), dtype=np.float32)
        v = self._model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True,
            show_progress_bar=len(texts) > 64, convert_to_numpy=True)
        return v.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        self._ensure()
        v = self._model.encode(
            [QUERY_INSTRUCTION + query], normalize_embeddings=True,
            convert_to_numpy=True)
        return v.astype(np.float32)[0]

    @property
    def dim(self) -> int:
        self._ensure()
        return int(self._model.get_sentence_embedding_dimension())
