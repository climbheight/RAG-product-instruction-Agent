"""双索引向量库：文本 chunk 一个 collection，图片一个 collection。

为什么分两个：
  - BGE 只吃文本，图片得先变成 search_text 才能嵌入；
  - 文本和图片的检索时机不同——文本在生成答案前检索，图片在答案生成后检索；
  - 分开才能做「先用文本召回圈定 doc 范围，再在圈内精排图片」，
    避免问扫地机配出华为手机的图。

图片侧的关键设计：二次检索**不做全库搜索**，只在第一轮命中 chunk 的 figure_ids 里精排。
"""
from __future__ import annotations

import json
import os

import numpy as np

from .embed import BgeEmbedder

CHUNK_COL = "manual_chunks"
FIGURE_COL = "manual_figures"


def _meta(v) -> str | int | float | bool:
    """Chroma 的 metadata 只接受标量，list/dict 一律转 JSON 字符串。"""
    if isinstance(v, (str, int, float, bool)):
        return v
    return json.dumps(v, ensure_ascii=False)


class DualIndex:
    def __init__(self, persist_dir: str, embedder: BgeEmbedder | None = None):
        import chromadb
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedder = embedder or BgeEmbedder()

    # ---------------- 建库 ----------------

    def build(self, chunks: list[dict], figures: list[dict], reset: bool = True,
              batch: int = 256) -> dict:
        if reset:
            for name in (CHUNK_COL, FIGURE_COL):
                try:
                    self.client.delete_collection(name)
                except Exception:
                    pass

        ccol = self.client.get_or_create_collection(
            CHUNK_COL, metadata={"hnsw:space": "cosine"})
        fcol = self.client.get_or_create_collection(
            FIGURE_COL, metadata={"hnsw:space": "cosine"})

        # 文本：剥掉图片占位符再嵌入（占位符不是自然语言，会污染向量）
        from ..ingestion.search_text import strip_fig_marks
        texts = [strip_fig_marks(c["text"]).strip() for c in chunks]
        for i in range(0, len(chunks), batch):
            sl = slice(i, i + batch)
            sub_t = texts[sl]
            emb = self.embedder.embed_documents(sub_t)
            ccol.add(
                ids=[c["chunk_id"] for c in chunks[sl]],
                embeddings=emb.tolist(),
                documents=sub_t,
                metadatas=[{
                    "chunk_id": c["chunk_id"],
                    "doc_id": c["doc_id"],
                    "doc_title": _meta(c.get("doc_title", "")),
                    "product": _meta(c.get("product", "")),
                    "section": _meta(c.get("section", "")),
                    "page_start": int(c.get("page_start", 0)),
                    "page_end": int(c.get("page_end", 0)),
                    "figure_ids": _meta(c.get("figure_ids", [])),
                    "char_count": int(c.get("char_count", 0)),
                } for c in chunks[sl]],
            )

        # 图片：用 search_text 嵌入
        figs = [f for f in figures if (f.get("search_text") or "").strip()]
        for i in range(0, len(figs), batch):
            sl = slice(i, i + batch)
            sub_f = figs[sl]
            emb = self.embedder.embed_documents([f["search_text"] for f in sub_f])
            fcol.add(
                ids=[f["figure_id"] for f in sub_f],
                embeddings=emb.tolist(),
                documents=[f["search_text"] for f in sub_f],
                metadatas=[{
                    "figure_id": f["figure_id"],
                    "doc_id": f["doc_id"],
                    "doc_title": _meta(f.get("doc_title", "")),
                    "page": int(f.get("page", 0)),
                    "path": _meta(f.get("path", "")),
                    "source": _meta(f.get("source", "")),
                    "region_kind": _meta(f.get("region_kind", "")),
                    "section": _meta(f.get("section", "")),
                    "nearby_text": _meta((f.get("nearby_text") or "")[:300]),
                    "vlm_caption": _meta((f.get("vlm_caption") or "")[:300]),
                    "chunk_ids": _meta(f.get("chunk_ids", [])),
                    "width": int(f.get("width", 0)),
                    "height": int(f.get("height", 0)),
                } for f in sub_f],
            )

        return {"chunks": len(chunks), "figures": len(figs)}

    # ---------------- 检索 ----------------

    def search_chunks(self, query: str, k: int = 6,
                      doc_filter: list[str] | None = None) -> list[dict]:
        col = self.client.get_collection(CHUNK_COL)
        qv = self.embedder.embed_query(query)
        where = {"doc_id": {"$in": doc_filter}} if doc_filter else None
        res = col.query(query_embeddings=[qv.tolist()], n_results=k,
                        where=where,
                        include=["documents", "metadatas", "distances"])
        out = []
        for i in range(len(res["ids"][0])):
            m = res["metadatas"][0][i]
            out.append({
                "chunk_id": m["chunk_id"],
                "doc_id": m["doc_id"],
                "doc_title": m.get("doc_title", ""),
                "product": m.get("product", ""),
                "section": m.get("section", ""),
                "page_start": m.get("page_start", 0),
                "page_end": m.get("page_end", 0),
                "figure_ids": json.loads(m.get("figure_ids", "[]")),
                "text": res["documents"][0][i],
                "score": 1.0 - float(res["distances"][0][i]),   # cosine 距离 → 相似度
            })
        return out

    def get_figures(self, ids: list[str]) -> dict[str, dict]:
        if not ids:
            return {}
        col = self.client.get_collection(FIGURE_COL)
        res = col.get(ids=ids, include=["metadatas", "documents", "embeddings"])
        out: dict[str, dict] = {}
        for i, fid in enumerate(res["ids"]):
            m = res["metadatas"][i]
            out[fid] = {
                "figure_id": fid,
                "doc_id": m.get("doc_id", ""),
                "doc_title": m.get("doc_title", ""),
                "page": m.get("page", 0),
                "path": m.get("path", ""),
                "source": m.get("source", ""),
                "region_kind": m.get("region_kind", ""),
                "section": m.get("section", ""),
                "nearby_text": m.get("nearby_text", ""),
                "vlm_caption": m.get("vlm_caption", ""),
                "chunk_ids": json.loads(m.get("chunk_ids", "[]")),
                "width": m.get("width", 0),
                "height": m.get("height", 0),
                "search_text": res["documents"][i] if res.get("documents") else "",
                "_emb": np.asarray(res["embeddings"][i], dtype=np.float32),
            }
        return out

    def figures_of_docs(self, doc_ids: list[str]) -> list[str]:
        """某个文档下的全部 figure_id。用于给候选池兜底。"""
        if not doc_ids:
            return []
        col = self.client.get_collection(FIGURE_COL)
        res = col.get(where={"doc_id": {"$in": list(doc_ids)}}, include=[])
        return list(res["ids"])

    def collect_candidates(self, chunk_hits: list[dict], min_pool: int = 12) -> list[str]:
        """组装图片候选池，**严格锁定单个产品**。

        先定目标文档 = top-1 命中 chunk 的 doc_id，再只在该文档内取图。
        为什么不用所有命中文档：实测「怎么把机器人从充电座上搬起来」这类问题
        会同时命中石头和宇树，若把两个文档的图都放进池子，精排就会把宇树的充电图
        配给扫地机——这正是要杜绝的配图幻觉。

        池子优先用命中 chunk 绑定的 figure_ids（原文证据最强）；
        太小就放宽到该文档的全部图（仍然同一产品）。
        """
        if not chunk_hits:
            return []
        target_doc = chunk_hits[0].get("doc_id")
        if not target_doc:
            return []

        ids: list[str] = []
        for h in chunk_hits:
            if h.get("doc_id") != target_doc:
                continue
            for fid in h.get("figure_ids", []):
                if fid not in ids:
                    ids.append(fid)

        if len(ids) < min_pool:
            for fid in self.figures_of_docs([target_doc]):
                if fid not in ids:
                    ids.append(fid)
        return ids

    def rerank_figures(self, query_text: str, cand_ids: list[str],
                       top_k: int = 2, threshold: float = 0.42) -> list[dict]:
        """在候选池内按语义相似度精排图片。

        候选池来自第一轮命中 chunk 的 figure_ids —— 天然落在命中的 doc 内，
        不会跨产品配图。低于阈值宁可不返回，避免「每问必凑图」。
        """
        figs = self.get_figures(list(dict.fromkeys(cand_ids)))
        if not figs:
            return []
        qv = self.embedder.embed_query(query_text)
        scored = []
        for f in figs.values():
            emb = f.pop("_emb")
            scored.append((float(np.dot(qv, emb)), f))
        scored.sort(key=lambda t: t[0], reverse=True)
        picked = []
        for s, f in scored:
            if s < threshold:
                break
            f["score"] = round(s, 4)
            picked.append(f)
            if len(picked) >= top_k:
                break
        return picked

    def stats(self) -> dict:
        out = {}
        for name in (CHUNK_COL, FIGURE_COL):
            try:
                out[name] = self.client.get_collection(name).count()
            except Exception:
                out[name] = 0
        return out
