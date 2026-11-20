"""LangGraph 问答链路。

图结构（5 个节点，2 条条件边）：

    START
      ↓
   retrieve          文本检索（BGE + Chroma）
      ↓
    grade            判断召回是否足够 ──不足──→ fallback → END
      ↓ 足够
   generate          拼上下文 + 本地模型生成答案
      ↓
  retrieve_images    用【问题 + 答案】在同一产品内精排图片
      ↓
     END

设计要点（对应课程验收点）：
1. **哪些步骤由程序控制、哪些由模型生成**：
   - 程序控制：检索、阈值判断、候选池限定、图片精排、兜底话术
   - 模型生成：只有 generate 节点里的自然语言答案
2. **无检索结果 / 工具失败必须有明确兜底**：fallback 节点 + 异常捕获
3. **不许编造**：prompt 强约束 + 检索为空直接走兜底，不给模型发挥空间
"""
from __future__ import annotations

import os
from typing import Any, TypedDict

from ..llm.base import (EMPTY_REPLY, FAILURE_REPLY, NO_RESULT_REPLY,
                          LLMBase, build_prompt)
from ..retrieval.vectorstore import DualIndex

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXTRACTED = os.path.join(ROOT, "data", "extracted")

# 召回判定阈值。这个值不是拍脑袋，是实测标定的：
#   相关问题 top-1：0.596(设虚拟墙) / 0.662(拆滤网) / 0.694(返航) / 0.741(连WiFi) / 0.778(测体温)
#   无关问题 top-1：0.248(红楼梦) / 0.375(猫绝育) / 0.423(股市) / 0.441(护照) / 0.476(量子计算) / 0.521(Python读文件)
# 0.56 正好落在 0.521 与 0.596 之间。
# 早先用 0.45 时，「量子计算机怎么搭建」会被判成召回充分并拿路由器手册作答——典型的幻觉入口。
MIN_CHUNK_SCORE = 0.56
MIN_CHUNKS = 1
# 图片精排阈值：低于此宁可不配图
FIGURE_SCORE = 0.42
TOP_K_CHUNKS = 6
TOP_K_FIGURES = 2


class AgentState(TypedDict, total=False):
    question: str
    doc_filter: list[str] | None
    chunks: list[dict]
    context_blocks: list[str]
    answer: str
    figure_ids: list[str]
    figures: list[dict]
    error: str
    path: list[str]        # 走过的节点，便于验收时展示"程序做了什么"
    note: str


def _abs_figure_path(fig: dict) -> dict:
    """把 figures.jsonl 里的相对路径补成绝对路径，供界面直接读图。"""
    p = fig.get("path") or ""
    if p and not os.path.isabs(p):
        fig["abs_path"] = os.path.join(EXTRACTED, fig.get("doc_id", ""), p)
    else:
        fig["abs_path"] = p
    return fig


class ManualAgent:
    def __init__(self, index: DualIndex, llm: LLMBase):
        self.index = index
        self.llm = llm
        self.graph = self._build()

    # ---------------- 节点 ----------------

    def _retrieve(self, state: AgentState) -> dict:
        q = state["question"].strip()
        path = state.get("path", []) + ["retrieve"]
        if not q:
            return {"chunks": [], "path": path, "error": "空问题"}
        try:
            hits = self.index.search_chunks(
                q, k=TOP_K_CHUNKS, doc_filter=state.get("doc_filter"))
        except Exception as e:
            return {"chunks": [], "path": path, "error": f"检索失败: {e}"}
        return {"chunks": hits, "path": path, "error": ""}

    def _grade(self, state: AgentState) -> dict:
        """程序判定召回是否足够，不交给模型。"""
        path = state.get("path", []) + ["grade"]
        if state.get("error"):
            return {"path": path, "note": "检索异常"}
        hits = state.get("chunks") or []
        good = [h for h in hits if h.get("score", 0) >= MIN_CHUNK_SCORE]
        if len(good) < MIN_CHUNKS:
            return {"path": path, "note": "召回不足",
                    "context_blocks": [], "chunks": good}
        # 拼上下文块：带产品名/章节/页码，便于模型引用
        blocks = []
        for i, h in enumerate(good, 1):
            blocks.append("[%d] （%s / %s / 第%d页）\n%s" % (
                i, h.get("product") or h.get("doc_title", ""),
                h.get("section", ""), h.get("page_start", 0), h.get("text", "")))
        return {"path": path, "note": "召回充分", "context_blocks": blocks,
                "chunks": good}

    def _route_after_grade(self, state: AgentState) -> str:
        """条件边：召回不足或出错 → fallback；否则 → generate。"""
        if state.get("error"):
            return "fallback"
        return "generate" if state.get("context_blocks") else "fallback"

    def _generate(self, state: AgentState) -> dict:
        path = state.get("path", []) + ["generate"]
        prompt = build_prompt(state["question"], state.get("context_blocks") or [])
        try:
            answer = self.llm.generate(prompt)
        except Exception as e:
            return {"path": path, "answer": FAILURE_REPLY,
                    "error": f"生成失败: {e}", "note": "模型调用异常"}
        return {"path": path, "answer": answer}

    def _retrieve_images(self, state: AgentState) -> dict:
        """答案生成后，用【问题+答案】在同一产品内精排图片。

        候选池来自命中 chunk 的 figure_ids（原文证据），不足时放宽到该文档全部图。
        始终锁定 top-1 命中文档，杜绝跨产品配图。
        """
        path = state.get("path", []) + ["retrieve_images"]
        hits = state.get("chunks") or []
        if not hits:
            return {"path": path, "figures": [], "figure_ids": []}
        try:
            cand = self.index.collect_candidates(hits)
            query = "问题：%s\n答案：%s" % (
                state["question"], (state.get("answer") or "")[:300])
            picked = self.index.rerank_figures(
                query, cand, top_k=TOP_K_FIGURES, threshold=FIGURE_SCORE)
        except Exception as e:
            return {"path": path, "figures": [], "figure_ids": [],
                    "note": (state.get("note", "") + " 图片检索异常").strip()}
        return {"path": path, "figures": [_abs_figure_path(f) for f in picked],
                "figure_ids": [f["figure_id"] for f in picked]}

    def _fallback(self, state: AgentState) -> dict:
        path = state.get("path", []) + ["fallback"]
        err = state.get("error") or ""
        if err == "空问题":
            msg = EMPTY_REPLY
        elif err:
            msg = FAILURE_REPLY
        else:
            msg = NO_RESULT_REPLY
        return {"path": path, "answer": msg, "figures": [], "figure_ids": []}

    # ---------------- 建图 ----------------

    def _build(self):
        from langgraph.graph import END, START, StateGraph

        g = StateGraph(AgentState)
        g.add_node("retrieve", self._retrieve)
        g.add_node("grade", self._grade)
        g.add_node("generate", self._generate)
        g.add_node("retrieve_images", self._retrieve_images)
        g.add_node("fallback", self._fallback)

        g.add_edge(START, "retrieve")
        g.add_edge("retrieve", "grade")
        g.add_conditional_edges("grade", self._route_after_grade,
                                {"generate": "generate", "fallback": "fallback"})
        g.add_edge("generate", "retrieve_images")
        g.add_edge("retrieve_images", END)
        g.add_edge("fallback", END)
        return g.compile()

    # ---------------- 对外 ----------------

    def ask(self, question: str, doc_filter: list[str] | None = None) -> AgentState:
        return self.graph.invoke({"question": question,
                                  "doc_filter": doc_filter,
                                  "path": []})
