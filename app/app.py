"""Gradio 界面。

设计对应课程验收点：
- 回答下方展示**手册原图**（不是模型生图）
- 显示"程序做了什么"：命中的片段、相似度、走过的节点，便于解释
  哪些步骤由程序控制、哪些由模型生成
- 空输入 / 无结果 / 模型异常都有明确提示，不静默失败
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.agent.graph import ManualAgent  # noqa: E402
from src.llm.base import load_llm  # noqa: E402
from src.retrieval.vectorstore import DualIndex  # noqa: E402

CFG = os.path.join(ROOT, "configs", "manuals.json")
VECTOR_DIR = os.path.join(ROOT, "vectorstore")

_AGENT: ManualAgent | None = None


def get_agent() -> ManualAgent:
    global _AGENT
    if _AGENT is None:
        llm = load_llm()
        index = DualIndex(VECTOR_DIR)
        _AGENT = ManualAgent(index, llm)
    return _AGENT


def load_products() -> tuple[list[tuple[str, str]], dict[str, str]]:
    """返回下拉选项 [(显示名, doc_id)] 和 显示名→doc_id 映射。"""
    if not os.path.exists(CFG):
        return [("全部产品", "")], {}
    manuals = json.load(open(CFG, encoding="utf-8"))
    opts = [("全部产品（跨手册检索）", "")]
    mapping: dict[str, str] = {}
    for meta in manuals.values():
        label = "%s %s" % (meta.get("brand", ""), meta.get("product", ""))
        opts.append((label, meta["doc_id"]))
        mapping[label] = meta["doc_id"]
    return opts, mapping


def _fmt_detail(state: dict) -> str:
    lines = ["**执行路径**：`%s`" % " → ".join(state.get("path", []))]
    note = state.get("note")
    if note:
        lines.append("**判定**：%s" % note)
    if state.get("error"):
        lines.append("**异常**：%s" % state["error"])

    hits = state.get("chunks") or []
    if hits:
        lines.append("\n**命中片段**（按相似度排序）：")
        for i, h in enumerate(hits, 1):
            lines.append(
                "%d. `%.3f` %s · %s · 第%d页" % (
                    i, h.get("score", 0), h.get("product") or h.get("doc_title", ""),
                    h.get("section", ""), h.get("page_start", 0)))
            snippet = (h.get("text") or "").replace("\n", " ")[:90]
            lines.append("   > %s…" % snippet)
    else:
        lines.append("\n**命中片段**：无")

    figs = state.get("figures") or []
    if figs:
        lines.append("\n**配图**：")
        for f in figs:
            desc = f.get("vlm_caption") or f.get("nearby_text") or ""
            lines.append("- `%.3f` %s 第%d页 · %s" % (
                f.get("score", 0), f.get("doc_title", ""), f.get("page", 0),
                desc[:60]))
    else:
        lines.append("\n**配图**：无（相关度未达阈值，宁可不配图）")
    return "\n".join(lines)


def answer_fn(question: str, product_label: str, mapping: dict):
    q = (question or "").strip()
    if not q:
        return ("请输入问题后再提交。", [], "**未执行**：问题为空。")
    try:
        agent = get_agent()
    except Exception as e:
        return ("系统初始化失败：%s" % e, [], "**初始化异常**：`%s`" % e)

    doc_id = mapping.get(product_label, "")
    try:
        state = agent.ask(q, doc_filter=[doc_id] if doc_id else None)
    except Exception as e:
        return ("链路执行异常：%s" % e, [], "**执行异常**：`%s`" % e)

    answer = state.get("answer") or "（无回答）"
    gallery = []
    for f in state.get("figures") or []:
        p = f.get("abs_path") or ""
        if p and os.path.exists(p):
            desc = f.get("vlm_caption") or f.get("nearby_text") or f.get("figure_id", "")
            gallery.append((p, "%s 第%d页｜%s" % (
                f.get("doc_title", ""), f.get("page", 0), desc[:40])))
    return answer, gallery, _fmt_detail(state)


def build_ui():
    import gradio as gr

    opts, mapping = load_products()

    with gr.Blocks(title="产品手册智能问答", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 产品手册智能问答\n"
            "基于 16 份官方产品手册（1242 页 / 1928 个文本片段 / 785 张插图）。\n"
            "回答严格依据手册原文，并在下方附上对应插图。")

        with gr.Row():
            with gr.Column(scale=3):
                question = gr.Textbox(
                    label="你的问题", lines=2,
                    placeholder="例如：石头扫地机启动前需要清理地面上的什么？")
            with gr.Column(scale=1):
                product = gr.Dropdown(
                    choices=[o[0] for o in opts], value=opts[0][0],
                    label="限定产品（可选）")

        with gr.Row():
            btn = gr.Button("提问", variant="primary")
            clear = gr.Button("清空")

        answer = gr.Markdown(label="回答")
        gr.Markdown("### 手册原图")
        gallery = gr.Gallery(label="相关插图", columns=2, height="auto",
                             object_fit="contain")

        with gr.Accordion("检索详情（程序做了什么）", open=False):
            detail = gr.Markdown()

        btn.click(answer_fn, [question, product, gr.State(mapping)],
                  [answer, gallery, detail])
        question.submit(answer_fn, [question, product, gr.State(mapping)],
                        [answer, gallery, detail])
        clear.click(lambda: ("", [], ""), None, [answer, gallery, detail])

        gr.Markdown(
            "> **说明**：检索、阈值判断、候选池限定、图片精排、兜底话术均由程序控制；\n"
            "> 只有最终自然语言答案由本地模型生成。相关度不足时不配图，避免硬凑。")
    return demo


def main():
    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)


if __name__ == "__main__":
    main()
