"""本地推理模型的可插拔封装。

课程指定 MiniMind，备选 Qwen3-0.6B / MiniCPM-1B。三者接口一致：
给一段 prompt，返回生成的文本。所以这里定义统一接口，具体实现可替换。

【重要】MockLLM 只用于验证链路结构（节点、状态流转、条件边、兜底分支），
**不能用来评估回答质量**。真实回答必须接 MiniMind 等本地模型。
"""
from __future__ import annotations

import os
import re

# 系统提示：强约束，不许编造。这是课程评分的重点（幻觉）。
SYSTEM_PROMPT = """你是产品使用手册助手。请严格依据下面提供的【资料】回答问题。

规则：
1. 只使用【资料】中的信息，不得补充资料以外的内容。
2. 如果【资料】不足以回答，直接说"手册中没有找到相关说明"，不要猜测。
3. 回答要具体，涉及按键、菜单路径、参数数值时照抄资料原文。
4. 回答末尾不要重复资料原文，不要写"根据资料"这类套话。"""

NO_RESULT_REPLY = (
    "抱歉，在已入库的 16 份产品手册中没有检索到与该问题相关的内容。\n\n"
    "建议：\n"
    "- 换一种说法描述问题（例如把「连不上」改成「如何连接 Wi-Fi」）\n"
    "- 确认问题所属的产品是否在支持范围内\n"
    "- 如果问题涉及具体型号，请在问题中说明型号"
)

FAILURE_REPLY = (
    "检索过程出现异常，暂时无法给出可靠回答。\n\n"
    "为避免给出错误信息，本次不作答。请稍后重试，或换一种说法提问。"
)

EMPTY_REPLY = "请先输入问题。例如：石头扫地机启动前需要清理地面上的什么？"


def build_prompt(question: str, context_blocks: list[str]) -> str:
    """拼 RAG prompt。资料带编号和来源标注，便于模型引用。"""
    if context_blocks:
        ctx = "\n\n".join(context_blocks)
    else:
        ctx = "（无可用资料）"
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"【资料】\n{ctx}\n\n"
        f"【问题】{question}\n\n"
        f"【回答】"
    )


class LLMBase:
    name = "base"

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        raise NotImplementedError

    @property
    def available(self) -> bool:
        return True


class MockLLM(LLMBase):
    """占位实现：不做真实推理，只把资料摘要出来。

    ⚠️ 只用于验证 LangGraph 链路（节点顺序、状态传递、条件边、兜底），
    回答质量无意义。验收前必须换成真实模型。
    """
    name = "mock"

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        m = re.search(r"【资料】\n(.*?)\n\n【问题】", prompt, re.S)
        ctx = m.group(1).strip() if m else ""
        if not ctx or ctx == "（无可用资料）":
            return NO_RESULT_REPLY
        # 取前两条资料的前 120 字，拼成"看起来像回答"的文本
        blocks = [b.strip() for b in ctx.split("\n\n") if b.strip()][:2]
        picked = []
        for b in blocks:
            body = re.sub(r"^\[\d+\]\s*", "", b)
            picked.append(body[:120].replace("\n", " "))
        return "（占位模型输出，仅用于链路验证）\n" + "\n".join(picked)


class HFLocalLLM(LLMBase):
    """transformers 本地推理。适配 MiniMind / Qwen3-0.6B / MiniCPM-1B。

    用法：
        llm = HFLocalLLM("models/MiniMind-768.pth")   # 权重目录或 HF 仓库名
    """

    def __init__(self, model_path: str, device: str | None = None,
                 torch_dtype: str = "auto"):
        self.model_path = model_path
        self._device = device
        self._dtype = torch_dtype
        self._model = None
        self._tok = None
        self.name = "hf:" + os.path.basename(model_path.rstrip("/\\"))

    def _ensure(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True)
        kwargs = {"trust_remote_code": True}
        if self._dtype != "auto":
            kwargs["torch_dtype"] = getattr(torch, self._dtype)
        else:
            kwargs["torch_dtype"] = "auto"
        self._model = AutoModelForCausalLM.from_pretrained(self.model_path, **kwargs)
        if self._device:
            self._model = self._model.to(self._device)
        else:
            self._model = self._model.to(
                "cuda" if torch.cuda.is_available() else "cpu")
        self._model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        self._ensure()
        import torch

        messages = [{"role": "user", "content": prompt}]
        try:
            text = self._tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = prompt
        inputs = self._tok(text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # 手册问答要稳定，不采样
                repetition_penalty=1.05,
                pad_token_id=self._tok.pad_token_id or self._tok.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self._tok.decode(gen, skip_special_tokens=True).strip()

    @property
    def available(self) -> bool:
        try:
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False


def load_llm(model_path: str | None = None, allow_mock: bool = True) -> LLMBase:
    """按优先级加载：显式路径 → 项目 models/ 下的本地模型 → Mock。"""
    if model_path:
        return HFLocalLLM(model_path)

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_dir = os.path.join(root, "models")
    if os.path.isdir(models_dir):
        for name in sorted(os.listdir(models_dir)):
            d = os.path.join(models_dir, name)
            if not os.path.isdir(d) or "bge" in name.lower():
                continue
            if os.path.exists(os.path.join(d, "config.json")):
                return HFLocalLLM(d)

    if allow_mock:
        import warnings
        warnings.warn(
            "未找到本地推理模型，使用 MockLLM。它只验证链路，不代表回答质量。\n"
            "接真实模型：把 MiniMind / Qwen3-0.6B 权重放到 models/ 下，"
            "或传入 model_path。",
            RuntimeWarning)
        return MockLLM()
    raise FileNotFoundError("未找到本地推理模型")
