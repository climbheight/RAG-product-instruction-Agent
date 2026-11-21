"""本地推理模型封装。"""

from .base import (  # noqa: F401
    EMPTY_REPLY,
    FAILURE_REPLY,
    NO_RESULT_REPLY,
    HFLocalLLM,
    LLMBase,
    MockLLM,
    build_prompt,
    load_llm,
)
