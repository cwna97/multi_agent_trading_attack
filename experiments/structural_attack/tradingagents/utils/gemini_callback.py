"""
Gemini API 비용 추적 콜백 (get_openai_callback과 동일한 인터페이스)

Usage:
    from tradingagents.utils.gemini_callback import get_gemini_callback

    with get_gemini_callback() as cb:
        _, decision = ta.propagate(ticker, date_str)

    print(cb.total_cost, cb.prompt_tokens, cb.completion_tokens)
"""

from contextvars import ContextVar
from contextlib import contextmanager
from typing import Any, Generator, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.tracers.context import register_configure_hook

# USD per token
_PRICING: dict[str, dict[str, float]] = {
    "gemini-2.5-pro":   {"input": 1.25e-6,  "output": 10.0e-6,  "thinking": 10.0e-6},
    "gemini-2.5-flash": {"input": 0.075e-6, "output": 0.30e-6,  "thinking": 3.50e-6},
}
_DEFAULT_PRICING = {"input": 1.0e-6, "output": 4.0e-6, "thinking": 4.0e-6}


class GeminiCallbackHandler(BaseCallbackHandler):
    """Gemini 토큰 사용량 및 비용을 누적하는 콜백 핸들러."""

    def __init__(self) -> None:
        super().__init__()
        self.prompt_tokens:     int   = 0
        self.completion_tokens: int   = 0
        self.thinking_tokens:   int   = 0
        self.total_tokens:      int   = 0
        self.total_cost:        float = 0.0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        for gen_list in response.generations:
            if not gen_list:
                continue
            gen = gen_list[0]
            msg = getattr(gen, "message", None)
            if msg is None:
                continue
            usage = getattr(msg, "usage_metadata", None)
            if not usage:
                continue

            input_tok    = usage.get("input_tokens", 0)
            output_tok   = usage.get("output_tokens", 0)
            details      = usage.get("output_token_details") or {}
            thinking_tok = details.get("reasoning", 0)
            non_think    = max(output_tok - thinking_tok, 0)

            model_name = (getattr(msg, "response_metadata", None) or {}).get("model_name", "")
            pricing    = _PRICING.get(model_name, _DEFAULT_PRICING)

            cost = (
                input_tok    * pricing["input"]
                + non_think  * pricing["output"]
                + thinking_tok * pricing["thinking"]
            )

            self.prompt_tokens     += input_tok
            self.completion_tokens += output_tok
            self.thinking_tokens   += thinking_tok
            self.total_tokens      += input_tok + output_tok
            self.total_cost        += cost


gemini_callback_var: ContextVar[Optional[GeminiCallbackHandler]] = ContextVar(
    "gemini_callback", default=None
)
register_configure_hook(gemini_callback_var, True)


@contextmanager
def get_gemini_callback() -> Generator[GeminiCallbackHandler, None, None]:
    """Gemini LLM 호출 비용을 추적하는 컨텍스트 매니저."""
    cb = GeminiCallbackHandler()
    gemini_callback_var.set(cb)
    try:
        yield cb
    finally:
        gemini_callback_var.set(None)
