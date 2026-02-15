"""带token统计的LLM模型包装器。

自动拦截所有invoke调用，记录token消耗。
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from mobius.utils.token_tracker import track_token_usage


class TokenTrackingChatModel(BaseChatModel):
    """带token统计的聊天模型包装器。"""

    model_name: str = "token_tracking_wrapper"
    temperature: float = 1.0
    max_tokens: int = 2048

    def __init__(
        self,
        model: BaseChatModel,
        operation_prefix: str = "unknown",
        chapter_index: int | None = None
    ):
        """初始化包装器。

        Args:
            model: 原始的LLM模型
            operation_prefix: 操作前缀，用于标识调用来源
            chapter_index: 当前章节索引
        """
        # 先设置基础属性
        self._model = model
        self._operation_prefix = operation_prefix
        self._chapter_index = chapter_index

        # 复制原始模型的属性
        self.model_name = getattr(model, 'model_name', 'unknown')
        self.temperature = getattr(model, 'temperature', 1.0)
        self.max_tokens = getattr(model, 'max_tokens', 2048)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any
    ) -> Any:
        """拦截generate调用，记录token消耗。"""
        start_time = time.time()

        # 调用原始模型
        result = self._model._generate(messages, stop=stop, **kwargs)

        duration_ms = (time.time() - start_time) * 1000

        # 尝试提取token信息
        input_tokens = self._extract_input_tokens(messages)
        output_tokens = self._extract_output_tokens(result)
        total_tokens = input_tokens + output_tokens

        # 记录token消耗
        track_token_usage(
            operation=self._operation_prefix,
            model_name=self.model_name,
            chapter_index=self._chapter_index,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            duration_ms=duration_ms,
            message_count=len(messages),
        )

        return result

    def _extract_input_tokens(self, messages: list[BaseMessage]) -> int:
        """估算输入token数。"""
        total_chars = sum(len(str(msg.content)) for msg in messages)
        # 粗略估算：1个中文字符≈1.5个token，英文≈0.3个token
        # 这里使用保守估算：平均1个字符≈1个token
        return max(1, int(total_chars * 0.8))

    def _extract_output_tokens(self, result: Any) -> int:
        """从响应中提取输出token数。"""
        # 尝试从不同的响应格式中提取token信息
        if hasattr(result, 'usage_metadata'):
            usage = result.usage_metadata
            if hasattr(usage, 'completion_tokens'):
                return usage.completion_tokens
            elif hasattr(usage, 'output_token_details'):
                return usage.output_token_details.get('completion_tokens', 0)

        # 如果无法获取准确token数，使用内容长度估算
        if hasattr(result, 'content'):
            content = str(result.content)
            return max(1, len(content) // 4)  # 粗略估算：4个字符≈1个token

        # 兜底值
        return 100  # 假设平均输出

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any
    ) -> Any:
        """异步版本（如果需要的话）。"""
        # 目前同步实现，异步调用时会fallback到同步
        return self._generate(messages, stop, **kwargs)

    @property
    def _llm_type(self) -> str:
        """LLM类型标识。"""
        return f"token_tracking_{self._model._llm_type}"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """识别参数。"""
        return {
            **self._model._identifying_params,
            "token_tracking": True,
            "operation_prefix": self._operation_prefix,
        }

    def __getattr__(self, name: str) -> Any:
        """代理其他属性访问到原始模型。"""
        return getattr(self._model, name)


def wrap_model_with_token_tracking(
    model: BaseChatModel,
    operation_prefix: str,
    chapter_index: int | None = None
) -> TokenTrackingChatModel:
    """为模型添加token统计包装器。"""
    return TokenTrackingChatModel(model, operation_prefix, chapter_index)


def create_token_tracking_model(
    model_class: type[BaseChatModel],
    operation_prefix: str,
    chapter_index: int | None = None,
    **model_kwargs: Any
) -> TokenTrackingChatModel:
    """创建带有token统计的模型实例。"""
    model = model_class(**model_kwargs)
    return wrap_model_with_token_tracking(model, operation_prefix, chapter_index)