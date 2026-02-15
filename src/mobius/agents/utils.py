"""Agent 通用工具函数。"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)

# 可重试的异常：网络/限流/临时故障
RETRYABLE_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    OSError,
)


def invoke_with_retry(
    model: BaseChatModel,
    messages: list[BaseMessage],
    max_retries: int = 2,
    base_delay: float = 2.0,
    operation_name: str = "invoke",
) -> Any:
    """带重试的 LLM 调用，应对网络抖动与限流。

    - 仅对可重试异常（连接、超时、OS 等）重试，其他异常直接抛出。
    - 重试间隔指数退避：base_delay, base_delay*2, ...
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return model.invoke(messages)
        except RETRYABLE_EXCEPTIONS as e:
            last_exc = e
            if attempt < max_retries:
                delay = base_delay * (2**attempt)
                logger.warning(
                    "%s 第 %d 次失败 (%s)，%s 秒后重试",
                    operation_name,
                    attempt + 1,
                    type(e).__name__,
                    delay,
                )
                time.sleep(delay)
            else:
                logger.error("%s 重试 %d 次后仍失败: %s", operation_name, max_retries, e)
                raise
        except Exception:
            raise
    if last_exc:
        raise last_exc
    raise RuntimeError("invoke_with_retry unexpected state")


def extract_text(content: str | list | Any) -> str:
    """从 LLM 响应中提取纯文本内容。

    不同模型提供商返回的 content 格式不同：
    - OpenAI: 直接返回 str
    - Google Gemini: 返回 list[dict]，每个 dict 包含 'type' 和 'text'

    此函数统一处理这些差异。
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content)


def extract_response_text(response: BaseMessage) -> str:
    """从 LLM 响应消息中提取纯文本。"""
    return extract_text(response.content)


def extract_json(text: str) -> Any:
    """从 LLM 输出中提取 JSON 数据。

    支持从 markdown 代码块和纯文本中提取。
    """
    # 尝试从 ```json ... ``` 代码块中提取
    try:
        if "```json" in text:
            start = text.index("```json") + len("```json")
            end = text.index("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.index("```") + 3
            # 跳过可能的语言标记行
            if "\n" in text[start : start + 20]:
                start = text.index("\n", start) + 1
            end = text.index("```", start)
            text = text[start:end].strip()
    except ValueError:
        # 代码块标记不完整（如只有开头没有结尾），尝试用 { } 定位
        pass

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 兜底：尝试找到第一个 { 和最后一个 }
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace > first_brace:
            return json.loads(text[first_brace : last_brace + 1])
        # 尝试找到第一个 [ 和最后一个 ]
        first_bracket = text.find("[")
        last_bracket = text.rfind("]")
        if first_bracket != -1 and last_bracket > first_bracket:
            return json.loads(text[first_bracket : last_bracket + 1])
        raise


def extract_json_safe(text: str, fallback_name: str = "") -> dict:
    """安全提取 JSON，带容错处理。"""
    try:
        return extract_json(text)
    except (json.JSONDecodeError, ValueError):
        # 尝试找到第一个 { 和最后一个 }
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            logger.warning("无法解析 JSON 输出 (%s)，使用原始文本", fallback_name)
            return {
                "character_name": fallback_name,
                "action_type": "mixed",
                "content": text[:500],
                "emotional_change": {},
                "relationship_change": {},
                "attribute_change": {},
                "new_memory": "",
            }
