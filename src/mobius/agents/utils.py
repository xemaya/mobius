"""Agent 通用工具函数。"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)


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

    return json.loads(text)


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
