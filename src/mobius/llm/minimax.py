"""MiniMax M2-her 角色扮演模型的 LangChain ChatModel 封装。

M2-her 是专为角色扮演优化的模型，API 与 OpenAI 不完全兼容：
- 端点: https://api.minimaxi.com/v1/text/chatcompletion_v2
- 特殊消息角色: user_system, group, sample_message_user, sample_message_ai
- max_completion_tokens 上限 2048
- temperature 默认 1.0
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterator, Optional

import httpx
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

logger = logging.getLogger(__name__)

_MINIMAX_API_URL = "https://api.minimaxi.com/v1/text/chatcompletion_v2"


class ChatMiniMax(BaseChatModel):
    """MiniMax M2-her 角色扮演模型。

    利用 MiniMax 独有的 user_system / sample_message 角色
    实现深度角色扮演，让角色具备独特的"思维质感"。
    """

    api_key: str = ""
    model: str = "M2-her"
    temperature: float = 1.0
    top_p: float = 0.95
    max_completion_tokens: int = 2048

    # ── 角色扮演专属配置 ──
    character_name: str = ""
    user_name: str = "导演"
    user_system_prompt: str = ""
    sample_dialogues: list[tuple[str, str]] = []

    # ── httpx ──
    timeout: float = 120.0

    class Config:  # noqa: D106
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "minimax-m2-her"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_completion_tokens": self.max_completion_tokens,
            "character_name": self.character_name,
        }

    # ------------------------------------------------------------------
    # Message 转换
    # ------------------------------------------------------------------

    def _convert_messages(self, messages: list[BaseMessage]) -> list[dict[str, Any]]:
        """将 LangChain 消息转换为 MiniMax API 格式。

        MiniMax 消息角色:
        - system: AI 角色人设 (name = character_name)
        - user_system: 用户/对话者人设
        - group: 对话名称
        - sample_message_user / sample_message_ai: 示例对话
        - user / assistant: 正常对话轮次
        """
        mm_messages: list[dict[str, Any]] = []

        # 1) 注入 system 消息（角色人设）
        for msg in messages:
            if isinstance(msg, SystemMessage):
                mm_messages.append({
                    "role": "system",
                    "name": self.character_name or "AI",
                    "content": str(msg.content),
                })
                break
        else:
            # 如果没有 SystemMessage，仍然发一个空 system 占位
            mm_messages.append({
                "role": "system",
                "name": self.character_name or "AI",
            })

        # 2) 注入 user_system（对话者人设）
        if self.user_system_prompt:
            mm_messages.append({
                "role": "user_system",
                "name": self.user_name,
                "content": self.user_system_prompt,
            })

        # 3) 注入示例对话
        for user_example, ai_example in self.sample_dialogues:
            mm_messages.append({
                "role": "sample_message_user",
                "name": self.user_name,
                "content": user_example,
            })
            mm_messages.append({
                "role": "sample_message_ai",
                "name": self.character_name or "AI",
                "content": ai_example,
            })

        # 4) 正常对话轮次
        for msg in messages:
            if isinstance(msg, SystemMessage):
                continue  # 已在第 1 步处理
            elif isinstance(msg, HumanMessage):
                mm_messages.append({
                    "role": "user",
                    "name": self.user_name,
                    "content": str(msg.content),
                })
            elif isinstance(msg, AIMessage):
                mm_messages.append({
                    "role": "assistant",
                    "name": self.character_name or "AI",
                    "content": str(msg.content),
                })
            else:
                # 其他消息类型作为 user 处理
                mm_messages.append({
                    "role": "user",
                    "name": self.user_name,
                    "content": str(msg.content),
                })

        return mm_messages

    # ------------------------------------------------------------------
    # 核心调用
    # ------------------------------------------------------------------

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """调用 MiniMax API 生成回复。"""
        mm_messages = self._convert_messages(messages)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": mm_messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_completion_tokens": self.max_completion_tokens,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(_MINIMAX_API_URL, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as e:
            logger.error("MiniMax API HTTP error: %s - %s", e.response.status_code, e.response.text)
            raise
        except Exception as e:
            logger.error("MiniMax API call failed: %s", e)
            raise

        # 检查 base_resp 错误
        base_resp = data.get("base_resp", {})
        if base_resp.get("status_code", 0) != 0:
            error_msg = base_resp.get("status_msg", "Unknown error")
            raise RuntimeError(f"MiniMax API error [{base_resp.get('status_code')}]: {error_msg}")

        # 提取内容
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("MiniMax API returned empty choices")

        content = choices[0].get("message", {}).get("content", "")
        ai_message = AIMessage(content=content)

        # 用量信息
        usage = data.get("usage", {})
        generation = ChatGeneration(
            message=ai_message,
            generation_info={
                "finish_reason": choices[0].get("finish_reason", ""),
                "usage": usage,
                "input_sensitive": data.get("input_sensitive", False),
                "output_sensitive": data.get("output_sensitive", False),
            },
        )

        return ChatResult(
            generations=[generation],
            llm_output={
                "model": data.get("model", self.model),
                "usage": usage,
            },
        )

    def with_character(
        self,
        character_name: str,
        user_system_prompt: str = "",
        sample_dialogues: list[tuple[str, str]] | None = None,
    ) -> "ChatMiniMax":
        """返回一个绑定了特定角色的新实例。

        便于为不同角色创建独立的 M2-her 调用者。
        """
        return ChatMiniMax(
            api_key=self.api_key,
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            max_completion_tokens=self.max_completion_tokens,
            character_name=character_name,
            user_name=self.user_name,
            user_system_prompt=user_system_prompt or self.user_system_prompt,
            sample_dialogues=sample_dialogues if sample_dialogues is not None else self.sample_dialogues,
            timeout=self.timeout,
        )
