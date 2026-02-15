"""文风控制器 (Style Governor)。

进一步去除 AI 味，实现张弛结构，强制执行比喻密度限制、抽象词压缩、哲学降频和普通句比例强制。
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from mobius.agents.utils import extract_response_text, invoke_with_retry
from mobius.models.chaos_engine import StyleGovernorConfig
from mobius.prompts import format_prompt

logger = logging.getLogger(__name__)


class StyleGovernor:
    """文风控制器：小说生成的最后一道关卡。"""

    def __init__(self, config: StyleGovernorConfig, model: Optional[BaseChatModel] = None):
        self.config = config
        self.model = model

    def process_chapter(self, title: str, content: str) -> str:
        """对整章内容进行文风后处理。"""
        if not self.config.enabled:
            return content

        processed_content = content

        # 1. 规则级清理（不依赖 LLM，立即生效）
        processed_content = self._regex_cool_down(processed_content)

        # 2. 若启用且提供模型，执行 LLM 改写（带重试）
        if self.model and self.config.use_llm_post_process:
            processed_content = self._llm_post_process(title, processed_content)

        return processed_content

    def _regex_cool_down(self, content: str) -> str:
        """基于规则的文风降温——不依赖 LLM，快速且确定性。"""

        # 1. 情绪总结句式替换（"她感到X" → 动作描写）
        emotion_patterns = [
            (r"她感到一阵(.{2,8})。", "她攥紧了拳头，呼吸变得急促。"),
            (r"她?觉得(.{2,12})。", "她移开了视线，没有说话。"),
            (r"他感到(.{2,8})。", "他移开了视线，喉结动了动。"),
            (r"她终于明白了?(.{2,15})。", "她在原地站了很久，然后转身离开了。"),
            (r"他终于明白了?(.{2,15})。", "他低下头，没有再说什么。"),
            (r"她?意识到(.{2,12})的真谛。", "她关掉了屏幕，在黑暗中坐了很久。"),
        ]
        for pattern, replacement in emotion_patterns:
            content = re.sub(pattern, replacement, content, count=1)

        # 2. 四字成语/文艺腔降级（规则来自 config，可通过 YAML 覆盖）
        for cliche, plain in self.config.cliche_replacements.items():
            content = content.replace(cliche, plain)

        # 3. 过度诗意的比喻词降级（规则来自 config，可通过 YAML 覆盖）
        for poetic, plain in self.config.poetic_downgrades.items():
            content = content.replace(poetic, plain)

        # 4. 统计并警告抽象词密度
        abstract_count = 0
        for noun in self.config.abstract_nouns_to_kill:
            abstract_count += content.count(noun)

        if abstract_count > self.config.max_abstract_nouns_per_chapter:
            logger.info(
                "StyleGovernor: 检测到 %d 个抽象词（上限 %d）",
                abstract_count, self.config.max_abstract_nouns_per_chapter,
            )

        return content

    def _llm_post_process(self, title: str, content: str) -> str:
        """使用 LLM 执行风格微调，严格保持原始篇幅。带重试。"""

        original_len = len(content)

        system_prompt = format_prompt(
            "style_governor_system",
            abstract_nouns=", ".join(self.config.abstract_nouns_to_kill),
            original_len=str(original_len),
        )

        human_msg = f"章节标题：{title}\n\n原始字数：{original_len} 字\n\n章节正文：\n{content}"

        # 带重试（最多2次）
        for attempt in range(2):
            try:
                response = invoke_with_retry(
                    self.model,
                    [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=human_msg),
                    ],
                    operation_name="StyleGovernor",
                )
                result = extract_response_text(response).strip()

                # 字数保护：如果处理后字数低于原始的 70%，说明 LLM 过度压缩了，放弃修改
                if len(result) < original_len * 0.7:
                    logger.warning(
                        "StyleGovernor 输出字数过少 (%d vs 原始 %d)，放弃修改",
                        len(result), original_len,
                    )
                    return content

                return result
            except Exception as e:
                if attempt == 0:
                    logger.warning("StyleGovernor 第1次尝试失败: %s，重试中...", e)
                else:
                    logger.error("StyleGovernor 处理失败（已重试）: %s", e)

        return content
