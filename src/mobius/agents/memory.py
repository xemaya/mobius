"""记忆蒸馏 Agent：将角色的短期记忆压缩为结构化长期摘要。

升级功能：
- 输出 StructuredMemorySummary（关键冲突/关系变化/未解悬念/信念转变/资源变动/情绪弧线）
- 保留纯文本 compressed_memory 给角色读取
- 特别关注"未兑现的欲望"和"未回收的伏笔"
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from mobius.agents.utils import extract_json, extract_response_text
from mobius.models.character import CharacterDynamicState
from mobius.models.review import StructuredMemorySummary
from mobius.prompts import load_prompt
from mobius.state.novel_state import NovelState

logger = logging.getLogger(__name__)

# 从外部文件加载提示词
DISTILL_SYSTEM_PROMPT = load_prompt("memory_distill_system")


def create_compress_memories_node(
    model: BaseChatModel,
    compress_threshold: int = 8,
    keep_recent: int = 5,
) -> Callable[[NovelState], dict[str, Any]]:
    """创建记忆蒸馏节点（升级版 compress_memories）。

    输出结构化的 StructuredMemorySummary + 纯文本 compressed_memory。
    """

    def distill_memory_node(state: NovelState) -> dict[str, Any]:
        char_states = dict(state.get("character_states", {}))
        memory_summaries = list(state.get("memory_summaries", []))
        updated_any = False

        for name, cs in char_states.items():
            if len(cs.memory) <= compress_threshold:
                continue

            old_memories = cs.memory[:-keep_recent] if keep_recent > 0 else cs.memory
            recent_memories = cs.memory[-keep_recent:] if keep_recent > 0 else []

            if not old_memories:
                continue

            existing_compressed = cs.compressed_memory or ""
            old_text = "\n".join(f"- {m}" for m in old_memories)

            # 构建角色状态上下文
            belief_ctx = ""
            if cs.beliefs:
                belief_ctx = "\n当前信念：" + "；".join(
                    f"「{b.statement}」({b.layer},{b.strength:.1f})" for b in cs.beliefs
                )
            desire_ctx = ""
            if cs.desires:
                desire_ctx = "\n当前欲望：" + "；".join(
                    f"「{d.description}」(优先级:{d.priority:.1f},{'已满足' if d.satisfied else '未满足'})"
                    for d in cs.desires
                )

            user_prompt = f"""# 角色：{name}

## 已有长期记忆摘要
{existing_compressed if existing_compressed else "（暂无）"}

## 需要蒸馏的新记忆条目
{old_text}

## 角色内在状态参考
{belief_ctx}
{desire_ctx}

---
请进行两项工作：

1. 输出结构化记忆分析（JSON 格式）：
```json
{{
    "key_conflicts": ["关键冲突1", "关键冲突2"],
    "relationship_changes": ["关系变化1"],
    "unresolved_tensions": ["未解决悬念1"],
    "belief_shifts": ["信念变化1"],
    "resource_shifts": ["资源变动1"],
    "emotional_arc": "这段时间的情绪弧线概括"
}}
```

2. 在 JSON 之后，输出更新后的纯文本记忆摘要（不超过 500 字，用 `---SUMMARY---` 分隔符标记开始）。"""

            messages = [
                SystemMessage(content=DISTILL_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]

            try:
                response = model.invoke(messages)
                text = extract_response_text(response).strip()

                # 尝试提取结构化摘要
                try:
                    json_data = extract_json(text)
                    summary = StructuredMemorySummary.model_validate(json_data)
                    memory_summaries.append(summary)
                except Exception:
                    logger.warning("角色「%s」结构化摘要解析失败，跳过", name)

                # 提取纯文本摘要
                new_compressed = text
                if "---SUMMARY---" in text:
                    new_compressed = text.split("---SUMMARY---", 1)[1].strip()
                elif "```" in text:
                    # JSON 在代码块里，摘要在代码块之后
                    parts = text.rsplit("```", 1)
                    if len(parts) > 1 and parts[1].strip():
                        new_compressed = parts[1].strip()

                # 如果纯文本摘要太短，退化为拼接
                if len(new_compressed) < 50:
                    new_compressed = (existing_compressed + "\n" + old_text).strip()[:500]

                updated_cs = cs.model_copy(deep=True)
                updated_cs.compressed_memory = new_compressed
                updated_cs.memory = recent_memories
                char_states[name] = updated_cs
                updated_any = True

                logger.info(
                    "角色「%s」记忆蒸馏完成: %d 条旧记忆 → %d 字摘要, 保留 %d 条近期记忆",
                    name, len(old_memories), len(new_compressed), len(recent_memories),
                )
            except Exception as e:
                logger.error("角色「%s」记忆蒸馏失败: %s", name, e)
                updated_cs = cs.model_copy(deep=True)
                overflow = "\n".join(old_memories)
                if updated_cs.compressed_memory:
                    updated_cs.compressed_memory += "\n" + overflow
                else:
                    updated_cs.compressed_memory = overflow
                updated_cs.memory = recent_memories
                char_states[name] = updated_cs
                updated_any = True

        # next_action 路由由 persist_memory 节点负责
        result: dict[str, Any] = {
            "memory_summaries": memory_summaries,
        }
        if updated_any:
            result["character_states"] = char_states
        return result

    return distill_memory_node
