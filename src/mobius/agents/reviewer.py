"""章节评审 Agent：融合主题守护、张力控制、自我批评。

单次 LLM 调用完成三项评审，输出 ChapterReview。
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from mobius.agents.utils import extract_json, extract_response_text, invoke_with_retry
from mobius.models.review import ChapterReview
from mobius.prompts import load_prompt
from mobius.state.novel_state import NovelState

logger = logging.getLogger(__name__)

# 从外部文件加载提示词
REVIEWER_SYSTEM_PROMPT = load_prompt("reviewer_system")
REVIEW_SCHEMA = load_prompt("reviewer_schema")


def create_review_chapter_node(
    model: BaseChatModel,
) -> Callable[[NovelState], dict[str, Any]]:
    """创建章节评审节点。

    输入: 章节文本 + theme + tension_curve + 角色信念状态
    输出: ChapterReview
    """

    def review_chapter_node(state: NovelState) -> dict[str, Any]:
        chapters = state.get("chapters", [])
        if not chapters:
            return {"next_action": "distill_memory"}

        latest_chapter = chapters[-1]
        theme = state.get("theme", "")
        tension_curve = state.get("tension_curve", [])
        chapter_plan = state.get("chapter_plan")

        # 构建角色信念/欲望摘要
        char_states = state.get("character_states", {})
        belief_summary_parts = []
        for name, cs in char_states.items():
            parts = [f"## {name}"]
            if cs.beliefs:
                core_beliefs = [b for b in cs.beliefs if b.layer == "core"]
                if core_beliefs:
                    parts.append("  核心信念: " + "; ".join(
                        f"「{b.statement}」({b.strength:.1f})" for b in core_beliefs
                    ))
            if cs.desires:
                top_desires = sorted(cs.desires, key=lambda d: d.priority, reverse=True)[:2]
                parts.append("  首要欲望: " + "; ".join(d.description for d in top_desires))
            belief_summary_parts.append("\n".join(parts))
        belief_summary = "\n".join(belief_summary_parts) if belief_summary_parts else "（无信念数据）"

        user_prompt = f"""# 核心主题
{theme or '（未设定）'}

# 章节内容（第{latest_chapter.chapter_index}章：{latest_chapter.title}）
{latest_chapter.content[:3000]}
{"...（已截断）" if len(latest_chapter.content) > 3000 else ""}

# 章节计划
{chapter_plan.summary if chapter_plan else '（无计划）'}

# 历史张力曲线
{tension_curve[-5:] if tension_curve else '（无历史数据）'}

# 角色内在状态
{belief_summary}

---
请对这一章进行全面评审。

{REVIEW_SCHEMA}"""

        try:
            response = invoke_with_retry(
                model,
                [
                    SystemMessage(content=REVIEWER_SYSTEM_PROMPT),
                    HumanMessage(content=user_prompt),
                ],
                operation_name="review_chapter",
            )
            text = extract_response_text(response)
            review_data = extract_json(text)
            review = ChapterReview.model_validate(review_data)

            # 更新张力曲线
            updated_tension = list(tension_curve)
            updated_tension.append(review.tension_score)

            # 添加到评审历史
            reviews = list(state.get("chapter_reviews", []))
            reviews.append(review)

            logger.info(
                "章节评审完成: 主题契合度=%.2f, 张力=%.2f, 主题推进=%.2f, 回收率=%.2f, 逻辑问题=%d个",
                review.theme_alignment,
                review.tension_score,
                review.theme_progression,
                review.thread_recovery_rate,
                len(review.logic_issues),
            )

            return {
                "chapter_reviews": reviews,
                "tension_curve": updated_tension,
                "next_action": "distill_memory",
            }
        except Exception as e:
            logger.error("章节评审失败: %s", e)
            return {"next_action": "distill_memory"}

    return review_chapter_node
