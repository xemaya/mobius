"""叙事生成 Agent：将角色行动和对话编译为流畅的小说文本。

升级功能：
- 潜台词文学化处理：利用 internal_monologue 创造言不由衷的深度
- 支线视角片段融合：在主线叙事中穿插非主角视角的观察
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from mobius.agents.utils import extract_response_text
from mobius.engine.chaos_engine import ChaosEngine
from mobius.models.chapter import Chapter, Scene
from mobius.models.character import CharacterAction
from mobius.models.viewpoint import ViewpointFragment
from mobius.prompts import load_prompt
from mobius.state.novel_state import NovelState

logger = logging.getLogger(__name__)

# 从外部文件加载提示词
NARRATOR_SYSTEM_PROMPT = load_prompt("narrator_system")
NARRATION_ONLY_PROMPT = load_prompt("narrator_narration_only")


def create_narration_node(model: BaseChatModel) -> Callable[[NovelState], dict[str, Any]]:
    """创建旁白/环境描写节点。用于不需要角色行动的纯叙事场景。"""

    def narration_node(state: NovelState) -> dict[str, Any]:
        scene: Scene | None = state.get("current_scene")
        if not scene:
            return {"next_action": "direct_scene"}

        chapter_plan = state.get("chapter_plan")
        chapter_title = chapter_plan.title if chapter_plan else ""

        # 环境状态注入
        env_ctx = ""
        env_state = state.get("environment")
        if env_state and env_state.variables:
            env_lines = [
                f"- {v.description or v.name}: {v.value:.0f}"
                for v in env_state.variables.values()
            ]
            env_ctx = f"\n\n环境氛围参考（不要直接提及数值）：\n" + "\n".join(env_lines)

        user_prompt = f"""# 章节：{chapter_title}

# 场景：{scene.title or scene.scene_id}
{scene.description}

地点：{scene.location or '未指定'}
氛围：{scene.mood or '未指定'}
导演指示：{scene.director_notes or '无'}{env_ctx}

请撰写 200-400 字的环境描写或旁白叙述。"""

        messages = [
            SystemMessage(content=NARRATION_ONLY_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        try:
            response = model.invoke(messages)
            narration = extract_response_text(response).strip()
            logger.info("旁白生成完成: %d 字", len(narration))
            return {
                "narrative_buffer": [narration],
                "next_action": "update_state",
            }
        except Exception as e:
            logger.error("旁白生成失败: %s", e)
            return {
                "narrative_buffer": [f"（{scene.description}）"],
                "next_action": "update_state",
            }

    return narration_node


def create_compile_chapter_node(
    model: BaseChatModel,
    chaos_engine: ChaosEngine | None = None,
) -> Callable[[NovelState], dict[str, Any]]:
    """创建章节编译节点。

    将所有场景的行动记录、叙事整合为完整章节。
    新增：潜台词素材 + 支线视角片段。
    """

    def compile_chapter_node(state: NovelState) -> dict[str, Any]:
        chapter_plan = state.get("chapter_plan")
        if not chapter_plan:
            return {"error": "缺少章节规划", "next_action": "end"}

        narrative_buffer = state.get("narrative_buffer", [])
        scene_actions = state.get("scene_actions", [])
        viewpoint_fragments = state.get("viewpoint_fragments", [])

        # 整理所有素材
        raw_materials: list[str] = []

        # 添加叙事缓冲中的文本
        for narration in narrative_buffer:
            raw_materials.append(narration)

        # 添加角色行动记录（外显行为）
        if scene_actions:
            action_texts: list[str] = []
            for action in scene_actions:
                action_texts.append(
                    f"[{action.character_name} - {action.action_type}]\n{action.content}"
                )
            raw_materials.append("\n\n".join(action_texts))

        all_materials = "\n\n---\n\n".join(raw_materials)

        # ── 构建潜台词素材（内心独白）──
        subtext_materials = ""
        monologue_actions = [a for a in scene_actions if a.internal_monologue]
        if monologue_actions:
            subtext_parts = [
                "\n# 潜台词素材（角色真实内心活动——不要直白写出，要用文学手法暗示）",
                "以下是角色在行动时的真实内心想法。请在叙事中巧妙暗示这些想法，",
                "但不要直接说出。用环境描写、微表情、语气词等手法来传达'言不由衷'的感觉。\n",
            ]
            for action in monologue_actions:
                dissonance_hint = ""
                if action.cognitive_dissonance > 0.3:
                    dissonance_hint = "（此角色的外在表现与内心想法有显著差异——重要的文学张力点！）"
                subtext_parts.append(
                    f"[{action.character_name}的潜台词] {dissonance_hint}\n"
                    f"外显行为：{action.content[:100]}...\n"
                    f"真实想法：{action.internal_monologue[:300]}"
                )
            subtext_materials = "\n\n".join(subtext_parts)

        # ── 构建支线视角素材 ──
        viewpoint_materials = ""
        if viewpoint_fragments:
            vp_parts = [
                "\n# 支线视角素材（请在合适位置插入，作为插叙或画外音）",
                "以下是从非主角视角观察到的片段。请在合适的位置自然地穿插这些内容，",
                "增加叙事的空间感和多层次感。注意保持视角切换的自然过渡。\n",
            ]
            for frag in viewpoint_fragments:
                vp_parts.append(
                    f"[视角：{frag.viewpoint_name}]\n{frag.content}"
                )
            viewpoint_materials = "\n\n".join(vp_parts)

        # ── 构建上一章结尾上下文 ──
        prev_chapter_ctx = ""
        completed_chapters = state.get("chapters", [])
        if completed_chapters:
            last_chapter = completed_chapters[-1]
            ending_snippet = last_chapter.content[-300:]
            prev_chapter_ctx = f"""# 上一章结尾（第{last_chapter.chapter_index}章「{last_chapter.title}」）
以下是上一章的最后一段，你必须确保本章开头与之自然衔接，并且绝对不能重复以下内容中的描写、意象和用词：

---
{ending_snippet}
---

"""

        user_prompt = f"""{prev_chapter_ctx}# 章节规划
标题：第{chapter_plan.chapter_index}章 {chapter_plan.title}
概要：{chapter_plan.summary}
关键事件：{'、'.join(chapter_plan.key_events)}
情感基调：{chapter_plan.emotional_tone or '未指定'}

# 原始素材（角色行动记录和叙事片段）
{all_materials}

{subtext_materials}

{viewpoint_materials}

---

请将以上原始素材编织成一篇完整、流畅的小说章节。要求：
1. 保持角色对话和行动的核心内容
2. 添加必要的过渡、环境描写和叙事衔接
3. 确保情节连贯，节奏得当
4. 章节开头要与上一章结尾自然衔接（如果有的话），不得重复上一章已有的环境描写和意象
5. 章节结尾要有悬念或总结
6. 篇幅至少 2500 字，写细写深，不要赶进度
7. 【潜台词处理】巧妙暗示角色的真实内心，不要直白说出——用文学手法让读者"感受到"言不由衷
8. 【支线视角融合】在合适位置自然插入支线视角的片段，增加空间感

直接输出小说正文，以章节标题开头。"""

        messages = [
            SystemMessage(content=NARRATOR_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        try:
            response = model.invoke(messages)
            chapter_text = extract_response_text(response).strip()

            # 【v2.1】应用去AI味处理
            if chaos_engine:
                chapter_text = chaos_engine.process_text_humanization(
                    chapter_text,
                    context=f"章节{chapter_plan.chapter_index}: {chapter_plan.title}"
                )
                logger.info("章节文本已应用去AI味处理")

            word_count = len(chapter_text)

            chapter = Chapter(
                chapter_index=chapter_plan.chapter_index,
                title=chapter_plan.title,
                content=chapter_text,
                summary=chapter_plan.summary,
                word_count=word_count,
            )

            logger.info(
                "章节编译完成: 第%d章「%s」, %d字 (潜台词素材:%d条, 支线视角:%d条)",
                chapter.chapter_index,
                chapter.title,
                word_count,
                len(monologue_actions),
                len(viewpoint_fragments),
            )

            next_idx = chapter_plan.chapter_index + 1
            total = state.get("total_chapters", 10)
            is_finished = next_idx > total

            return {
                "chapters": [chapter],
                "current_chapter_index": next_idx,
                "scene_actions": [],
                "narrative_buffer": [],
                "viewpoint_fragments": [],
                "chapter_plan": None,
                "scene_queue": [],
                "current_scene": None,
                "next_action": "review_chapter" if not is_finished else "end",
            }
        except Exception as e:
            logger.error("章节编译失败: %s", e)
            return {"error": str(e), "next_action": "end"}

    return compile_chapter_node
