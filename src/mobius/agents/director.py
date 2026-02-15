"""导演 Agent（编排者）：负责章节规划、场景编排和触发响应。

导演的角色从"剧本作者"变成"编排者"——他不再凭空创造剧情，
而是从观察者提供的 narrative_candidates 中选择和组织素材。
冲突来自角色欲望碰撞，不要人为制造冲突。
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from mobius.agents.utils import extract_json, extract_response_text, invoke_with_retry
from mobius.models.chapter import ChapterPlan, Scene
from mobius.models.character import CharacterDynamicState, CharacterProfile
from mobius.models.desire import DesireProposal
from mobius.models.review import ChapterReview, WorldEvent
from mobius.models.worldview import PlotOutline, WorldView
from mobius.prompts import load_prompt
from mobius.state.novel_state import NovelState

logger = logging.getLogger(__name__)

# 从外部文件加载提示词
DIRECTOR_SYSTEM_PROMPT = load_prompt("director_system")
CHAPTER_PLAN_SCHEMA = load_prompt("director_chapter_plan_schema")
SCENE_SILENCE_NOTES = load_prompt("scene_silence_notes")
SCENE_DAILY_NOTES = load_prompt("scene_daily_notes")
SCENE_DAILY_AUTO_INJECT = load_prompt("scene_daily_auto_inject")


def _build_worldview_context(state: NovelState) -> str:
    """构建世界观上下文摘要。"""
    wv: WorldView = state["worldview"]
    po: PlotOutline = state["plot_outline"]
    parts = [
        f"# 世界观：{wv.name}",
        wv.description,
        f"\n时代背景：{wv.era}" if wv.era else "",
        f"力量体系：{wv.power_system}" if wv.power_system else "",
        f"社会结构：{wv.social_structure}" if wv.social_structure else "",
        f"\n# 小说：{po.title}（{po.genre}）",
        f"主题：{po.theme}" if po.theme else "",
        f"概要：{po.synopsis}",
        f"起：{po.beginning}",
        f"承：{po.development}",
        f"转：{po.twist}",
        f"合：{po.conclusion}",
    ]
    if wv.key_factions:
        parts.append(f"\n关键势力：{'、'.join(wv.key_factions)}")
    return "\n".join(p for p in parts if p)


def _build_characters_summary(state: NovelState) -> str:
    """构建角色概况摘要（含信念、欲望、资源、动态状态）。"""
    profiles: dict[str, CharacterProfile] = state["character_profiles"]
    dyn_states: dict[str, CharacterDynamicState] = state.get("character_states", {})

    lines: list[str] = ["# 角色一览"]
    for name, profile in profiles.items():
        ds = dyn_states.get(name)
        line = f"\n## {name}（{profile.role}）\n性格：{profile.personality}"
        if profile.goals:
            line += f"\n目标：{'、'.join(profile.goals)}"
        if ds:
            if ds.emotional_state:
                emotions = ", ".join(f"{k}:{v:.1f}" for k, v in ds.emotional_state.items())
                line += f"\n当前情感：{emotions}"
            if ds.relationship_scores:
                rels = ", ".join(f"{k}:{v:+.1f}" for k, v in ds.relationship_scores.items())
                line += f"\n人际关系：{rels}"
            # 信念摘要
            if ds.beliefs:
                core = [b for b in ds.beliefs if b.layer == "core"]
                if core:
                    line += f"\n核心信念：" + "；".join(
                        f"「{b.statement}」({b.strength:.1f})" for b in core
                    )
            # 首要欲望
            if ds.desires:
                top = sorted(ds.desires, key=lambda d: d.priority, reverse=True)[:2]
                line += f"\n首要欲望：" + "；".join(d.description for d in top)
            # 资源概况
            if ds.resources:
                pool = ds.resources
                critical = []
                if pool.emotional_energy < 30:
                    critical.append(f"情绪能量危急({pool.emotional_energy:.0f})")
                if pool.reputation < 20:
                    critical.append(f"声誉危急({pool.reputation:.0f})")
                if pool.time < 20:
                    critical.append(f"时间不足({pool.time:.0f})")
                if critical:
                    line += f"\n⚠️ 资源警告：" + "、".join(critical)
            # 记忆
            if ds.memory:
                recent = ds.memory[-3:]
                line += f"\n近期记忆：{'；'.join(recent)}"
        lines.append(line)
    return "\n".join(lines)


def _build_previous_chapters_summary(state: NovelState) -> str:
    """构建已完成章节的摘要。"""
    chapters = state.get("chapters", [])
    if not chapters:
        return "（这是第一章，尚无已完成章节。）"
    lines = ["# 已完成章节摘要"]
    for ch in chapters:
        lines.append(f"\n## 第{ch.chapter_index}章：{ch.title}")
        lines.append(ch.summary if ch.summary else "（无摘要）")
    return "\n".join(lines)


def _build_narrative_candidates_context(state: NovelState) -> str:
    """构建观察者选出的高价值叙事事件上下文。"""
    world_events = state.get("world_events", [])
    selected = [e for e in world_events if e.selected_for_narrative]
    if not selected:
        return "（没有观察者标记的叙事候选事件，请根据角色欲望自行编排。）"

    lines = ["# 观察者标记的高价值事件（请从中选择素材编排本章）"]
    for i, evt in enumerate(selected):
        lines.append(
            f"{i+1}. [{evt.conflict_type}] {evt.description} "
            f"(叙事价值:{evt.narrative_value:.2f}, 参与者:{', '.join(evt.participants)})"
        )
    return "\n".join(lines)


def _build_desire_proposals_context(state: NovelState) -> str:
    """构建角色欲望提案上下文。"""
    proposals = state.get("desire_proposals", [])
    if not proposals:
        return ""
    lines = ["# 角色欲望提案"]
    for p in proposals:
        lines.append(f"- {p.character_name}：想要「{p.proposed_action}」(紧迫度:{p.urgency:.1f})")
    return "\n".join(lines)


def _build_review_feedback_context(state: NovelState) -> str:
    """构建上一章评审反馈。"""
    reviews = state.get("chapter_reviews", [])
    if not reviews:
        return ""
    last_review = reviews[-1]
    lines = [
        "# 上一章评审反馈",
        f"主题契合度：{last_review.theme_alignment:.2f}",
    ]
    if last_review.theme_drift_notes:
        lines.append(f"主题偏移：{last_review.theme_drift_notes}")
    lines.append(f"张力评分：{last_review.tension_score:.2f}")
    if last_review.pacing_notes:
        lines.append(f"节奏建议：{last_review.pacing_notes}")
    if last_review.logic_issues:
        lines.append(f"待修复逻辑问题：" + "；".join(last_review.logic_issues))
    if last_review.suggestions_for_next:
        lines.append(f"编排建议：{last_review.suggestions_for_next}")
    return "\n".join(lines)


# ────────────────────────────────────────────
# 导演节点函数
# ────────────────────────────────────────────

def create_director_plan_chapter(model: BaseChatModel, chaos_engine: Optional[ChaosEngine] = None):
    """创建导演的「章节规划」节点函数。
    
    编排者从 narrative_candidates 和欲望提案中组织本章素材。
    """

    def director_plan_chapter(state: NovelState) -> dict[str, Any]:
        current_idx = state.get("current_chapter_index", 1)
        total = state.get("total_chapters", 10)

        worldview_ctx = _build_worldview_context(state)
        characters_ctx = _build_characters_summary(state)
        prev_chapters_ctx = _build_previous_chapters_summary(state)

        # 1. 节奏控制：张弛算法
        rhythm_instruction = ""
        if chaos_engine and chaos_engine.config.rhythm_config.enabled:
            cycle = chaos_engine.config.rhythm_config.cycle
            rhythm_state = cycle[(current_idx - 1) % len(cycle)]
            
            rhythm_descriptions = {
                "high_tension": "高张力段：冲突爆发，情绪达到顶点，节奏紧凑。",
                "cooling": "冷却段：冲突后的余震，情绪的回落，节奏放缓。",
                "daily": "日常段：琐碎的生活细节，平淡的叙述，没有宏大主题。",
                "unexpected": "意外段：平淡中的突发小插曲，非预期的转折。",
                "silence": "沉默段：极少的对话，大量的空白和动作描写。",
                "outbreak": "再爆发：蓄势后的再次冲突，情绪的重新拉升。"
            }
            rhythm_instruction = f"\n# 节奏要求（张弛算法）\n当前章节节奏状态：**{rhythm_state}**\n说明：{rhythm_descriptions.get(rhythm_state, '')}"
            
            if chaos_engine.config.rhythm_config.low_energy_segment_required_after_high_density:
                rhythm_instruction += "\n注意：在本章的场景编排中，如果前一个场景是高张力/高密度情绪，请务必插入一个‘低能量场景’（无冲突、无主题，仅环境或简单动作）。"

        # 2. 角色失误：Human Error Engine
        error_instructions = []
        if chaos_engine:
            for char_name in state.get("character_profiles", {}):
                err_msg = chaos_engine.get_character_error_instruction(current_idx, char_name)
                if err_msg:
                    error_instructions.append(f"- {char_name}: {err_msg}")
        
        error_ctx = ""
        if error_instructions:
            error_ctx = "\n# 角色执行修正（Human Error Engine）\n本章强制要求的角色失误：\n" + "\n".join(error_instructions)

        # 观察者标记的叙事候选
        narrative_ctx = _build_narrative_candidates_context(state)
        # 角色欲望提案
        desire_ctx = _build_desire_proposals_context(state)
        # 评审反馈
        review_ctx = _build_review_feedback_context(state)

        # 触发事件
        triggered = state.get("triggered_events", [])
        trigger_ctx = ""
        if triggered:
            trigger_lines = [
                f"- 角色「{evt.character_name}」触发了事件：{evt.rule.event_description}"
                f"（属性「{evt.rule.attribute}」当前值={evt.current_value}）"
                for evt in triggered
            ]
            trigger_ctx = "\n# 待融入的触发事件\n" + "\n".join(trigger_lines)

        # 环境状态
        env_state = state.get("environment")
        env_ctx = ""
        if env_state and env_state.variables:
            env_lines = [
                f"- {v.description or v.name}: {v.value:.0f}"
                for v in env_state.variables.values()
            ]
            env_ctx = "\n# 当前环境状态\n" + "\n".join(env_lines)

        user_prompt = f"""{worldview_ctx}

{characters_ctx}

{prev_chapters_ctx}

{rhythm_instruction}

{error_ctx}

{narrative_ctx}

{desire_ctx}

{review_ctx}

{trigger_ctx}

{env_ctx}

---
请编排第 {current_idx} 章（共 {total} 章）。

要求：
1. **严格遵守上述节奏要求**，在场景编排中体现张弛结构。
2. **强制执行角色失误指令**（如有），确保错误产生真实后果。
3. **从 narrative_candidates 中选择素材**组织本章（如有）。
4. **尊重角色欲望提案**，让冲突自然发生。
5. 给出章节标题和内容概要。
6. 列出本章的关键事件（3-5个）。
7. 指定参与角色和场景。
8. 设定情感基调。
9. 将章节拆分为 2-5 个场景。
{"10. 请务必将触发事件自然融入章节剧情中" if triggered else ""}
{"11. 参考评审反馈调整方向" if review_ctx else ""}

{CHAPTER_PLAN_SCHEMA}

请直接输出 JSON 对象（不要加 markdown 代码块标记之外的任何文字）。"""

        messages = [
            SystemMessage(content=DIRECTOR_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        # 带重试的章节规划（最多 2 次尝试，容错 LLM 输出格式问题）
        last_error = None
        for plan_attempt in range(2):
            try:
                response = invoke_with_retry(model, messages, operation_name="director_plan_chapter")
                content = extract_response_text(response)
                plan_data = extract_json(content)
                chapter_plan = ChapterPlan.model_validate(plan_data)
                chapter_plan.chapter_index = current_idx

                for i, scene in enumerate(chapter_plan.scenes):
                    if not scene.scene_id:
                        scene.scene_id = f"ch{current_idx}_scene{i + 1}"

                # ── 节奏强制：如果导演没有安排 silence/daily 场景，自动注入 ──
                has_breathing_scene = any(
                    s.scene_type in ("silence", "daily") for s in chapter_plan.scenes
                )
                if not has_breathing_scene and len(chapter_plan.scenes) >= 2:
                    # 在第2个场景之后插入一个 daily 场景
                    breathing_scene = Scene(
                        scene_id=f"ch{current_idx}_breathing",
                        title="日常间隙",
                        description="角色在紧张事件之间的日常片刻。",
                        location="",
                        participating_characters=[chapter_plan.participating_characters[0]]
                            if chapter_plan.participating_characters else [],
                        scene_type="daily",
                        mood="平静",
                        objectives=["给读者喘息空间", "展示角色的普通一面"],
                        director_notes=SCENE_DAILY_AUTO_INJECT,
                    )
                    insert_pos = min(2, len(chapter_plan.scenes))
                    chapter_plan.scenes.insert(insert_pos, breathing_scene)
                    logger.info("节奏引擎: 为第%d章自动注入日常场景", current_idx)

                logger.info(
                    "编排者规划完成: 第%d章「%s」, %d个场景",
                    current_idx,
                    chapter_plan.title,
                    len(chapter_plan.scenes),
                )

                return {
                    "chapter_plan": chapter_plan,
                    "scene_queue": list(chapter_plan.scenes),
                    "current_scene": None,
                    "scene_actions": [],
                    "narrative_buffer": [],
                    "triggered_events": [],
                    "next_action": "direct_scene",
                }
            except Exception as e:
                last_error = e
                if plan_attempt == 0:
                    logger.warning("编排者规划第%d章第 1 次尝试失败 (%s)，重试中...", current_idx, e)
                else:
                    logger.error("编排者规划章节失败（已重试）: %s", e)

        return {"error": str(last_error), "next_action": "end"}

    return director_plan_chapter


def create_director_direct_scene(model: BaseChatModel):
    """创建导演的「场景编排」节点函数。"""

    def director_direct_scene(state: NovelState) -> dict[str, Any]:
        scene_queue = list(state.get("scene_queue", []))
        if not scene_queue:
            logger.info("所有场景已完成，准备编译章节")
            return {"next_action": "compile_chapter"}

        current_scene = scene_queue.pop(0)
        logger.info(
            "编排场景: %s（%s）, 参与角色: %s",
            current_scene.title or current_scene.scene_id,
            current_scene.scene_type,
            ", ".join(current_scene.participating_characters),
        )

        scene_type = current_scene.scene_type
        if scene_type == "interaction" and len(current_scene.participating_characters) >= 2:
            next_action = "character_interact"
        elif scene_type in ("narration", "silence", "daily"):
            # silence 和 daily 场景都由 narration 节点处理，
            # 但通过 director_notes 传达不同的写作指令
            if scene_type == "silence" and not current_scene.director_notes:
                current_scene.director_notes = SCENE_SILENCE_NOTES
            elif scene_type == "daily" and not current_scene.director_notes:
                current_scene.director_notes = SCENE_DAILY_NOTES
            next_action = "narration"
        else:
            next_action = "character_action"

        return {
            "current_scene": current_scene,
            "scene_queue": scene_queue,
            "next_action": next_action,
        }

    return director_direct_scene


def create_director_handle_trigger(model: BaseChatModel):
    """创建导演的「触发响应」节点函数。"""

    def director_handle_trigger(state: NovelState) -> dict[str, Any]:
        triggered = state.get("triggered_events", [])
        if not triggered:
            return {"next_action": "direct_scene"}

        worldview_ctx = _build_worldview_context(state)
        characters_ctx = _build_characters_summary(state)

        trigger_lines = [
            f"- 角色「{evt.character_name}」: {evt.rule.event_description}"
            f"（{evt.rule.attribute}={evt.current_value}）"
            for evt in triggered
        ]
        triggers_text = "\n".join(trigger_lines)

        user_prompt = f"""{worldview_ctx}

{characters_ctx}

# 触发的事件
{triggers_text}

请为这些触发事件生成具体的剧情描述。每个事件用 1-2 段话描述将如何在故事中呈现。

请严格按以下 JSON 数组格式返回：
```json
[
    {{"character_name": "角色名", "description": "剧情描述"}}
]
```"""

        messages = [
            SystemMessage(content=DIRECTOR_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        try:
            response = invoke_with_retry(model, messages, operation_name="director_direct_scene")
            result = extract_json(extract_response_text(response))
            narrations = []
            if isinstance(result, list):
                for item in result:
                    desc = item.get("description", "")
                    if desc:
                        narrations.append(desc)
            elif isinstance(result, dict) and "description" in result:
                narrations.append(result["description"])

            logger.info("编排者处理了 %d 个触发事件", len(triggered))
            return {
                "narrative_buffer": narrations,
                "triggered_events": [],
                "next_action": "direct_scene",
            }
        except Exception as e:
            logger.error("编排者处理触发事件失败: %s", e)
            return {"triggered_events": [], "next_action": "direct_scene"}

    return director_handle_trigger
