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
from mobius.models.architecture import ChapterContract, NovelBlueprint
from mobius.models.chapter import (
    ChapterOutline,
    ChapterPlan,
    ChapterStoryboard,
    Scene,
    SettingCharacterProfile,
    SettingEntity,
    SettingPack,
    SettingRule,
    StoryboardScene,
    TimelineEvent,
)
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
OUTLINE_SYSTEM_PROMPT = load_prompt("director_outline_system")
OUTLINE_SCHEMA = load_prompt("director_outline_schema")
SETTING_PACK_SYSTEM_PROMPT = load_prompt("director_setting_pack_system")
SETTING_PACK_SCHEMA = load_prompt("director_setting_pack_schema")
STORYBOARD_SYSTEM_PROMPT = load_prompt("director_storyboard_system")
STORYBOARD_SCHEMA = load_prompt("director_storyboard_schema")
SCENE_SILENCE_NOTES = load_prompt("scene_silence_notes")
SCENE_DAILY_NOTES = load_prompt("scene_daily_notes")
SCENE_DAILY_AUTO_INJECT = load_prompt("scene_daily_auto_inject")
BLUEPRINT_SYSTEM_PROMPT = load_prompt("director_blueprint_system")
BLUEPRINT_SCHEMA = load_prompt("director_blueprint_schema")


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


def _build_blueprint_context(state: NovelState) -> str:
    """构建全书蓝图上下文。"""
    blueprint = state.get("novel_blueprint")
    if not blueprint:
        return "（尚未建立全书蓝图）"
    lines = [
        "# 全书蓝图",
        f"主命题：{blueprint.thesis}",
        f"反命题：{blueprint.antithesis}",
        f"综合立场：{blueprint.synthesis}",
    ]
    if blueprint.chapter_missions:
        lines.append("章节职责：")
        for idx, mission in enumerate(blueprint.chapter_missions, start=1):
            lines.append(f"- 第{idx}章：{mission}")
    if blueprint.character_arcs:
        lines.append("角色哲学弧线：")
        for name, arc in blueprint.character_arcs.items():
            lines.append(f"- {name}：{arc}")
    return "\n".join(lines)


def _build_thread_ledger_context(state: NovelState) -> str:
    """构建线索账本上下文。"""
    ledger = state.get("payoff_ledger", [])
    if not ledger:
        return "（暂无线索账本）"
    lines = ["# 线索账本（优先回收到期项）"]
    for item in ledger:
        if item.status != "resolved":
            lines.append(
                f"- {item.thread}（来源:第{item.source_chapter}章, 最晚回收:第{item.due_chapter}章, 状态:{item.status}）"
            )
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
    lines.append(f"主题推进度：{last_review.theme_progression:.2f}")
    lines.append(f"线索回收率：{last_review.thread_recovery_rate:.2f}")
    lines.append(f"章节必要性：{last_review.chapter_necessity:.2f}")
    if last_review.pacing_notes:
        lines.append(f"节奏建议：{last_review.pacing_notes}")
    if last_review.theme_progression_notes:
        lines.append(f"主题推进说明：{last_review.theme_progression_notes}")
    if last_review.chapter_necessity_notes:
        lines.append(f"章节必要性说明：{last_review.chapter_necessity_notes}")
    if last_review.logic_issues:
        lines.append(f"待修复逻辑问题：" + "；".join(last_review.logic_issues))
    if last_review.unrecovered_threads:
        lines.append(f"未回收线索：" + "；".join(last_review.unrecovered_threads))
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
        # 全书蓝图
        blueprint_ctx = _build_blueprint_context(state)
        # 线索账本
        ledger_ctx = _build_thread_ledger_context(state)

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

        # ── 信息流管理：已揭露的信息 ──
        revealed = state.get("revealed_information", [])
        revealed_ctx = ""
        if revealed:
            revealed_ctx = (
                "\n# 已向读者揭露的信息（不要重复这些，安排新的信息增量）\n- "
                + "\n- ".join(revealed)
            )

        # ── 剧情阶段感知 ──
        stage_ctx = ""
        if total > 1:
            progress = current_idx / total
            if progress <= 0.25:
                stage = "起"
                stage_desc = "建立世界观，介绍角色，铺垫冲突种子。重点：让读者理解设定。"
            elif progress <= 0.5:
                stage = "承"
                stage_desc = "发展角色关系，升级冲突，揭示更多背景。重点：加深角色塑造。"
            elif progress <= 0.75:
                stage = "转"
                stage_desc = "重大转折，信念崩塌，不可逆事件。重点：打破读者预期。"
            else:
                stage = "合"
                stage_desc = "收束线索，角色弧线完成，余韵留白。重点：不要强行圆满。"
            stage_ctx = f"\n# 剧情阶段\n当前处于「{stage}」（第{current_idx}章/共{total}章，进度{progress:.0%}）\n{stage_desc}"

        user_prompt = f"""{worldview_ctx}

{characters_ctx}

{prev_chapters_ctx}

{rhythm_instruction}

{error_ctx}

{narrative_ctx}

{desire_ctx}

{review_ctx}

{blueprint_ctx}

{ledger_ctx}

{trigger_ctx}

{env_ctx}

{revealed_ctx}

{stage_ctx}

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
10. 明确声明本章如何服务全书主命题（chapter_purpose/theme_move/philosophical_beat）。
11. 明确本章要回收哪些线索（required_payoffs）以及本章新埋哪些承诺（new_promises）。
{"- 请务必将触发事件自然融入章节剧情中" if triggered else ""}
{"- 参考评审反馈调整方向" if review_ctx else ""}

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
                if isinstance(plan_data, dict) and isinstance(plan_data.get("scenes"), list):
                    for scene in plan_data["scenes"]:
                        if not isinstance(scene, dict):
                            continue
                        signal = str(scene.get("thesis_signal", "")).strip().lower()
                        # LLM 常把 thesis_signal 错写为 invert，这里归并为 challenge。
                        if signal == "invert":
                            scene["thesis_signal"] = "challenge"
                        elif signal not in {"support", "challenge", "suspend"}:
                            scene["thesis_signal"] = "suspend"
                chapter_plan = ChapterPlan.model_validate(plan_data)
                chapter_plan.chapter_index = current_idx

                for i, scene in enumerate(chapter_plan.scenes):
                    if not scene.scene_id:
                        scene.scene_id = f"ch{current_idx}_scene{i + 1}"

                # ── 节奏弹性控制：仅在连续高张力后才注入日常场景 ──
                if len(chapter_plan.scenes) >= 3:
                    # 检测是否存在连续 2 个高张力场景（非 silence/daily/narration）
                    consecutive_intense = 0
                    needs_breathing = False
                    for s in chapter_plan.scenes:
                        if s.scene_type in ("silence", "daily", "narration"):
                            consecutive_intense = 0
                        else:
                            consecutive_intense += 1
                            if consecutive_intense >= 2:
                                needs_breathing = True
                                break

                    has_breathing = any(
                        s.scene_type in ("silence", "daily") for s in chapter_plan.scenes
                    )
                    if needs_breathing and not has_breathing:
                        breathing_scene = Scene(
                            scene_id=f"ch{current_idx}_breathing",
                            title="日常间隙",
                            description="角色在紧张事件之间的日常片刻。",
                            location="",
                            participating_characters=[chapter_plan.participating_characters[0]]
                                if chapter_plan.participating_characters else [],
                            scene_type="daily",
                            mood="平静",
                            objectives=["给读者喘息空间"],
                            director_notes=SCENE_DAILY_AUTO_INJECT,
                        )
                        insert_pos = min(2, len(chapter_plan.scenes))
                        chapter_plan.scenes.insert(insert_pos, breathing_scene)
                        logger.info("节奏引擎: 第%d章连续高张力，自动注入日常场景", current_idx)

                # ── 收集本章各场景的 information_revelation 到全局状态 ──
                new_revelations = [
                    s.information_revelation
                    for s in chapter_plan.scenes
                    if s.information_revelation.strip()
                ]

                logger.info(
                    "编排者规划完成: 第%d章「%s」, %d个场景, %d条新信息揭露",
                    current_idx,
                    chapter_plan.title,
                    len(chapter_plan.scenes),
                    len(new_revelations),
                )

                return {
                    "chapter_plan": chapter_plan,
                    "chapter_contract": ChapterContract(
                        chapter_index=current_idx,
                        chapter_purpose=chapter_plan.chapter_purpose,
                        theme_move=chapter_plan.theme_move,
                        required_payoffs=chapter_plan.required_payoffs,
                        new_promises=chapter_plan.new_promises,
                        philosophical_beat=chapter_plan.philosophical_beat,
                    ),
                    "scene_queue": list(chapter_plan.scenes),
                    "current_scene": None,
                    "scene_actions": [],
                    "narrative_buffer": [],
                    "triggered_events": [],
                    "revealed_information": new_revelations,
                    "next_action": "chapter_contract",
                }
            except Exception as e:
                last_error = e
                if plan_attempt == 0:
                    logger.warning("编排者规划第%d章第 1 次尝试失败 (%s)，重试中...", current_idx, e)
                else:
                    logger.error("编排者规划章节失败（已重试）: %s", e)

        return {"error": str(last_error), "next_action": "end"}

    return director_plan_chapter


def create_director_blueprint_refresh(model: BaseChatModel):
    """创建/刷新全书蓝图节点。"""

    def blueprint_refresh(state: NovelState) -> dict[str, Any]:
        # 仅在首章强制生成；后续章节可按评审反馈渐进刷新
        existing = state.get("novel_blueprint")
        current_idx = state.get("current_chapter_index", 1)
        if existing and current_idx > 1:
            return {"next_action": "plan_chapter"}

        worldview_ctx = _build_worldview_context(state)
        characters_ctx = _build_characters_summary(state)
        prev_ctx = _build_previous_chapters_summary(state)
        total = state.get("total_chapters", 10)

        user_prompt = f"""{worldview_ctx}

{characters_ctx}

{prev_ctx}

---
请为这部共 {total} 章的小说生成“全书蓝图”。
要求：
1. 给出主命题、反命题、最终综合立场。
2. 为每章指定一句“章节职责”（chapter_missions，长度应等于总章节数）。
3. 给出主要角色的哲学弧线（character_arcs）。

{BLUEPRINT_SCHEMA}
"""
        messages = [
            SystemMessage(content=BLUEPRINT_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        try:
            response = invoke_with_retry(model, messages, operation_name="director_blueprint_refresh")
            payload = extract_json(extract_response_text(response))
            blueprint = NovelBlueprint.model_validate(payload)
            if len(blueprint.chapter_missions) < total:
                # 不足时补齐，避免后续索引越界
                blueprint.chapter_missions.extend(
                    ["围绕主命题推进冲突与角色弧线"] * (total - len(blueprint.chapter_missions))
                )
            logger.info("全书蓝图已建立：主命题=%s", blueprint.thesis)
            return {"novel_blueprint": blueprint, "next_action": "plan_chapter"}
        except Exception as e:
            logger.warning("全书蓝图生成失败，使用降级蓝图：%s", e)
            fallback = NovelBlueprint(
                thesis=state.get("theme", "人与AI之间的权力与爱的悖论"),
                antithesis="控制欲与被控制欲互换后的失衡",
                synthesis="爱无法通过完全控制得到，只会在失控中显形",
                chapter_missions=["推进冲突与主题"] * total,
                character_arcs={name: "从执念走向自我认知裂变" for name in state.get("character_profiles", {})},
            )
            return {"novel_blueprint": fallback, "next_action": "plan_chapter"}

    return blueprint_refresh


def create_director_chapter_contract():
    """章节合同校验节点：强约束章节结构完整性。"""

    def chapter_contract_node(state: NovelState) -> dict[str, Any]:
        chapter_plan = state.get("chapter_plan")
        if chapter_plan is None:
            return {"next_action": "plan_chapter"}

        required_text_fields = [
            chapter_plan.chapter_purpose.strip(),
            chapter_plan.philosophical_beat.strip(),
            chapter_plan.summary.strip(),
        ]
        scene_ok = all(
            s.title.strip() and s.description.strip() and s.causal_to.strip()
            for s in chapter_plan.scenes
        )
        chapter_ok = all(required_text_fields) and bool(chapter_plan.new_promises) and scene_ok
        if chapter_ok:
            contract = ChapterContract(
                chapter_index=chapter_plan.chapter_index,
                chapter_purpose=chapter_plan.chapter_purpose,
                theme_move=chapter_plan.theme_move,
                required_payoffs=chapter_plan.required_payoffs,
                new_promises=chapter_plan.new_promises,
                philosophical_beat=chapter_plan.philosophical_beat,
            )
            metadata = dict(state.get("metadata", {}))
            metadata["chapter_contract_retries"] = 0
            return {
                "chapter_contract": contract,
                "metadata": metadata,
                "next_action": "direct_scene",
            }

        retries = int((state.get("metadata") or {}).get("chapter_contract_retries", 0))
        if retries >= 1:
            logger.warning("章节合同校验未通过，但已重试一次，降级放行到 direct_scene")
            metadata = dict(state.get("metadata", {}))
            metadata["chapter_contract_retries"] = 0
            return {"metadata": metadata, "next_action": "direct_scene"}

        logger.info("章节合同校验未通过，回退重规划")
        metadata = dict(state.get("metadata", {}))
        metadata["chapter_contract_retries"] = retries + 1
        return {"metadata": metadata, "next_action": "plan_chapter"}

    return chapter_contract_node


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


def create_generate_outlines_node(model: BaseChatModel):
    """生成全书章节概要（Phase1）。"""

    def generate_outlines_node(state: NovelState) -> dict[str, Any]:
        total = state.get("total_chapters", 10)
        blueprint = state.get("novel_blueprint")
        setting_pack: SettingPack | None = state.get("setting_pack")
        worldview_ctx = _build_worldview_context(state)
        characters_ctx = _build_characters_summary(state)
        blueprint_ctx = _build_blueprint_context(state)
        guardrails = state.get("global_guardrails", [])
        guardrails_ctx = "\n".join(f"- {g}" for g in guardrails) if guardrails else "（无）"

        mission_lines: list[str] = []
        if blueprint and blueprint.chapter_missions:
            for idx, mission in enumerate(blueprint.chapter_missions[:total], start=1):
                mission_lines.append(f"- 第{idx}章：{mission}")
        else:
            mission_lines = [f"- 第{i}章：围绕主命题推进冲突并产生不可逆后果" for i in range(1, total + 1)]

        setting_ctx = "（尚未生成结构化设定集）"
        if setting_pack:
            setting_ctx = (
                f"# 结构化设定集\n"
                f"主旨：{setting_pack.theme}\n"
                f"世界规则：{'；'.join(setting_pack.worldview_rules) if setting_pack.worldview_rules else '（无）'}\n"
                f"关键时间线：{'；'.join(setting_pack.core_events_timeline) if setting_pack.core_events_timeline else '（无）'}\n"
                f"待补完：{'；'.join(setting_pack.missing_items) if setting_pack.missing_items else '（无）'}"
            )

        user_prompt = f"""{worldview_ctx}

{characters_ctx}

{blueprint_ctx}

{setting_ctx}

# 章节职责（必须逐章覆盖）
{chr(10).join(mission_lines)}

# 全书硬约束
{guardrails_ctx}

---
请一次性生成 1 到 {total} 章的概要，要求：
1. 每章 300-500 字 core_plot。
2. 每章必须有明确且不可逆的变化（irreversible_change）。
3. 每章职责不能重复，且要形成连续推进。
4. 每章至少提供 1 条 must_payoffs 与 1 条 new_promises（如确实没有可填“（无）”）。
5. character_arc_delta 必须反映角色阶段性变化，而非抽象空话。

{OUTLINE_SCHEMA}
"""

        messages = [
            SystemMessage(content=OUTLINE_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        try:
            response = invoke_with_retry(model, messages, operation_name="generate_outlines")
            payload = extract_json(extract_response_text(response))
            raw_outlines = payload.get("outlines", []) if isinstance(payload, dict) else payload
            if not isinstance(raw_outlines, list):
                raise ValueError("outlines 输出格式错误：必须是数组")

            outlines: list[ChapterOutline] = []
            for idx, item in enumerate(raw_outlines[:total], start=1):
                if isinstance(item, dict):
                    item.setdefault("chapter_index", idx)
                    outlines.append(ChapterOutline.model_validate(item))

            if len(outlines) < total:
                # LLM 少给章节时补齐占位，避免流程中断
                for i in range(len(outlines) + 1, total + 1):
                    outlines.append(
                        ChapterOutline(
                            chapter_index=i,
                            title=f"第{i}章",
                            purpose=f"推进第{i}章主线职责",
                            core_plot=f"第{i}章核心剧情待补充。",
                            irreversible_change="角色关系发生不可逆变化（待细化）",
                            character_arc_delta="角色弧线出现阶段性位移（待细化）",
                            must_payoffs=["（无）"],
                            new_promises=["（无）"],
                        )
                    )

            logger.info("全书概要生成完成: %d 章", len(outlines))
            return {
                "chapter_outlines": outlines,
                "outline_approved": False,
                "next_action": "persist_outlines",
            }
        except Exception as e:
            logger.error("全书概要生成失败: %s", e)
            return {"error": str(e), "next_action": "end"}

    return generate_outlines_node


def create_generate_setting_pack_node(model: BaseChatModel):
    """第一层：设定集结构化与补完。"""

    def generate_setting_pack_node(state: NovelState) -> dict[str, Any]:
        worldview_ctx = _build_worldview_context(state)
        characters_ctx = _build_characters_summary(state)
        total = state.get("total_chapters", 10)
        outlines = state.get("chapter_outlines", [])
        outline_ctx = "（尚无章节概要）"
        if outlines:
            outline_lines = []
            for out in outlines:
                outline_lines.append(
                    f"- 第{out.chapter_index}章 {out.title} | 职责:{out.purpose} | 不可逆:{out.irreversible_change}"
                )
            outline_ctx = "\n".join(outline_lines)

        user_prompt = f"""{worldview_ctx}

{characters_ctx}

# 已有章节概要（用于反向补完设定）
{outline_ctx}

---
请为这部预计 {total} 章的小说生成结构化设定集（SettingPack）。
要求：
1. 提炼主旨命题。
2. 提炼世界规则（硬约束）。
3. 输出关键事件时间线（至少8条）。
4. 输出角色细化档案，尤其要包含：思考习惯、性格特征、衣着打扮、首次出场章节与出场约束。
5. 输出组织与关键物件设定，并给出约束与待补完问题。
6. 输出 theme_longform 与 worldview_longform（建议千字级深描）。
7. 给出缺失项补完建议（missing_items），用于后续分镜与扩写阶段持续迭代。

{SETTING_PACK_SCHEMA}
"""
        messages = [
            SystemMessage(content=SETTING_PACK_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        try:
            response = invoke_with_retry(model, messages, operation_name="generate_setting_pack")
            payload = extract_json(extract_response_text(response))
            entities_raw = payload.get("entities", []) if isinstance(payload, dict) else []
            entities = []
            for item in entities_raw:
                if isinstance(item, dict):
                    entities.append(SettingEntity.model_validate(item))
            rules_raw = payload.get("detailed_rules", []) if isinstance(payload, dict) else []
            detailed_rules = []
            for item in rules_raw:
                if isinstance(item, dict):
                    detailed_rules.append(SettingRule.model_validate(item))
            timeline_raw = payload.get("timeline_events", []) if isinstance(payload, dict) else []
            timeline_events = []
            for item in timeline_raw:
                if isinstance(item, dict):
                    timeline_events.append(TimelineEvent.model_validate(item))
            chars_raw = payload.get("characters", []) if isinstance(payload, dict) else []
            character_profiles = []
            for item in chars_raw:
                if isinstance(item, dict):
                    character_profiles.append(SettingCharacterProfile.model_validate(item))
            org_raw = payload.get("organizations", []) if isinstance(payload, dict) else []
            organizations = []
            for item in org_raw:
                if isinstance(item, dict):
                    organizations.append(SettingEntity.model_validate(item))
            items_raw = payload.get("items", []) if isinstance(payload, dict) else []
            items = []
            for item in items_raw:
                if isinstance(item, dict):
                    items.append(SettingEntity.model_validate(item))
            setting_pack = SettingPack(
                title=payload.get("title") if isinstance(payload, dict) else state["plot_outline"].title,
                theme=payload.get("theme") if isinstance(payload, dict) else state.get("theme", ""),
                theme_longform=payload.get("theme_longform", "") if isinstance(payload, dict) else "",
                worldview_longform=payload.get("worldview_longform", "") if isinstance(payload, dict) else "",
                worldview_rules=payload.get("worldview_rules", []) if isinstance(payload, dict) else [],
                detailed_rules=detailed_rules,
                core_events_timeline=payload.get("core_events_timeline", []) if isinstance(payload, dict) else [],
                timeline_events=timeline_events,
                entities=entities,
                characters=character_profiles,
                organizations=organizations,
                items=items,
                missing_items=payload.get("missing_items", []) if isinstance(payload, dict) else [],
                author_notes=payload.get("author_notes", "") if isinstance(payload, dict) else "",
            )
            return {
                "setting_pack": setting_pack,
                "setting_approved": False,
                "next_action": "persist_setting_pack",
            }
        except Exception as e:
            logger.error("结构化设定集生成失败: %s", e)
            return {"error": str(e), "next_action": "end"}

    return generate_setting_pack_node


def create_generate_storyboards_node(model: BaseChatModel):
    """第三层：按章节概要生成分镜。"""

    def generate_storyboards_node(state: NovelState) -> dict[str, Any]:
        outlines = state.get("chapter_outlines", [])
        setting_pack: SettingPack | None = state.get("setting_pack")
        guardrails = state.get("global_guardrails", [])
        if not outlines:
            return {"error": "缺少章节概要，无法生成分镜", "next_action": "end"}

        setting_ctx = "（无）"
        if setting_pack:
            char_guardrails = "（无）"
            if setting_pack.characters:
                char_guardrails = "\n".join(
                    f"- {c.name}: 首次出场>=第{c.first_appearance_chapter}章；时机={c.first_appearance_moment or '未注明'}"
                    for c in setting_pack.characters
                )
            setting_ctx = (
                f"主题：{setting_pack.theme}\n"
                f"世界规则：{'；'.join(setting_pack.worldview_rules) if setting_pack.worldview_rules else '（无）'}\n"
                f"关键时间线：{'；'.join(setting_pack.core_events_timeline[:10]) if setting_pack.core_events_timeline else '（无）'}\n"
                f"角色出场约束：\n{char_guardrails}"
            )
        guardrails_ctx = "\n".join(f"- {g}" for g in guardrails) if guardrails else "（无）"

        storyboards: list[ChapterStoryboard] = []
        for outline in outlines:
            outline_ctx = (
                f"章节：第{outline.chapter_index}章 {outline.title}\n"
                f"职责：{outline.purpose}\n"
                f"核心剧情：{outline.core_plot}\n"
                f"不可逆变化：{outline.irreversible_change}\n"
                f"必兑现线索：{'、'.join(outline.must_payoffs) if outline.must_payoffs else '（无）'}\n"
                f"新承诺：{'、'.join(outline.new_promises) if outline.new_promises else '（无）'}"
            )
            user_prompt = f"""# 结构化设定摘要
{setting_ctx}

# 全书硬约束
{guardrails_ctx}

# 本章概要
{outline_ctx}

---
请为本章生成 4-8 个场景分镜，且必须至少包含 1 个 daily/silence/narration 降密场景。
场景必须覆盖主线推进并体现因果链。

{STORYBOARD_SCHEMA}
"""
            messages = [
                SystemMessage(content=STORYBOARD_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
            response = invoke_with_retry(model, messages, operation_name="generate_storyboard")
            payload = extract_json(extract_response_text(response))
            scene_items = payload.get("scenes", []) if isinstance(payload, dict) else []
            scenes: list[StoryboardScene] = []
            for idx, item in enumerate(scene_items, start=1):
                if isinstance(item, dict):
                    item.setdefault("scene_index", idx)
                    scenes.append(StoryboardScene.model_validate(item))
            if not scenes:
                scenes = [
                    StoryboardScene(
                        scene_index=1,
                        scene_type="plot_progress",
                        title="主线推进",
                        objective="推进本章核心冲突",
                        conflict_type="价值冲突",
                        causal_from="承接上章后果",
                        causal_to="引出下一场景",
                        info_gain=outline.core_plot[:80],
                        style_notes="克制叙述，强调行动后果",
                        expected_beats=["冲突启动", "关键抉择"],
                    ),
                    StoryboardScene(
                        scene_index=2,
                        scene_type="daily",
                        title="降密场景",
                        objective="提供节奏缓冲与心理过渡",
                        conflict_type="内在张力",
                        causal_from="冲突余波",
                        causal_to="回到主线",
                        info_gain="角色情绪与关系微调",
                        style_notes="环境描写+心理描写",
                        expected_beats=["静默观察", "情绪转折"],
                    ),
                ]
            has_low_density = any(s.scene_type in {"daily", "silence", "narration"} for s in scenes)
            if not has_low_density:
                scenes.append(
                    StoryboardScene(
                        scene_index=len(scenes) + 1,
                        scene_type="daily",
                        title="降密补偿场景",
                        objective="稀释剧情密度并增强生活质感",
                        conflict_type="弱冲突",
                        causal_from="高压冲突后的回落",
                        causal_to="重返主线推进",
                        info_gain="角色关系细微变化",
                        style_notes="减少论述感，提升具象细节",
                        expected_beats=["呼吸间隙", "微小承诺"],
                    )
                )

            storyboards.append(
                ChapterStoryboard(
                    chapter_index=outline.chapter_index,
                    title=outline.title,
                    purpose=outline.purpose,
                    irreversible_change=outline.irreversible_change,
                    must_payoffs=outline.must_payoffs,
                    scenes=scenes,
                )
            )

        return {
            "chapter_storyboards": storyboards,
            "storyboard_approved": False,
            "next_action": "persist_storyboards",
        }

    return generate_storyboards_node
