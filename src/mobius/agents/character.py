"""角色 Agent 工厂：为每个角色动态创建独立的 Agent 节点。

支持：
- 双阶段生成（Stage 1: 内心独白 -> Stage 2: 外显行动）
- 双模型（reasoning_model + roleplay_model / MiniMax M2-her）
- 信念/欲望/资源/环境系统 prompt 注入
- 认知黑箱（内心独白不向其他角色公开）
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from mobius.agents.utils import extract_json_safe, extract_response_text
from mobius.engine.chaos_engine import ChaosEngine
from mobius.models.character import CharacterAction, CharacterDynamicState, CharacterProfile
from mobius.models.chapter import Scene
from mobius.models.environment import EnvironmentState
from mobius.prompts import load_prompt
from mobius.state.character_state import get_active_behavior_effects
from mobius.state.novel_state import NovelState

logger = logging.getLogger(__name__)

# 从外部文件加载提示词
CHARACTER_ACTION_FORMAT = load_prompt("character_action_format")
INTERNAL_MONOLOGUE_PROMPT = load_prompt("character_internal_monologue")
STAGE2_ACTION_PROMPT = load_prompt("character_stage2_action")


# ──────────────────────────────────────────
# NPC 临时生成
# ──────────────────────────────────────────


def _create_npc_profile(name: str, scene: Scene) -> CharacterProfile:
    """为未注册的 NPC 角色创建临时档案。"""
    npc_hint = scene.director_notes or ""
    return CharacterProfile(
        name=name,
        role="NPC（临时角色）",
        personality=f"在当前场景中出现的配角。{npc_hint}",
        background=f"一个出现在「{scene.location or '此地'}」的角色。",
        abilities=[],
        speech_style="",
        goals=[],
        weaknesses=[],
        extra={},
    )


def _create_npc_state(name: str, scene: Scene) -> CharacterDynamicState:
    """为未注册的 NPC 角色创建临时动态状态。"""
    return CharacterDynamicState(
        name=name,
        emotional_state={"快乐": 0.5, "悲伤": 0.0, "愤怒": 0.0, "恐惧": 0.0},
        relationship_scores={},
        custom_attributes={},
        trigger_rules=[],
        memory=[],
        current_location=scene.location or "",
        is_alive=True,
    )


# ──────────────────────────────────────────
# System Prompt 构建
# ──────────────────────────────────────────


def build_character_prompt(profile: CharacterProfile) -> str:
    """根据角色档案构建完整的 system prompt。"""
    parts = [
        f"你是小说中的角色「{profile.name}」。你必须完全沉浸在这个角色中，"
        f"以第一人称视角思考和行动。\n",
        f"# 基本信息",
        f"- 姓名：{profile.name}",
        f"- 定位：{profile.role}",
    ]

    if profile.age:
        parts.append(f"- 年龄：{profile.age}")
    if profile.gender:
        parts.append(f"- 性别：{profile.gender}")
    if profile.appearance:
        parts.append(f"- 外貌：{profile.appearance}")

    parts.append(f"\n# 性格特点\n{profile.personality}")

    if profile.background:
        parts.append(f"\n# 背景故事\n{profile.background}")
    if profile.abilities:
        parts.append(f"\n# 能力/技能\n- " + "\n- ".join(profile.abilities))
    if profile.speech_style:
        parts.append(f"\n# 说话风格\n{profile.speech_style}")
    if profile.goals:
        parts.append(f"\n# 目标/动机\n- " + "\n- ".join(profile.goals))
    if profile.weaknesses:
        parts.append(f"\n# 弱点/缺陷\n- " + "\n- ".join(profile.weaknesses))

    for key, value in profile.extra.items():
        parts.append(f"\n# {key}\n{value}")

    parts.append(
        "\n# 行动规则\n"
        "1. 始终保持角色一致性，根据你的性格、背景和当前情感状态来反应\n"
        "2. 你的回应需要包含：对话、动作描写、和/或内心独白\n"
        "3. 注意当前的情感状态和关系，它们会影响你的反应方式\n"
        "4. 你的每个行动都有代价——考虑你的资源状况再做决定\n"
        "5. 你可以质疑和推翻自己的中层/表层信念\n"
        "6. 你必须以 JSON 格式输出你的行动结果"
    )

    return "\n".join(parts)


# ──────────────────────────────────────────
# 状态上下文构建
# ──────────────────────────────────────────


def _relationship_label(score: float) -> str:
    """根据关系分数返回文字描述。"""
    if score >= 0.8:
        return "亲密挚友"
    if score >= 0.5:
        return "友好"
    if score >= 0.2:
        return "略有好感"
    if score >= -0.2:
        return "中立"
    if score >= -0.5:
        return "不太信任"
    if score >= -0.8:
        return "敌意"
    return "深仇大恨"


def _build_state_context(
    char_state: CharacterDynamicState,
    scene: Scene,
    recent_actions: list[str] | None = None,
    env_state: EnvironmentState | None = None,
) -> str:
    """构建角色当前状态和场景上下文。

    包含：场景信息、情感状态、完整关系图谱、信念系统、
    欲望优先级、资源面板、环境状态、长期/短期记忆。
    """
    parts = [f"# 当前场景\n{scene.description}"]

    if scene.location:
        parts.append(f"地点：{scene.location}")
    if scene.mood:
        parts.append(f"氛围：{scene.mood}")
    if scene.director_notes:
        parts.append(f"导演指示：{scene.director_notes}")

    # ── 情感状态 ──
    parts.append(f"\n# 你的当前状态")
    if char_state.emotional_state:
        emotions = ", ".join(f"{k}: {v:.1f}" for k, v in char_state.emotional_state.items())
        parts.append(f"情感：{emotions}")
    if char_state.current_location:
        parts.append(f"位置：{char_state.current_location}")

    # ── 信念系统 ──
    if char_state.beliefs:
        belief_groups: dict[str, list[str]] = {"core": [], "middle": [], "surface": []}
        for b in char_state.beliefs:
            label = f"「{b.statement}」(强度:{b.strength:.2f}, {b.category})"
            belief_groups.setdefault(b.layer, []).append(label)
        belief_parts: list[str] = []
        if belief_groups.get("core"):
            belief_parts.append("  核心信念（极难改变）：\n    " + "\n    ".join(belief_groups["core"]))
        if belief_groups.get("middle"):
            belief_parts.append("  中层信念（可被剧情动摇）：\n    " + "\n    ".join(belief_groups["middle"]))
        if belief_groups.get("surface"):
            belief_parts.append("  行为策略（动态更新）：\n    " + "\n    ".join(belief_groups["surface"]))
        if belief_parts:
            parts.append("\n# 你的信念体系\n" + "\n".join(belief_parts))

    # ── 欲望优先级 ──
    if char_state.desires:
        sorted_desires = sorted(char_state.desires, key=lambda d: d.priority, reverse=True)
        desire_lines = []
        for d in sorted_desires:
            status = "已满足" if d.satisfied else ("被阻碍: " + d.blocked_by if d.blocked_by else "进行中")
            desire_lines.append(f"  - [{d.priority:.1f}] {d.description} ({status})")
        parts.append("\n# 你的欲望（按优先级排序）\n" + "\n".join(desire_lines))

    # ── 恐惧列表 ──
    if char_state.fears:
        fear_lines = [f"  - {f.description} (强度:{f.intensity:.1f})" for f in char_state.fears]
        parts.append("\n# 你的恐惧\n" + "\n".join(fear_lines))

    # ── 资源面板 ──
    pool = char_state.resources
    resource_lines = [
        f"  时间精力: {pool.time:.0f}/100",
        f"  社会声誉: {pool.reputation:.0f}/100",
        f"  权力: {pool.power:.0f}",
        f"  信息量: {pool.information:.0f}",
        f"  情绪能量: {pool.emotional_energy:.0f}/100",
        f"  财富: {pool.wealth:.0f}",
    ]
    for name, val in pool.custom.items():
        resource_lines.append(f"  {name}: {val:.0f}")
    parts.append("\n# 你的资源（行动需要消耗资源）\n" + "\n".join(resource_lines))

    # ── 关系图谱 ──
    if char_state.relationship_scores:
        present_chars = set(scene.participating_characters) - {char_state.name}
        present_rels: list[str] = []
        absent_rels: list[str] = []
        for other_name, score in char_state.relationship_scores.items():
            label = _relationship_label(score)
            entry = f"  - {other_name}: {score:+.1f}（{label}）"
            if other_name in present_chars:
                present_rels.append(entry)
            else:
                absent_rels.append(entry)
        rel_parts: list[str] = []
        if present_rels:
            rel_parts.append("在场角色：\n" + "\n".join(present_rels))
        if absent_rels:
            rel_parts.append("不在场但已知的角色：\n" + "\n".join(absent_rels))
        if rel_parts:
            parts.append("\n# 关系图谱\n" + "\n".join(rel_parts))

    if char_state.custom_attributes:
        attrs = ", ".join(f"{k}: {v:.0f}" for k, v in char_state.custom_attributes.items())
        parts.append(f"特殊属性：{attrs}")

    # ── 环境状态总线 ──
    if env_state and env_state.variables:
        env_parts = []
        for var_name, var in env_state.variables.items():
            pct = (var.value - var.min_val) / max(var.max_val - var.min_val, 1) * 100
            if pct > 80:
                level = "危险"
            elif pct > 50:
                level = "警戒"
            else:
                level = "正常"
            env_parts.append(f"  - {var.description or var.name}: {var.value:.0f} ({level})")
        parts.append("\n# 当前环境状态\n" + "\n".join(env_parts))

        # 注入行为修饰
        effects = get_active_behavior_effects(char_state.name, env_state)
        for effect in effects:
            parts.append(f"\n⚠️ 环境压力: {effect}")

    # ── 长期记忆（压缩摘要）──
    if char_state.compressed_memory:
        parts.append(f"\n# 长期记忆（过往经历摘要）\n{char_state.compressed_memory}")

    # ── 短期记忆 ──
    if char_state.memory:
        recent = char_state.memory[-10:]
        parts.append(f"\n# 近期记忆\n- " + "\n- ".join(recent))

    if recent_actions:
        parts.append(f"\n# 刚刚发生的事\n" + "\n".join(recent_actions))

    return "\n".join(parts)


# ──────────────────────────────────────────
# JSON 输出格式
# ──────────────────────────────────────────


# ──────────────────────────────────────────
# 角色行动节点（双阶段生成）
# ──────────────────────────────────────────


def create_character_action_node(
    profiles: dict[str, CharacterProfile],
    reasoning_model: BaseChatModel,
    roleplay_model: BaseChatModel | None = None,
    enable_internal_monologue: bool = True,
    chaos_engine: ChaosEngine | None = None,
) -> Callable[[NovelState], dict[str, Any]]:
    """创建角色单独行动的节点函数。

    支持双阶段生成：
    Stage 1: 用 roleplay_model（M2-her）或 reasoning_model 生成内心独白
    Stage 2: 用 reasoning_model 生成结构化行动 JSON
    """

    def character_action_node(state: NovelState) -> dict[str, Any]:
        scene: Scene | None = state.get("current_scene")
        if not scene:
            return {"next_action": "direct_scene"}

        char_states = state.get("character_states", {})
        env_state: EnvironmentState | None = state.get("environment")
        actions: list[CharacterAction] = []

        for char_name in scene.participating_characters:
            profile = profiles.get(char_name)
            char_state = char_states.get(char_name)

            # 动态生成 NPC
            if not profile:
                logger.info("角色「%s」未注册，自动创建 NPC 临时档案", char_name)
                profile = _create_npc_profile(char_name, scene)
            if not char_state:
                char_state = _create_npc_state(char_name, scene)
                char_states[char_name] = char_state

            system_prompt = build_character_prompt(profile)
            context = _build_state_context(char_state, scene, env_state=env_state)

            # ── Stage 1: 内心独白 ──
            internal_monologue = ""
            if enable_internal_monologue and roleplay_model:
                try:
                    monologue_prompt = INTERNAL_MONOLOGUE_PROMPT.format(
                        character_name=char_name,
                        context=context,
                    )
                    # 如果 roleplay_model 支持 with_character，绑定角色
                    model_for_monologue = roleplay_model
                    if hasattr(roleplay_model, "with_character"):
                        model_for_monologue = roleplay_model.with_character(
                            character_name=char_name,
                        )
                    mono_resp = model_for_monologue.invoke([
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=monologue_prompt),
                    ])
                    internal_monologue = extract_response_text(mono_resp)
                    logger.info("角色 %s 内心独白生成完成 (%d字)", char_name, len(internal_monologue))
                except Exception as e:
                    logger.warning("角色 %s 内心独白生成失败: %s，使用默认模型", char_name, e)
                    # 降级到 reasoning_model
                    try:
                        monologue_prompt = INTERNAL_MONOLOGUE_PROMPT.format(
                            character_name=char_name,
                            context=context,
                        )
                        mono_resp = reasoning_model.invoke([
                            SystemMessage(content=system_prompt),
                            HumanMessage(content=monologue_prompt),
                        ])
                        internal_monologue = extract_response_text(mono_resp)
                    except Exception:
                        internal_monologue = ""
            elif enable_internal_monologue:
                # 没有 roleplay_model 时用 reasoning_model 生成
                try:
                    monologue_prompt = INTERNAL_MONOLOGUE_PROMPT.format(
                        character_name=char_name,
                        context=context,
                    )
                    mono_resp = reasoning_model.invoke([
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=monologue_prompt),
                    ])
                    internal_monologue = extract_response_text(mono_resp)
                except Exception:
                    internal_monologue = ""

            # ── Stage 2: 外显行动 ──
            if internal_monologue:
                user_prompt = STAGE2_ACTION_PROMPT.format(
                    internal_monologue=internal_monologue,
                    action_format=CHARACTER_ACTION_FORMAT,
                )
            else:
                user_prompt = f"""{context}

现在轮到你行动了。请根据场景描述和你的性格、状态做出反应。

{CHARACTER_ACTION_FORMAT}"""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]

            try:
                response = reasoning_model.invoke(messages)
                text = extract_response_text(response)
                action_data = extract_json_safe(text, char_name)
                action = CharacterAction.model_validate(action_data)
                action.character_name = char_name
                action.internal_monologue = internal_monologue
                # 计算认知失调度（简化：有内心独白就标记 0.5，留给后续优化）
                if internal_monologue:
                    action.cognitive_dissonance = 0.5

                # 【v2.1】应用失控引擎：让行动带有缺陷
                if chaos_engine:
                    chapter_idx = state.get("current_chapter_index", 1)
                    chaos_result = chaos_engine.process_character_action(
                        action.content, char_state, chapter_idx
                    )

                    # 如果有缺陷行动，用它替换原始行动
                    if chaos_result.get('flawed_action'):
                        flawed = chaos_result['flawed_action']
                        action.content = flawed.actual_action
                        action.cognitive_dissonance = max(action.cognitive_dissonance, flawed.optimality_loss)

                        logger.info("角色 %s 行动被失控引擎修改: 原始最优性 %.2f, 实际最优性 %.2f",
                                  char_name, 1.0, 1.0 - flawed.optimality_loss)

                actions.append(action)
                logger.info("角色 %s 完成行动: %s", char_name, action.action_type)
            except Exception as e:
                logger.error("角色 %s 行动失败: %s", char_name, e)
                actions.append(
                    CharacterAction(
                        character_name=char_name,
                        action_type="action",
                        content=f"（{char_name}沉默了片刻，没有做出特别的反应。）",
                        internal_monologue=internal_monologue,
                        new_memory="",
                    )
                )

        result: dict[str, Any] = {
            "scene_actions": actions,
            "next_action": "update_state",
        }
        if any(name not in state.get("character_states", {}) for name in char_states):
            result["character_states"] = char_states
        return result

    return character_action_node


# ──────────────────────────────────────────
# 角色交互节点（双阶段生成）
# ──────────────────────────────────────────


def create_character_interact_node(
    profiles: dict[str, CharacterProfile],
    reasoning_model: BaseChatModel,
    roleplay_model: BaseChatModel | None = None,
    max_rounds: int = 5,
    enable_internal_monologue: bool = True,
    chaos_engine: ChaosEngine | None = None,
) -> Callable[[NovelState], dict[str, Any]]:
    """创建角色交互节点函数。"""

    def character_interact_node(state: NovelState) -> dict[str, Any]:
        scene: Scene | None = state.get("current_scene")
        if not scene or len(scene.participating_characters) < 2:
            return {"next_action": "update_state"}

        char_states = state.get("character_states", {})
        env_state: EnvironmentState | None = state.get("environment")
        actions: list[CharacterAction] = []
        interaction_log: list[str] = []

        participants = scene.participating_characters[:4]

        for round_num in range(max_rounds):
            for char_name in participants:
                profile = profiles.get(char_name)
                char_state = char_states.get(char_name)

                if not profile:
                    logger.info("互动中角色「%s」未注册，自动创建 NPC 临时档案", char_name)
                    profile = _create_npc_profile(char_name, scene)
                if not char_state:
                    char_state = _create_npc_state(char_name, scene)
                    char_states[char_name] = char_state

                system_prompt = build_character_prompt(profile)
                context = _build_state_context(
                    char_state,
                    scene,
                    recent_actions=interaction_log[-6:],
                    env_state=env_state,
                )

                # ── Stage 1: 内心独白 ──
                internal_monologue = ""
                if enable_internal_monologue and (roleplay_model or reasoning_model):
                    try:
                        mono_model = roleplay_model or reasoning_model
                        if hasattr(mono_model, "with_character"):
                            mono_model = mono_model.with_character(character_name=char_name)
                        monologue_prompt = INTERNAL_MONOLOGUE_PROMPT.format(
                            character_name=char_name,
                            context=context,
                        )
                        mono_resp = mono_model.invoke([
                            SystemMessage(content=system_prompt),
                            HumanMessage(content=monologue_prompt),
                        ])
                        internal_monologue = extract_response_text(mono_resp)
                    except Exception as e:
                        logger.warning("互动中角色 %s 内心独白失败: %s", char_name, e)

                # ── Stage 2: 外显行动 ──
                is_last_round = round_num == max_rounds - 1
                round_hint = ""
                if is_last_round:
                    round_hint = "\n（这是最后一轮互动，请做出总结性的反应。）"

                if internal_monologue:
                    action_context = (
                        f"{context}\n\n"
                        f"这是第 {round_num + 1}/{max_rounds} 轮互动。"
                        f"你正在与{'、'.join(c for c in participants if c != char_name)}互动。{round_hint}\n\n"
                    )
                    user_prompt = action_context + STAGE2_ACTION_PROMPT.format(
                        internal_monologue=internal_monologue,
                        action_format=CHARACTER_ACTION_FORMAT,
                    )
                else:
                    user_prompt = f"""{context}

这是第 {round_num + 1}/{max_rounds} 轮互动。你正在与{'、'.join(c for c in participants if c != char_name)}互动。{round_hint}

请根据之前的互动和你的性格做出反应。

{CHARACTER_ACTION_FORMAT}"""

                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]

                try:
                    response = reasoning_model.invoke(messages)
                    text = extract_response_text(response)
                    action_data = extract_json_safe(text, char_name)
                    action = CharacterAction.model_validate(action_data)
                    action.character_name = char_name
                    action.internal_monologue = internal_monologue
                    if internal_monologue:
                        action.cognitive_dissonance = 0.5

                    # 【v2.1】应用失控引擎：让互动行动也带有缺陷
                    if chaos_engine:
                        chapter_idx = state.get("current_chapter_index", 1)
                        chaos_result = chaos_engine.process_character_action(
                            action.content, char_state, chapter_idx
                        )

                        # 如果有缺陷行动，用它替换原始行动
                        if chaos_result.get('flawed_action'):
                            flawed = chaos_result['flawed_action']
                            action.content = flawed.actual_action
                            action.cognitive_dissonance = max(action.cognitive_dissonance, flawed.optimality_loss)

                    actions.append(action)
                    # 注意：interaction_log 只包含外显内容，不包含 internal_monologue
                    interaction_log.append(f"[{char_name}]: {action.content[:200]}")
                    logger.info(
                        "互动第%d轮 - 角色 %s: %s", round_num + 1, char_name, action.action_type,
                    )
                except Exception as e:
                    logger.error("角色 %s 互动失败: %s", char_name, e)
                    fallback = CharacterAction(
                        character_name=char_name,
                        action_type="action",
                        content=f"（{char_name}犹豫了一下。）",
                        internal_monologue=internal_monologue,
                        new_memory="",
                    )
                    actions.append(fallback)
                    interaction_log.append(f"[{char_name}]: {fallback.content}")

        result: dict[str, Any] = {
            "scene_actions": actions,
            "next_action": "update_state",
        }
        if any(name not in state.get("character_states", {}) for name in char_states):
            result["character_states"] = char_states
        return result

    return character_interact_node


# ──────────────────────────────────────────
# 欲望提案收集节点
# ──────────────────────────────────────────


def create_desire_tick_node(
    profiles: dict[str, CharacterProfile],
    model: BaseChatModel,
) -> Callable[[NovelState], dict[str, Any]]:
    """创建欲望提案收集节点。

    每章开始前，每个角色根据当前欲望优先级和资源状况提出行动提案。
    """

    DESIRE_TICK_PROMPT = load_prompt("character_desire_tick")

    def desire_tick_node(state: NovelState) -> dict[str, Any]:
        from mobius.models.desire import DesireProposal

        char_states = state.get("character_states", {})
        env_state: EnvironmentState | None = state.get("environment")
        proposals: list[DesireProposal] = []

        # 创建一个临时 scene 用于构建上下文
        temp_scene = Scene(
            scene_id="desire_tick",
            title="欲望评估",
            description="角色评估自身欲望和资源，提出行动提案。",
            participating_characters=list(char_states.keys()),
        )

        for char_name, char_state in char_states.items():
            if not char_state.is_alive or not char_state.desires:
                continue

            profile = profiles.get(char_name)
            if not profile:
                continue

            context = _build_state_context(char_state, temp_scene, env_state=env_state)
            prompt = DESIRE_TICK_PROMPT.format(
                character_name=char_name,
                context=context,
            )

            try:
                response = model.invoke([
                    SystemMessage(content=build_character_prompt(profile)),
                    HumanMessage(content=prompt),
                ])
                text = extract_response_text(response)
                data = extract_json_safe(text, char_name)
                proposal = DesireProposal.model_validate(data)
                proposal.character_name = char_name
                proposals.append(proposal)
                logger.info("角色 %s 提出欲望提案: %s", char_name, proposal.proposed_action)
            except Exception as e:
                logger.warning("角色 %s 欲望提案失败: %s", char_name, e)

        return {
            "desire_proposals": proposals,
            "next_action": "conflict_detect",
        }

    return desire_tick_node
