"""主图定义与编排（双循环架构）。

内循环（世界模拟）：env_update -> desire_tick -> conflict_detect -> observer_mark
外循环（叙事呈现）：plan_chapter -> direct_scene -> execute -> compile -> review -> distill -> 下一章
"""

from __future__ import annotations

import logging
from typing import Any

import yaml
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from mobius.agents.character import (
    create_character_action_node,
    create_character_interact_node,
    create_desire_tick_node,
)
from mobius.agents.director import (
    create_director_direct_scene,
    create_director_handle_trigger,
    create_director_plan_chapter,
)
from mobius.agents.memory import create_compress_memories_node
from mobius.agents.narrator import create_compile_chapter_node, create_narration_node
from mobius.agents.observer import create_observer_node, create_secondary_viewpoint_node
from mobius.agents.reviewer import create_review_chapter_node
from mobius.config.settings import NovelConfig
from mobius.graph.routing import route_after_trigger_check, route_by_next_action
from mobius.output.manager import OutputManager
from mobius.models.belief import Belief
from mobius.models.character import CharacterDynamicState, CharacterProfile
from mobius.models.desire import Desire, Fear
from mobius.models.environment import EnvironmentBehaviorRule, EnvironmentState, EnvironmentVariable
from mobius.models.resource import ResourcePool
from mobius.models.triggers import TriggerRule
from mobius.models.viewpoint import SecondaryViewpoint
from mobius.models.worldview import PlotOutline, WorldView
from mobius.state.character_state import (
    apply_action_to_state,
    check_triggers,
    decay_emotions,
    recover_resources,
)
from mobius.state.conflict_engine import detect_conflicts
from mobius.state.novel_state import NovelState

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────
# 内部节点：环境更新
# ────────────────────────────────────────────


def _create_env_update_node(config: NovelConfig):
    """创建环境变量更新节点。

    在每章开始时：
    1. 环境变量按 decay_rate 自然衰减
    2. 角色资源按 recovery_rate 自然回复
    """

    def env_update_node(state: NovelState) -> dict[str, Any]:
        # 环境变量衰减
        env_state = state.get("environment")
        if env_state and env_state.variables:
            updated_env = env_state.model_copy(deep=True)
            for var_name, var in updated_env.variables.items():
                if var.decay_rate > 0:
                    # 向 min_val 方向衰减
                    if var.value > var.min_val:
                        var.value = max(var.min_val, var.value - var.decay_rate)
                    elif var.value < var.min_val:
                        var.value = min(var.min_val, var.value + var.decay_rate)
            result_env = updated_env
        else:
            result_env = env_state

        # 角色资源回复
        char_states = dict(state.get("character_states", {}))
        for name, cs in char_states.items():
            char_states[name] = recover_resources(cs, config.resource_recovery_rate)

        result: dict[str, Any] = {
            "character_states": char_states,
            "next_action": "desire_tick",
        }
        if result_env is not None:
            result["environment"] = result_env
        return result

    return env_update_node


# ────────────────────────────────────────────
# 内部节点：冲突检测
# ────────────────────────────────────────────


def _create_conflict_detect_node():
    """创建冲突检测节点。从欲望提案中检测冲突并生成世界事件。"""

    def conflict_detect_node(state: NovelState) -> dict[str, Any]:
        proposals = state.get("desire_proposals", [])
        char_states = state.get("character_states", {})

        if not proposals:
            return {"world_events": [], "next_action": "observer_mark"}

        # 整理提案
        all_proposals: dict[str, list] = {}
        for p in proposals:
            all_proposals.setdefault(p.character_name, []).append(p)

        # 整理资源
        all_resources: dict[str, ResourcePool] = {
            name: cs.resources for name, cs in char_states.items()
        }

        # 检测冲突
        events = detect_conflicts(all_proposals, all_resources)

        logger.info("冲突引擎检测到 %d 个世界事件", len(events))

        return {
            "world_events": events,
            "next_action": "observer_mark",
        }

    return conflict_detect_node


# ────────────────────────────────────────────
# 状态更新节点
# ────────────────────────────────────────────


def _create_update_state_node(config: NovelConfig):
    """创建状态更新节点。

    应用角色行动的状态变化（包括信念、资源、环境）、情感衰减、触发检查。
    """

    def update_state_node(state: NovelState) -> dict[str, Any]:
        actions = state.get("scene_actions", [])
        char_states = dict(state.get("character_states", {}))
        env_state = state.get("environment")
        all_triggered = []
        acted_chars: set[str] = set()

        for action in actions:
            char_name = action.character_name
            if char_name not in char_states:
                continue

            acted_chars.add(char_name)

            # 应用状态变化（含信念、欲望、资源）
            updated = apply_action_to_state(
                char_states[char_name],
                action,
                max_memory=config.max_memory_entries,
            )
            char_states[char_name] = updated

            # 应用环境变量变化
            if action.environment_change and env_state:
                updated_env = env_state.model_copy(deep=True)
                for var_name, delta in action.environment_change.items():
                    if var_name in updated_env.variables:
                        var = updated_env.variables[var_name]
                        var.value = max(var.min_val, min(var.max_val, var.value + delta))
                env_state = updated_env

            # 检查触发条件
            triggered = check_triggers(updated)
            all_triggered.extend(triggered)

        # 情感衰减
        for char_name in acted_chars:
            char_states[char_name] = decay_emotions(
                char_states[char_name],
                rate=config.emotion_decay_rate,
            )

        if all_triggered:
            for evt in all_triggered:
                logger.info(
                    "触发事件: 角色「%s」- %s (值=%.1f)",
                    evt.character_name,
                    evt.rule.event_description,
                    evt.current_value,
                )

        result: dict[str, Any] = {
            "character_states": char_states,
            "triggered_events": all_triggered,
            "next_action": "check_triggers",
        }
        if env_state is not None:
            result["environment"] = env_state
        return result

    return update_state_node


# ────────────────────────────────────────────
# 触发检查路由
# ────────────────────────────────────────────


def _check_triggers_router(state: NovelState) -> dict[str, Any]:
    """触发检查路由节点。"""
    triggered = state.get("triggered_events", [])
    if triggered:
        return {"next_action": "handle_trigger"}
    return {"next_action": "direct_scene"}


# ────────────────────────────────────────────
# 持久化节点：每章产出立即落盘
# ────────────────────────────────────────────


def _create_persist_chapter_node(output_manager: OutputManager | None):
    """创建章节持久化节点。

    每生成一章立即写入磁盘，同时保存事件、角色状态快照。
    """

    def persist_chapter_node(state: NovelState) -> dict[str, Any]:
        if output_manager is None:
            return {}

        chapters = state.get("chapters", [])
        if not chapters:
            return {}

        latest_chapter = chapters[-1]
        chapter_idx = latest_chapter.chapter_index

        # 1. 保存章节
        output_manager.save_chapter(latest_chapter)

        # 2. 保存世界事件和欲望提案
        output_manager.save_chapter_events(
            chapter_index=chapter_idx,
            world_events=state.get("world_events"),
            desire_proposals=state.get("desire_proposals"),
        )

        # 3. 保存角色状态快照
        output_manager.save_character_states(
            chapter_index=chapter_idx,
            character_states=state.get("character_states", {}),
        )

        logger.info("持久化完成: 第%d章及相关数据已写入磁盘", chapter_idx)
        return {}

    return persist_chapter_node


def _create_persist_review_node(output_manager: OutputManager | None):
    """创建评审结果持久化节点。在评审完成后保存评审记录。"""

    def persist_review_node(state: NovelState) -> dict[str, Any]:
        if output_manager is None:
            return {"next_action": "distill_memory"}

        reviews = state.get("chapter_reviews", [])
        chapters = state.get("chapters", [])
        if reviews and chapters:
            latest_chapter = chapters[-1]
            latest_review = reviews[-1]
            output_manager.save_chapter_review(
                chapter_index=latest_chapter.chapter_index,
                review=latest_review,
            )

        return {"next_action": "distill_memory"}

    return persist_review_node


def _create_persist_memory_node(output_manager: OutputManager | None):
    """在记忆蒸馏后保存记忆摘要。"""

    def persist_memory_node(state: NovelState) -> dict[str, Any]:
        if output_manager is None:
            # 原始路由逻辑
            current_idx = state.get("current_chapter_index", 1)
            total = state.get("total_chapters", 10)
            return {"next_action": "end" if current_idx > total else "env_update"}

        memory_summaries = state.get("memory_summaries", [])
        chapters = state.get("chapters", [])
        if memory_summaries and chapters:
            latest_chapter = chapters[-1]
            output_manager.save_chapter_memory(
                chapter_index=latest_chapter.chapter_index,
                memory_summaries=memory_summaries,
            )

        # 原始路由逻辑
        current_idx = state.get("current_chapter_index", 1)
        total = state.get("total_chapters", 10)
        return {"next_action": "end" if current_idx > total else "env_update"}

    return persist_memory_node


# ────────────────────────────────────────────
# 图构建
# ────────────────────────────────────────────


def build_novel_graph(
    director_model: BaseChatModel,
    character_model: BaseChatModel,
    narrator_model: BaseChatModel,
    character_profiles: dict[str, CharacterProfile],
    config: NovelConfig | None = None,
    roleplay_model: BaseChatModel | None = None,
    reviewer_model: BaseChatModel | None = None,
    observer_model: BaseChatModel | None = None,
    secondary_viewpoints: list[SecondaryViewpoint] | None = None,
    output_manager: OutputManager | None = None,
) -> StateGraph:
    """构建完整的小说生成主图（双循环架构）。"""
    if config is None:
        config = NovelConfig()

    effective_reviewer = reviewer_model or director_model
    effective_observer = observer_model or director_model

    # ── 创建节点函数 ──

    # 内循环节点
    env_update = _create_env_update_node(config)
    desire_tick = create_desire_tick_node(character_profiles, character_model)
    conflict_detect = _create_conflict_detect_node()
    observer_mark = create_observer_node(effective_observer)

    # 外循环节点
    director_plan = create_director_plan_chapter(director_model)
    director_scene = create_director_direct_scene(director_model)
    director_trigger = create_director_handle_trigger(director_model)

    char_action = create_character_action_node(
        character_profiles,
        reasoning_model=character_model,
        roleplay_model=roleplay_model,
        enable_internal_monologue=config.enable_internal_monologue,
    )
    char_interact = create_character_interact_node(
        character_profiles,
        reasoning_model=character_model,
        roleplay_model=roleplay_model,
        max_rounds=config.max_interaction_rounds,
        enable_internal_monologue=config.enable_internal_monologue,
    )

    narration = create_narration_node(narrator_model)
    compile_chapter = create_compile_chapter_node(narrator_model)
    review_chapter = create_review_chapter_node(effective_reviewer)
    distill_memory = create_compress_memories_node(
        narrator_model,
        compress_threshold=config.memory_compress_threshold,
        keep_recent=config.memory_keep_recent,
    )
    update_state = _create_update_state_node(config)

    # 持久化节点
    persist_chapter = _create_persist_chapter_node(output_manager)
    persist_review = _create_persist_review_node(output_manager)
    persist_memory = _create_persist_memory_node(output_manager)

    # 支线视角节点（可选）
    viewpoint_node = None
    if secondary_viewpoints and config.enable_secondary_viewpoints:
        viewpoint_node = create_secondary_viewpoint_node(
            secondary_viewpoints, effective_observer
        )

    # ── 构建图 ──
    workflow = StateGraph(NovelState)

    # 内循环节点
    workflow.add_node("env_update", env_update)
    workflow.add_node("desire_tick", desire_tick)
    workflow.add_node("conflict_detect", conflict_detect)
    workflow.add_node("observer_mark", observer_mark)

    # 外循环节点
    workflow.add_node("plan_chapter", director_plan)
    workflow.add_node("direct_scene", director_scene)
    workflow.add_node("character_action", char_action)
    workflow.add_node("character_interact", char_interact)
    workflow.add_node("narration", narration)
    workflow.add_node("update_state", update_state)
    workflow.add_node("check_triggers", _check_triggers_router)
    workflow.add_node("handle_trigger", director_trigger)
    workflow.add_node("compile_chapter", compile_chapter)
    workflow.add_node("persist_chapter", persist_chapter)
    workflow.add_node("review_chapter", review_chapter)
    workflow.add_node("persist_review", persist_review)
    workflow.add_node("distill_memory", distill_memory)
    workflow.add_node("persist_memory", persist_memory)

    if viewpoint_node:
        workflow.add_node("secondary_viewpoints", viewpoint_node)

    # ── 设置边 ──

    # 入口 → 环境更新（双循环起点）
    workflow.add_edge(START, "env_update")

    # 内循环：env_update → desire_tick → conflict_detect → observer_mark → plan_chapter
    workflow.add_edge("env_update", "desire_tick")
    workflow.add_conditional_edges("desire_tick", route_by_next_action)
    workflow.add_conditional_edges("conflict_detect", route_by_next_action)
    workflow.add_conditional_edges("observer_mark", route_by_next_action)

    # 外循环：章节规划 → 场景编排
    workflow.add_conditional_edges("plan_chapter", route_by_next_action)
    workflow.add_conditional_edges("direct_scene", route_by_next_action)

    # 角色行动/互动/旁白 → 状态更新
    workflow.add_edge("character_action", "update_state")
    workflow.add_edge("character_interact", "update_state")
    workflow.add_edge("narration", "update_state")

    # 状态更新 → 触发检查
    workflow.add_edge("update_state", "check_triggers")
    workflow.add_conditional_edges("check_triggers", route_after_trigger_check)
    workflow.add_edge("handle_trigger", "direct_scene")

    # 章节编译 → 持久化落盘 → 支线视角（如有） → 评审 → 持久化评审 → 蒸馏 → 持久化记忆
    workflow.add_edge("compile_chapter", "persist_chapter")

    if viewpoint_node:
        workflow.add_edge("persist_chapter", "secondary_viewpoints")
        workflow.add_edge("secondary_viewpoints", "review_chapter")
    else:
        workflow.add_edge("persist_chapter", "review_chapter")

    workflow.add_edge("review_chapter", "persist_review")
    workflow.add_edge("persist_review", "distill_memory")
    workflow.add_edge("distill_memory", "persist_memory")
    workflow.add_conditional_edges("persist_memory", route_by_next_action)

    return workflow


def compile_novel_graph(
    director_model: BaseChatModel,
    character_model: BaseChatModel,
    narrator_model: BaseChatModel,
    character_profiles: dict[str, CharacterProfile],
    config: NovelConfig | None = None,
    roleplay_model: BaseChatModel | None = None,
    reviewer_model: BaseChatModel | None = None,
    observer_model: BaseChatModel | None = None,
    secondary_viewpoints: list[SecondaryViewpoint] | None = None,
    output_manager: OutputManager | None = None,
):
    """构建并编译主图，带 Checkpointer。"""
    workflow = build_novel_graph(
        director_model=director_model,
        character_model=character_model,
        narrator_model=narrator_model,
        character_profiles=character_profiles,
        config=config,
        roleplay_model=roleplay_model,
        reviewer_model=reviewer_model,
        observer_model=observer_model,
        secondary_viewpoints=secondary_viewpoints,
        output_manager=output_manager,
    )
    checkpointer = InMemorySaver()
    return workflow.compile(checkpointer=checkpointer)


# ────────────────────────────────────────────
# 初始化辅助函数
# ────────────────────────────────────────────


def create_initial_state(
    worldview: WorldView,
    plot_outline: PlotOutline,
    character_profiles: dict[str, CharacterProfile],
    initial_character_states: dict[str, CharacterDynamicState] | None = None,
    environment: EnvironmentState | None = None,
    secondary_viewpoints: list[SecondaryViewpoint] | None = None,
) -> NovelState:
    """根据设定集创建初始状态。"""
    if initial_character_states is None:
        initial_character_states = {}
        for name, profile in character_profiles.items():
            initial_character_states[name] = CharacterDynamicState(
                name=name,
                emotional_state={"快乐": 0.5, "悲伤": 0.0, "愤怒": 0.0, "恐惧": 0.0},
                relationship_scores={},
                custom_attributes={},
                trigger_rules=[],
                memory=[],
                current_location="",
                is_alive=True,
            )

    return {
        "worldview": worldview,
        "plot_outline": plot_outline,
        "character_profiles": character_profiles,
        "theme": plot_outline.theme or "",
        "total_chapters": plot_outline.total_chapters,
        "current_chapter_index": 1,
        "chapter_plan": None,
        "chapters": [],
        "scene_queue": [],
        "current_scene": None,
        "character_states": initial_character_states,
        "scene_actions": [],
        "narrative_buffer": [],
        "triggered_events": [],
        "desire_proposals": [],
        "world_events": [],
        "environment": environment or EnvironmentState(),
        "secondary_viewpoints": secondary_viewpoints or [],
        "viewpoint_fragments": [],
        "tension_curve": [],
        "chapter_reviews": [],
        "memory_summaries": [],
        "next_action": "env_update",
        "error": None,
        "metadata": {},
    }


def load_setting_from_yaml(path: str) -> dict[str, Any]:
    """从 YAML 文件加载设定集。

    Returns:
        包含 worldview, plot_outline, character_profiles, character_states,
        environment, secondary_viewpoints 的字典。
    """
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    worldview = WorldView.model_validate(data["worldview"])
    plot_outline = PlotOutline.model_validate(data["plot_outline"])

    # 解析角色档案
    character_profiles: dict[str, CharacterProfile] = {}
    for char_data in data.get("characters", []):
        profile = CharacterProfile.model_validate(char_data)
        character_profiles[profile.name] = profile

    # 解析角色初始状态
    character_states: dict[str, CharacterDynamicState] = {}
    for char_data in data.get("characters", []):
        name = char_data["name"]
        init_state = char_data.get("initial_state", {})

        # 解析触发规则
        trigger_rules = [
            TriggerRule.model_validate(tr) for tr in init_state.get("trigger_rules", [])
        ]

        # 解析信念
        beliefs = [
            Belief.model_validate(b) for b in init_state.get("beliefs", [])
        ]

        # 解析欲望
        desires = [
            Desire.model_validate(d) for d in init_state.get("desires", [])
        ]

        # 解析恐惧
        fears = [
            Fear.model_validate(f) for f in init_state.get("fears", [])
        ]

        # 解析资源池
        resources_data = init_state.get("resources", {})
        resources = ResourcePool.model_validate(resources_data) if resources_data else ResourcePool()

        character_states[name] = CharacterDynamicState(
            name=name,
            emotional_state=init_state.get(
                "emotional_state", {"快乐": 0.5, "悲伤": 0.0, "愤怒": 0.0, "恐惧": 0.0}
            ),
            relationship_scores=init_state.get("relationship_scores", {}),
            custom_attributes=init_state.get("custom_attributes", {}),
            trigger_rules=trigger_rules,
            beliefs=beliefs,
            desires=desires,
            fears=fears,
            resources=resources,
            memory=init_state.get("memory", []),
            compressed_memory=init_state.get("compressed_memory", ""),
            current_location=init_state.get("current_location", ""),
            is_alive=init_state.get("is_alive", True),
        )

    # 解析环境状态
    environment = None
    env_data = data.get("environment")
    if env_data:
        variables = {}
        for var_name, var_data in env_data.get("variables", {}).items():
            var_data["name"] = var_name
            variables[var_name] = EnvironmentVariable.model_validate(var_data)

        modifiers = [
            EnvironmentBehaviorRule.model_validate(m)
            for m in env_data.get("behavior_modifiers", [])
        ]

        environment = EnvironmentState(variables=variables, behavior_modifiers=modifiers)

    # 解析支线视角
    secondary_viewpoints = [
        SecondaryViewpoint.model_validate(vp)
        for vp in data.get("secondary_viewpoints", [])
    ]

    return {
        "worldview": worldview,
        "plot_outline": plot_outline,
        "character_profiles": character_profiles,
        "character_states": character_states,
        "environment": environment,
        "secondary_viewpoints": secondary_viewpoints,
    }
