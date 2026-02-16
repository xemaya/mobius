"""主图定义与编排（双循环架构）。

内循环（世界模拟）：env_update -> desire_tick -> conflict_detect -> observer_mark
外循环（叙事呈现）：plan_chapter -> direct_scene -> execute -> compile -> review -> distill -> 下一章
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
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
    create_director_blueprint_refresh,
    create_director_chapter_contract,
    create_director_direct_scene,
    create_generate_setting_pack_node,
    create_generate_storyboards_node,
    create_generate_outlines_node,
    create_director_handle_trigger,
    create_director_plan_chapter,
)
from mobius.agents.memory import create_compress_memories_node
from mobius.agents.narrator import (
    create_compile_chapter_node,
    create_expand_storyboard_chapter_node,
    create_narration_node,
)
from mobius.agents.observer import create_observer_node, create_secondary_viewpoint_node
from mobius.agents.reviewer import create_review_chapter_node
from mobius.config.settings import NovelConfig
from mobius.engine.chaos_engine import ChaosEngine
from mobius.engine.style_governor import StyleGovernor
from mobius.graph.routing import route_after_trigger_check, route_by_next_action
from mobius.models.chaos_engine import ChaosEngineConfig
from mobius.models.architecture import ThreadLedgerItem
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


def _create_style_governor_node(chaos_engine: ChaosEngine | None, model: BaseChatModel | None = None):
    """创建文风控制器节点。

    注意：此节点通过 in-place 修改最新章节内容来工作，
    **不能**返回 {"chapters": ...}，否则 add reducer 会导致章节重复。
    """

    def style_governor_node(state: NovelState) -> dict[str, Any]:
        if chaos_engine is None:
            return {}

        chapters = state.get("chapters", [])
        if not chapters:
            return {}

        latest_chapter = chapters[-1]
        
        # 初始化 StyleGovernor
        governor = StyleGovernor(chaos_engine.config.style_governor, model=model)
        
        # 处理章节内容（in-place 修改，不返回 chapters 列表）
        original_content = latest_chapter.content
        processed_content = governor.process_chapter(latest_chapter.title, original_content)
        
        # in-place 更新章节内容
        if processed_content != original_content:
            logger.info("第%d章内容已由 StyleGovernor 优化", latest_chapter.chapter_index)
            latest_chapter.content = processed_content
            latest_chapter.word_count = len(processed_content)

        # 关键：返回空字典。chapters 使用 Annotated[list, add] reducer，
        # 如果返回 {"chapters": chapters} 会把整个列表追加一次，导致指数级重复。
        return {}

    return style_governor_node


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
            return {"next_action": "end" if current_idx > total else "clue_ledger"}

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
        return {"next_action": "end" if current_idx > total else "clue_ledger"}

    return persist_memory_node


def _create_persist_outlines_node(output_manager: OutputManager | None):
    """保存全书概要。"""

    def persist_outlines_node(state: NovelState) -> dict[str, Any]:
        if output_manager is None:
            return {"next_action": "end"}

        outlines = state.get("chapter_outlines", [])
        if not outlines:
            return {"next_action": "end"}

        for item in outlines:
            output_manager.save_outline(item)
        output_manager.save_full_outline(outlines)
        return {"next_action": "end"}

    return persist_outlines_node


def _create_persist_setting_pack_node(output_manager: OutputManager | None):
    """保存结构化设定集。"""

    def persist_setting_pack_node(state: NovelState) -> dict[str, Any]:
        if output_manager is None:
            return {"next_action": "end"}
        setting_pack = state.get("setting_pack")
        if not setting_pack:
            return {"next_action": "end"}
        output_manager.save_setting_pack(setting_pack)
        return {"next_action": "end"}

    return persist_setting_pack_node


def _create_persist_storyboards_node(output_manager: OutputManager | None):
    """保存全书分镜。"""

    def persist_storyboards_node(state: NovelState) -> dict[str, Any]:
        if output_manager is None:
            return {"next_action": "end"}
        storyboards = state.get("chapter_storyboards", [])
        if not storyboards:
            return {"next_action": "end"}
        for sb in storyboards:
            output_manager.save_storyboard(sb)
        output_manager.save_full_storyboards(storyboards)
        return {"next_action": "end"}

    return persist_storyboards_node


def _extract_keywords(text: str) -> list[str]:
    """从中文/英文短语中抽取关键词。"""
    if not text:
        return []
    tokens = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{3,}", text)
    stop = {"必须", "本章", "剧情", "角色", "变化", "系统", "章节", "推进", "线索"}
    uniq: list[str] = []
    for t in tokens:
        t = t.strip()
        if not t or t in stop:
            continue
        if t not in uniq:
            uniq.append(t)
    return uniq[:12]


def _requirement_hit(content: str, requirement: str) -> bool:
    """判断正文是否兑现某条要求。"""
    if not requirement or requirement.strip() in {"（无）", "无"}:
        return True
    keys = _extract_keywords(requirement)
    if not keys:
        return True
    return any(k in content for k in keys)


def _check_guardrail_violations(content: str, guardrails: list[str]) -> list[str]:
    """检测明显硬约束违背（规则匹配版）。"""
    violations: list[str] = []
    banned_patterns = {
        "只允许信息逆流，禁止物质逆流": ["物质回到过去", "实体回到过去", "肉身回到过去"],
        "存档点是共振锚点，不是重置按钮": ["按下重置", "一键回档", "重置时间线"],
        "只能信息逆流，不能物质逆流": ["物质回到过去", "实体回到过去", "时间回滚实体"],
    }
    for g in guardrails:
        g = g.strip()
        for key, patterns in banned_patterns.items():
            if key in g:
                for p in patterns:
                    if p in content:
                        violations.append(f"触发禁用叙述: {p}")
    return violations


def _create_storyboard_quality_gate_node():
    """章节扩写后的质量闸门（分镜覆盖+设定一致性+推进命中）。"""

    def storyboard_quality_gate_node(state: NovelState) -> dict[str, Any]:
        storyboards = state.get("chapter_storyboards", [])
        chapters = state.get("chapters", [])
        guardrails = state.get("global_guardrails", [])
        setting_pack = state.get("setting_pack")
        if not storyboards or not chapters:
            return {"next_action": "persist_expand_chapter"}

        latest = chapters[-1]
        if latest.chapter_index <= 0 or latest.chapter_index > len(storyboards):
            return {"next_action": "persist_expand_chapter"}
        storyboard = storyboards[latest.chapter_index - 1]

        irrev_ok = _requirement_hit(latest.content, storyboard.irreversible_change)
        payoff_targets = [p for p in storyboard.must_payoffs if p.strip() not in {"", "（无）", "无"}]
        payoff_ok = True if not payoff_targets else any(_requirement_hit(latest.content, p) for p in payoff_targets)
        violations = _check_guardrail_violations(latest.content, guardrails)
        guardrail_ok = len(violations) == 0
        scene_cover_hits = sum(
            1 for s in storyboard.scenes if _requirement_hit(latest.content, s.objective or s.title)
        )
        coverage_ratio = scene_cover_hits / max(len(storyboard.scenes), 1)
        coverage_ok = coverage_ratio >= 0.5
        setting_ok = True
        if setting_pack and setting_pack.worldview_rules:
            # 至少命中一条设定规则关键词，防止写飞
            setting_ok = any(_requirement_hit(latest.content, rule) for rule in setting_pack.worldview_rules)

        reasons: list[str] = []
        if not irrev_ok:
            reasons.append("正文未体现分镜定义的不可逆变化")
        if not payoff_ok:
            reasons.append("正文未体现本章必兑现线索")
        if not guardrail_ok:
            reasons.append("正文触发全书硬约束冲突")
        if not coverage_ok:
            reasons.append(f"正文场景覆盖率不足({coverage_ratio:.0%} < 50%)")
        if not setting_ok:
            reasons.append("正文与结构化设定锚点关联过弱")

        metadata = dict(state.get("metadata", {}))
        attempts = dict(metadata.get("chapter_rewrite_attempts", {}))
        key = str(latest.chapter_index)
        used = int(attempts.get(key, 0))

        if reasons and used < 1:
            attempts[key] = used + 1
            metadata["chapter_rewrite_attempts"] = attempts
            metadata["chapter_rewrite_reason"] = "；".join(reasons) + "。请基于当前稿补足缺失，不要偏离分镜。"
            logger.warning("第%d章质量闸门未通过，触发重写: %s", latest.chapter_index, "；".join(reasons))
            return {
                "metadata": metadata,
                "next_action": "expand_storyboard_chapter",
            }

        if reasons:
            logger.warning("第%d章质量闸门仍有问题但已达重试上限，降级放行: %s", latest.chapter_index, "；".join(reasons))
        else:
            logger.info("第%d章通过质量闸门", latest.chapter_index)
        metadata.pop("chapter_rewrite_reason", None)
        return {"metadata": metadata, "next_action": "persist_expand_chapter"}

    return storyboard_quality_gate_node


def _create_persist_expand_chapter_node(output_manager: OutputManager | None):
    """保存扩写章节并推进到下一章。"""

    def persist_expand_chapter_node(state: NovelState) -> dict[str, Any]:
        chapters = state.get("chapters", [])
        storyboards = state.get("chapter_storyboards", [])
        if not chapters:
            return {"next_action": "end"}
        latest = chapters[-1]
        if output_manager is not None:
            output_manager.save_chapter(latest)

        next_idx = latest.chapter_index + 1
        total = state.get("total_chapters", len(storyboards))
        return {
            "current_chapter_index": next_idx,
            "next_action": "end" if next_idx > total else "expand_storyboard_chapter",
        }

    return persist_expand_chapter_node


def _create_clue_ledger_node():
    """线索账本结算：开线、回收、逾期标注与主题推进记录。"""

    def clue_ledger_node(state: NovelState) -> dict[str, Any]:
        chapters = state.get("chapters", [])
        if not chapters:
            return {"next_action": "env_update"}

        latest_chapter = chapters[-1]
        chapter_plan = state.get("chapter_plan")
        review_list = state.get("chapter_reviews", [])
        latest_review = review_list[-1] if review_list else None

        ledger = list(state.get("payoff_ledger", []))
        open_threads = list(state.get("open_threads", []))
        theme_log = list(state.get("theme_progress_log", []))
        total = state.get("total_chapters", 10)

        # 1) 新承诺入账
        if chapter_plan and chapter_plan.new_promises:
            for promise in chapter_plan.new_promises:
                promise = promise.strip()
                if not promise:
                    continue
                due = min(total, latest_chapter.chapter_index + 2)
                if not any(item.thread == promise for item in ledger):
                    ledger.append(
                        ThreadLedgerItem(
                            thread=promise,
                            source_chapter=latest_chapter.chapter_index,
                            due_chapter=due,
                            status="open",
                            notes="由 chapter_plan.new_promises 生成",
                            updated_at=datetime.now().isoformat(),
                        )
                    )

        # 2) 回收标记（不可变更新，避免污染历史状态）
        payoffs = []
        if chapter_plan and chapter_plan.required_payoffs:
            payoffs = [p.strip() for p in chapter_plan.required_payoffs if p.strip()]

        resolved_ledger: list[ThreadLedgerItem] = []
        for item in ledger:
            if item.status == "resolved":
                resolved_ledger.append(item)
                continue

            matched = any(payoff in item.thread or item.thread in payoff for payoff in payoffs)
            if matched:
                resolved_ledger.append(
                    item.model_copy(
                        update={
                            "status": "resolved",
                            "resolved_chapter": latest_chapter.chapter_index,
                            "updated_at": datetime.now().isoformat(),
                        }
                    )
                )
            else:
                resolved_ledger.append(item)

        # 3) 逾期检查（不可变更新）
        ledger = []
        for item in resolved_ledger:
            if item.status == "open" and latest_chapter.chapter_index > item.due_chapter:
                ledger.append(
                    item.model_copy(
                        update={
                            "status": "overdue",
                            "updated_at": datetime.now().isoformat(),
                        }
                    )
                )
            else:
                ledger.append(item)

        # 4) open_threads 刷新
        unresolved_from_review = []
        if latest_review:
            unresolved_from_review.extend(latest_review.unresolved_threads)
            unresolved_from_review.extend(latest_review.unrecovered_threads)
            theme_log.append(
                f"第{latest_chapter.chapter_index}章 主题推进={latest_review.theme_progression:.2f} "
                f"回收率={latest_review.thread_recovery_rate:.2f} 必要性={latest_review.chapter_necessity:.2f}"
            )
        open_threads = sorted(
            {
                *[item.thread for item in ledger if item.status in ("open", "overdue")],
                *[t for t in unresolved_from_review if t.strip()],
            }
        )

        # 5) 决定是否结束
        current_idx = state.get("current_chapter_index", 1)
        next_action = "end" if current_idx > total else "env_update"
        return {
            "payoff_ledger": ledger,
            "open_threads": open_threads,
            "theme_progress_log": theme_log,
            "next_action": next_action,
        }

    return clue_ledger_node


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
    chaos_engine: ChaosEngine | None = None,
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
    blueprint_refresh = create_director_blueprint_refresh(director_model)

    # 外循环节点
    director_plan = create_director_plan_chapter(director_model, chaos_engine)
    chapter_contract = create_director_chapter_contract()
    director_scene = create_director_direct_scene(director_model)
    director_trigger = create_director_handle_trigger(director_model)

    char_action = create_character_action_node(
        character_profiles,
        reasoning_model=character_model,
        roleplay_model=roleplay_model,
        enable_internal_monologue=config.enable_internal_monologue,
        chaos_engine=chaos_engine,
    )
    char_interact = create_character_interact_node(
        character_profiles,
        reasoning_model=character_model,
        roleplay_model=roleplay_model,
        max_rounds=config.max_interaction_rounds,
        enable_internal_monologue=config.enable_internal_monologue,
        chaos_engine=chaos_engine,
    )

    narration = create_narration_node(narrator_model)
    compile_chapter = create_compile_chapter_node(
        narrator_model,
        chaos_engine,
        chapter_min_words=config.chapter_min_words,
    )
    style_governor = _create_style_governor_node(chaos_engine, narrator_model)
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
    clue_ledger = _create_clue_ledger_node()

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
    workflow.add_node("blueprint_refresh", blueprint_refresh)

    # 外循环节点
    workflow.add_node("plan_chapter", director_plan)
    workflow.add_node("chapter_contract", chapter_contract)
    workflow.add_node("direct_scene", director_scene)
    workflow.add_node("character_action", char_action)
    workflow.add_node("character_interact", char_interact)
    workflow.add_node("narration", narration)
    workflow.add_node("update_state", update_state)
    workflow.add_node("check_triggers", _check_triggers_router)
    workflow.add_node("handle_trigger", director_trigger)
    workflow.add_node("compile_chapter", compile_chapter)
    workflow.add_node("style_governor", style_governor)
    workflow.add_node("persist_chapter", persist_chapter)
    workflow.add_node("review_chapter", review_chapter)
    workflow.add_node("persist_review", persist_review)
    workflow.add_node("distill_memory", distill_memory)
    workflow.add_node("persist_memory", persist_memory)
    workflow.add_node("clue_ledger", clue_ledger)

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
    workflow.add_conditional_edges("blueprint_refresh", route_by_next_action)

    # 外循环：章节规划 → 场景编排
    workflow.add_conditional_edges("plan_chapter", route_by_next_action)
    workflow.add_conditional_edges("chapter_contract", route_by_next_action)
    workflow.add_conditional_edges("direct_scene", route_by_next_action)

    # 角色行动/互动/旁白 → 状态更新
    workflow.add_edge("character_action", "update_state")
    workflow.add_edge("character_interact", "update_state")
    workflow.add_edge("narration", "update_state")

    # 状态更新 → 触发检查
    workflow.add_edge("update_state", "check_triggers")
    workflow.add_conditional_edges("check_triggers", route_after_trigger_check)
    workflow.add_edge("handle_trigger", "direct_scene")

    # 章节编译 → 文风控制 → 持久化落盘 → 支线视角（如有） → 评审 → 持久化评审 → 蒸馏 → 持久化记忆
    workflow.add_edge("compile_chapter", "style_governor")
    workflow.add_edge("style_governor", "persist_chapter")

    if viewpoint_node:
        workflow.add_edge("persist_chapter", "secondary_viewpoints")
        workflow.add_edge("secondary_viewpoints", "review_chapter")
    else:
        workflow.add_edge("persist_chapter", "review_chapter")

    workflow.add_edge("review_chapter", "persist_review")
    workflow.add_edge("persist_review", "distill_memory")
    workflow.add_edge("distill_memory", "persist_memory")
    workflow.add_conditional_edges("persist_memory", route_by_next_action)
    workflow.add_conditional_edges("clue_ledger", route_by_next_action)

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
    chaos_engine: ChaosEngine | None = None,
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
        chaos_engine=chaos_engine,
    )
    checkpointer = InMemorySaver()
    return workflow.compile(checkpointer=checkpointer)


def build_outline_graph(
    director_model: BaseChatModel,
    character_profiles: dict[str, CharacterProfile],
    config: NovelConfig | None = None,
    output_manager: OutputManager | None = None,
) -> StateGraph:
    """构建 Phase1：全书概要生成图。"""
    if config is None:
        config = NovelConfig()

    blueprint_refresh_base = create_director_blueprint_refresh(director_model)

    def blueprint_refresh(state: NovelState) -> dict[str, Any]:
        result = blueprint_refresh_base(state)
        if result.get("next_action") == "plan_chapter":
            result["next_action"] = "generate_outlines"
        return result

    generate_outlines = create_generate_outlines_node(director_model)
    persist_outlines = _create_persist_outlines_node(output_manager)

    workflow = StateGraph(NovelState)
    workflow.add_node("blueprint_refresh", blueprint_refresh)
    workflow.add_node("generate_outlines", generate_outlines)
    workflow.add_node("persist_outlines", persist_outlines)

    workflow.add_edge(START, "blueprint_refresh")
    workflow.add_conditional_edges("blueprint_refresh", route_by_next_action)
    workflow.add_conditional_edges("generate_outlines", route_by_next_action)
    workflow.add_conditional_edges("persist_outlines", route_by_next_action)
    return workflow


def compile_outline_graph(
    director_model: BaseChatModel,
    character_profiles: dict[str, CharacterProfile],
    config: NovelConfig | None = None,
    output_manager: OutputManager | None = None,
):
    """编译 Phase1 概要图。"""
    workflow = build_outline_graph(
        director_model=director_model,
        character_profiles=character_profiles,
        config=config,
        output_manager=output_manager,
    )
    checkpointer = InMemorySaver()
    return workflow.compile(checkpointer=checkpointer)


def build_setting_pack_graph(
    director_model: BaseChatModel,
    output_manager: OutputManager | None = None,
) -> StateGraph:
    """构建 Phase1：结构化设定集生成图。"""
    generate_setting_pack = create_generate_setting_pack_node(director_model)
    persist_setting_pack = _create_persist_setting_pack_node(output_manager)

    workflow = StateGraph(NovelState)
    workflow.add_node("generate_setting_pack", generate_setting_pack)
    workflow.add_node("persist_setting_pack", persist_setting_pack)
    workflow.add_edge(START, "generate_setting_pack")
    workflow.add_conditional_edges("generate_setting_pack", route_by_next_action)
    workflow.add_conditional_edges("persist_setting_pack", route_by_next_action)
    return workflow


def compile_setting_pack_graph(
    director_model: BaseChatModel,
    output_manager: OutputManager | None = None,
):
    """编译 Phase1 设定集图。"""
    workflow = build_setting_pack_graph(
        director_model=director_model,
        output_manager=output_manager,
    )
    checkpointer = InMemorySaver()
    return workflow.compile(checkpointer=checkpointer)


def build_storyboard_graph(
    director_model: BaseChatModel,
    output_manager: OutputManager | None = None,
) -> StateGraph:
    """构建 Phase3：章节分镜生成图。"""
    generate_storyboards = create_generate_storyboards_node(director_model)
    persist_storyboards = _create_persist_storyboards_node(output_manager)

    workflow = StateGraph(NovelState)
    workflow.add_node("generate_storyboards", generate_storyboards)
    workflow.add_node("persist_storyboards", persist_storyboards)
    workflow.add_edge(START, "generate_storyboards")
    workflow.add_conditional_edges("generate_storyboards", route_by_next_action)
    workflow.add_conditional_edges("persist_storyboards", route_by_next_action)
    return workflow


def compile_storyboard_graph(
    director_model: BaseChatModel,
    output_manager: OutputManager | None = None,
):
    """编译 Phase3 分镜图。"""
    workflow = build_storyboard_graph(
        director_model=director_model,
        output_manager=output_manager,
    )
    checkpointer = InMemorySaver()
    return workflow.compile(checkpointer=checkpointer)


def build_expand_graph(
    narrator_model: BaseChatModel,
    config: NovelConfig | None = None,
    output_manager: OutputManager | None = None,
    chaos_engine: ChaosEngine | None = None,
) -> StateGraph:
    """构建 Phase4：按分镜扩写图。"""
    if config is None:
        config = NovelConfig()

    expand_storyboard_chapter = create_expand_storyboard_chapter_node(
        narrator_model,
        chaos_engine=chaos_engine,
        chapter_min_words=config.chapter_min_words,
    )
    quality_gate = _create_storyboard_quality_gate_node()
    persist_expand = _create_persist_expand_chapter_node(output_manager)

    workflow = StateGraph(NovelState)
    workflow.add_node("expand_storyboard_chapter", expand_storyboard_chapter)
    workflow.add_node("storyboard_quality_gate", quality_gate)
    workflow.add_node("persist_expand_chapter", persist_expand)

    workflow.add_edge(START, "expand_storyboard_chapter")
    workflow.add_conditional_edges("expand_storyboard_chapter", route_by_next_action)
    workflow.add_conditional_edges("storyboard_quality_gate", route_by_next_action)
    workflow.add_conditional_edges("persist_expand_chapter", route_by_next_action)
    return workflow


def compile_expand_graph(
    narrator_model: BaseChatModel,
    config: NovelConfig | None = None,
    output_manager: OutputManager | None = None,
    chaos_engine: ChaosEngine | None = None,
):
    """编译 Phase4 扩写图。"""
    workflow = build_expand_graph(
        narrator_model=narrator_model,
        config=config,
        output_manager=output_manager,
        chaos_engine=chaos_engine,
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
                emotional_state={},
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
        "chapter_contract": None,
        "chapters": [],
        "chapter_outlines": [],
        "outline_approved": False,
        "global_guardrails": [],
        "setting_pack": None,
        "setting_approved": False,
        "chapter_storyboards": [],
        "storyboard_approved": False,
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
        "revealed_information": [],
        "novel_blueprint": None,
        "open_threads": [],
        "payoff_ledger": [],
        "theme_progress_log": [],
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
                "emotional_state", {}
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

    # 可选：小说生成参数（覆盖 NovelConfig 默认值，用于提速等）
    novel_config_overrides = data.get("novel_config") or {}

    return {
        "worldview": worldview,
        "plot_outline": plot_outline,
        "character_profiles": character_profiles,
        "character_states": character_states,
        "environment": environment,
        "secondary_viewpoints": secondary_viewpoints,
        "novel_config_overrides": novel_config_overrides,
    }
