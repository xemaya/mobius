"""条件路由逻辑（双循环架构）。"""

from __future__ import annotations

from langgraph.graph import END

from mobius.state.novel_state import NovelState


# 所有合法的节点名称
VALID_ACTIONS = {
    # 内循环（世界模拟）
    "env_update",
    "desire_tick",
    "conflict_detect",
    "observer_mark",
    "secondary_viewpoints",
    # 外循环（叙事呈现）
    "plan_chapter",
    "direct_scene",
    "character_action",
    "character_interact",
    "narration",
    "update_state",
    "check_triggers",
    "handle_trigger",
    "compile_chapter",
    "review_chapter",
    "distill_memory",
    # 持久化节点
    "persist_chapter",
    "persist_review",
    "persist_memory",
    # 终止
    "end",
}


def route_by_next_action(state: NovelState) -> str:
    """通用路由：根据 state['next_action'] 决定下一个节点。"""
    action = state.get("next_action", "end")

    if action == "end":
        return END
    if action in VALID_ACTIONS:
        return action
    return END


def route_after_trigger_check(state: NovelState) -> str:
    """触发检查后的路由。"""
    triggered = state.get("triggered_events", [])
    if triggered:
        return "handle_trigger"
    return "direct_scene"
