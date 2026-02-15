"""状态定义与管理。"""

from mobius.state.character_state import (
    apply_action_to_state,
    apply_belief_change,
    apply_resource_cost,
    check_resource_depletion,
    check_triggers,
    decay_emotions,
    evaluate_condition,
    evaluate_env_rule,
    get_active_behavior_effects,
    get_attribute_value,
    recover_resources,
)
from mobius.state.conflict_engine import detect_conflicts
from mobius.state.novel_state import NovelState

__all__ = [
    "NovelState",
    "apply_action_to_state",
    "apply_belief_change",
    "apply_resource_cost",
    "check_resource_depletion",
    "check_triggers",
    "decay_emotions",
    "detect_conflicts",
    "evaluate_condition",
    "evaluate_env_rule",
    "get_active_behavior_effects",
    "get_attribute_value",
    "recover_resources",
]
