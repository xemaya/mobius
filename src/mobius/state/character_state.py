"""角色动态状态管理与触发系统。

包含信念系统逻辑、资源消耗、环境规则评估等。
"""

from __future__ import annotations

import logging

from mobius.models.belief import (
    BELIEF_COLLAPSE_THRESHOLD,
    BELIEF_RESISTANCE,
    BELIEF_SOLIDIFY_THRESHOLD,
)
from mobius.models.character import CharacterAction, CharacterDynamicState
from mobius.models.environment import EnvironmentBehaviorRule, EnvironmentState
from mobius.models.triggers import TriggerRule, TriggeredEvent

logger = logging.getLogger(__name__)

# 情感的"中性"基准值：正面情感趋向 0.5，负面情感趋向 0.0
_EMOTION_NEUTRAL: dict[str, float] = {
    "快乐": 0.5,
    "悲伤": 0.0,
    "愤怒": 0.0,
    "恐惧": 0.0,
    "焦虑": 0.0,
    "兴奋": 0.3,
    "平静": 0.5,
    "厌恶": 0.0,
}


# ──────────────────────────────────────────
# 触发系统
# ──────────────────────────────────────────


def evaluate_condition(value: float, operator: str, threshold: float) -> bool:
    """评估触发条件是否满足。"""
    match operator:
        case "gte" | ">=":
            return value >= threshold
        case "lte" | "<=":
            return value <= threshold
        case "gt" | ">":
            return value > threshold
        case "lt" | "<":
            return value < threshold
        case "eq" | "==":
            return abs(value - threshold) < 1e-6
        case _:
            return False


def get_attribute_value(state: CharacterDynamicState, attribute: str) -> float:
    """从角色动态状态中获取属性值。

    支持从 emotional_state、relationship_scores、custom_attributes、
    beliefs（通过 "belief:<id>" 格式）、fears（通过 "fear:<id>"）、
    resources（通过 "resource:<field>"）中查找。
    """
    # 信念值
    if attribute.startswith("belief:"):
        belief_id = attribute[7:]
        for b in state.beliefs:
            if b.id == belief_id:
                return b.strength
        return 0.0

    # 恐惧值
    if attribute.startswith("fear:"):
        fear_id = attribute[5:]
        for f in state.fears:
            if f.id == fear_id:
                return f.intensity
        return 0.0

    # 资源值
    if attribute.startswith("resource:"):
        resource_name = attribute[9:]
        pool = state.resources
        if hasattr(pool, resource_name):
            return getattr(pool, resource_name)
        return pool.custom.get(resource_name, 0.0)

    if attribute in state.emotional_state:
        return state.emotional_state[attribute]
    if attribute in state.relationship_scores:
        return state.relationship_scores[attribute]
    if attribute in state.custom_attributes:
        return state.custom_attributes[attribute]
    return 0.0


def check_triggers(state: CharacterDynamicState) -> list[TriggeredEvent]:
    """检查角色状态是否触发了任何规则。"""
    events: list[TriggeredEvent] = []
    for rule in state.trigger_rules:
        if rule.triggered and rule.one_shot:
            continue
        value = get_attribute_value(state, rule.attribute)
        if evaluate_condition(value, rule.operator, rule.threshold):
            rule.triggered = True
            events.append(
                TriggeredEvent(
                    character_name=state.name,
                    rule=rule,
                    current_value=value,
                )
            )
    return events


# ──────────────────────────────────────────
# 信念系统
# ──────────────────────────────────────────


def apply_belief_change(
    state: CharacterDynamicState,
    belief_id: str,
    delta: float,
) -> CharacterDynamicState:
    """对指定信念施加变化，根据层级施加抗变系数。

    - core 层仅接受 30% 的变化量
    - middle 层接受 70%
    - surface 层接受 100%

    同时检查"固化"和"崩解"条件。
    """
    updated = state.model_copy(deep=True)
    for belief in updated.beliefs:
        if belief.id != belief_id:
            continue
        resistance = BELIEF_RESISTANCE.get(belief.layer, 1.0)
        actual_delta = delta * resistance
        belief.strength = max(0.0, min(1.0, belief.strength + actual_delta))

        # 固化：surface 信念 >= 0.95 -> 提升为 middle
        if belief.layer == "surface" and belief.strength >= BELIEF_SOLIDIFY_THRESHOLD:
            logger.info(
                "信念固化: %s 的 '%s' 从 surface 升级为 middle",
                state.name,
                belief.statement,
            )
            belief.layer = "middle"

        # 崩解：core 信念 <= 0.2 -> 降级为 middle（重大角色成长时刻）
        if belief.layer == "core" and belief.strength <= BELIEF_COLLAPSE_THRESHOLD:
            logger.info(
                "信念崩解: %s 的核心信念 '%s' 崩解为 middle！这是重大转折点。",
                state.name,
                belief.statement,
            )
            belief.layer = "middle"

        break
    return updated


# ──────────────────────────────────────────
# 资源系统
# ──────────────────────────────────────────


def apply_resource_cost(
    state: CharacterDynamicState,
    costs: dict[str, float],
) -> CharacterDynamicState:
    """扣减角色资源，返回更新后的状态。"""
    updated = state.model_copy(deep=True)
    pool = updated.resources
    for resource_name, delta in costs.items():
        if hasattr(pool, resource_name) and resource_name != "custom":
            current = getattr(pool, resource_name)
            setattr(pool, resource_name, max(0.0, current + delta))
        else:
            current = pool.custom.get(resource_name, 0.0)
            pool.custom[resource_name] = max(0.0, current + delta)
    return updated


def recover_resources(
    state: CharacterDynamicState,
    recovery_rates: dict[str, float],
) -> CharacterDynamicState:
    """每章开始时资源自然回复。"""
    updated = state.model_copy(deep=True)
    pool = updated.resources
    for resource_name, amount in recovery_rates.items():
        if hasattr(pool, resource_name) and resource_name != "custom":
            current = getattr(pool, resource_name)
            setattr(pool, resource_name, min(100.0, current + amount))
        else:
            current = pool.custom.get(resource_name, 0.0)
            pool.custom[resource_name] = min(100.0, current + amount)
    return updated


def check_resource_depletion(state: CharacterDynamicState) -> list[str]:
    """检查哪些资源已耗尽（<= 0）。"""
    depleted: list[str] = []
    pool = state.resources
    for field_name in ["time", "reputation", "power", "information", "emotional_energy", "wealth"]:
        if getattr(pool, field_name, 0.0) <= 0:
            depleted.append(field_name)
    for name, val in pool.custom.items():
        if val <= 0:
            depleted.append(name)
    return depleted


# ──────────────────────────────────────────
# 环境规则评估
# ──────────────────────────────────────────


def evaluate_env_rule(
    rule: EnvironmentBehaviorRule,
    env_state: EnvironmentState,
) -> bool:
    """检查环境行为规则是否被触发。"""
    var = env_state.variables.get(rule.variable_name)
    if var is None:
        return False
    return evaluate_condition(var.value, rule.operator, rule.threshold)


def get_active_behavior_effects(
    character_name: str,
    env_state: EnvironmentState | None,
) -> list[str]:
    """获取当前对指定角色生效的环境行为修饰文本列表。"""
    if env_state is None:
        return []
    effects: list[str] = []
    for rule in env_state.behavior_modifiers:
        if rule.affected_characters and character_name not in rule.affected_characters:
            continue
        if evaluate_env_rule(rule, env_state):
            effects.append(rule.behavior_effect)
    return effects


# ──────────────────────────────────────────
# 综合行动应用
# ──────────────────────────────────────────


def apply_action_to_state(
    state: CharacterDynamicState,
    action: CharacterAction,
    max_memory: int = 20,
) -> CharacterDynamicState:
    """将角色行动产生的状态变化应用到角色动态状态上。"""
    updated = state.model_copy(deep=True)

    # 应用情感变化
    for emotion, delta in action.emotional_change.items():
        current = updated.emotional_state.get(emotion, 0.0)
        updated.emotional_state[emotion] = max(0.0, min(1.0, current + delta))

    # 应用关系变化
    for char_name, delta in action.relationship_change.items():
        current = updated.relationship_scores.get(char_name, 0.0)
        updated.relationship_scores[char_name] = max(-1.0, min(1.0, current + delta))

    # 应用自定义属性变化
    for attr, delta in action.attribute_change.items():
        current = updated.custom_attributes.get(attr, 0.0)
        updated.custom_attributes[attr] = current + delta

    # 应用信念变化
    for belief_id, delta in action.belief_change.items():
        updated = apply_belief_change(updated, belief_id, delta)

    # 应用欲望优先级变化
    for desire_id, delta in action.desire_change.items():
        for desire in updated.desires:
            if desire.id == desire_id:
                desire.priority = max(0.0, min(1.0, desire.priority + delta))
                break

    # 应用资源消耗
    if action.resource_cost:
        updated = apply_resource_cost(updated, action.resource_cost)

    # 添加新记忆
    if action.new_memory:
        updated.memory.append(action.new_memory)
        if len(updated.memory) > max_memory:
            updated.memory = updated.memory[-max_memory:]

    return updated


def decay_emotions(
    state: CharacterDynamicState,
    rate: float = 0.05,
) -> CharacterDynamicState:
    """对角色的情感状态施加自然衰减，使极端情绪逐渐回归中性。"""
    updated = state.model_copy(deep=True)
    for emotion, value in updated.emotional_state.items():
        neutral = _EMOTION_NEUTRAL.get(emotion, 0.0)
        if abs(value - neutral) < 1e-6:
            continue
        if value > neutral:
            updated.emotional_state[emotion] = max(neutral, value - rate)
        else:
            updated.emotional_state[emotion] = min(neutral, value + rate)
    return updated
