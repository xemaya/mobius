"""世界观察者 Agent + 支线观察者视角。

世界观察者不干预世界，只做三件事：记录、评估叙事价值、裁剪呈现。
支线观察者从非主角视角生成叙事片段，增加空间感和悬疑感。
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from mobius.agents.utils import extract_json, extract_response_text, invoke_with_retry
from mobius.models.environment import EnvironmentState
from mobius.models.review import WorldEvent
from mobius.models.viewpoint import SecondaryViewpoint, ViewpointFragment
from mobius.prompts import load_prompt
from mobius.state.character_state import evaluate_condition
from mobius.state.novel_state import NovelState

logger = logging.getLogger(__name__)

# 从外部文件加载提示词
OBSERVER_SYSTEM_PROMPT = load_prompt("observer_system")


def create_observer_node(
    model: BaseChatModel,
) -> Callable[[NovelState], dict[str, Any]]:
    """创建世界观察者节点。

    输入: world_events（世界模拟产生的所有事件）
    处理: 用 LLM 评估每个事件的叙事价值
    输出: 标记了 narrative_value 和 selected_for_narrative 的事件列表
    """

    def observer_mark_node(state: NovelState) -> dict[str, Any]:
        world_events = state.get("world_events", [])
        theme = state.get("theme", "")
        tension_curve = state.get("tension_curve", [])

        if not world_events:
            return {"next_action": "plan_chapter"}

        # 构建事件列表描述
        event_descriptions = []
        for i, evt in enumerate(world_events):
            event_descriptions.append(
                f"{i+1}. [{evt.conflict_type}] {evt.description} "
                f"(参与者: {', '.join(evt.participants)})"
            )

        recent_tension = tension_curve[-3:] if tension_curve else []

        user_prompt = f"""# 核心主题
{theme or '（未设定）'}

# 近期张力曲线
{recent_tension if recent_tension else '（无数据）'}

# 待评估的世界事件
{chr(10).join(event_descriptions)}

请为每个事件评估叙事价值（0-1），并决定是否应该写入小说。

以 JSON 数组格式返回：
```json
[
    {{
        "event_index": 1,
        "narrative_value": 0.8,
        "selected": true,
        "reason": "简要说明选择/不选择的原因"
    }}
]
```

选择标准：
- 冲突强度：涉及多少角色？欲望碰撞有多剧烈？
- 情感密度：事件的情感落差有多大？
- 主题相关度：与核心主题的关联度
- 悬念贡献：是否为后续剧情埋下伏笔？
- 新鲜度：是否与最近的事件有足够差异？"""

        try:
            response = invoke_with_retry(
                model,
                [
                    SystemMessage(content=OBSERVER_SYSTEM_PROMPT),
                    HumanMessage(content=user_prompt),
                ],
                operation_name="observer_mark",
            )
            text = extract_response_text(response)
            evaluations = extract_json(text)

            if isinstance(evaluations, list):
                for eval_item in evaluations:
                    idx = eval_item.get("event_index", 0) - 1
                    if 0 <= idx < len(world_events):
                        world_events[idx].narrative_value = eval_item.get("narrative_value", 0.5)
                        world_events[idx].selected_for_narrative = eval_item.get("selected", False)

            selected_count = sum(1 for e in world_events if e.selected_for_narrative)
            logger.info(
                "观察者评估完成: %d个事件中选出%d个高价值事件",
                len(world_events),
                selected_count,
            )
        except Exception as e:
            logger.error("观察者评估失败: %s，使用默认值", e)
            # 降级：按初始 narrative_value 排序，选前 3 个
            for evt in sorted(world_events, key=lambda x: x.narrative_value, reverse=True)[:3]:
                evt.selected_for_narrative = True

        return {
            "world_events": world_events,
            "next_action": "plan_chapter",
        }

    return observer_mark_node


# ──────────────────────────────────────────
# 支线观察者视角
# ──────────────────────────────────────────


VIEWPOINT_SYSTEM_PROMPT = load_prompt("observer_viewpoint")


def create_secondary_viewpoint_node(
    viewpoints: list[SecondaryViewpoint],
    model: BaseChatModel,
) -> Callable[[NovelState], dict[str, Any]]:
    """创建支线观察者视角生成节点。

    从非主角视角描述场景片段，增加叙事空间感。
    """

    def secondary_viewpoint_node(state: NovelState) -> dict[str, Any]:
        if not viewpoints:
            return {}

        env_state: EnvironmentState | None = state.get("environment")
        scene_actions = state.get("scene_actions", [])
        current_scene = state.get("current_scene")

        if not scene_actions and not current_scene:
            return {}

        # 构建当前章节的事件摘要
        action_summary = []
        for act in scene_actions[-5:]:
            action_summary.append(f"[{act.character_name}] {act.content[:100]}")
        summary_text = "\n".join(action_summary) if action_summary else "（尚无明显事件）"

        fragments: list[ViewpointFragment] = []

        for vp in viewpoints:
            # 检查触发条件
            if vp.trigger_condition and env_state:
                if not _evaluate_viewpoint_trigger(vp.trigger_condition, env_state):
                    continue

            # 检查可观察范围
            if vp.can_observe and current_scene:
                participants = set(current_scene.participating_characters)
                if not participants.intersection(set(vp.can_observe)):
                    continue

            system_prompt = VIEWPOINT_SYSTEM_PROMPT.format(
                viewpoint_name=vp.name,
                perspective_type=vp.perspective_type,
                voice_style=vp.voice_style,
            )

            location = current_scene.location if current_scene else "未知地点"
            user_prompt = f"""地点：{location}

最近发生的事件：
{summary_text}

请从你的视角描述你观察到的场景。记住保持你独特的叙述风格。"""

            try:
                response = invoke_with_retry(
                    model,
                    [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_prompt),
                    ],
                    operation_name="secondary_viewpoint",
                )
                content = extract_response_text(response)
                if content:
                    fragment = ViewpointFragment(
                        viewpoint_id=vp.id,
                        viewpoint_name=vp.name,
                        content=content,
                        related_characters=[
                            a.character_name for a in scene_actions[-5:]
                        ],
                    )
                    fragments.append(fragment)
                    logger.info("支线视角「%s」生成完成", vp.name)
            except Exception as e:
                logger.warning("支线视角「%s」生成失败: %s", vp.name, e)

        return {"viewpoint_fragments": fragments}

    return secondary_viewpoint_node


def _evaluate_viewpoint_trigger(
    condition: str,
    env_state: EnvironmentState,
) -> bool:
    """简单评估视角触发条件，格式如 'alert_level > 70'。"""
    try:
        parts = condition.split()
        if len(parts) != 3:
            return True  # 格式不对就默认触发
        var_name, op, threshold_str = parts
        var = env_state.variables.get(var_name)
        if var is None:
            return False
        threshold = float(threshold_str)
        return evaluate_condition(var.value, op, threshold)
    except Exception:
        return True  # 解析失败默认触发
