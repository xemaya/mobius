"""冲突引擎：从欲望碰撞中自动检测冲突，生成世界事件。

冲突 = 欲望相撞。当两个角色的欲望在同一资源上冲突时，
冲突引擎自动标记这是一个高叙事价值事件。
"""

from __future__ import annotations

import uuid

from mobius.models.desire import DesireProposal
from mobius.models.resource import ResourcePool
from mobius.models.review import WorldEvent


def detect_conflicts(
    all_proposals: dict[str, list[DesireProposal]],
    all_resources: dict[str, ResourcePool],
) -> list[WorldEvent]:
    """遍历所有角色的欲望提案，检测冲突。

    检测三种类型：
    1. 欲望碰撞：两个角色想要互斥的东西
    2. 资源争夺：多个角色争夺同一有限资源
    3. 恐惧触发：某角色的行动恰好触发另一角色的恐惧

    Returns:
        WorldEvent 列表。
    """
    events: list[WorldEvent] = []
    all_proposals_flat: list[DesireProposal] = []
    for proposals in all_proposals.values():
        all_proposals_flat.extend(proposals)

    # 1) 欲望碰撞检测：检查不同角色提案之间的直接冲突
    for i, pa in enumerate(all_proposals_flat):
        for pb in all_proposals_flat[i + 1 :]:
            if pa.character_name == pb.character_name:
                continue
            # 检查资源重叠
            common_resources = set(pa.resource_cost.keys()) & set(pb.resource_cost.keys())
            if common_resources:
                events.append(
                    WorldEvent(
                        event_id=f"conflict_{uuid.uuid4().hex[:8]}",
                        description=(
                            f"{pa.character_name}想要「{pa.proposed_action}」，"
                            f"而{pb.character_name}想要「{pb.proposed_action}」，"
                            f"双方在 {', '.join(common_resources)} 上产生争夺。"
                        ),
                        participants=[pa.character_name, pb.character_name],
                        conflict_type="desire_clash",
                        narrative_value=min(1.0, (pa.urgency + pb.urgency) / 2 + 0.2),
                    )
                )

    # 2) 资源枯竭检测
    for char_name, pool in all_resources.items():
        depleted: list[str] = []
        for field in ["time", "reputation", "power", "emotional_energy", "wealth"]:
            if getattr(pool, field, 0.0) <= 10:
                depleted.append(field)
        if depleted:
            events.append(
                WorldEvent(
                    event_id=f"depletion_{uuid.uuid4().hex[:8]}",
                    description=(
                        f"{char_name}的 {', '.join(depleted)} 即将耗尽，"
                        f"面临资源危机。"
                    ),
                    participants=[char_name],
                    conflict_type="resource_scarcity",
                    narrative_value=0.6,
                )
            )

    # 3) 恐惧触发检测：某角色的行动可能触发另一角色的恐惧
    for pa in all_proposals_flat:
        if pa.fear_override:
            events.append(
                WorldEvent(
                    event_id=f"fear_{uuid.uuid4().hex[:8]}",
                    description=(
                        f"{pa.character_name}决定克服恐惧「{pa.fear_override}」"
                        f"去执行「{pa.proposed_action}」——这是一个勇气时刻。"
                    ),
                    participants=[pa.character_name],
                    conflict_type="fear_trigger",
                    narrative_value=0.7,
                )
            )

    return events
