"""欲望驱动系统数据模型。

冲突 = 欲望相撞。当两个角色的欲望在同一资源上冲突时，
冲突引擎自动标记这是一个高叙事价值事件。
"""

from pydantic import BaseModel, Field


class Desire(BaseModel):
    """角色的欲望。剧情不是触发器，冲突 = 欲望相撞。"""

    id: str = Field(description="欲望唯一标识")
    description: str = Field(description="欲望描述，如'找到并拯救陆辰'")
    priority: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="优先级 0.0-1.0，动态调整",
    )
    category: str = Field(
        default="belonging",
        description="分类: 'survival' | 'belonging' | 'power' | 'self_actualization'",
    )
    satisfied: bool = Field(default=False, description="是否已满足")
    blocked_by: str = Field(default="", description="当前阻碍")
    required_resources: dict[str, float] = Field(
        default_factory=dict,
        description="所需资源，如 {'time': 10, 'reputation': -20}",
    )
    conflicts_with: list[str] = Field(
        default_factory=list,
        description="与哪些其他角色的欲望冲突（欲望 ID 列表）",
    )


class Fear(BaseModel):
    """角色的恐惧。恐惧往往是欲望的反面。"""

    id: str = Field(description="恐惧唯一标识")
    description: str = Field(description="恐惧描述，如'害怕失去对陆辰的控制'")
    intensity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="恐惧强度 0.0-1.0",
    )
    linked_desire: str = Field(
        default="", description="关联的欲望 ID（恐惧往往是欲望的反面）"
    )
    avoidance_cost: float = Field(
        default=0.0,
        description="为逃避恐惧愿意付出的资源代价",
    )


class DesireProposal(BaseModel):
    """角色在每轮 desire_tick 中提出的行动提案。"""

    character_name: str = Field(description="角色名称")
    desire_id: str = Field(description="驱动该提案的欲望 ID")
    proposed_action: str = Field(
        description="提议的行动，如'我想独自潜入数字冥河寻找陆辰'"
    )
    urgency: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="紧迫度 0.0-1.0",
    )
    resource_cost: dict[str, float] = Field(
        default_factory=dict,
        description="预估资源代价",
    )
    fear_override: str = Field(
        default="",
        description="如果该行动需要克服的恐惧 ID",
    )
