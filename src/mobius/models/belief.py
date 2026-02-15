"""信念系统数据模型。

角色成长 = 信念层级重排。当 surface 信念被反复强化到 0.95+，
它可以"固化"为 middle；当 core 信念被削弱到 0.2 以下，它"崩解"为 middle。
"""

from pydantic import BaseModel, Field


# 各层级的抗变系数：core 仅接受 30% 变化量，middle 70%，surface 100%
BELIEF_RESISTANCE: dict[str, float] = {
    "core": 0.3,
    "middle": 0.7,
    "surface": 1.0,
}

# 信念固化/崩解阈值
BELIEF_SOLIDIFY_THRESHOLD = 0.95   # surface -> middle
BELIEF_COLLAPSE_THRESHOLD = 0.2    # core -> middle


class Belief(BaseModel):
    """单条信念。

    信念分三层：
    - core:    核心信念，极难改变（如"人性本善"）
    - middle:  中层信念，可被剧情动摇（如"AI不应拥有情感"）
    - surface: 表层/行为策略，动态更新（如"先观察再行动"）
    """

    id: str = Field(description="信念唯一标识")
    statement: str = Field(description="信念的文字描述，如'掌控一切才能保护所爱的人'")
    layer: str = Field(
        default="surface",
        description="信念层级: 'core' | 'middle' | 'surface'",
    )
    strength: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="信念强度 0.0-1.0",
    )
    category: str = Field(
        default="行为策略",
        description="信念类别: '价值观' | '世界认知' | '行为策略'",
    )
