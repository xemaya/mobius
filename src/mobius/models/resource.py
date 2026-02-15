"""资源系统数据模型。

没有资源，就没有真实冲突。行动必须有成本，决策 = 有限资源下的选择。
"""

from pydantic import BaseModel, Field


class ResourcePool(BaseModel):
    """角色持有的资源池。行动需要消耗资源，决策 = 有限资源下的选择。

    当 emotional_energy 降到 20 以下时，角色不会再理性决策，
    而是开始做出冲动行为（由 trigger_rules 控制）。
    当 reputation 归零，角色无法再调动社会资源。
    """

    time: float = Field(default=100.0, description="时间精力（每章消耗，自然回复一部分）")
    reputation: float = Field(
        default=50.0, description="社会声誉（0-100，影响可调动的社会关系）"
    )
    power: float = Field(default=0.0, description="权力/影响力（职位、地位带来的硬权力）")
    information: float = Field(
        default=0.0, description="信息量（知道多少真相，信息不对称的核心）"
    )
    emotional_energy: float = Field(
        default=80.0, description="情绪能量（过低时角色会崩溃或做出非理性决策）"
    )
    wealth: float = Field(default=50.0, description="经济资源")
    custom: dict[str, float] = Field(
        default_factory=dict, description="自定义资源维度"
    )


class ResourceCost(BaseModel):
    """一次行动的资源消耗。"""

    costs: dict[str, float] = Field(
        default_factory=dict,
        description="资源消耗，如 {'time': -10, 'reputation': -5, 'emotional_energy': -15}",
    )
    gains: dict[str, float] = Field(
        default_factory=dict,
        description="可能的资源收益，如 {'information': +20}",
    )


class ResourceEvent(BaseModel):
    """资源变动事件，用于追踪。"""

    character_name: str = Field(description="角色名称")
    action_description: str = Field(description="导致资源变动的行动描述")
    costs: dict[str, float] = Field(default_factory=dict, description="实际消耗")
    remaining: dict[str, float] = Field(
        default_factory=dict, description="变动后剩余资源"
    )
    is_depleted: list[str] = Field(
        default_factory=list, description="哪些资源被耗尽"
    )
