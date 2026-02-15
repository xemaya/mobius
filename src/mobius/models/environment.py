"""环境交互引擎数据模型（物理状态总线）。

环境不再仅仅是描写，而是能够影响 Agent 决策的变量。
例如"算力余量"、"警报等级"、"雨势大小"。
"""

from pydantic import BaseModel, Field


class EnvironmentVariable(BaseModel):
    """单个环境变量。"""

    name: str = Field(description="变量名，如 'alert_level', 'compute_power'")
    value: float = Field(default=0.0, description="当前值")
    min_val: float = Field(default=0.0, description="最小值")
    max_val: float = Field(default=100.0, description="最大值")
    description: str = Field(
        default="",
        description="人类可读描述，如'系统警报等级，0=平静，100=紧急'",
    )
    decay_rate: float = Field(
        default=0.0,
        description="每章自然衰减量（值向 min_val 方向移动）",
    )


class EnvironmentBehaviorRule(BaseModel):
    """环境变量对角色行为的修改规则。

    当某个环境变量达到阈值时，自动向特定角色的 prompt 注入行为修饰。
    """

    variable_name: str = Field(description="监控的环境变量名")
    threshold: float = Field(description="触发阈值")
    operator: str = Field(
        default=">=",
        description="比较操作符: '>=' | '<=' | '>' | '<' | '=='",
    )
    affected_characters: list[str] = Field(
        default_factory=list,
        description="受影响的角色列表（空列表 = 所有角色）",
    )
    behavior_effect: str = Field(
        description="注入角色 prompt 的行为修饰文本",
    )
    resource_effect: dict[str, float] = Field(
        default_factory=dict,
        description="额外的资源影响，如 {'emotional_energy': -15}",
    )


class EnvironmentState(BaseModel):
    """全局物理环境状态。"""

    variables: dict[str, EnvironmentVariable] = Field(
        default_factory=dict, description="所有环境变量"
    )
    behavior_modifiers: list[EnvironmentBehaviorRule] = Field(
        default_factory=list, description="环境变量对角色行为的影响规则"
    )
