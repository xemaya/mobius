"""支线观察者视角数据模型。

为了打破场景重复感、增加叙事空间感，引入非主角视角的 Agent。
例如"系统自动日志"、"匿名路人"、"监控摄像头"等。
"""

from pydantic import BaseModel, Field


class SecondaryViewpoint(BaseModel):
    """非主角叙事视角定义。"""

    id: str = Field(description="视角唯一标识，如 'system_log', 'bystander_01'")
    name: str = Field(description="视角名称，如 'MindForge 系统自动日志'")
    perspective_type: str = Field(
        default="bystander",
        description="视角类型: 'system_log' | 'bystander' | 'surveillance' | 'inner_voice'",
    )
    voice_style: str = Field(
        description="叙述风格，如'冷冰冰的机器语言，只记录事实' / '好奇的邻居视角'"
    )
    can_observe: list[str] = Field(
        default_factory=list,
        description="可观察的角色/区域名称列表（空 = 全部）",
    )
    trigger_condition: str = Field(
        default="",
        description="激活条件表达式，如 'alert_level > 70'",
    )


class ViewpointFragment(BaseModel):
    """支线视角生成的叙事片段。"""

    viewpoint_id: str = Field(description="来源视角 ID")
    viewpoint_name: str = Field(description="来源视角名称")
    content: str = Field(description="从该视角描述的叙事片段")
    related_characters: list[str] = Field(
        default_factory=list, description="相关角色"
    )
    insertion_hint: str = Field(
        default="",
        description="建议插入的叙事位置，如'在主角崩溃之后'",
    )
