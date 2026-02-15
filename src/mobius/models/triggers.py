"""触发规则相关数据模型。"""

from pydantic import BaseModel, Field


class TriggerRule(BaseModel):
    """角色的触发规则。当某个数值属性达到阈值时触发特定剧情事件。"""

    id: str = Field(description="触发规则唯一标识")
    attribute: str = Field(description="监控的属性名，如 '黑化值'、'好感度'")
    threshold: float = Field(description="触发阈值")
    operator: str = Field(
        default="gte",
        description="比较操作符: 'gte'(>=), 'lte'(<=), 'eq'(==), 'gt'(>), 'lt'(<)",
    )
    event_description: str = Field(description="触发后的剧情事件描述")
    one_shot: bool = Field(default=True, description="是否只触发一次")
    triggered: bool = Field(default=False, description="是否已被触发")


class TriggeredEvent(BaseModel):
    """已触发的事件。"""

    character_name: str = Field(description="触发事件的角色名")
    rule: TriggerRule = Field(description="触发的规则")
    current_value: float = Field(description="触发时的实际数值")
    description: str = Field(default="", description="由导演生成的具体事件描述")
