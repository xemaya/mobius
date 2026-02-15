"""章节评审与世界事件数据模型。

世界模拟层产生事件，观察者标记叙事价值，
评审 Agent 评估章节质量。
"""

from pydantic import BaseModel, Field


class WorldEvent(BaseModel):
    """世界模拟层产生的事件（不一定全部被写入小说）。

    世界继续跑，观察者只选择哪一段被写进"小说"。
    裁剪 ≠ 干预。
    """

    event_id: str = Field(description="事件唯一标识")
    description: str = Field(description="事件描述")
    participants: list[str] = Field(default_factory=list, description="参与角色")
    conflict_type: str = Field(
        default="",
        description="冲突类型: 'desire_clash' | 'resource_scarcity' | 'belief_crisis' | 'fear_trigger'",
    )
    narrative_value: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="叙事价值 0-1，由观察者标记",
    )
    tension_contribution: float = Field(
        default=0.0, description="对张力的贡献"
    )
    selected_for_narrative: bool = Field(
        default=False, description="是否被选入小说"
    )


class ChapterReview(BaseModel):
    """章节评审结果，融合主题守护、张力控制、自我批评。"""

    theme_alignment: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="主题契合度 0-1",
    )
    theme_drift_notes: str = Field(default="", description="主题偏移备注")
    tension_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="张力评分 0-1",
    )
    pacing_notes: str = Field(default="", description="节奏备注")
    logic_issues: list[str] = Field(
        default_factory=list, description="逻辑漏洞列表"
    )
    character_voice_issues: list[str] = Field(
        default_factory=list, description="角色声音一致性问题"
    )
    unresolved_threads: list[str] = Field(
        default_factory=list, description="未解决的叙事线索"
    )
    suggestions_for_next: str = Field(
        default="", description="对下一章的建议"
    )


class StructuredMemorySummary(BaseModel):
    """记忆蒸馏的结构化输出。"""

    key_conflicts: list[str] = Field(
        default_factory=list, description="关键冲突"
    )
    relationship_changes: list[str] = Field(
        default_factory=list, description="关系变化"
    )
    unresolved_tensions: list[str] = Field(
        default_factory=list, description="未解决悬念"
    )
    belief_shifts: list[str] = Field(
        default_factory=list, description="信念转变"
    )
    resource_shifts: list[str] = Field(
        default_factory=list, description="资源变动，如'角色A声誉崩盘，从80降至12'"
    )
    emotional_arc: str = Field(default="", description="情绪弧线概括")
