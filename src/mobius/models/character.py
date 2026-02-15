"""角色相关数据模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field

from mobius.models.belief import Belief
from mobius.models.desire import Desire, Fear
from mobius.models.resource import ResourcePool
from mobius.models.triggers import TriggerRule


class CharacterProfile(BaseModel):
    """角色的静态档案（不随剧情变化）。"""

    name: str = Field(description="角色名称")
    role: str = Field(
        default="supporting",
        description="角色定位: 'protagonist'(主角), 'antagonist'(反派), 'supporting'(配角)",
    )
    age: str = Field(default="", description="年龄")
    gender: str = Field(default="", description="性别")
    appearance: str = Field(default="", description="外貌描述")
    personality: str = Field(description="性格特点")
    background: str = Field(default="", description="背景故事")
    abilities: list[str] = Field(default_factory=list, description="能力/技能")
    speech_style: str = Field(default="", description="说话风格/口头禅")
    goals: list[str] = Field(default_factory=list, description="角色目标/动机")
    weaknesses: list[str] = Field(default_factory=list, description="弱点/缺陷")
    extra: dict[str, str] = Field(default_factory=dict, description="其他自定义属性")


class CharacterDynamicState(BaseModel):
    """角色的动态状态（随剧情变化）。"""

    name: str = Field(description="角色名称")
    emotional_state: dict[str, float] = Field(
        default_factory=dict,
        description="情感状态数值，如 {'愤怒': 0.3, '悲伤': 0.1, '快乐': 0.6}",
    )
    relationship_scores: dict[str, float] = Field(
        default_factory=dict,
        description="与其他角色的关系分数，如 {'角色B': 0.7, '角色C': -0.3}",
    )
    custom_attributes: dict[str, float] = Field(
        default_factory=dict,
        description="自定义数值属性，如 {'黑化值': 45, '修为': 100}",
    )
    trigger_rules: list[TriggerRule] = Field(
        default_factory=list, description="触发规则列表"
    )
    memory: list[str] = Field(
        default_factory=list, description="近期记忆片段（短期记忆，保留最近若干条原始记录）"
    )
    compressed_memory: str = Field(
        default="",
        description="压缩后的长期记忆摘要（由 LLM 在章节边界处将旧记忆压缩而成）",
    )
    current_location: str = Field(default="", description="当前所在位置")
    is_alive: bool = Field(default=True, description="是否存活")

    # ── 信念系统 ──
    beliefs: list[Belief] = Field(
        default_factory=list, description="角色信念列表（core/middle/surface 三层）"
    )

    # ── 欲望驱动系统 ──
    desires: list[Desire] = Field(
        default_factory=list, description="角色欲望列表（按优先级排序）"
    )
    fears: list[Fear] = Field(
        default_factory=list, description="角色恐惧列表"
    )

    # ── 资源系统 ──
    resources: ResourcePool = Field(
        default_factory=ResourcePool, description="角色持有的资源池"
    )


class CharacterAction(BaseModel):
    """角色在场景中的一次行动输出。"""

    character_name: str = Field(description="角色名称")
    action_type: str = Field(
        description="行动类型: 'dialogue'(对话), 'action'(动作), 'thought'(内心独白), 'mixed'(混合)"
    )
    content: str = Field(description="行动内容（包含对话、动作描写等）——外显行为")
    emotional_change: dict[str, float] = Field(
        default_factory=dict,
        description="情感状态变化量，如 {'愤怒': +0.2}",
    )
    relationship_change: dict[str, float] = Field(
        default_factory=dict,
        description="关系分数变化量，如 {'角色B': -0.1}",
    )
    attribute_change: dict[str, float] = Field(
        default_factory=dict,
        description="自定义属性变化量，如 {'黑化值': +5}",
    )
    new_memory: str = Field(default="", description="本次行动产生的新记忆摘要")

    # ── 潜台词与认知黑箱 ──
    internal_monologue: str = Field(
        default="",
        description="私有心理活动——角色真实的内心想法，不向其他角色开放",
    )
    cognitive_dissonance: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="言行不一致度 0-1（潜台词强度指标）",
    )

    # ── 信念/欲望/资源变化 ──
    belief_change: dict[str, float] = Field(
        default_factory=dict,
        description="信念强度变化量，如 {'belief_id': +0.1}",
    )
    desire_change: dict[str, float] = Field(
        default_factory=dict,
        description="欲望优先级变化量，如 {'desire_id': +0.2}",
    )
    resource_cost: dict[str, float] = Field(
        default_factory=dict,
        description="资源消耗，如 {'time': -10, 'emotional_energy': -5}",
    )

    # ── 环境影响 ──
    environment_change: dict[str, float] = Field(
        default_factory=dict,
        description="对环境变量的影响，如 {'alert_level': +30}",
    )
