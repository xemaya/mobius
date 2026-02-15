"""失控型叙事引擎数据模型。

Mobius v2.1 的核心：让角色带着偏见做错事。
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class CognitiveBiasType(str, Enum):
    """认知偏差类型。"""
    CONFIRMATION_BIAS = "confirmation_bias"  # 确认偏误：只看到支持自己信念的信息
    EMOTION_DRIVEN = "emotion_driven"  # 情绪驱动：情感压倒理性
    POSSESSIVE_RATIONALIZATION = "possessive_rationalization"  # 占有欲合理化
    FEAR_AMPLIFICATION = "fear_amplification"  # 恐惧放大
    OPTIMISM_BIAS = "optimism_bias"  # 乐观偏误
    RECENCY_BIAS = "recency_bias"  # 近期偏误：过分重视最近事件


class ImpulseType(str, Enum):
    """冲动类型。"""
    SUDDEN_DESIRE = "sudden_desire"  # 突然欲望
    IRRATIONAL_FEAR = "irrational_fear"  # 非理性恐惧
    POSSESSIVE_URGE = "possessive_urge"  # 占有冲动
    DESTRUCTIVE_IMPULSE = "destructive_impulse"  # 破坏冲动
    ESCAPE_DESIRE = "escape_desire"  # 逃避欲望


class ConsequenceType(str, Enum):
    """后果类型。"""
    RELATIONSHIP_RUPTURE = "relationship_rupture"  # 关系破裂
    RESOURCE_LOSS = "resource_loss"  # 资源损失
    COLLATERAL_DAMAGE = "collateral_damage"  # 附带损害
    THIRD_PARTY_AWAKENING = "third_party_awakening"  # 第三方觉醒
    SYSTEM_EXPOSURE = "system_exposure"  # 系统暴露
    POWER_BACKLASH = "power_backlash"  # 权力反噬
    INFORMATION_LEAK = "information_leak"  # 信息泄露


class BeliefMutationType(str, Enum):
    """信念变异类型。"""
    EXTREME_REINFORCEMENT = "extreme_reinforcement"  # 极端信念强化
    DISTORTED_VERSION = "distorted_version"  # 歪曲版本
    SELF_DENIAL = "self_denial"  # 自我否认
    PARANOID_SHIFT = "paranoid_shift"  # 偏执转变
    NIHILISTIC_COLLAPSE = "nihilistic_collapse"  # 虚无主义崩溃


# ────────────────────────────────────────────
# 核心引擎数据模型
# ────────────────────────────────────────────

class HumanNoiseLayer(BaseModel):
    """去AI味模块：让文字更像人。

    通过注入犹豫、限制抽象语言、制造节奏不均等方式，
    让AI生成的文字失去机械感。
    """
    hesitation_injection: bool = Field(
        default=True,
        description="是否注入犹豫噪声（无关小动作、转移话题等）"
    )
    abstract_language_limit: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="抽象语言密度上限（哲学化表达限制）"
    )
    rhythm_variance: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="节奏变异度（句子长度随机性）"
    )
    incomplete_thought_ratio: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="不完整思想比例（句子被打断）"
    )
    self_deception_probability: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="自我欺骗概率（角色说服自己相信谎言）"
    )
    poetic_density_limit: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="诗意密度上限（防止过度文艺）"
    )


class CognitiveBiasEngine(BaseModel):
    """错误决策强制机制：角色永远不会做最优选择。

    通过各种认知偏差，让角色的决策带有系统性缺陷。
    """
    enabled: bool = Field(default=True, description="是否启用认知偏差")
    primary_bias: CognitiveBiasType = Field(
        default=CognitiveBiasType.CONFIRMATION_BIAS,
        description="主要认知偏差类型"
    )
    distortion_strength: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="偏差扭曲强度（0.0=理性，1.0=完全失控）"
    )
    blind_spots: List[str] = Field(
        default_factory=list,
        description="认知盲点（角色无法正确评估的领域）"
    )
    bias_triggers: Dict[str, float] = Field(
        default_factory=dict,
        description="偏差触发条件（情绪状态→偏差强度）"
    )


class ChaosPropagator(BaseModel):
    """不可控后果生成器：每次行动必须产生无法掌控的副作用。

    确保没有行动是"干净"的，每件事都会引发连锁反应。
    """
    unpredictability_factor: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="不可预测性因子（后果的随机程度）"
    )
    collateral_damage_rate: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="附带损害率（行动影响无关方的概率）"
    )
    information_leak_probability: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="信息泄露概率（机密被意外暴露）"
    )
    consequence_delay_range: tuple[int, int] = Field(
        default=(0, 3),
        description="后果延迟范围（立即到N章后生效）"
    )
    cascade_probability: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="连锁反应概率（后果引发更多后果）"
    )


class IrreversibleMark(BaseModel):
    """不可逆印记系统：每章至少产生一个无法修复的损伤。

    确保故事的不可逆转性，角色无法简单"重置"。
    """
    relationship_damage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="关系损害程度（信任无法完全修复）"
    )
    psychological_crack: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="心理创伤深度（角色永久改变）"
    )
    structural_shift: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="结构性转变（权力/社会地位永久变化）"
    )
    mark_description: str = Field(
        default="",
        description="印记的具体描述"
    )
    recovery_impossible: bool = Field(
        default=False,
        description="是否完全不可恢复"
    )


class Impulse(BaseModel):
    """冲动：行动前的原始驱动力。

    v2.1的核心：行动前增加冲动阶段，让决策不再理性。
    """
    impulse_type: ImpulseType = Field(description="冲动类型")
    intensity: float = Field(
        ge=0.0,
        le=1.0,
        description="冲动强度"
    )
    trigger_condition: str = Field(
        default="",
        description="触发条件描述"
    )
    rationalization_needed: bool = Field(
        default=True,
        description="是否需要合理化（将冲动包装成理性决定）"
    )


class FlawedAction(BaseModel):
    """有缺陷的行动：永远不是最优选择。

    强制让角色的行动带有系统性缺陷。
    """
    original_intent: str = Field(description="原始意图")
    actual_action: str = Field(description="实际执行的有缺陷行动")
    bias_applied: CognitiveBiasType = Field(description="应用的认知偏差")
    flaw_description: str = Field(description="缺陷的具体描述")
    optimality_loss: float = Field(
        ge=0.0,
        le=1.0,
        description="最优性损失（1.0=完全错误的选择）"
    )


class UnintendedConsequence(BaseModel):
    """意外后果：无法掌控的连锁反应。

    每次重大行动的必然产物。
    """
    consequence_type: ConsequenceType = Field(description="后果类型")
    description: str = Field(description="后果的具体描述")
    affected_parties: List[str] = Field(
        default_factory=list,
        description="受影响的各方"
    )
    severity: float = Field(
        ge=0.0,
        le=1.0,
        description="严重程度"
    )
    controllability: float = Field(
        ge=0.0,
        le=1.0,
        description="可控程度（0.0=完全不可控）"
    )
    delayed: bool = Field(
        default=False,
        description="是否延迟生效"
    )
    cascade_triggered: bool = Field(
        default=False,
        description="是否触发了连锁反应"
    )


class BeliefMutation(BaseModel):
    """信念变异：非线性的信念变化。

    信念不再平滑衰减，而是可能畸形生长。
    """
    mutation_type: BeliefMutationType = Field(description="变异类型")
    original_belief: str = Field(description="原始信念")
    mutated_belief: str = Field(description="变异后的信念")
    trauma_trigger: str = Field(description="触发变异的创伤事件")
    instability_factor: float = Field(
        ge=0.0,
        le=1.0,
        description="不稳定因子（变异剧烈程度）"
    )


class NarrativeStressIndex(BaseModel):
    """叙事压力指数 v2：主动制造不稳定。

    计算方式：
    stress = unresolved_conflict + irreversible_damage +
             resource_loss + belief_instability
    """
    unresolved_conflicts: float = Field(
        default=0.0,
        ge=0.0,
        description="未解决冲突权重"
    )
    irreversible_damage: float = Field(
        default=0.0,
        ge=0.0,
        description="不可逆损害权重"
    )
    resource_loss: float = Field(
        default=0.0,
        ge=0.0,
        description="资源损失权重"
    )
    belief_instability: float = Field(
        default=0.0,
        ge=0.0,
        description="信念不稳定权重"
    )
    external_shock_probability: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="需要外部冲击的概率"
    )

    @property
    def total_stress(self) -> float:
        """计算总压力指数。"""
        return (
            self.unresolved_conflicts +
            self.irreversible_damage +
            self.resource_loss +
            self.belief_instability
        )


class LossOfControlEvent(BaseModel):
    """角色失控事件：随机触发失控行为。

    让角色偶尔做出无法预测的举动。
    """
    character_name: str = Field(description="失控的角色")
    trigger_reason: str = Field(description="触发原因")
    uncontrolled_action: str = Field(description="失控行为描述")
    consequence_description: str = Field(description="后果描述")
    recovery_chapters: int = Field(
        default=1,
        ge=1,
        description="恢复所需章节数"
    )


class AuthorIntentOverride(BaseModel):
    """作者意图覆盖：让角色拒绝剧情安排。

    当角色欲望强度 + 情绪压力超过阈值时，覆盖剧情方向。
    """
    author_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="作者意图权重"
    )
    character_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="角色自主权重"
    )
    override_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="覆盖阈值（角色权重超过此值时可覆盖剧情）"
    )
    override_probability: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="实际覆盖概率（即使超过阈值也不总是覆盖）"
    )


class ChapterCompletionCriteria(BaseModel):
    """章节完成条件升级：必须包含错误决策+不可逆后果+不总结情绪。

    确保每章都符合失控型叙事的要求。
    """
    requires_flawed_decision: bool = Field(
        default=True,
        description="是否必须包含错误决策"
    )
    requires_irreversible_consequence: bool = Field(
        default=True,
        description="是否必须包含不可逆后果"
    )
    forbids_emotion_summary: bool = Field(
        default=True,
        description="是否禁止情绪总结（必须用行为描写）"
    )
    minimum_hesitation_events: int = Field(
        default=1,
        ge=0,
        description="最少犹豫事件数量"
    )
    maximum_poetic_density: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="最大诗意密度"
    )


class StyleGovernorConfig(BaseModel):
    """文风控制器配置：进一步去AI味，实现张弛结构。

    所有替换规则都通过配置字段定义，不在代码中硬编码，可通过 YAML 覆盖。
    """
    enabled: bool = Field(default=True, description="是否启用文风控制器")
    use_llm_post_process: bool = Field(
        default=True,
        description="是否使用 LLM 做整章改写（关闭则仅做规则清理，显著提速）",
    )

    # 1. 比喻密度限制
    max_metaphors_per_1000_words: int = Field(default=1, description="每千字最大比喻数")
    forbid_consecutive_imagery: bool = Field(default=True, description="禁止连续段落使用意象句")
    
    # 2. 抽象词压缩
    abstract_nouns_to_kill: List[str] = Field(
        default_factory=lambda: ["神性", "逻辑", "变量", "深渊", "命运", "毁灭", "救赎", "存在", "本质", "意识"],
        description="需要杀死的抽象词列表"
    )
    max_abstract_nouns_per_chapter: int = Field(default=5, description="每章最大抽象词出现次数")
    
    # 3. 哲学台词降频
    max_philosophical_golden_sentences: int = Field(default=3, description="每章最大总结性金句数")
    forbid_consecutive_world_explaining: bool = Field(default=True, description="禁止连续页面出现解释世界的台词")
    
    # 4. 普通句比例强制
    min_plain_sentence_ratio: float = Field(default=0.4, ge=0.0, le=1.0, description="最小普通句比例")
    
    # 5. 删减策略
    beautiful_sentence_pruning_ratio: float = Field(default=0.2, description="删掉读着最‘爽’的句子的比例")


    # 6. 成语/文艺腔替换规则
    cliche_replacements: Dict[str, str] = Field(
        default_factory=lambda: {
            "\u5982\u9ca0\u5728\u5589": "\u8bf4\u4e0d\u51fa\u8bdd",
            "\u6cea\u5982\u96e8\u4e0b": "\u773c\u6cea\u6389\u4e0b\u6765",
            "\u5fc3\u5982\u6b7b\u7070": "\u4ec0\u4e48\u4e5f\u4e0d\u60f3\u505a\u4e86",
            "\u4e07\u7c41\u4ff1\u5bc2": "\u5f88\u5b89\u9759",
            "\u4e0d\u5bd2\u800c\u6817": "\u6253\u4e86\u4e2a\u51b7\u6218",
            "\u604d\u5982\u9694\u4e16": "\u597d\u50cf\u8fc7\u4e86\u5f88\u4e45",
            "\u523b\u9aa8\u94ed\u5fc3": "\u5fd8\u4e0d\u6389",
            "\u6495\u5fc3\u88c2\u80ba": "\u5f88\u75db",
            "\u809d\u80a0\u5bf8\u65ad": "\u96be\u53d7\u5f97\u8981\u547d",
            "\u9b42\u98de\u9b44\u6563": "\u5413\u4e86\u4e00\u8df3",
        },
        description="\u6210\u8bed/\u6587\u827a\u8154\u2192\u53e3\u8bed\u5316\u66ff\u6362\u8868\uff08\u53ef\u5728 YAML \u4e2d\u8986\u76d6\uff09",
    )

    # 7. \u8fc7\u5ea6\u8bd7\u610f\u6bd4\u55bb\u8bcd\u964d\u7ea7\u89c4\u5219
    poetic_downgrades: Dict[str, str] = Field(
        default_factory=lambda: {
            "\u5b9b\u5982": "\u50cf",
            "\u4eff\u4f5b": "\u597d\u50cf",
            "\u72b9\u5982": "\u50cf",
            "\u604d\u82e5": "\u597d\u50cf",
            "\u597d\u4f3c": "\u50cf",
            "\u4e00\u5982": "\u50cf",
        },
        description="\u8bd7\u610f\u6bd4\u55bb\u8bcd\u2192\u6734\u7d20\u8bcd\u66ff\u6362\u8868\uff08\u53ef\u5728 YAML \u4e2d\u8986\u76d6\uff09",
    )


class RhythmConfig(BaseModel):
    """节奏模块配置：张弛算法。"""
    enabled: bool = Field(default=True, description="是否启用张弛节奏控制")
    # 节奏序列：高张力 -> 冷却 -> 日常 -> 意外 -> 沉默 -> 再爆发
    cycle: List[str] = Field(
        default_factory=lambda: ["high_tension", "cooling", "daily", "unexpected", "silence", "outbreak"]
    )
    low_energy_segment_required_after_high_density: bool = Field(
        default=True, description="高密度情绪段后必须插入低能量段"
    )


class HumanErrorEngineConfig(BaseModel):
    """角色失误模块配置。"""
    enabled: bool = Field(default=True, description="是否启用角色失误模块")
    error_frequency_chapters: int = Field(default=2, description="每隔几章必须犯一次非战略性错误")
    error_types: List[str] = Field(
        default_factory=lambda: ["误判他人", "情绪失控", "推理错误", "选择逃避", "不必要的撒谎"]
    )
    require_real_consequences: bool = Field(default=True, description="错误必须产生真实后果")


# ────────────────────────────────────────────
# 失控引擎配置
# ────────────────────────────────────────────

class ChaosEngineConfig(BaseModel):
    """失控型叙事引擎总配置。"""
    human_noise_layer: HumanNoiseLayer = Field(
        default_factory=HumanNoiseLayer,
        description="去AI味模块配置"
    )
    cognitive_bias_engine: CognitiveBiasEngine = Field(
        default_factory=CognitiveBiasEngine,
        description="认知偏差引擎配置"
    )
    chaos_propagator: ChaosPropagator = Field(
        default_factory=ChaosPropagator,
        description="混沌传播器配置"
    )
    author_intent_override: AuthorIntentOverride = Field(
        default_factory=AuthorIntentOverride,
        description="作者意图覆盖配置"
    )
    chapter_criteria: ChapterCompletionCriteria = Field(
        default_factory=ChapterCompletionCriteria,
        description="章节完成条件"
    )
    
    # v2.2 新增
    style_governor: StyleGovernorConfig = Field(
        default_factory=StyleGovernorConfig,
        description="文风控制器配置"
    )
    rhythm_config: RhythmConfig = Field(
        default_factory=RhythmConfig,
        description="节奏控制配置"
    )
    human_error_engine: HumanErrorEngineConfig = Field(
        default_factory=HumanErrorEngineConfig,
        description="角色失误模块配置"
    )

    # 全局参数
    loss_of_control_base_probability: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="角色失控基础概率"
    )
    no_immediate_redemption: bool = Field(
        default=True,
        description="禁止立即救赎规则"
    )
    consequence_visibility: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="后果可见度（1.0=完全可见，无抽象层）"
    )