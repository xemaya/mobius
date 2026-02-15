"""失控型叙事引擎实现。

Mobius v2.1 的核心：让角色带着偏见做错事。
"""

from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional

from mobius.models.chaos_engine import (
    BeliefMutation,
    BeliefMutationType,
    ChaosEngineConfig,
    ChaosPropagator,
    CognitiveBiasEngine,
    CognitiveBiasType,
    FlawedAction,
    HumanNoiseLayer,
    Impulse,
    ImpulseType,
    IrreversibleMark,
    LossOfControlEvent,
    NarrativeStressIndex,
    UnintendedConsequence,
)
from mobius.models.chaos_engine import ConsequenceType  # 重新导入避免循环导入
from mobius.models.character import CharacterDynamicState
from mobius.state.novel_state import NovelState


class HumanNoiseInjector:
    """去AI味注入器：让文字更像人。"""

    def __init__(self, config: HumanNoiseLayer):
        self.config = config

    def inject_hesitation(self, text: str, context: str = "") -> str:
        """注入犹豫噪声。"""
        if not self.config.hesitation_injection or random.random() > 0.3:
            return text

        hesitation_patterns = [
            "...她顿了顿，盯着地板看了一会儿。",
            "...他突然想起昨天的天气，摇了摇头。",
            "...她转移了话题，假装对墙上的画感兴趣。",
            "...他摸了摸口袋里的钥匙，犹豫着。",
            "...她咬了咬嘴唇，没有立刻回答。",
            "...他清了清嗓子，试图整理思绪。",
            "...她看了看窗外，雨好像小了些。",
            "...他揉了揉太阳穴，感觉头有点晕。",
        ]

        # 在合适的位置插入犹豫
        sentences = text.split("。")
        if len(sentences) > 2:
            insert_pos = random.randint(1, len(sentences) - 1)
            hesitation = random.choice(hesitation_patterns)
            sentences.insert(insert_pos, hesitation)
            return "。".join(sentences)

        return text

    def degrade_poetic_density(self, text: str) -> str:
        """降低诗意密度，防止过度文艺。"""
        if random.random() > self.config.poetic_density_limit:
            return text

        # 找到过于诗意的句子并降级
        poetic_indicators = ["如梦", "似水", "宛如", "仿佛", "像极了", "如诗", "似歌"]
        sentences = text.split("。")

        for i, sentence in enumerate(sentences):
            if any(indicator in sentence for indicator in poetic_indicators):
                if random.random() < 0.5:  # 50%概率降级
                    # 用更平实的语言替换
                    sentences[i] = sentence.replace("宛如", "像").replace("仿佛", "好像")

        return "。".join(sentences)

    def inject_incomplete_thoughts(self, text: str) -> str:
        """注入不完整思想。"""
        if random.random() > self.config.incomplete_thought_ratio:
            return text

        # 在句子中间插入打断
        sentences = text.split("。")
        if len(sentences) > 1:
            target_idx = random.randint(0, len(sentences) - 1)
            sentence = sentences[target_idx]

            # 找到合适的位置打断
            words = sentence.split("，")
            if len(words) > 2:
                break_pos = random.randint(1, len(words) - 1)
                interrupted = "，".join(words[:break_pos])
                sentences[target_idx] = f"{interrupted}——不，这不重要。"

        return "。".join(sentences)

    def prevent_emotion_summary(self, text: str) -> str:
        """防止情绪总结，强制用行为描写。"""
        # 检测并重写情绪总结句式
        emotion_patterns = [
            (r"她感到(.+?)。", "她攥紧了拳头，呼吸变得急促。"),
            (r"他觉得(.+?)。", "他移开了视线，喉结动了动。"),
            (r"(.+?)觉得(.+?)。", "\\1咬了咬牙，没有说话。"),
            (r"(.+?)感到(.+?)。", "\\1揉了揉眉心，深吸一口气。"),
        ]

        for pattern, replacement in emotion_patterns:
            if re.search(pattern, text):
                if random.random() < 0.7:  # 70%概率重写
                    text = re.sub(pattern, replacement, text)
                    break

        return text

    def process_text(self, text: str, context: str = "") -> str:
        """处理文本，应用所有去AI味规则。"""
        text = self.inject_hesitation(text, context)
        text = self.degrade_poetic_density(text)
        text = self.inject_incomplete_thoughts(text)
        text = self.prevent_emotion_summary(text)
        return text


class CognitiveBiasProcessor:
    """认知偏差处理器：强制让决策有缺陷。"""

    def __init__(self, config: CognitiveBiasEngine):
        self.config = config

    def apply_bias(self, action: str, character_state: CharacterDynamicState) -> FlawedAction:
        """应用认知偏差，让行动带有缺陷。"""
        if not self.config.enabled:
            return FlawedAction(
                original_intent=action,
                actual_action=action,
                bias_applied=CognitiveBiasType.CONFIRMATION_BIAS,
                flaw_description="无偏差（理性决策）",
                optimality_loss=0.0
            )

        # 根据角色当前状态选择合适的偏差
        bias_type = self._select_bias_type(character_state)

        # 计算偏差强度
        strength = self._calculate_bias_strength(character_state)

        # 生成有缺陷的行动
        flawed_action = self._generate_flawed_action(action, bias_type, strength)

        return FlawedAction(
            original_intent=action,
            actual_action=flawed_action,
            bias_applied=bias_type,
            flaw_description=self._describe_flaw(bias_type, strength),
            optimality_loss=strength
        )

    def _select_bias_type(self, character_state: CharacterDynamicState) -> CognitiveBiasType:
        """根据角色状态选择偏差类型。"""
        # 基于情感状态选择
        if character_state.emotional_state.get("恐惧", 0) > 0.7:
            return CognitiveBiasType.FEAR_AMPLIFICATION
        elif character_state.emotional_state.get("愤怒", 0) > 0.6:
            return CognitiveBiasType.EMOTION_DRIVEN
        elif any(d.priority > 0.8 for d in character_state.desires):
            return CognitiveBiasType.POSSESSIVE_RATIONALIZATION
        else:
            return self.config.primary_bias

    def _calculate_bias_strength(self, character_state: CharacterDynamicState) -> float:
        """计算偏差强度。"""
        base_strength = self.config.distortion_strength

        # 情绪压力增加偏差
        emotional_pressure = sum(abs(v) for v in character_state.emotional_state.values()) / len(character_state.emotional_state)
        base_strength += emotional_pressure * 0.2

        # 资源压力增加偏差
        resource_pressure = 1.0 - (character_state.resources.emotional_energy / 100.0)
        base_strength += resource_pressure * 0.3

        return min(base_strength, 0.9)  # 最高0.9，保留一点理性

    def _generate_flawed_action(self, original_action: str, bias_type: CognitiveBiasType, strength: float) -> str:
        """生成有缺陷的行动。"""
        if strength < 0.3:
            return original_action  # 偏差不明显

        flaw_modifiers = {
            CognitiveBiasType.CONFIRMATION_BIAS: [
                "只关注支持自己想法的部分",
                "忽略了相反的证据",
                "过度相信自己的判断",
            ],
            CognitiveBiasType.EMOTION_DRIVEN: [
                "冲动行事",
                "情绪压倒了理性",
                "没有考虑后果",
            ],
            CognitiveBiasType.POSSESSIVE_RATIONALIZATION: [
                "用占有欲合理化行为",
                "低估了别人的感受",
                "过于坚持自己的权利",
            ],
            CognitiveBiasType.FEAR_AMPLIFICATION: [
                "把小问题想得很大",
                "过度担心最坏情况",
                "行动变得过于谨慎",
            ],
        }

        modifier = random.choice(flaw_modifiers.get(bias_type, ["做出次优选择"]))

        # 根据强度调整行动
        if strength > 0.7:
            # 严重偏差：完全改变行动
            return f"{original_action}（但{modifier}，结果变得更糟）"
        elif strength > 0.5:
            # 中等偏差：部分改变行动
            return f"{original_action}（不过{modifier}）"
        else:
            # 轻微偏差：轻微调整
            return f"{original_action}（带着些许犹豫）"

    def _describe_flaw(self, bias_type: CognitiveBiasType, strength: float) -> str:
        """描述偏差缺陷。"""
        descriptions = {
            CognitiveBiasType.CONFIRMATION_BIAS: f"确认偏误导致的信息过滤，强度{strength:.1f}",
            CognitiveBiasType.EMOTION_DRIVEN: f"情绪驱动的冲动决策，强度{strength:.1f}",
            CognitiveBiasType.POSSESSIVE_RATIONALIZATION: f"占有欲的合理化辩护，强度{strength:.1f}",
            CognitiveBiasType.FEAR_AMPLIFICATION: f"恐惧放大导致的过度反应，强度{strength:.1f}",
        }
        return descriptions.get(bias_type, f"未知偏差，强度{strength:.1f}")


class ChaosPropagatorEngine:
    """混沌传播器：生成不可控后果。"""

    def __init__(self, config: ChaosPropagator):
        self.config = config

    def generate_consequences(self, action: str, character_name: str, chapter_index: int) -> List[UnintendedConsequence]:
        """为行动生成意外后果。"""
        consequences = []

        # 基础后果概率
        if random.random() < self.config.collateral_damage_rate:
            consequences.append(self._generate_collateral_damage(action, character_name))

        if random.random() < self.config.information_leak_probability:
            consequences.append(self._generate_information_leak(action, character_name))

        # 随机额外后果
        if random.random() < self.config.unpredictability_factor:
            consequences.append(self._generate_random_consequence(action, character_name))

        # 处理延迟后果
        for consequence in consequences:
            if random.random() < 0.4:  # 40%概率延迟
                delay = random.randint(*self.config.consequence_delay_range)
                consequence.delayed = delay > 0

        # 连锁反应
        if consequences and random.random() < self.config.cascade_probability:
            cascade = self._generate_cascade_effect(consequences[0], character_name)
            if cascade:
                consequences.append(cascade)

        return consequences

    def _generate_collateral_damage(self, action: str, character_name: str) -> UnintendedConsequence:
        """生成附带损害。"""
        damage_types = [
            "无意中伤害了无关的人",
            "破坏了重要的物品",
            "造成了经济损失",
            "损害了个人声誉",
        ]

        return UnintendedConsequence(
            consequence_type=ConsequenceType.COLLATERAL_DAMAGE,
            description=f"{character_name}的行动{action}导致：{random.choice(damage_types)}",
            affected_parties=[character_name],
            severity=random.uniform(0.2, 0.7),
            controllability=0.2,  # 很难控制
        )

    def _generate_information_leak(self, action: str, character_name: str) -> UnintendedConsequence:
        """生成信息泄露。"""
        leak_types = [
            "私密信息被意外听到",
            "重要文件被误放位置",
            "通话被意外录音",
            "秘密被无意中提及",
        ]

        return UnintendedConsequence(
            consequence_type=ConsequenceType.INFORMATION_LEAK,
            description=f"{character_name}在{action}时：{random.choice(leak_types)}",
            affected_parties=[character_name, "第三方"],
            severity=random.uniform(0.3, 0.8),
            controllability=0.1,  # 几乎无法控制
        )

    def _generate_random_consequence(self, action: str, character_name: str) -> UnintendedConsequence:
        """生成随机后果。"""
        consequence_types = [
            (ConsequenceType.RELATIONSHIP_RUPTURE, "关系破裂"),
            (ConsequenceType.RESOURCE_LOSS, "资源损失"),
            (ConsequenceType.THIRD_PARTY_AWAKENING, "第三方觉醒"),
            (ConsequenceType.SYSTEM_EXPOSURE, "系统暴露"),
            (ConsequenceType.POWER_BACKLASH, "权力反噬"),
        ]

        cons_type, desc = random.choice(consequence_types)

        return UnintendedConsequence(
            consequence_type=cons_type,
            description=f"{character_name}的行动意外引发：{desc}",
            affected_parties=[character_name],
            severity=random.uniform(0.1, 0.9),
            controllability=random.uniform(0.0, 0.5),
        )

    def _generate_cascade_effect(self, trigger: UnintendedConsequence, character_name: str) -> Optional[UnintendedConsequence]:
        """生成连锁反应。"""
        if trigger.consequence_type == ConsequenceType.INFORMATION_LEAK:
            return UnintendedConsequence(
                consequence_type=ConsequenceType.RELATIONSHIP_RUPTURE,
                description=f"信息泄露引发连锁反应：信任崩塌",
                affected_parties=[character_name, "相关各方"],
                severity=min(trigger.severity + 0.2, 1.0),
                controllability=0.0,
                cascade_triggered=True
            )
        return None


class BeliefMutator:
    """信念变异器：非线性信念变化。"""

    def mutate_belief(self, belief_statement: str, trauma_event: str) -> BeliefMutation:
        """让信念发生非线性变异。"""
        mutation_type = random.choice(list(BeliefMutationType))

        original = belief_statement
        mutated = self._apply_mutation(original, mutation_type, trauma_event)

        return BeliefMutation(
            mutation_type=mutation_type,
            original_belief=original,
            mutated_belief=mutated,
            trauma_trigger=trauma_event,
            instability_factor=random.uniform(0.3, 0.9)
        )

    def _apply_mutation(self, belief: str, mutation_type: BeliefMutationType, trauma: str) -> str:
        """应用具体的变异逻辑。"""
        if mutation_type == BeliefMutationType.EXTREME_REINFORCEMENT:
            # 极端强化：把正常信念推向极端
            return f"{belief}（现在变得无比坚定，甚至到了偏执的地步）"

        elif mutation_type == BeliefMutationType.DISTORTED_VERSION:
            # 歪曲版本：信念变得扭曲
            return f"{belief}（但现在变成了病态的版本）"

        elif mutation_type == BeliefMutationType.SELF_DENIAL:
            # 自我否认：开始怀疑自己的信念
            return f"我一直相信的'{belief}'可能是错的"

        elif mutation_type == BeliefMutationType.PARANOID_SHIFT:
            # 偏执转变：变得多疑
            return f"{belief}（但现在怀疑所有人都在反对这个信念）"

        elif mutation_type == BeliefMutationType.NIHILISTIC_COLLAPSE:
            # 虚无主义崩溃：完全放弃
            return f"什么'{belief}'，一切都没有意义"

        return belief  # 兜底


class IrreversibleMarkGenerator:
    """不可逆印记生成器。"""

    def generate_mark(self, chapter_events: List[Any], character_states: Dict[str, CharacterDynamicState]) -> IrreversibleMark:
        """为章节生成不可逆印记。"""
        # 分析章节中的重大事件
        damage_levels = self._analyze_damage(chapter_events, character_states)

        mark_type = self._select_mark_type(damage_levels)

        return IrreversibleMark(
            relationship_damage=damage_levels['relationship'],
            psychological_crack=damage_levels['psychological'],
            structural_shift=damage_levels['structural'],
            mark_description=self._generate_mark_description(mark_type, damage_levels),
            recovery_impossible=random.random() < 0.3  # 30%概率完全不可恢复
        )

    def _analyze_damage(self, events: List[Any], states: Dict[str, CharacterDynamicState]) -> Dict[str, float]:
        """分析各类损害程度。"""
        relationship_damage = 0.0
        psychological_damage = 0.0
        structural_damage = 0.0

        # 从事件中分析损害
        for event in events:
            if hasattr(event, 'consequence_type'):
                if event.consequence_type == ConsequenceType.RELATIONSHIP_RUPTURE:
                    relationship_damage += event.severity
                elif event.consequence_type in [ConsequenceType.THIRD_PARTY_AWAKENING, ConsequenceType.SYSTEM_EXPOSURE]:
                    structural_damage += event.severity

        # 从角色状态分析损害
        for state in states.values():
            # 信念变化表示心理创伤
            if len(state.memory) > 10:  # 记忆积累太多
                psychological_damage += 0.2

            # 资源严重不足
            if state.resources.emotional_energy < 30:
                psychological_damage += 0.3

        return {
            'relationship': min(relationship_damage, 1.0),
            'psychological': min(psychological_damage, 1.0),
            'structural': min(structural_damage, 1.0),
        }

    def _select_mark_type(self, damage_levels: Dict[str, float]) -> str:
        """选择印记类型。"""
        max_damage = max(damage_levels.values())
        if damage_levels['relationship'] == max_damage:
            return "relationship"
        elif damage_levels['psychological'] == max_damage:
            return "psychological"
        else:
            return "structural"

    def _generate_mark_description(self, mark_type: str, levels: Dict[str, float]) -> str:
        """生成印记描述。"""
        templates = {
            "relationship": [
                "一段重要的关系出现了不可修复的裂痕",
                "信任被永久破坏",
                "曾经亲密的人变得陌生",
            ],
            "psychological": [
                "心理上留下了永久的创伤",
                "一个重要的信念被永远改变",
                "角色的人格发生了不可逆转变",
            ],
            "structural": [
                "权力结构被永久改变",
                "社会地位不可挽回地下降",
                "重要的机会永远失去",
            ]
        }

        return random.choice(templates.get(mark_type, ["产生了不可逆的影响"]))


class LossOfControlEngine:
    """角色失控引擎。"""

    def __init__(self, base_probability: float = 0.1):
        self.base_probability = base_probability

    def check_loss_of_control(self, character_state: CharacterDynamicState, stress_index: NarrativeStressIndex) -> Optional[LossOfControlEvent]:
        """检查角色是否失控。"""
        # 计算失控概率
        probability = self.base_probability
        probability += stress_index.total_stress * 0.2  # 压力增加概率
        probability += (1.0 - character_state.resources.emotional_energy / 100.0) * 0.3  # 情绪耗尽增加概率

        if random.random() < probability:
            return self._generate_loss_of_control_event(character_state.name)

        return None

    def _generate_loss_of_control_event(self, character_name: str) -> LossOfControlEvent:
        """生成失控事件。"""
        actions = [
            "突然说出不该说的话",
            "做出冲动破坏性行为",
            "说出内心真实想法而不顾后果",
            "做出极端情绪化反应",
            "忽略所有警告继续行动",
        ]

        consequences = [
            "这让情况变得更糟",
            "引发了更多冲突",
            "暴露了重要秘密",
            "损害了重要关系",
            "造成了无法挽回的后果",
        ]

        return LossOfControlEvent(
            character_name=character_name,
            trigger_reason="压力过大导致失控",
            uncontrolled_action=random.choice(actions),
            consequence_description=random.choice(consequences),
            recovery_chapters=random.randint(1, 3)
        )


# ────────────────────────────────────────────
# 主引擎整合
# ────────────────────────────────────────────

class ChaosEngine:
    """失控型叙事引擎总控制器。"""

    def __init__(self, config: ChaosEngineConfig):
        self.config = config

        # 初始化各个子引擎
        self.human_noise = HumanNoiseInjector(config.human_noise_layer)
        self.bias_processor = CognitiveBiasProcessor(config.cognitive_bias_engine)
        self.chaos_propagator = ChaosPropagatorEngine(config.chaos_propagator)
        self.belief_mutator = BeliefMutator()
        self.mark_generator = IrreversibleMarkGenerator()
        self.loss_of_control = LossOfControlEngine(config.loss_of_control_base_probability)

    def process_character_action(self, action: str, character_state: CharacterDynamicState, chapter_index: int) -> Dict[str, Any]:
        """处理角色行动，应用所有失控机制。"""
        result = {
            'original_action': action,
            'flawed_action': None,
            'consequences': [],
            'belief_mutations': [],
            'loss_of_control_event': None,
        }

        # 1. 应用认知偏差
        flawed_action = self.bias_processor.apply_bias(action, character_state)
        result['flawed_action'] = flawed_action

        # 2. 生成意外后果
        consequences = self.chaos_propagator.generate_consequences(
            flawed_action.actual_action, character_state.name, chapter_index
        )
        result['consequences'] = consequences

        # 3. 检查信念变异（如果有重大创伤）
        if any(c.severity > 0.7 for c in consequences):
            for belief in character_state.beliefs:
                if random.random() < 0.3:  # 30%概率变异
                    mutation = self.belief_mutator.mutate_belief(
                        belief.statement,
                        f"{flawed_action.actual_action}的后果"
                    )
                    result['belief_mutations'].append(mutation)

        # 4. 检查失控事件
        stress_index = self._calculate_stress_index(character_state, consequences)
        loss_of_control_event = self.loss_of_control.check_loss_of_control(character_state, stress_index)
        result['loss_of_control_event'] = loss_of_control_event

        return result

    def generate_chapter_mark(self, chapter_events: List[Any], character_states: Dict[str, CharacterDynamicState]) -> IrreversibleMark:
        """为章节生成不可逆印记。"""
        return self.mark_generator.generate_mark(chapter_events, character_states)

    def process_text_humanization(self, text: str, context: str = "") -> str:
        """应用去AI味处理。"""
        return self.human_noise.process_text(text, context)

    def check_author_override(self, character_drive_strength: float, author_intent_strength: float) -> bool:
        """检查是否应该覆盖作者意图。"""
        config = self.config.author_intent_override

        # 计算角色权重
        character_weight = character_drive_strength * config.character_weight
        author_weight = author_intent_strength * config.author_weight

        # 检查阈值
        if character_weight / (character_weight + author_weight) > config.override_threshold:
            return random.random() < config.override_probability

        return False

    def _calculate_stress_index(self, character_state: CharacterDynamicState, consequences: List[UnintendedConsequence]) -> NarrativeStressIndex:
        """计算叙事压力指数。"""
        stress = NarrativeStressIndex()

        # 未解决冲突：欲望未满足
        stress.unresolved_conflicts = sum(
            d.priority for d in character_state.desires if not d.satisfied
        ) / max(len(character_state.desires), 1)

        # 不可逆损害：严重后果
        stress.irreversible_damage = sum(c.severity for c in consequences if c.severity > 0.6)

        # 资源损失：资源不足
        stress.resource_loss = 1.0 - (
            character_state.resources.emotional_energy / 100.0 +
            character_state.resources.reputation / 100.0
        ) / 2.0

        # 信念不稳定：信念数量太多
        stress.belief_instability = min(len(character_state.beliefs) / 10.0, 1.0)

        # 外部冲击概率
        stress.external_shock_probability = 0.0
        if stress.total_stress < 0.7:
            stress.external_shock_probability = 0.8 - stress.total_stress

        return stress