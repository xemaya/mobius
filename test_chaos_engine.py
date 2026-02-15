#!/usr/bin/env python3
"""测试失控型叙事引擎 v2.1 的功能。"""

from mobius.engine.chaos_engine import ChaosEngine, HumanNoiseInjector, CognitiveBiasProcessor
from mobius.models.chaos_engine import (
    ChaosEngineConfig,
    CognitiveBiasType,
    CognitiveBiasEngine,
    HumanNoiseLayer,
)
from mobius.models.character import CharacterDynamicState
from mobius.models.character import Belief, Desire, Fear, ResourcePool


def test_human_noise_injector():
    """测试去AI味注入器。"""
    print("🧠 测试去AI味注入器")
    print("-" * 40)

    config = HumanNoiseLayer(
        hesitation_injection=True,
        abstract_language_limit=0.3,
        rhythm_variance=0.4,
        incomplete_thought_ratio=0.15,
        self_deception_probability=0.25
    )

    injector = HumanNoiseInjector(config)

    # 测试犹豫注入
    text1 = "她决定告诉他真相。"
    result1 = injector.inject_hesitation(text1)
    print(f"原文: {text1}")
    print(f"犹豫注入: {result1}")

    # 测试情绪总结重写
    text2 = "她感到很开心。"
    result2 = injector.prevent_emotion_summary(text2)
    print(f"\n原文: {text2}")
    print(f"情绪重写: {result2}")

    print("✅ 去AI味注入器测试完成\n")


def test_cognitive_bias_processor():
    """测试认知偏差处理器。"""
    print("🧠 测试认知偏差处理器")
    print("-" * 40)

    config = CognitiveBiasEngine(
        enabled=True,
        primary_bias=CognitiveBiasType.CONFIRMATION_BIAS,
        distortion_strength=0.6
    )

    processor = CognitiveBiasProcessor(config)

    # 创建测试角色状态
    char_state = CharacterDynamicState(
        name="测试角色",
        emotional_state={"恐惧": 0.8, "愤怒": 0.3},
        resources=ResourcePool(emotional_energy=30)  # 情绪能量低
    )

    # 测试偏差应用
    action = "我应该谨慎处理这个问题"
    flawed_action = processor.apply_bias(action, char_state)

    print(f"原始行动: {action}")
    print(f"偏差类型: {flawed_action.bias_applied}")
    print(f"有缺陷行动: {flawed_action.actual_action}")
    print(f"最优性损失: {flawed_action.optimality_loss:.2f}")
    print(f"缺陷描述: {flawed_action.flaw_description}")

    print("✅ 认知偏差处理器测试完成\n")


def test_chaos_propagator():
    """测试混沌传播器。"""
    print("🌀 测试混沌传播器")
    print("-" * 40)

    from mobius.engine.chaos_engine import ChaosPropagatorEngine
    from mobius.models.chaos_engine import ChaosPropagator

    config = ChaosPropagator(
        unpredictability_factor=0.6,
        collateral_damage_rate=0.4,
        information_leak_probability=0.3
    )

    propagator = ChaosPropagatorEngine(config)

    # 测试后果生成
    action = "秘密告诉最好的朋友"
    consequences = propagator.generate_consequences(action, "小明", 1)

    print(f"行动: {action}")
    print(f"生成后果数量: {len(consequences)}")

    for i, cons in enumerate(consequences, 1):
        print(f"  后果{i}: {cons.description}")
        print(f"    类型: {cons.consequence_type}")
        print(f"    严重程度: {cons.severity:.2f}")
        print(f"    可控程度: {cons.controllability:.2f}")

    print("✅ 混沌传播器测试完成\n")


def test_full_chaos_engine():
    """测试完整失控引擎。"""
    print("🔥 测试完整失控引擎")
    print("-" * 40)

    # 创建引擎配置
    config = ChaosEngineConfig()
    config.cognitive_bias_engine.distortion_strength = 0.5
    config.chaos_propagator.collateral_damage_rate = 0.3

    engine = ChaosEngine(config)

    # 创建测试角色状态
    char_state = CharacterDynamicState(
        name="测试受害者",
        emotional_state={"恐惧": 0.7, "孤独": 0.8},
        beliefs=[Belief(id="trust_others", statement="别人是值得信任的", layer="core", strength=0.8, category="价值观")],
        desires=[Desire(id="make_friends", description="交到真心朋友", priority=0.9, category="社交")],
        fears=[Fear(id="abandonment", description="被抛弃", intensity=0.8)],
        resources=ResourcePool(emotional_energy=40, reputation=60)
    )

    # 测试完整行动处理
    action = "向新认识的人敞开心扉"
    result = engine.process_character_action(action, char_state, 1)

    print(f"原始行动: {action}")
    print(f"有缺陷行动: {result['flawed_action'].actual_action if result.get('flawed_action') else '无'}")
    print(f"后果数量: {len(result.get('consequences', []))}")
    print(f"信念变异: {len(result.get('belief_mutations', []))}")
    print(f"失控事件: {'是' if result.get('loss_of_control_event') else '否'}")

    # 测试文本去AI味
    text = "她感到非常难过，内心充满了悲伤。"
    humanized = engine.process_text_humanization(text)
    print(f"\n原文: {text}")
    print(f"人性化: {humanized}")

    print("✅ 完整失控引擎测试完成\n")


def main():
    """主测试函数。"""
    print("🚀 Mobius v2.1 - 失控型叙事引擎功能测试")
    print("=" * 60)

    try:
        test_human_noise_injector()
        test_cognitive_bias_processor()
        test_chaos_propagator()
        test_full_chaos_engine()

        print("🎉 所有测试通过！失控型叙事引擎 v2.1 准备就绪")
        print("\n核心特性:")
        print("  • 角色不再做最优选择，而是带着偏见做错事")
        print("  • 每个行动都产生不可控的意外后果")
        print("  • 信念会发生非线性畸形变异")
        print("  • 文本经过去AI味处理，更像真实人性")
        print("  • 叙事主动制造不稳定和不可逆损伤")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()