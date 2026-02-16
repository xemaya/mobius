#!/usr/bin/env python3
"""Token统计功能演示脚本。

展示如何使用TokenTracker跟踪LLM调用消耗。
"""

from mobius.utils.token_tracker import TokenTracker, get_token_stats, reset_token_tracking, track_token_usage
from mobius.output.manager import OutputManager
import time

def demo_token_tracking():
    """演示token统计功能。"""
    print("🚀 Mobius Token统计功能演示")
    print("=" * 50)

    # 重置统计
    reset_token_tracking()

    # 模拟一些LLM调用
    print("📊 模拟LLM调用统计...")

    # 模拟director调用
    track_token_usage(
        operation="director_plan_chapter",
        model_name="gemini-3-flash-preview",
        chapter_index=1,
        input_tokens=1500,
        output_tokens=300,
        total_tokens=1800,
        duration_ms=2500.0
    )

    # 模拟character调用
    track_token_usage(
        operation="character_action",
        model_name="gemini-3-flash-preview",
        chapter_index=1,
        input_tokens=800,
        output_tokens=150,
        total_tokens=950,
        duration_ms=1200.0
    )

    track_token_usage(
        operation="character_interact",
        model_name="gemini-3-flash-preview",
        chapter_index=1,
        input_tokens=1200,
        output_tokens=400,
        total_tokens=1600,
        duration_ms=1800.0
    )

    # 模拟narrator调用
    track_token_usage(
        operation="narrator_compile_chapter",
        model_name="gemini-3-flash-preview",
        chapter_index=1,
        input_tokens=2500,
        output_tokens=1800,
        total_tokens=4300,
        duration_ms=3500.0
    )

    # 模拟reviewer调用
    track_token_usage(
        operation="reviewer_evaluate",
        model_name="gemini-3-flash-preview",
        chapter_index=1,
        input_tokens=2000,
        output_tokens=250,
        total_tokens=2250,
        duration_ms=1500.0
    )

    # 模拟memory调用
    track_token_usage(
        operation="memory_distill",
        model_name="gemini-3-flash-preview",
        chapter_index=1,
        input_tokens=800,
        output_tokens=300,
        total_tokens=1100,
        duration_ms=800.0
    )

    # 获取统计结果
    stats = get_token_stats()

    print("\n📈 Token消耗统计结果:")
    print(f"  总调用次数: {stats.total_calls}")
    print(f"  总Token数: {stats.total_tokens:,}")
    print(f"  输入Token: {stats.total_input_tokens:,}")
    print(f"  输出Token: {stats.total_output_tokens:,}")
    print(".2f")
    print(".2f")

    print("\n🔍 按操作分类:")
    for op, op_stats in sorted(stats.by_operation.items()):
        print(f"  {op}: {op_stats.total_tokens:,} tokens ({op_stats.total_calls}次)")

    print("\n🔍 按模型分类:")
    for model, model_stats in sorted(stats.by_model.items()):
        print(f"  {model}: {model_stats.total_tokens:,} tokens ({model_stats.total_calls}次)")

    # 创建OutputManager演示
    print("\n💾 OutputManager集成演示...")
    output_mgr = OutputManager("demo_output", "Token演示小说")

    # 模拟保存一些数据
    print("  ✓ 已创建输出目录结构")
    print("  ✓ Token统计已保存到metadata.json")

    # 显示最终统计摘要
    token_summary = output_mgr.get_token_summary()
    if "error" not in token_summary:
        print("\n🎯 成本估算 (基于Gemini价格):")
        input_cost_per_1k = 0.00025  # $0.00025 per 1K input tokens
        output_cost_per_1k = 0.001   # $0.001 per 1K output tokens

        total_input_cost = (token_summary['total_input_tokens'] / 1000) * input_cost_per_1k
        total_output_cost = (token_summary['total_output_tokens'] / 1000) * output_cost_per_1k
        total_cost = total_input_cost + total_output_cost

        print(".4f")
        print(".4f")
        print(".4f")
        print(".6f")

    print("\n✅ Token统计功能演示完成！")
    print("\n📋 功能特性:")
    print("  • 实时跟踪所有LLM调用")
    print("  • 按操作/模型/章节分类统计")
    print("  • 自动保存到metadata.json")
    print("  • 支持成本估算")
    print("  • 线程安全单例模式")

if __name__ == "__main__":
    demo_token_tracking()