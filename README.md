# 🔥 Mobius v2.1 - 失控型叙事引擎

> **核心理念**: 角色不是执行主题。角色是带着偏见在做错事。

基于 LangGraph 的 AI 小说创作引擎，通过导演 Agent 与多个角色 Sub-Agent 的协作，动态生成小说剧情。

## 🚀 v2.1 重大升级

从"可控叙事"转向"失控演化"：
- **去AI味模块**: 犹豫注入、情绪重写，让文字更像真实人性
- **认知偏差强制**: 角色永远不会做最优选择
- **不可控后果生成**: 每个行动都产生无法掌控的连锁反应
- **不可逆印记系统**: 每章至少产生一个无法修复的损伤
- **信念畸形变异**: 非线性信念变化，可能极端强化或自我否认
- **作者意图覆盖**: 角色可以拒绝剧情安排

## 架构

- **导演 Agent**：管理世界观、剧情梗概，推动每一章节发展
- **角色 Agent**：每个角色拥有独立信念系统、欲望驱动、认知偏差
- **失控引擎**：自动应用认知偏差、生成意外后果、畸变信念
- **评审系统**：主题守护 + 张力控制 + 自我批评

## 安装

```bash
pip install -e .
```

## 使用

```bash
# 使用预设设定集生成小说（自动应用v2.1失控引擎）
mobius generate presets/ai_love_story.yaml -o output/my_novel

# 交互式模式（逐章生成，可在章节间调整）
mobius generate presets/ai_love_story.yaml -o output/my_novel --interactive

# 查看token消耗统计
cat output/my_novel/metadata.json | jq '.token_usage.summary'
```

## 设定集格式

参见 `presets/ai_love_story.yaml` 了解完整的设定集格式。

**v2.1特性**: 所有生成自动应用失控机制，无需额外配置。

## 环境变量

默认使用 Google Gemini 模型，设置对应的 API Key：

```bash
export GOOGLE_API_KEY=your_key_here
```

也支持通过环境变量切换模型：

```bash
# 使用 OpenAI
export OPENAI_API_KEY=your_key
MOBIUS_PROVIDER=openai MOBIUS_MODEL=gpt-4o mobius generate presets/ai_love_story.yaml
```
