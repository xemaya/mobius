# Mobius - 小说创作多 Agent 系统

基于 LangGraph 的 AI 小说创作引擎，通过导演 Agent 与多个角色 Sub-Agent 的协作，动态生成小说剧情。

## 架构

- **导演 Agent**：管理世界观、剧情梗概、角色关系，推动每一章节发展
- **角色 Agent**：每个重要角色拥有独立 Agent，记忆角色特点、属性设定、关键情节
- **触发系统**：角色数值累积到阈值时自动触发特定剧情

## 安装

```bash
pip install -e .
```

## 使用

```bash
# 使用示例设定集生成小说
mobius generate examples/wuxia_setting.yaml

# 交互式模式（逐章生成，可在章节间调整）
mobius generate examples/wuxia_setting.yaml --interactive
```

## 设定集格式

参见 `examples/wuxia_setting.yaml` 了解完整的设定集格式。

## 环境变量

默认使用 Google Gemini 模型，设置对应的 API Key：

```bash
export GOOGLE_API_KEY=your_key_here
```

也支持通过环境变量切换模型：

```bash
# 使用 OpenAI
export OPENAI_API_KEY=your_key
MOBIUS_PROVIDER=openai MOBIUS_MODEL=gpt-4o mobius generate examples/wuxia_setting.yaml
```
