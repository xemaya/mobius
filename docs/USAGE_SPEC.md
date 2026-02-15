# Mobius 使用规范 (Usage Specification)

## 1. 快速开始

```bash
# 安装
cd mobius
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 设置 API Key
export GOOGLE_API_KEY="your-key-here"

# 生成小说
mobius generate presets/ai_love_story.yaml -o output/my_novel -v
```

---

## 2. 命令行接口 (CLI)

### `mobius generate`

```bash
mobius generate <setting> [options]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `<setting>` | 设定集 YAML 文件路径 | (必填) |
| `-o, --output` | 输出目录 | `output` |
| `-i, --interactive` | 交互模式（逐章暂停） | 关闭 |
| `-v, --verbose` | 详细日志 | 关闭 |

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `GOOGLE_API_KEY` | Google Gemini API Key | — |
| `MINIMAX_API_KEY` | MiniMax API Key（角色扮演） | — |
| `MOBIUS_MODEL` | 覆盖模型名称 | `gemini-3-flash-preview` |
| `MOBIUS_PROVIDER` | 覆盖模型提供商 | `google` |
| `MOBIUS_TEMPERATURE` | 覆盖温度参数 | `0.8` |

---

## 3. 设定集 YAML 规范

设定集位于 `presets/` 目录下，YAML 格式。

### 结构概览

```yaml
worldview:           # 世界观
  name: string
  description: string
  era: string
  geography: string
  power_system: string
  social_structure: string
  key_factions: [string]

plot_outline:        # 剧情梗概
  title: string
  genre: string
  theme: string
  synopsis: string   # 支持【写作风格要求】段落
  total_chapters: int
  beginning: string
  development: string
  twist: string
  conclusion: string

characters:          # 角色列表
  - name: string
    role: string     # protagonist / antagonist / supporting
    personality: string
    background: string
    age: int
    gender: string
    appearance: string
    abilities: [string]
    speech_style: string
    goals: [string]
    weaknesses: [string]
    extra: {}
    initial_state:
      emotional_state: {快乐: 0.5, 悲伤: 0.0, ...}
      relationship_scores: {角色名: float}
      custom_attributes: {}
      beliefs: [...]         # 信念系统
      desires: [...]         # 欲望系统
      fears: [...]           # 恐惧系统
      resources: {...}       # 资源池
      trigger_rules: [...]   # 触发规则

environment:         # 环境状态总线
  variables:
    variable_name:
      value: float
      min_val: float
      max_val: float
      description: string
      decay_rate: float
  behavior_modifiers: [...]

secondary_viewpoints:  # 支线观察者视角
  - id: string
    name: string
    perspective_type: string
    voice_style: string
    can_observe: [string]
    trigger_condition: string
```

### 完整示例

参见 `presets/ai_love_story.yaml`。

---

## 4. 产出物目录结构

生成过程中，所有产出物按以下严格目录结构写入磁盘：

```
output/<novel_name>/
├── chapters/                      # 逐章 Markdown 文件
│   ├── chapter_001.md             # 生成后立即写入
│   ├── chapter_002.md
│   └── ...
│
├── events/                        # 世界事件过程记录
│   ├── chapter_001_events.json    # 世界事件（冲突检测结果）
│   ├── chapter_001_desires.json   # 角色欲望提案
│   ├── chapter_002_events.json
│   ├── chapter_002_desires.json
│   └── ...
│
├── reviews/                       # 章节评审记录
│   ├── chapter_001_review.json    # 主题/张力/逻辑评审
│   ├── chapter_002_review.json
│   └── ...
│
├── memory/                        # 记忆蒸馏记录
│   ├── chapter_001_memory.json    # 结构化记忆摘要
│   └── ...
│
├── state/                         # 角色状态快照
│   ├── chapter_001_state.json     # 每章结束时的角色完整状态
│   └── ...
│
├── full_novel.md                  # 完整小说合并文件（最终生成）
└── metadata.json                  # 生成元数据
```

### 各文件格式说明

#### `chapters/chapter_NNN.md`
```markdown
# 第N章 章节标题

小说正文...
```

#### `events/chapter_NNN_events.json`
```json
{
  "chapter_index": 1,
  "world_events": [
    {
      "event_id": "evt_001",
      "description": "事件描述",
      "participants": ["角色A", "角色B"],
      "conflict_type": "desire_clash",
      "narrative_value": 0.85,
      "tension_contribution": 0.3,
      "selected_for_narrative": true
    }
  ]
}
```

#### `events/chapter_NNN_desires.json`
```json
{
  "chapter_index": 1,
  "desire_proposals": [
    {
      "character_name": "林晚晴",
      "desire_id": "d_protect",
      "proposed_action": "想做的事",
      "urgency": 0.8,
      "resource_cost": {"emotional_energy": -15},
      "fear_override": ""
    }
  ]
}
```

#### `reviews/chapter_NNN_review.json`
```json
{
  "chapter_index": 1,
  "review": {
    "theme_alignment": 0.85,
    "theme_drift_notes": "",
    "tension_score": 0.7,
    "pacing_notes": "节奏良好",
    "logic_issues": [],
    "character_voice_issues": [],
    "unresolved_threads": ["线索A"],
    "suggestions_for_next": "建议..."
  }
}
```

#### `state/chapter_NNN_state.json`
```json
{
  "chapter_index": 1,
  "character_states": {
    "角色名": {
      "emotional_state": {"快乐": 0.3, "悲伤": 0.6},
      "relationship_scores": {"其他角色": 0.5},
      "beliefs": [...],
      "desires": [...],
      "resources": {...},
      "memory": [...]
    }
  }
}
```

#### `metadata.json`
```json
{
  "novel_title": "她的造物",
  "created_at": "2026-02-15T12:00:00",
  "updated_at": "2026-02-15T13:30:00",
  "chapters_generated": 10,
  "total_words": 25000,
  "full_novel_saved": true,
  "generation_log": [
    {"type": "chapter", "chapter_index": 1, "title": "...", "word_count": 2500, "timestamp": "..."},
    {"type": "events", "chapter_index": 1, "event_count": 5, "desire_count": 4, "timestamp": "..."}
  ],
  "token_usage": {
    "summary": {
      "total_calls": 120,
      "total_input_tokens": 450000,
      "total_output_tokens": 85000,
      "total_tokens": 535000,
      "total_duration_ms": 180000,
      "avg_duration_ms": 1500
    },
    "by_operation": { "director": {...}, "character": {...}, "narrator": {...} },
    "by_model": { "gemini-3-flash-preview": {...}, "M2-her": {...} },
    "by_chapter": { "1": {...}, "2": {...} }
  }
}
```

**Token 消耗查看**：
```bash
cat output/<novel_name>/metadata.json | jq '.token_usage.summary'
```

---

## 5. Token 消耗统计

系统自动统计所有 LLM 调用的 token 消耗，并在生成结束时写入 `metadata.json` 的 `token_usage` 字段。

| 统计维度 | 说明 |
|----------|------|
| `summary` | 总调用次数、总 input/output/total tokens、总耗时 |
| `by_operation` | 按操作类型（director、character、narrator、reviewer 等） |
| `by_model` | 按模型（Gemini、MiniMax M2-her 等） |
| `by_chapter` | 按章节索引 |

**成本估算**（参考，以实际 API 定价为准）：
- Gemini Flash：约 $0.075/1M input tokens，$0.30/1M output tokens
- 10 章小说（约 3 万字）通常消耗 50–80 万 tokens，成本约 $0.05–0.15

---

## 6. v2.2 文风控制器与节奏引擎

自 v2.2 起，系统引入了**文风控制器（Style Governor）**和**张弛节奏引擎（Rhythm Engine）**，旨在进一步提升文学质感，去除 AI 味。

### 核心功能

- **文风后处理**：
    - **比喻限制**：每 1000 字最多 1 个强比喻，禁止连续段落使用意象。
    - **抽象词压缩**：强制替换“神性、本质、变量”等抽象词汇。
    - **普通句强制**：确保 40% 以上的句子为简单的动作或场景描写。
    - **漂亮话删减**：自动剪掉最华丽、最像 AI 金句的 20% 句子。
- **张弛算法**：
    - 章节节奏按 `爆发 -> 冷却 -> 日常 -> 意外 -> 沉默 -> 再爆发` 循环。
    - 高能量情节后强制插入“低能量段落”（纯动作或环境，无主题）。
- **角色失误引擎**：
    - 角色每 2 章强制犯一个非战略性错误（误判、失控、撒谎）。
    - 错误会产生真实、不可逆的后果。

### 自动激活

这些功能已集成在 `ChaosEngine` 中并自动激活。配置项详见 `src/mobius/models/chaos_engine.py` 中的 `StyleGovernorConfig` 和 `RhythmConfig`。

---

## 7. 生成流水线（双循环架构）

每一章的生成经历以下完整管线：

```
START
  │
  ▼
┌─────────────────── 内循环（世界模拟） ───────────────────┐
│  env_update        环境变量衰减 + 资源自然回复            │
│  ↓                                                       │
│  desire_tick       每个角色提出欲望提案                    │
│  ↓                                                       │
│  conflict_detect   从欲望碰撞中检测冲突，生成世界事件      │
│  ↓                                                       │
│  observer_mark     观察者评估事件叙事价值，选择高价值事件   │
└──────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────── 外循环（叙事呈现） ───────────────────┐
│  plan_chapter      导演从素材中编排章节规划                │
│  ↓                                                       │
│  direct_scene      调度下一个场景                          │
│  ↓                                                       │
│  character_action / character_interact / narration        │
│                    角色执行行动（双阶段：内心独白→外显行为） │
│  ↓                                                       │
│  update_state      应用状态变化（信念/资源/关系/环境）      │
│  ↓                                                       │
│  check_triggers    检查是否触发事件                        │
│  ↓                                                       │
│  [循环直到所有场景完成]                                    │
│  ↓                                                       │
│  compile_chapter   叙事 Agent 编译完整章节                 │
│  ↓                                                       │
│  persist_chapter   ★ 立即写入磁盘（章节 + 事件 + 状态）   │
│  ↓                                                       │
│  [secondary_viewpoints]  支线视角片段（可选）              │
│  ↓                                                       │
│  review_chapter    评审：主题守护 + 张力控制 + 自我批评     │
│  ↓                                                       │
│  persist_review    ★ 写入评审记录                          │
│  ↓                                                       │
│  distill_memory    记忆蒸馏：压缩长期记忆                  │
│  ↓                                                       │
│  persist_memory    ★ 写入记忆摘要，路由到下一章或结束       │
└──────────────────────────────────────────────────────────┘
  │
  ▼
下一章 env_update / END
```

**关键特性：** 
- 每生成一章立即写入 `chapters/`，不等最终汇总
- 世界事件和欲望提案每章记录到 `events/`
- 评审结果每章记录到 `reviews/`
- 角色状态快照每章记录到 `state/`
- 记忆蒸馏结果每章记录到 `memory/`

---

## 8. 提示词管理

所有 Agent 的 System Prompt 与代码分离，存放在 `src/mobius/prompts/` 目录：

```
src/mobius/prompts/
├── __init__.py                        # load_prompt() / format_prompt()
├── director_system.txt                # 导演/编排者系统提示
├── director_chapter_plan_schema.txt   # 章节规划 JSON Schema
├── character_action_format.txt        # 角色行动 JSON 格式
├── character_internal_monologue.txt   # 内心独白生成提示
├── character_stage2_action.txt        # 外显行动生成提示
├── character_desire_tick.txt          # 欲望提案提示
├── narrator_system.txt                # 叙事 Agent 系统提示（含文风要求）
├── narrator_narration_only.txt        # 旁白/环境描写提示
├── memory_distill_system.txt          # 记忆蒸馏系统提示
├── reviewer_system.txt                # 评审系统提示
├── reviewer_schema.txt                # 评审输出 JSON Schema
├── observer_system.txt                # 世界观察者系统提示
└── observer_viewpoint.txt             # 支线视角提示模板
```

### 修改提示词

直接编辑对应的 `.txt` 文件即可，无需修改 Python 代码。

### 代码中使用

```python
from mobius.prompts import load_prompt, format_prompt

# 加载原始提示词
prompt = load_prompt("director_system")

# 加载并填充模板变量
prompt = format_prompt("character_internal_monologue",
    character_name="林晚晴",
    context="当前场景描述..."
)
```

---

## 9. 项目目录总览

```
mobius/
├── presets/                     # 设定集（预设故事配置）
│   └── ai_love_story.yaml
│
├── src/mobius/
│   ├── agents/                  # Agent 实现
│   │   ├── director.py          # 导演/编排者
│   │   ├── character.py         # 角色 Agent 工厂
│   │   ├── narrator.py          # 叙事 Agent
│   │   ├── memory.py            # 记忆蒸馏
│   │   ├── reviewer.py          # 评审（主题+张力+自批评）
│   │   ├── observer.py          # 世界观察者 + 支线视角
│   │   └── utils.py             # LLM 响应解析工具
│   │
│   ├── prompts/                 # 提示词（与代码分离）
│   │   ├── director_system.txt
│   │   ├── narrator_system.txt
│   │   └── ...
│   │
│   ├── output/                  # 产出物管理
│   │   └── manager.py           # OutputManager
│   │
│   ├── graph/                   # LangGraph 图定义
│   │   ├── novel_graph.py       # 双循环主图 + 持久化节点
│   │   └── routing.py           # 条件路由
│   │
│   ├── models/                  # Pydantic 数据模型
│   │   ├── belief.py            # 信念系统
│   │   ├── desire.py            # 欲望/恐惧系统
│   │   ├── resource.py          # 资源系统
│   │   ├── environment.py       # 环境状态总线
│   │   ├── viewpoint.py         # 支线视角
│   │   ├── review.py            # 评审/记忆摘要
│   │   └── ...
│   │
│   ├── state/                   # 状态管理
│   │   ├── novel_state.py       # 全局状态 TypedDict
│   │   ├── character_state.py   # 角色状态操作
│   │   └── conflict_engine.py   # 冲突检测引擎
│   │
│   ├── llm/                     # LLM 封装
│   │   └── minimax.py           # MiniMax M2-her 适配器
│   │
│   ├── config/                  # 配置
│   │   └── settings.py
│   │
│   └── main.py                  # CLI 入口
│
├── docs/                        # 文档
│   ├── ARCHITECTURE.md          # 架构设计文档
│   └── USAGE_SPEC.md            # 本文档
│
├── pyproject.toml
└── README.md
```

---

## 10. 后期优化指南

### 利用 events/ 数据优化

`events/` 目录记录了每章的世界事件和角色欲望提案：

1. **冲突分析**：检查 `chapter_NNN_events.json` 中的 `conflict_type` 分布，判断冲突是否多样
2. **欲望追踪**：对比各章 `chapter_NNN_desires.json`，观察角色欲望优先级的变化轨迹
3. **叙事价值**：统计 `narrative_value` 分布，调整观察者的筛选阈值

### 利用 reviews/ 数据优化

1. **主题漂移检测**：追踪各章 `theme_alignment` 值，发现趋势性偏移
2. **张力曲线**：绘制各章 `tension_score`，调整高潮间距
3. **逻辑修复**：汇总所有 `logic_issues`，在后续生成中通过评审反馈修正

### 利用 state/ 数据优化

1. **角色弧线**：对比首末章的角色状态，量化角色成长
2. **资源消耗**：追踪资源变化，确保资源系统产生有意义的约束
3. **信念演化**：对比信念变化轨迹，验证角色成长是否符合预期
