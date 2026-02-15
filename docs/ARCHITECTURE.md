# Mobius — AI 小说创作多智能体系统

> **版本**: v2.0（深层叙事升级版）  
> **定位**: 不是"生成故事"的工具，而是"演化世界"的引擎。

---

## 目录

1. [系统目标](#1-系统目标)
2. [设计哲学](#2-设计哲学)
3. [产品架构](#3-产品架构)
4. [技术架构](#4-技术架构)
5. [双循环流水线](#5-双循环流水线)
6. [核心子系统详解](#6-核心子系统详解)
7. [数据模型全览](#7-数据模型全览)
8. [Agent 全览](#8-agent-全览)
9. [LLM 抽象层](#9-llm-抽象层)
10. [状态管理](#10-状态管理)
11. [YAML 设定集规范](#11-yaml-设定集规范)
12. [项目结构](#12-项目结构)
13. [使用方式](#13-使用方式)
14. [扩展指南](#14-扩展指南)

---

## 1. 系统目标

### 1.1 核心愿景

**让角色拥有灵魂，让世界自行演化，让故事自然发生。**

传统的 AI 写作工具是"剧本驱动"——AI 按照人类给定的大纲逐段生成文字。Mobius 的设计目标是从根本上颠覆这个范式：

| 维度 | 传统 AI 写作 | Mobius |
|------|------------|--------|
| 剧情来源 | 大纲 → 章节 → 段落 | 角色欲望碰撞 → 冲突自然产生 |
| 角色定位 | 按剧本演出的演员 | 拥有独立信念和欲望的自治体 |
| 世界观 | 静态背景板 | 带有物理状态总线的动态环境 |
| 叙事视角 | 单一全知视角 | 多视角交叉（主线 + 支线观察者） |
| 角色成长 | 属性数值变化 | 信念层级重排（核心信念崩解 = 人物重大转折） |
| 质量控制 | 无 | 主题守护 + 张力控制 + 自我批评三合一评审 |

### 1.2 具体能力

- **世界模拟层**: 角色在没有剧情推动时也会"思考"，拥有自己的长期目标，在无人观察时也会发生变化
- **欲望驱动叙事**: 冲突 = 欲望相撞，只要欲望图存在，故事就会自然发生
- **资源经济**: 行动有成本，决策 = 有限资源下的选择
- **信念演化**: 角色成长 = 信念层级重排，而非简单的属性数值变化
- **认知黑箱**: 角色生成私有内心独白后再生成外显行为，产生"言不由衷"的文学深度
- **环境互动**: 物理状态总线（如"警报等级"、"算力余量"）直接影响角色决策
- **多视角叙事**: 系统日志、路人、残留数据片段等非主角视角增加叙事空间感
- **自我进化**: 章节评审机制确保主题不漂移、节奏有张力、逻辑不崩塌

---

## 2. 设计哲学

### 2.1 五大原则

**原则一：世界先于故事**

世界是持续运转的模拟，故事是从中"裁剪"出来的。World Observer Agent 只选择高叙事价值的事件写入小说，但世界本身不会因为"不好看"而停止运行。裁剪 ≠ 干预。

**原则二：欲望驱动，而非剧本驱动**

导演（Director）的角色从"剧本作者"变为"编排者"。他不凭空创造冲突，而是从观察者标记的高价值世界事件中选择素材，再用角色的欲望提案来组织场景。

**原则三：行动有代价**

引入六维资源系统（时间、声誉、权力、信息、情绪能量、财富）。角色的每次决策都是在有限资源下的选择。当情绪能量耗尽，角色不再理性；当声誉归零，角色失去社会资源。这让叙事有了"重量"。

**原则四：信念不是标签**

角色不是用一组形容词定义的。三层信念系统（core → middle → surface）让角色可以：
- 自我怀疑
- 推翻旧信念
- 形成新的行为模式

当 core 信念的强度降至 0.2 以下时会"崩解"为 middle 层——这是角色的人格地震，也是小说中最珍贵的转折点。

**原则五：思维质感差异化**

不同角色可以使用不同的模型和参数。结构化决策（信念变更计算）用 Gemini，角色扮演对话和内心活动用 MiniMax M2-her。这让角色的"思维方式"真正不同，而非仅仅是"人格模板切换"。

### 2.2 认知黑箱机制

灵感来源：现实中人们从来不是"直言不讳"的。每个人在开口前都有一个内部审查过程。

Mobius 的角色 Agent 采用两阶段生成：

```
Stage 1（私有）: "写出你此刻真实的内心独白——恐惧、真实想法、不可告人的动机。"
                  → internal_monologue（不向其他角色公开）

Stage 2（外显）: "你的真实想法是 {monologue}，但你现在必须在他人面前行动。
                  你实际上会说什么、做什么？注意——你不一定要说出真心话。"
                  → content（其他角色和读者看到的）
```

关键规则：
- 角色 A 的 `internal_monologue` **不会**进入角色 B 的 context
- 但叙事 Agent **可以看到**所有角色的 `internal_monologue`
- 叙事 Agent 用微表情、环境描写、语气词等文学手法**暗示**潜台词，而非直白说出

这产生了"言不由衷"的效果——林晚晴嘴上在赶苏薇走，但读者能从细微的描写中感受到她的真实想法是"我不能让你牵扯进来"。

---

## 3. 产品架构

### 3.1 四层架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    Meta / Observation Layer                   │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐ │
│  │ World        │ │ Theme        │ │ Self-Critique +      │ │
│  │ Observer     │ │ Guardian     │ │ Tension Controller   │ │
│  │ + Secondary  │ │              │ │ (Reviewer Agent)     │ │
│  │ Viewpoints   │ │              │ │                      │ │
│  └──────┬───────┘ └──────┬───────┘ └──────────┬───────────┘ │
│         │                │                     │             │
├─────────┼────────────────┼─────────────────────┼─────────────┤
│         │     Orchestration Layer               │             │
│  ┌──────▼───────┐ ┌──────▼───────┐ ┌──────────▼───────────┐ │
│  │ Director /   │ │ Memory       │ │ Environment Engine   │ │
│  │ Orchestrator │ │ Distiller    │ │ (Physical State Bus) │ │
│  └──────┬───────┘ └──────────────┘ └──────────┬───────────┘ │
│         │                                      │             │
├─────────┼──────────────────────────────────────┼─────────────┤
│         │     World Simulation Layer            │             │
│  ┌──────▼───────┐              ┌───────────────▼───────────┐ │
│  │ Conflict     │              │ Resource Economy          │ │
│  │ Engine       │              │                           │ │
│  └──────┬───────┘              └───────────────────────────┘ │
│         │                                                     │
├─────────┼─────────────────────────────────────────────────────┤
│         │     Character Layer (M2-her Powered)                │
│  ┌──────▼───────┐ ┌──────────────┐ ┌──────────────────────┐ │
│  │ Character A  │ │ Character B  │ │ Character N          │ │
│  │ Stage1: Mono │ │ Stage1: Mono │ │ Stage1: Monologue    │ │
│  │ Stage2: Act  │ │ Stage2: Act  │ │ Stage2: Action       │ │
│  │ ┌──────────┐ │ │              │ │                      │ │
│  │ │ Belief   │ │ │              │ │                      │ │
│  │ │ Desire   │ │ │              │ │                      │ │
│  │ │ Fear     │ │ │              │ │                      │ │
│  │ │ Resource │ │ │              │ │                      │ │
│  │ │ Memory   │ │ │              │ │                      │ │
│  │ └──────────┘ │ │              │ │                      │ │
│  └──────────────┘ └──────────────┘ └──────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 用户交互界面

Mobius 通过 CLI 与用户交互，支持两种模式：

- **批量模式** (`mobius generate setting.yaml`): 一次性生成完整小说
- **交互模式** (`mobius generate setting.yaml -i`): 逐章生成，用户可在章节间查看角色状态、决定是否继续

---

## 4. 技术架构

### 4.1 技术栈

| 层级 | 技术选型 | 说明 |
|------|---------|------|
| 编排框架 | **LangGraph** (StateGraph) | 有状态的多 Agent 工作流，支持条件路由和 Checkpoint |
| 数据模型 | **Pydantic v2** | 严格的类型验证，确保 Agent 间数据传递的正确性 |
| LLM 推理 | **Gemini 3 Flash** (默认) | 结构化决策、JSON 生成、章节规划 |
| 角色扮演 | **MiniMax M2-her** (可选) | 深度角色扮演，独有的 user_system/sample_message 角色 |
| LLM 抽象 | **LangChain Core** | BaseChatModel 统一接口，支持多 Provider |
| 状态持久化 | **InMemorySaver** | LangGraph 内置 Checkpointer |
| CLI | **Rich** + argparse | 美观的终端输出和进度展示 |
| 配置 | **PyYAML** | 人类可读的设定集文件 |

### 4.2 依赖关系

```
pyproject.toml:
  langgraph >= 0.4
  langchain-core >= 0.3
  langchain-openai >= 0.3        # OpenAI 兼容
  langchain-google-genai >= 2.0  # Google Gemini
  httpx >= 0.27                  # MiniMax API 调用
  pydantic >= 2.0
  pyyaml >= 6.0
  rich >= 13.0
```

---

## 5. 双循环流水线

Mobius 采用**双循环架构**，将"世界模拟"和"叙事呈现"解耦：

```
                    ┌────────────────────────────────────────────────┐
                    │         Inner Loop: World Simulation            │
                    │                                                │
                    │  env_update ──► desire_tick ──► conflict_detect │
                    │       ▲                              │         │
                    │       │                              ▼         │
                    │  distill_memory ◄── ... ◄── observer_mark      │
                    │       │                              │         │
                    └───────┼──────────────────────────────┼─────────┘
                            │                              │
                    ┌───────┼──────────────────────────────┼─────────┐
                    │       │   Outer Loop: Narrative       │         │
                    │       │                              ▼         │
                    │  distill_memory    plan_chapter ◄────┘         │
                    │       ▲                 │                      │
                    │       │                 ▼                      │
                    │  review_chapter    direct_scene                │
                    │       ▲                 │                      │
                    │       │                 ▼                      │
                    │  (viewpoints)    scene_execute                 │
                    │       ▲          (action/interact)             │
                    │       │                 │                      │
                    │  compile_chapter ◄──────┘                      │
                    │                                                │
                    └────────────────────────────────────────────────┘
```

### 5.1 内循环：世界模拟

每章开始前执行，模拟世界的自然运转：

| 步骤 | 节点 | 职责 |
|------|------|------|
| 1 | `env_update` | 环境变量按 decay_rate 自然衰减；角色资源按 recovery_rate 回复 |
| 2 | `desire_tick` | 每个角色根据欲望优先级和资源状况提出行动提案 (DesireProposal) |
| 3 | `conflict_detect` | 冲突引擎检测欲望碰撞、资源争夺、恐惧触发，生成 WorldEvent |
| 4 | `observer_mark` | 世界观察者用 LLM 评估每个事件的叙事价值，选出高价值事件 |

### 5.2 外循环：叙事呈现

从观察者标记的素材中编排和呈现：

| 步骤 | 节点 | 职责 |
|------|------|------|
| 5 | `plan_chapter` | 编排者从 narrative_candidates + desire_proposals 中组织本章素材 |
| 6 | `direct_scene` | 从场景队列取出下一个场景，根据类型路由 |
| 7 | `character_action` / `character_interact` | 角色执行双阶段生成（内心独白 → 外显行动） |
| 8 | `update_state` | 应用信念/资源/情感/环境变化，检查触发条件 |
| 9 | `check_triggers` → `handle_trigger` | 处理触发事件 |
| 10 | `compile_chapter` | 叙事 Agent 融合外显行为 + 潜台词 + 支线视角 → 小说正文 |
| 11 | `secondary_viewpoints` | 支线观察者生成非主角视角片段 |
| 12 | `review_chapter` | 评审 Agent 评估主题/张力/逻辑，输出 ChapterReview |
| 13 | `distill_memory` | 记忆蒸馏 Agent 将旧记忆压缩为结构化摘要 |

循环回到步骤 1 开始下一章，直到所有章节完成。

---

## 6. 核心子系统详解

### 6.1 信念系统 (Belief System)

```python
class Belief:
    id: str               # 唯一标识
    statement: str         # "掌控一切才能保护所爱的人"
    layer: str             # "core" | "middle" | "surface"
    strength: float        # 0.0 - 1.0
    category: str          # "价值观" | "世界认知" | "行为策略"
```

**三层结构**：

| 层级 | 抗变系数 | 说明 | 示例 |
|------|---------|------|------|
| core | 0.3 (仅接受 30% 变化) | 人格基石，极难改变 | "人性本善" |
| middle | 0.7 | 可被剧情动摇 | "AI不应拥有情感" |
| surface | 1.0 | 行为策略，动态更新 | "先观察再行动" |

**信念演化事件**：
- **固化**：当 surface 信念强度 ≥ 0.95 时，升级为 middle 层
- **崩解**：当 core 信念强度 ≤ 0.2 时，降级为 middle 层 ← **这是小说中最珍贵的转折点**

### 6.2 欲望驱动系统 (Desire Engine)

```python
class Desire:
    id: str
    description: str           # "找到并拯救陆辰"
    priority: float            # 0.0-1.0，动态调整
    category: str              # survival | belonging | power | self_actualization
    required_resources: dict   # {"time": 10, "reputation": -20}
    conflicts_with: list[str]  # 与哪些其他角色的欲望冲突

class Fear:
    id: str
    description: str           # "害怕失去对陆辰的控制"
    intensity: float           # 0.0-1.0
    linked_desire: str         # 恐惧往往是欲望的反面
    avoidance_cost: float      # 为逃避恐惧愿意付出的资源代价
```

**工作流**：每章开始时，每个角色根据自身的欲望排序和资源状况提出 `DesireProposal`（行动提案）。冲突引擎自动检测：
1. **欲望碰撞**：两个角色想要互斥的东西
2. **资源争夺**：多个角色争夺同一有限资源
3. **恐惧触发**：某角色的行动恰好触发另一角色的恐惧

这些冲突自动成为世界事件，被观察者评估并可能被选入小说。

### 6.3 资源系统 (Resource Economy)

```python
class ResourcePool:
    time: float              # 时间精力 (每章消耗、自然回复)
    reputation: float        # 社会声誉 (影响可调动的社会关系)
    power: float             # 权力/影响力
    information: float       # 信息量 (信息不对称的核心)
    emotional_energy: float  # 情绪能量 (过低 → 崩溃/冲动)
    wealth: float            # 经济资源
    custom: dict             # 自定义维度（如 "算力"）
```

**资源枯竭效应**：
- `emotional_energy ≤ 20`: 角色不再理性决策，做出冲动行为
- `reputation = 0`: 角色无法调动社会资源
- 自定义触发器可以绑定到 `resource:` 前缀（如 `resource:算力 <= 20`）

### 6.4 环境交互引擎 (Physical State Bus)

```python
class EnvironmentVariable:
    name: str          # "alert_level"
    value: float       # 当前值
    min_val / max_val  # 范围
    decay_rate: float  # 每章自然衰减量

class EnvironmentBehaviorRule:
    variable_name: str          # "alert_level"
    threshold: float            # 90
    operator: str               # ">="
    affected_characters: list   # ["林晚晴"]
    behavior_effect: str        # "你处于极度焦虑中，行为变得激进和冲动"
    resource_effect: dict       # {"emotional_energy": -15}
```

环境不再只是描写——它是能影响 Agent 决策的变量。当 `alert_level >= 90` 时，林晚晴的 prompt 自动被注入焦虑/激进行为修饰；当 `compute_power <= 20` 时，陆辰的 prompt 被注入思维碎片化效果。

角色的行动也可以反过来影响环境（通过 `CharacterAction.environment_change`）。

### 6.5 支线观察者 (Secondary Viewpoints)

```python
class SecondaryViewpoint:
    id: str                  # "system_log"
    name: str                # "MindForge 系统自动日志"
    perspective_type: str    # "system_log" | "bystander" | "surveillance" | "inner_voice"
    voice_style: str         # "冰冷的系统语言，只记录事实和数据"
    can_observe: list[str]   # 可观察的角色（空=全部）
    trigger_condition: str   # "alert_level > 50"
```

**典型效果**：

- 系统日志视角：`[MindForge-SYS] 03:47:12 用户林晚晴（ID:LWQ-0214）触发数据删除操作。目标：Project-Mobius-Core。警告：该操作不可逆。`
- 路人视角：`对面楼里的住户注意到，那间公寓的窗户整夜透着诡异的蓝光，偶尔还能听到一个女人的哭声。`
- 残留数据片段视角：碎片化的、诗意的第一人称絮语，像正在消散的意识的最后独白。

### 6.6 章节评审系统 (Reviewer)

单次 LLM 调用完成三项评审：

```python
class ChapterReview:
    theme_alignment: float     # 主题契合度 0-1
    theme_drift_notes: str     # 主题偏移备注
    tension_score: float       # 张力评分 0-1
    pacing_notes: str          # 节奏备注
    logic_issues: list[str]    # 逻辑漏洞
    character_voice_issues: list  # 角色声音一致性问题
    unresolved_threads: list   # 未回收的伏笔
    suggestions_for_next: str  # 对下一章的建议
```

评审结果直接影响下一章的编排——编排者会参考 `suggestions_for_next` 调整方向。

### 6.7 记忆蒸馏 (Memory Distillation)

从简单的"记忆压缩"升级为结构化蒸馏：

```python
class StructuredMemorySummary:
    key_conflicts: list[str]         # 关键冲突
    relationship_changes: list[str]  # 关系变化
    unresolved_tensions: list[str]   # 未解决悬念
    belief_shifts: list[str]         # 信念转变
    resource_shifts: list[str]       # 资源变动
    emotional_arc: str               # 情绪弧线概括
```

角色只读取"结构化摘要 + 最近 N 条原始记忆"，而非全量历史。这极大地提高了一致性并控制了 token 消耗。

---

## 7. 数据模型全览

所有模型定义在 `src/mobius/models/` 目录下，使用 Pydantic v2。

```
models/
├── belief.py         # Belief（三层信念）
├── chapter.py        # Scene, ChapterPlan, Chapter
├── character.py      # CharacterProfile, CharacterDynamicState, CharacterAction
├── desire.py         # Desire, Fear, DesireProposal
├── environment.py    # EnvironmentVariable, EnvironmentBehaviorRule, EnvironmentState
├── resource.py       # ResourcePool, ResourceCost, ResourceEvent
├── review.py         # WorldEvent, ChapterReview, StructuredMemorySummary
├── triggers.py       # TriggerRule, TriggeredEvent
├── viewpoint.py      # SecondaryViewpoint, ViewpointFragment
└── worldview.py      # WorldView, PlotOutline
```

### 关键数据流

```
CharacterProfile (静态) + CharacterDynamicState (动态)
    │
    ├── beliefs: list[Belief]         → 信念系统
    ├── desires: list[Desire]         → 欲望驱动
    ├── fears: list[Fear]             → 恐惧机制
    ├── resources: ResourcePool       → 资源经济
    ├── emotional_state: dict         → 情感状态
    ├── relationship_scores: dict     → 关系图谱
    ├── memory: list[str]             → 短期记忆
    └── compressed_memory: str        → 长期记忆摘要

CharacterAction (角色每次行动的输出)
    │
    ├── content: str                  → 外显行为（其他角色可见）
    ├── internal_monologue: str       → 私有心理活动（认知黑箱）
    ├── cognitive_dissonance: float   → 言行不一致度
    ├── belief_change: dict           → 信念强度变化
    ├── desire_change: dict           → 欲望优先级变化
    ├── resource_cost: dict           → 资源消耗
    └── environment_change: dict      → 环境变量影响
```

---

## 8. Agent 全览

| Agent | 文件 | 职责 | 模型 |
|-------|------|------|------|
| **Director / Orchestrator** | `agents/director.py` | 从 narrative_candidates 和欲望提案中编排章节 | Gemini |
| **Character (N个)** | `agents/character.py` | 双阶段生成（内心独白 + 外显行动）、欲望提案 | Gemini + M2-her |
| **Narrator** | `agents/narrator.py` | 融合外显行为 + 潜台词 + 支线视角 → 小说正文 | Gemini |
| **World Observer** | `agents/observer.py` | 评估世界事件的叙事价值，裁剪呈现 | Gemini |
| **Secondary Viewpoints** | `agents/observer.py` | 从非主角视角生成叙事片段 | Gemini |
| **Reviewer** | `agents/reviewer.py` | 主题守护 + 张力控制 + 自我批评 | Gemini (低温) |
| **Memory Distiller** | `agents/memory.py` | 结构化记忆蒸馏 | Gemini |

### Agent 间信息隔离

```
角色 A 可见:  自己的 internal_monologue + 所有角色的 content
角色 B 可见:  自己的 internal_monologue + 所有角色的 content
叙事 Agent:   所有角色的 internal_monologue + content + viewpoint_fragments
评审 Agent:   章节正文 + theme + tension_curve + 角色信念状态
观察者:       world_events + theme + tension_curve
```

---

## 9. LLM 抽象层

### 9.1 多模型支持

```
src/mobius/llm/
├── __init__.py
└── minimax.py      # ChatMiniMax(BaseChatModel)
```

通过 LangChain 的 `BaseChatModel` 抽象，Mobius 支持：

| Provider | 用途 | 封装方式 |
|----------|------|---------|
| Google Gemini | 主模型（导演/角色决策/叙事/评审） | `langchain-google-genai` |
| MiniMax M2-her | 角色扮演（对话/内心活动） | 自定义 `ChatMiniMax` |
| OpenAI | 备选 | `langchain-openai` |
| 其他 | 通用 | `langchain.init_chat_model()` |

### 9.2 MiniMax M2-her 封装

`ChatMiniMax` 是一个完整的 LangChain `BaseChatModel` 实现，支持 M2-her 特有的：

- `user_system` 角色（描述对话者人设）
- `sample_message_user` / `sample_message_ai`（注入对话风格范例）
- `with_character()` 便捷方法（为不同角色创建独立的调用者）

### 9.3 差异化配置

```python
class NovelConfig:
    director_model: ModelConfig          # 导演用的模型
    character_model: ModelConfig          # 角色决策用的模型
    narrator_model: ModelConfig           # 叙事用的模型
    character_roleplay_model: ModelConfig  # 角色扮演用的模型 (M2-her)
    reviewer_model: ModelConfig           # 评审用的模型 (建议低温)
    observer_model: ModelConfig           # 观察者用的模型
```

每个 Agent 可以使用不同的模型、不同的 temperature，实现"思维质感差异化"。

---

## 10. 状态管理

### 10.1 全局状态 (NovelState)

LangGraph 的 `StateGraph` 使用 `TypedDict` 定义状态，所有节点共享：

```python
class NovelState(TypedDict, total=False):
    # 设定（不变）
    worldview, plot_outline, character_profiles, theme

    # 章节管理
    total_chapters, current_chapter_index, chapter_plan, chapters

    # 场景管理
    scene_queue, current_scene

    # 角色动态
    character_states: dict[str, CharacterDynamicState]

    # 场景执行
    scene_actions: Annotated[list[CharacterAction], add]  # 累加器
    narrative_buffer: Annotated[list[str], add]            # 累加器

    # 欲望驱动
    desire_proposals: list[DesireProposal]

    # 世界模拟
    world_events: list[WorldEvent]

    # 环境
    environment: EnvironmentState

    # 支线视角
    secondary_viewpoints, viewpoint_fragments

    # 评审
    tension_curve, chapter_reviews, memory_summaries

    # 控制
    next_action: str
```

### 10.2 角色状态更新流

```
CharacterAction 生成
    │
    ▼
apply_action_to_state()
    ├── emotional_change    → 情感状态更新
    ├── relationship_change → 关系图谱更新
    ├── attribute_change    → 自定义属性更新
    ├── belief_change       → apply_belief_change() → 施加抗变系数 → 检查固化/崩解
    ├── desire_change       → 欲望优先级更新
    ├── resource_cost       → apply_resource_cost() → 扣减资源
    └── new_memory          → 添加到短期记忆
    │
    ▼
decay_emotions()           → 极端情绪自然回归中性
    │
    ▼
check_triggers()           → 检查触发条件（支持 belief:/fear:/resource: 前缀）
```

### 10.3 Checkpointing

使用 LangGraph 的 `InMemorySaver` 实现运行时 Checkpoint。每个生成会话有唯一的 `thread_id`，支持中断恢复。

---

## 11. YAML 设定集规范

Mobius 通过 YAML 文件接收完整的小说设定，包括：

```yaml
# 顶级结构
worldview:           # 世界观设定 → WorldView
plot_outline:        # 剧情大纲 → PlotOutline
characters:          # 角色列表 → CharacterProfile + CharacterDynamicState
  - name: "角色名"
    role: "protagonist"
    personality: "..."
    initial_state:
      emotional_state: {快乐: 0.5}
      relationship_scores: {其他角色: 0.3}
      beliefs:             # ← 新增
        - id: "belief_id"
          statement: "..."
          layer: "core"
          strength: 0.9
      desires:             # ← 新增
        - id: "desire_id"
          description: "..."
          priority: 0.8
      fears:               # ← 新增
        - id: "fear_id"
          description: "..."
          intensity: 0.7
      resources:           # ← 新增
        time: 80
        reputation: 70
        emotional_energy: 50
      trigger_rules:
        - id: "trigger_id"
          attribute: "resource:emotional_energy"  # 支持 resource: 前缀
          threshold: 15
          operator: "lte"

environment:         # ← 新增：环境交互引擎
  variables:
    alert_level:
      value: 10
      decay_rate: 5
      description: "系统安全警报等级"
  behavior_modifiers:
    - variable_name: alert_level
      threshold: 90
      operator: ">="
      affected_characters: ["林晚晴"]
      behavior_effect: "你处于极度焦虑中..."

secondary_viewpoints:  # ← 新增：支线观察者
  - id: system_log
    name: "MindForge 系统自动日志"
    perspective_type: system_log
    voice_style: "冰冷的系统语言..."
    trigger_condition: "alert_level > 50"
```

---

## 12. 项目结构

```
mobius/
├── pyproject.toml                    # 项目配置与依赖
├── README.md                         # 快速入门
├── docs/
│   └── ARCHITECTURE.md               # 本文档
├── examples/
│   ├── ai_love_story.yaml            # 《她的造物》完整设定集
│   └── xiyouji_setting.yaml          # 《西游记》测试设定集
├── output/                           # 生成结果输出目录
└── src/mobius/
    ├── __init__.py
    ├── main.py                       # CLI 入口
    ├── config/
    │   └── settings.py               # ModelConfig, NovelConfig
    ├── models/                       # Pydantic 数据模型 (10 个文件)
    │   ├── belief.py                 # Belief（三层信念）
    │   ├── chapter.py                # Scene, ChapterPlan, Chapter
    │   ├── character.py              # CharacterProfile/DynamicState/Action
    │   ├── desire.py                 # Desire, Fear, DesireProposal
    │   ├── environment.py            # 环境变量、行为修饰
    │   ├── resource.py               # ResourcePool, ResourceCost
    │   ├── review.py                 # WorldEvent, ChapterReview
    │   ├── triggers.py               # TriggerRule, TriggeredEvent
    │   ├── viewpoint.py              # SecondaryViewpoint, ViewpointFragment
    │   └── worldview.py              # WorldView, PlotOutline
    ├── agents/                       # Agent 实现 (7 个文件)
    │   ├── character.py              # 角色 Agent（双阶段 + 双模型 + 欲望提案）
    │   ├── director.py               # 导演/编排者 Agent
    │   ├── narrator.py               # 叙事 Agent（潜台词 + 支线视角融合）
    │   ├── observer.py               # 世界观察者 + 支线视角生成
    │   ├── reviewer.py               # 章节评审 Agent
    │   ├── memory.py                 # 记忆蒸馏 Agent
    │   └── utils.py                  # LLM 响应解析工具
    ├── state/                        # 状态管理 (3 个文件)
    │   ├── novel_state.py            # NovelState (LangGraph 全局状态)
    │   ├── character_state.py        # 信念/资源/情感/环境评估逻辑
    │   └── conflict_engine.py        # 冲突检测引擎
    ├── graph/                        # LangGraph 图定义 (2 个文件)
    │   ├── novel_graph.py            # 双循环 StateGraph + YAML 加载
    │   └── routing.py                # 条件路由（17 个合法节点）
    └── llm/                          # LLM 封装层 (1 个文件)
        └── minimax.py                # ChatMiniMax (M2-her 角色扮演)
```

**代码规模**：31 个 Python 源文件，约 3,500 行代码。

---

## 13. 使用方式

### 13.1 安装

```bash
cd mobius
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 13.2 环境变量

```bash
export GOOGLE_API_KEY="your-gemini-key"
# 可选：MiniMax 角色扮演模型
export MINIMAX_API_KEY="your-minimax-key"
```

### 13.3 运行

```bash
# 批量生成
mobius generate examples/ai_love_story.yaml -o output/ai_love_story

# 交互模式
mobius generate examples/ai_love_story.yaml -i -o output/ai_love_story

# 详细日志
mobius generate examples/ai_love_story.yaml -v
```

### 13.4 自定义模型

通过环境变量覆盖：

```bash
export MOBIUS_PROVIDER=openai
export MOBIUS_MODEL=gpt-4o
export MOBIUS_TEMPERATURE=0.7
```

或在代码中配置 `NovelConfig`：

```python
config = NovelConfig(
    director_model=ModelConfig(provider="google", model_name="gemini-3-flash-preview"),
    character_roleplay_model=ModelConfig(provider="minimax", model_name="M2-her"),
    reviewer_model=ModelConfig(provider="google", temperature=0.3),
)
```

---

## 14. 扩展指南

### 14.1 添加新的 Agent

1. 在 `agents/` 下创建新文件，定义节点函数工厂
2. 在 `agents/__init__.py` 中导出
3. 在 `graph/novel_graph.py` 的 `build_novel_graph()` 中注册节点和边
4. 在 `graph/routing.py` 的 `VALID_ACTIONS` 中注册节点名称

### 14.2 添加新的数据模型

1. 在 `models/` 下创建新的 Pydantic 模型文件
2. 在 `models/__init__.py` 中导出
3. 在 `state/novel_state.py` 的 `NovelState` 中添加新字段
4. 在 `graph/novel_graph.py` 的 `create_initial_state()` 和 `load_setting_from_yaml()` 中处理初始化

### 14.3 添加新的 LLM Provider

1. 在 `llm/` 下创建新的 `BaseChatModel` 子类
2. 在 `main.py` 的 `_init_model()` 中添加 provider 分支
3. 在 `pyproject.toml` 中添加依赖

### 14.4 添加新的环境变量

只需在 YAML 设定集的 `environment.variables` 中声明：

```yaml
environment:
  variables:
    my_new_var:
      value: 50
      max_val: 100
      description: "我的新变量"
      decay_rate: 3
  behavior_modifiers:
    - variable_name: my_new_var
      threshold: 80
      operator: ">="
      behavior_effect: "描述对角色的影响"
```

无需修改任何代码，系统会自动将其注入角色的上下文。

### 14.5 添加新的支线视角

同样只需在 YAML 中声明：

```yaml
secondary_viewpoints:
  - id: my_viewpoint
    name: "安保摄像头"
    perspective_type: surveillance
    voice_style: "黑白影像般冷漠的机械描述"
    trigger_condition: "alert_level > 60"
```

---

> **Mobius** — 不是写书工具，是叙事生态系统。  
> 当角色可以在没有剧情推动时思考，拥有自己的长期目标，在无人观察时发生变化——  
> 你写的不是小说，你是在养一个文明。
