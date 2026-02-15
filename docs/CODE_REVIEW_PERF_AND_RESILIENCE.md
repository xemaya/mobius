# Code Review：性能与容错（简要结论与改动）

## 一、性能瓶颈结论

1. **单章 LLM 调用量极大**
   - 欲望提案：每角色 1 次
   - 观察者：1 次
   - 导演规划：1 次
   - 每场景：导演编排 1 次；若为互动场景则 `max_rounds × 参与角色数 × (1 内心独白 + 1 外显行动)`，例如 5 轮×2 人×2 ≈ 20 次
   - 章节编译：1 次
   - 文风控制器：1 次（整章改写）
   - 评审：1 次；记忆蒸馏：每角色 1 次  
   因此「互动轮数」和「每章场景数」对总耗时影响最大。

2. **无重试**
   - 所有 `model.invoke()` 一次失败即导致该节点失败，网络抖动或限流会直接中断流程。

---

## 二、已实施的改动

### 1. 容错：LLM 调用统一加重试

- **新增** `mobius.agents.utils.invoke_with_retry(model, messages, max_retries=2, base_delay=2.0, operation_name=...)`
  - 仅对可重试异常（`ConnectionError`, `TimeoutError`, `OSError`）重试
  - 指数退避：2s、4s
- **接入节点**：director（规划 + 场景编排）、narrator（旁白 + 章节编译）、character（行动 + 互动 + 内心独白 + desire_tick）、observer、reviewer、memory、style_governor

### 2. 性能：降低默认调用量

- **NovelConfig 默认值**
  - `max_interaction_rounds`: **5 → 3**
  - `max_scenes_per_chapter`: **6 → 4**
- 效果：每章互动场景的 LLM 调用次数明显减少（例如 2 角色时每场景约 20 次 → 约 12 次），整体生成更快。

### 3. 性能：文风控制器可选关闭 LLM

- **StyleGovernorConfig** 新增 `use_llm_post_process: bool = True`
  - 为 `True`：保持现状，整章再走一遍 LLM 改写
  - 为 `False`：仅做规则清理（抽象词检测等），不再调用 LLM，**每章少 1 次大请求**
- 在 `ChaosEngineConfig.style_governor` 下可配置（未来若从 YAML 读 chaos 配置即可生效）。

### 4. 可配置性：YAML 覆盖 NovelConfig

- **load_setting_from_yaml** 支持可选键 `novel_config`，返回 `novel_config_overrides` 字典
- **main** 中若存在 `novel_config_overrides`，则与 `NovelConfig()` 默认值合并，再用于图构建
- 预设中已加注释示例（`presets/human_ai_love.yaml` 末尾），可按需取消注释并调整 `max_interaction_rounds`、`max_scenes_per_chapter` 等

---

## 三、使用建议

- **追求速度**：保持默认（3 轮互动、4 场景），或将 `style_governor.use_llm_post_process` 设为 `False`（仅规则冷却）。
- **追求质量**：在 YAML 中设置 `novel_config.max_interaction_rounds: 5`、`max_scenes_per_chapter: 6`，并保持文风控制器 LLM 开启。
- **网络不稳**：无需改配置，重试已统一接入；若遇 4xx/5xx 等非连接类错误，可考虑后续对特定异常做有限次重试（当前未做，避免无限重试）。

---

## 四、未改动项（保持简单）

- **并行化**：角色按顺序行动/互动，以保证剧情连贯；未做并行 LLM 调用。
- **提前结束互动**：未实现「若本轮无实质冲突则提前结束」，仍按固定轮数执行。
- **HTTP 超时**：由各 LangChain 模型封装决定，未在应用层统一设置 timeout。
