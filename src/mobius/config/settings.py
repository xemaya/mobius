"""全局配置。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """LLM 模型配置。"""

    provider: str = Field(
        default="google",
        description="模型提供商: 'google', 'openai', 'minimax', 'anthropic' 等",
    )
    model_name: str = Field(default="gemini-3-flash-preview", description="模型名称")
    temperature: float = Field(default=0.8, description="生成温度")
    max_tokens: int = Field(default=8192, description="最大 token 数")
    api_key: str = Field(
        default="",
        description="模型 API key（可选，优先使用环境变量）",
    )


class NovelConfig(BaseModel):
    """小说生成全局配置。"""

    # ── 模型配置 ──
    director_model: ModelConfig = Field(
        default_factory=ModelConfig,
        description="导演 Agent 使用的模型",
    )
    character_model: ModelConfig = Field(
        default_factory=ModelConfig,
        description="角色 Agent 使用的模型（结构化决策、JSON 输出）",
    )
    narrator_model: ModelConfig = Field(
        default_factory=ModelConfig,
        description="叙事 Agent 使用的模型",
    )
    character_roleplay_model: ModelConfig | None = Field(
        default=None,
        description="角色扮演专用模型（MiniMax M2-her），用于生成对话和内心活动。"
        "若为 None 则使用 character_model 进行所有生成。",
    )
    reviewer_model: ModelConfig | None = Field(
        default=None,
        description="评审 Agent 使用的模型（建议低 temperature 如 0.3）。"
        "若为 None 则使用 director_model。",
    )
    observer_model: ModelConfig | None = Field(
        default=None,
        description="世界观察者使用的模型。若为 None 则使用 director_model。",
    )

    # ── 生成参数 ──
    max_interaction_rounds: int = Field(
        default=3, description="角色互动最大对话轮数（降低可显著提速，默认 3）"
    )
    max_scenes_per_chapter: int = Field(
        default=4, description="每章最大场景数（降低可提速，默认 4）"
    )
    max_memory_entries: int = Field(
        default=50, description="每个角色保留的最大记忆条数（含未压缩的原始记录）"
    )
    memory_compress_threshold: int = Field(
        default=8,
        description="短期记忆超过此数量时，在章节边界触发 LLM 压缩",
    )
    memory_keep_recent: int = Field(
        default=5,
        description="压缩时保留的最近 N 条原始记忆（短期记忆窗口）",
    )
    emotion_decay_rate: float = Field(
        default=0.05,
        description="每个场景结束后情感向中性衰减的幅度",
    )
    chapter_min_words: int = Field(
        default=2500, description="每章最少字数"
    )

    # ── 信念系统 ──
    belief_resistance: dict[str, float] = Field(
        default_factory=lambda: {"core": 0.3, "middle": 0.7, "surface": 1.0},
        description="各层级信念的抗变系数",
    )

    # ── 资源系统 ──
    resource_recovery_rate: dict[str, float] = Field(
        default_factory=lambda: {"time": 20.0, "emotional_energy": 10.0},
        description="每章开始时各资源的自然回复量",
    )

    # ── 功能开关 ──
    enable_desire_proposals: bool = Field(
        default=True, description="是否启用欲望提案系统"
    )
    enable_chapter_review: bool = Field(
        default=True, description="是否启用章节评审"
    )
    enable_world_observer: bool = Field(
        default=True, description="是否启用世界观察者"
    )
    enable_internal_monologue: bool = Field(
        default=True, description="是否启用潜台词/认知黑箱机制"
    )
    enable_environment_engine: bool = Field(
        default=True, description="是否启用环境交互引擎"
    )
    enable_secondary_viewpoints: bool = Field(
        default=True, description="是否启用支线观察者视角"
    )

    # ── 输出配置 ──
    output_dir: str = Field(default="output", description="输出目录")
    language: str = Field(default="zh", description="输出语言")
