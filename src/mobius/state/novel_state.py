"""全局小说状态定义（LangGraph StateGraph 状态）。"""

from __future__ import annotations

from operator import add
from typing import Annotated, Any

from typing_extensions import TypedDict

from mobius.models.chapter import Chapter, ChapterPlan, Scene
from mobius.models.character import CharacterAction, CharacterDynamicState, CharacterProfile
from mobius.models.desire import DesireProposal
from mobius.models.environment import EnvironmentState
from mobius.models.review import ChapterReview, StructuredMemorySummary, WorldEvent
from mobius.models.triggers import TriggeredEvent
from mobius.models.viewpoint import SecondaryViewpoint, ViewpointFragment
from mobius.models.worldview import PlotOutline, WorldView


class NovelState(TypedDict, total=False):
    """LangGraph 主图的全局状态。

    使用 total=False 使所有字段可选，便于在节点中做部分更新。
    """

    # ── 设定（初始化后不变）──
    worldview: WorldView
    plot_outline: PlotOutline
    character_profiles: dict[str, CharacterProfile]
    theme: str  # 核心主题（从 plot_outline.theme 提取）

    # ── 章节管理 ──
    total_chapters: int
    current_chapter_index: int
    chapter_plan: ChapterPlan | None
    chapters: Annotated[list[Chapter], add]

    # ── 场景管理 ──
    scene_queue: list[Scene]
    current_scene: Scene | None

    # ── 角色动态状态 ──
    character_states: dict[str, CharacterDynamicState]

    # ── 场景执行结果 ──
    scene_actions: Annotated[list[CharacterAction], add]
    narrative_buffer: Annotated[list[str], add]

    # ── 触发事件 ──
    triggered_events: list[TriggeredEvent]

    # ── 欲望驱动系统 ──
    desire_proposals: list[DesireProposal]

    # ── 世界模拟 ──
    world_events: list[WorldEvent]

    # ── 环境交互引擎（物理状态总线）──
    environment: EnvironmentState

    # ── 支线观察者 ──
    secondary_viewpoints: list[SecondaryViewpoint]
    viewpoint_fragments: list[ViewpointFragment]

    # ── 张力与评审 ──
    tension_curve: list[float]
    chapter_reviews: list[ChapterReview]

    # ── 记忆蒸馏 ──
    memory_summaries: list[StructuredMemorySummary]

    # ── 控制流 ──
    next_action: str  # 路由控制

    # ── 元数据 ──
    error: str | None
    metadata: dict[str, Any]
