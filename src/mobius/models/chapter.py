"""章节和场景相关数据模型。"""

from typing import Literal
from pydantic import BaseModel, Field


# 合法的场景类型
VALID_SCENE_TYPES = {
    "single_action",   # 单角色行动
    "interaction",      # 多角色互动
    "narration",        # 旁白/环境描写
    "silence",          # 沉默场景：无对话，纯环境+内心流动，300-500字呼吸间隙
    "daily",            # 日常场景：琐碎生活细节，无宏大主题，展示角色的普通一面
}


class Scene(BaseModel):
    """一个场景的定义。"""

    scene_id: str = Field(description="场景唯一标识")
    title: str = Field(default="", description="场景标题")
    description: str = Field(default="", description="场景描述/目标")
    location: str = Field(default="", description="场景发生地点")
    participating_characters: list[str] = Field(description="参与角色名称列表")
    scene_type: str = Field(
        default="single_action",
        description=(
            "场景类型: 'single_action'(单角色行动), 'interaction'(多角色互动), "
            "'narration'(旁白/环境), 'silence'(沉默场景), 'daily'(日常场景)"
        ),
    )
    mood: str = Field(default="", description="场景氛围/基调")
    objectives: list[str] = Field(
        default_factory=list, description="场景需要达成的剧情目标"
    )
    information_revelation: str = Field(
        default="",
        description="本场景必须向读者揭示的背景信息或逻辑解释（信息流管理）",
    )
    causal_from: str = Field(
        default="",
        description="本场景由上一场景什么因果触发",
    )
    causal_to: str = Field(
        default="",
        description="本场景将导致下一场景什么后果",
    )
    thesis_signal: Literal["support", "challenge", "suspend"] = Field(
        default="suspend",
        description="本场景对主命题的立场：support/challenge/suspend",
    )
    thread_ops: list[Literal["opening", "deepening", "payoff", "none"]] = Field(
        default_factory=list,
        description="本场景对线索执行的操作",
    )
    director_notes: str = Field(default="", description="导演对场景的额外指示")


class ChapterPlan(BaseModel):
    """章节规划（导演输出）。"""

    chapter_index: int = Field(description="章节序号（从1开始）")
    title: str = Field(description="章节标题")
    summary: str = Field(description="章节内容概要")
    chapter_purpose: str = Field(default="", description="本章在全书中的唯一职责")
    theme_move: Literal["reinforce", "question", "invert", "synthesize"] = Field(
        default="question",
        description="本章对全书主题的推进方式",
    )
    required_payoffs: list[str] = Field(
        default_factory=list,
        description="本章必须回收的线索",
    )
    new_promises: list[str] = Field(
        default_factory=list,
        description="本章新埋设的承诺",
    )
    philosophical_beat: str = Field(
        default="",
        description="本章要呈现的核心哲学冲突句",
    )
    key_events: list[str] = Field(description="本章关键事件列表")
    participating_characters: list[str] = Field(description="本章参与的角色列表")
    emotional_tone: str = Field(default="", description="本章情感基调")
    scenes: list[Scene] = Field(default_factory=list, description="场景列表")


class Chapter(BaseModel):
    """完成的章节。"""

    chapter_index: int = Field(description="章节序号")
    title: str = Field(description="章节标题")
    content: str = Field(description="章节正文内容")
    summary: str = Field(default="", description="章节摘要（供后续章节参考）")
    word_count: int = Field(default=0, description="字数统计")


class ChapterOutline(BaseModel):
    """章节概要（Phase1 产物）。"""

    chapter_index: int = Field(description="章节序号（从1开始）")
    title: str = Field(description="章节标题")
    purpose: str = Field(description="本章在全书中的职责")
    core_plot: str = Field(description="本章300-500字核心剧情概要")
    irreversible_change: str = Field(description="本章不可逆变化")
    character_arc_delta: str = Field(
        default="",
        description="本章角色弧线变化摘要",
    )
    must_payoffs: list[str] = Field(
        default_factory=list,
        description="本章必须兑现/推进的线索",
    )
    new_promises: list[str] = Field(
        default_factory=list,
        description="本章新埋设承诺",
    )


class SettingEntity(BaseModel):
    """设定集中可复用的实体（人物/组织/地点/物件）。"""

    name: str = Field(description="实体名称")
    category: Literal["character", "organization", "location", "artifact", "concept"] = Field(
        description="实体类型",
    )
    description: str = Field(description="实体描述")
    constraints: list[str] = Field(default_factory=list, description="硬约束/使用边界")
    unresolved_questions: list[str] = Field(
        default_factory=list,
        description="待补完问题",
    )


class SettingCharacterProfile(BaseModel):
    """设定集中的角色细化档案。"""

    name: str = Field(description="角色名")
    role: str = Field(default="", description="角色定位")
    personality_traits: list[str] = Field(default_factory=list, description="性格特征")
    inner_thinking_habits: list[str] = Field(
        default_factory=list,
        description="内心思考习惯",
    )
    outfit_style: str = Field(default="", description="衣着与外观特征")
    first_appearance_chapter: int = Field(default=1, description="首次正式出场章节")
    first_appearance_moment: str = Field(default="", description="首次出场时机描述")
    first_appearance_constraints: list[str] = Field(
        default_factory=list,
        description="首次出场前必须完成的铺垫",
    )
    arc_seed: str = Field(default="", description="角色弧线种子")


class SettingRule(BaseModel):
    """世界规则细化条目。"""

    rule_id: str = Field(description="规则编号")
    statement: str = Field(description="规则描述")
    rationale: str = Field(default="", description="规则成立原因")
    forbidden_cases: list[str] = Field(default_factory=list, description="禁止叙述案例")


class TimelineEvent(BaseModel):
    """关键事件时间线条目。"""

    event_id: str = Field(description="事件编号")
    title: str = Field(description="事件标题")
    description: str = Field(description="事件描述")
    chapter_hint: int = Field(default=0, description="建议发生章节（0 表示待定）")
    dependencies: list[str] = Field(default_factory=list, description="前置依赖事件")
    irreversible_impact: str = Field(default="", description="不可逆影响")


class SettingPack(BaseModel):
    """结构化设定集（第一层产物）。"""

    title: str = Field(description="小说标题")
    theme: str = Field(description="主旨命题")
    theme_longform: str = Field(
        default="",
        description="主旨命题深描（建议千字级）",
    )
    worldview_longform: str = Field(
        default="",
        description="世界观深描（建议千字级）",
    )
    worldview_rules: list[str] = Field(default_factory=list, description="世界规则")
    detailed_rules: list[SettingRule] = Field(default_factory=list, description="细化世界规则")
    core_events_timeline: list[str] = Field(
        default_factory=list,
        description="关键事件时间线",
    )
    timeline_events: list[TimelineEvent] = Field(default_factory=list, description="结构化时间线")
    entities: list[SettingEntity] = Field(default_factory=list, description="核心实体")
    characters: list[SettingCharacterProfile] = Field(default_factory=list, description="角色细化档案")
    organizations: list[SettingEntity] = Field(default_factory=list, description="组织设定")
    items: list[SettingEntity] = Field(default_factory=list, description="关键物品/技术设定")
    missing_items: list[str] = Field(default_factory=list, description="缺失项与补完建议")
    author_notes: str = Field(default="", description="设定补充备注")


class StoryboardScene(BaseModel):
    """章节分镜中的单个场景。"""

    scene_index: int = Field(description="章节内场景序号（从1开始）")
    scene_type: Literal["plot_progress", "interaction", "narration", "daily", "silence"] = Field(
        description="场景类型",
    )
    title: str = Field(description="场景标题")
    objective: str = Field(description="场景目标")
    conflict_type: str = Field(default="", description="冲突类型")
    location: str = Field(default="", description="场景地点")
    participating_characters: list[str] = Field(default_factory=list, description="参与角色")
    causal_from: str = Field(default="", description="场景因果起点")
    causal_to: str = Field(default="", description="场景因果终点")
    info_gain: str = Field(default="", description="本场景信息增量")
    style_notes: str = Field(default="", description="风格注记（环境/心理/节奏）")
    expected_beats: list[str] = Field(default_factory=list, description="关键节拍")


class ChapterStoryboard(BaseModel):
    """章节分镜（第三层产物）。"""

    chapter_index: int = Field(description="章节序号")
    title: str = Field(description="章节标题")
    purpose: str = Field(description="章节职责")
    irreversible_change: str = Field(description="本章不可逆变化")
    must_payoffs: list[str] = Field(default_factory=list, description="本章必兑现线索")
    scenes: list[StoryboardScene] = Field(default_factory=list, description="场景分镜列表")
