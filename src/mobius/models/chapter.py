"""章节和场景相关数据模型。"""

from pydantic import BaseModel, Field


class Scene(BaseModel):
    """一个场景的定义。"""

    scene_id: str = Field(description="场景唯一标识")
    title: str = Field(default="", description="场景标题")
    description: str = Field(description="场景描述/目标")
    location: str = Field(default="", description="场景发生地点")
    participating_characters: list[str] = Field(description="参与角色名称列表")
    scene_type: str = Field(
        default="single_action",
        description="场景类型: 'single_action'(单角色行动), 'interaction'(多角色互动), 'narration'(旁白/环境)",
    )
    mood: str = Field(default="", description="场景氛围/基调")
    objectives: list[str] = Field(
        default_factory=list, description="场景需要达成的剧情目标"
    )
    director_notes: str = Field(default="", description="导演对场景的额外指示")


class ChapterPlan(BaseModel):
    """章节规划（导演输出）。"""

    chapter_index: int = Field(description="章节序号（从1开始）")
    title: str = Field(description="章节标题")
    summary: str = Field(description="章节内容概要")
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
