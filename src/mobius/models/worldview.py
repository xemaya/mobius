"""世界观相关数据模型。"""

from pydantic import BaseModel, Field


class WorldView(BaseModel):
    """小说的世界观设定。"""

    name: str = Field(description="世界观名称")
    description: str = Field(description="世界观总体描述")
    era: str = Field(default="", description="时代背景")
    geography: str = Field(default="", description="地理环境")
    power_system: str = Field(default="", description="力量体系（如修仙、魔法等）")
    social_structure: str = Field(default="", description="社会结构")
    key_factions: list[str] = Field(default_factory=list, description="关键势力/阵营")
    rules: list[str] = Field(default_factory=list, description="世界运行的核心规则")
    extra: dict[str, str] = Field(default_factory=dict, description="其他自定义世界观设定")


class PlotOutline(BaseModel):
    """剧情大纲。"""

    title: str = Field(description="小说标题")
    genre: str = Field(default="", description="小说类型（如玄幻、都市、科幻）")
    theme: str = Field(default="", description="核心主题")
    synopsis: str = Field(description="一句话概要")
    beginning: str = Field(description="起：故事开端")
    development: str = Field(description="承：情节发展")
    twist: str = Field(description="转：矛盾高潮")
    conclusion: str = Field(description="合：结局走向")
    total_chapters: int = Field(default=10, description="预计总章节数")
    extra_plot_points: list[str] = Field(
        default_factory=list, description="额外的关键情节点"
    )
