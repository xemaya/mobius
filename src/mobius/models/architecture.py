"""全书叙事架构模型：蓝图、章节合同、线索账本。"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


ThemeMove = Literal["reinforce", "question", "invert", "synthesize"]
ThesisSignal = Literal["support", "challenge", "suspend"]
ThreadOp = Literal["opening", "deepening", "payoff", "none"]
ThreadStatus = Literal["open", "resolved", "overdue"]


class NovelBlueprint(BaseModel):
    """全书蓝图：统一思想主线和章节职责。"""

    thesis: str = Field(default="", description="主命题")
    antithesis: str = Field(default="", description="反命题")
    synthesis: str = Field(default="", description="最终综合立场")
    chapter_missions: list[str] = Field(
        default_factory=list,
        description="按章节序号对应的章节职责列表",
    )
    character_arcs: dict[str, str] = Field(
        default_factory=dict,
        description="角色哲学弧线：角色名 -> 弧线描述",
    )


class ChapterContract(BaseModel):
    """章节合同：连接全书蓝图与单章执行。"""

    chapter_index: int = Field(description="章节序号")
    chapter_purpose: str = Field(default="", description="本章在全书中的唯一职责")
    theme_move: ThemeMove = Field(
        default="question",
        description="本章对主题的推进方式",
    )
    required_payoffs: list[str] = Field(
        default_factory=list,
        description="本章必须回收的线索",
    )
    new_promises: list[str] = Field(
        default_factory=list,
        description="本章新埋下的承诺",
    )
    philosophical_beat: str = Field(
        default="",
        description="本章明确提出的哲学冲突句",
    )


class ThreadLedgerItem(BaseModel):
    """单条线索账本记录。"""

    thread: str = Field(description="线索/承诺文本")
    source_chapter: int = Field(description="开线章节")
    due_chapter: int = Field(description="最晚回收章节")
    status: ThreadStatus = Field(default="open", description="线索状态")
    resolved_chapter: int | None = Field(default=None, description="回收章节")
    notes: str = Field(default="", description="补充说明")
    updated_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="更新时间",
    )
