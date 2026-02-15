"""OutputManager：负责所有文件产出物的写入与管理。

严格目录结构：
output/<novel_name>/
├── chapters/                  # 逐章 Markdown 文件
├── events/                    # 世界事件 / 冲突 / 欲望提案
├── reviews/                   # 章节评审
├── memory/                    # 记忆蒸馏
├── state/                     # 角色状态快照
├── full_novel.md              # 完整小说
└── metadata.json              # 生成元数据
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from mobius.models.chapter import Chapter

logger = logging.getLogger(__name__)


class OutputManager:
    """管理小说生成的全部产出物。

    每生成一章就立即写入磁盘，同时记录世界事件、评审、记忆等过程数据。
    """

    def __init__(self, output_dir: str | Path, novel_title: str = "untitled"):
        self.root = Path(output_dir)
        self.novel_title = novel_title

        # 子目录
        self.chapters_dir = self.root / "chapters"
        self.events_dir = self.root / "events"
        self.reviews_dir = self.root / "reviews"
        self.memory_dir = self.root / "memory"
        self.state_dir = self.root / "state"

        # 创建所有子目录
        for d in [
            self.chapters_dir,
            self.events_dir,
            self.reviews_dir,
            self.memory_dir,
            self.state_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

        # 初始化元数据
        self._metadata: dict[str, Any] = {
            "novel_title": novel_title,
            "created_at": datetime.now().isoformat(),
            "chapters_generated": 0,
            "total_words": 0,
            "generation_log": [],
            "token_usage": {},
        }
        self._save_metadata()

    # ────────────────────────────────────────────
    # 章节输出
    # ────────────────────────────────────────────

    def save_chapter(self, chapter: Chapter) -> Path:
        """保存单个章节，立即写入磁盘。

        Returns:
            写入的文件路径。
        """
        filename = f"chapter_{chapter.chapter_index:03d}.md"
        filepath = self.chapters_dir / filename

        content = f"# 第{chapter.chapter_index}章 {chapter.title}\n\n{chapter.content}\n"
        filepath.write_text(content, encoding="utf-8")

        # 更新元数据
        self._metadata["chapters_generated"] = max(
            self._metadata["chapters_generated"], chapter.chapter_index
        )
        self._metadata["total_words"] += chapter.word_count
        self._metadata["generation_log"].append({
            "type": "chapter",
            "chapter_index": chapter.chapter_index,
            "title": chapter.title,
            "word_count": chapter.word_count,
            "timestamp": datetime.now().isoformat(),
        })
        self._save_metadata()

        logger.info(
            "📄 章节已写入磁盘: %s (%d字)", filepath.name, chapter.word_count
        )
        return filepath

    def save_full_novel(self, chapters: list[Chapter]) -> Path:
        """将所有章节合并为完整小说。"""
        filepath = self.root / "full_novel.md"

        parts = [f"# {self.novel_title}\n"]
        for chapter in sorted(chapters, key=lambda c: c.chapter_index):
            parts.append(f"\n\n---\n\n## 第{chapter.chapter_index}章 {chapter.title}\n\n")
            parts.append(chapter.content)

        filepath.write_text("".join(parts), encoding="utf-8")
        self._metadata["full_novel_saved"] = True
        self._save_metadata()

        logger.info("📖 完整小说已保存: %s", filepath)
        return filepath

    # ────────────────────────────────────────────
    # 世界事件记录
    # ────────────────────────────────────────────

    def save_chapter_events(
        self,
        chapter_index: int,
        world_events: list[Any] | None = None,
        desire_proposals: list[Any] | None = None,
    ) -> None:
        """保存本章相关的世界事件和欲望提案。"""
        prefix = f"chapter_{chapter_index:03d}"

        if world_events:
            events_data = []
            for evt in world_events:
                if hasattr(evt, "model_dump"):
                    events_data.append(evt.model_dump())
                elif isinstance(evt, dict):
                    events_data.append(evt)
            self._write_json(
                self.events_dir / f"{prefix}_events.json",
                {"chapter_index": chapter_index, "world_events": events_data},
            )

        if desire_proposals:
            proposals_data = []
            for p in desire_proposals:
                if hasattr(p, "model_dump"):
                    proposals_data.append(p.model_dump())
                elif isinstance(p, dict):
                    proposals_data.append(p)
            self._write_json(
                self.events_dir / f"{prefix}_desires.json",
                {"chapter_index": chapter_index, "desire_proposals": proposals_data},
            )

        self._metadata["generation_log"].append({
            "type": "events",
            "chapter_index": chapter_index,
            "event_count": len(world_events) if world_events else 0,
            "desire_count": len(desire_proposals) if desire_proposals else 0,
            "timestamp": datetime.now().isoformat(),
        })
        self._save_metadata()

    # ────────────────────────────────────────────
    # 评审记录
    # ────────────────────────────────────────────

    def save_chapter_review(self, chapter_index: int, review: Any) -> None:
        """保存章节评审结果。"""
        prefix = f"chapter_{chapter_index:03d}"
        review_data = review.model_dump() if hasattr(review, "model_dump") else review
        self._write_json(
            self.reviews_dir / f"{prefix}_review.json",
            {"chapter_index": chapter_index, "review": review_data},
        )

    # ────────────────────────────────────────────
    # 记忆蒸馏记录
    # ────────────────────────────────────────────

    def save_chapter_memory(
        self, chapter_index: int, memory_summaries: list[Any]
    ) -> None:
        """保存本章记忆蒸馏结果。"""
        prefix = f"chapter_{chapter_index:03d}"
        summaries_data = []
        for s in memory_summaries:
            if hasattr(s, "model_dump"):
                summaries_data.append(s.model_dump())
            elif isinstance(s, dict):
                summaries_data.append(s)
        self._write_json(
            self.memory_dir / f"{prefix}_memory.json",
            {"chapter_index": chapter_index, "memory_summaries": summaries_data},
        )

    # ────────────────────────────────────────────
    # 角色状态快照
    # ────────────────────────────────────────────

    def save_character_states(
        self, chapter_index: int, character_states: dict[str, Any]
    ) -> None:
        """保存角色状态快照。"""
        prefix = f"chapter_{chapter_index:03d}"
        states_data = {}
        for name, cs in character_states.items():
            if hasattr(cs, "model_dump"):
                states_data[name] = cs.model_dump()
            elif isinstance(cs, dict):
                states_data[name] = cs
        self._write_json(
            self.state_dir / f"{prefix}_state.json",
            {"chapter_index": chapter_index, "character_states": states_data},
        )

    # ────────────────────────────────────────────
    # 内部工具
    # ────────────────────────────────────────────

    def _write_json(self, filepath: Path, data: Any) -> None:
        """写入 JSON 文件。"""
        filepath.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    def _save_metadata(self) -> None:
        """更新 metadata.json。"""
        # 更新token统计信息
        try:
            from mobius.utils.token_tracker import TokenTracker
            tracker = TokenTracker.get_instance()
            self._metadata["token_usage"] = tracker.to_dict()
        except ImportError:
            # 如果token_tracker模块不存在，跳过
            pass

        self._metadata["updated_at"] = datetime.now().isoformat()
        self._write_json(self.root / "metadata.json", self._metadata)

    def get_token_summary(self) -> dict[str, Any]:
        """获取token消耗摘要。"""
        try:
            from mobius.utils.token_tracker import get_token_stats
            stats = get_token_stats()
            return {
                "total_calls": stats.total_calls,
                "total_tokens": stats.total_tokens,
                "total_input_tokens": stats.total_input_tokens,
                "total_output_tokens": stats.total_output_tokens,
                "total_duration_ms": stats.total_duration_ms,
                "by_operation": {
                    op: {
                        "calls": s.total_calls,
                        "tokens": s.total_tokens,
                        "avg_duration_ms": s.total_duration_ms / max(s.total_calls, 1),
                    }
                    for op, s in stats.by_operation.items()
                },
                "by_model": {
                    model: {
                        "calls": s.total_calls,
                        "tokens": s.total_tokens,
                    }
                    for model, s in stats.by_model.items()
                },
            }
        except ImportError:
            return {"error": "Token tracking not available"}
