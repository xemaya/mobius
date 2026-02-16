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

from mobius.models.chapter import Chapter, ChapterOutline, ChapterStoryboard, SettingPack

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
        self.outlines_dir = self.root / "outlines"
        self.storyboards_dir = self.root / "storyboards"
        self.setting_pack_dir = self.root / "setting_pack"

        # 创建所有子目录
        for d in [
            self.chapters_dir,
            self.events_dir,
            self.reviews_dir,
            self.memory_dir,
            self.state_dir,
            self.outlines_dir,
            self.storyboards_dir,
            self.setting_pack_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

        # 初始化或加载元数据
        metadata_file = self.root / "metadata.json"
        if metadata_file.exists():
            try:
                self._metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
                self._metadata.setdefault("outline", {
                    "generated": False,
                    "approved": False,
                    "generated_chapters": 0,
                })
                self._metadata.setdefault("setting", {"generated": False, "approved": False})
                self._metadata.setdefault("storyboard", {
                    "generated": False,
                    "approved": False,
                    "generated_chapters": 0,
                })
                self._metadata["novel_title"] = self._metadata.get("novel_title") or novel_title
            except Exception:
                self._metadata = {
                    "novel_title": novel_title,
                    "created_at": datetime.now().isoformat(),
                    "chapters_generated": 0,
                    "total_words": 0,
                    "generation_log": [],
                    "token_usage": {},
                    "outline": {
                        "generated": False,
                        "approved": False,
                        "generated_chapters": 0,
                    },
                    "setting": {"generated": False, "approved": False},
                    "storyboard": {
                        "generated": False,
                        "approved": False,
                        "generated_chapters": 0,
                    },
                }
        else:
            self._metadata = {
                "novel_title": novel_title,
                "created_at": datetime.now().isoformat(),
                "chapters_generated": 0,
                "total_words": 0,
                "generation_log": [],
                "token_usage": {},
                "outline": {
                    "generated": False,
                    "approved": False,
                    "generated_chapters": 0,
                },
                "setting": {"generated": False, "approved": False},
                "storyboard": {
                    "generated": False,
                    "approved": False,
                    "generated_chapters": 0,
                },
            }
        self._save_metadata()

    # ────────────────────────────────────────────
    # 章节输出
    # ────────────────────────────────────────────

    def save_outline(self, outline: ChapterOutline) -> Path:
        """保存单章概要。"""
        prefix = f"chapter_{outline.chapter_index:03d}"
        json_path = self.outlines_dir / f"{prefix}_outline.json"
        md_path = self.outlines_dir / f"{prefix}.md"
        self._write_json(json_path, outline.model_dump())

        md_text = (
            f"# 第{outline.chapter_index}章 {outline.title}\n\n"
            f"## 章节职责\n{outline.purpose}\n\n"
            f"## 核心剧情概要\n{outline.core_plot}\n\n"
            f"## 不可逆变化\n{outline.irreversible_change}\n\n"
            f"## 角色弧线变化\n{outline.character_arc_delta or '（无）'}\n\n"
            f"## 必兑现线索\n- " + "\n- ".join(outline.must_payoffs or ["（无）"]) + "\n\n"
            f"## 新承诺\n- " + "\n- ".join(outline.new_promises or ["（无）"]) + "\n"
        )
        md_path.write_text(md_text, encoding="utf-8")
        return json_path

    def save_full_outline(self, outlines: list[ChapterOutline]) -> Path:
        """保存全书概要汇总。"""
        path = self.root / "full_outline.md"
        sorted_outlines = sorted(outlines, key=lambda o: o.chapter_index)
        parts = [f"# {self.novel_title} - 全书章节概要\n"]
        for item in sorted_outlines:
            parts.append(
                "\n\n---\n\n"
                f"## 第{item.chapter_index}章 {item.title}\n\n"
                f"**章节职责**：{item.purpose}\n\n"
                f"**核心剧情**：{item.core_plot}\n\n"
                f"**不可逆变化**：{item.irreversible_change}\n\n"
                f"**角色弧线变化**：{item.character_arc_delta or '（无）'}\n\n"
                f"**必兑现线索**：{'、'.join(item.must_payoffs) if item.must_payoffs else '（无）'}\n\n"
                f"**新承诺**：{'、'.join(item.new_promises) if item.new_promises else '（无）'}\n"
            )

        path.write_text("".join(parts), encoding="utf-8")
        self._metadata["outline"]["generated"] = True
        self._metadata["outline"]["generated_chapters"] = len(sorted_outlines)
        self._save_metadata()
        return path

    def save_setting_pack(self, setting_pack: SettingPack) -> Path:
        """保存结构化设定集。"""
        json_path = self.setting_pack_dir / "setting_pack.json"
        md_path = self.setting_pack_dir / "setting_pack.md"
        self._write_json(json_path, setting_pack.model_dump())

        lines = [
            f"# {setting_pack.title} - 结构化设定集",
            "",
            f"## 主旨命题",
            setting_pack.theme,
            "",
            "## 主旨深描",
            setting_pack.theme_longform or "（无）",
            "",
            "## 世界观深描",
            setting_pack.worldview_longform or "（无）",
            "",
            "## 世界规则",
            *[f"- {r}" for r in (setting_pack.worldview_rules or ["（无）"])],
            "",
            "## 细化世界规则",
        ]
        if setting_pack.detailed_rules:
            for rule in setting_pack.detailed_rules:
                lines.extend([
                    f"### {rule.rule_id} {rule.statement}",
                    f"- 成立原因: {rule.rationale or '（无）'}",
                    f"- 禁止案例: {'；'.join(rule.forbidden_cases) if rule.forbidden_cases else '（无）'}",
                    "",
                ])
        else:
            lines.extend(["- （无）", ""])
        lines.extend([
            "## 关键事件时间线",
            *[f"- {evt}" for evt in (setting_pack.core_events_timeline or ["（无）"])],
            "",
            "## 结构化时间线",
        ])
        if setting_pack.timeline_events:
            for event in setting_pack.timeline_events:
                lines.extend([
                    f"### {event.event_id} {event.title}",
                    event.description,
                    f"- 建议章节: {event.chapter_hint}",
                    f"- 前置依赖: {'、'.join(event.dependencies) if event.dependencies else '（无）'}",
                    f"- 不可逆影响: {event.irreversible_impact or '（无）'}",
                    "",
                ])
        else:
            lines.extend(["- （无）", ""])
        lines.extend([
            "## 人物细化档案",
        ])
        if setting_pack.characters:
            for c in setting_pack.characters:
                lines.extend([
                    f"### {c.name} ({c.role or '未标注'})",
                    f"- 性格: {'、'.join(c.personality_traits) if c.personality_traits else '（无）'}",
                    f"- 思考习惯: {'、'.join(c.inner_thinking_habits) if c.inner_thinking_habits else '（无）'}",
                    f"- 衣着打扮: {c.outfit_style or '（无）'}",
                    f"- 首次出场章节: 第{c.first_appearance_chapter}章",
                    f"- 首次出场时机: {c.first_appearance_moment or '（无）'}",
                    f"- 出场约束: {'；'.join(c.first_appearance_constraints) if c.first_appearance_constraints else '（无）'}",
                    f"- 弧线种子: {c.arc_seed or '（无）'}",
                    "",
                ])
        else:
            lines.extend(["- （无）", ""])
        lines.extend([
            "## 组织设定",
        ])
        if setting_pack.organizations:
            for org in setting_pack.organizations:
                lines.extend([
                    f"### {org.name}",
                    org.description,
                    f"- 约束: {'；'.join(org.constraints) if org.constraints else '（无）'}",
                    f"- 待补完: {'；'.join(org.unresolved_questions) if org.unresolved_questions else '（无）'}",
                    "",
                ])
        else:
            lines.extend(["- （无）", ""])
        lines.extend([
            "## 物品与技术设定",
        ])
        if setting_pack.items:
            for it in setting_pack.items:
                lines.extend([
                    f"### {it.name}",
                    it.description,
                    f"- 约束: {'；'.join(it.constraints) if it.constraints else '（无）'}",
                    f"- 待补完: {'；'.join(it.unresolved_questions) if it.unresolved_questions else '（无）'}",
                    "",
                ])
        else:
            lines.extend(["- （无）", ""])
        lines.extend([
            "## 核心实体",
        ])
        if setting_pack.entities:
            for entity in setting_pack.entities:
                lines.extend([
                    f"### {entity.name} ({entity.category})",
                    entity.description,
                    f"- 约束: {'；'.join(entity.constraints) if entity.constraints else '（无）'}",
                    f"- 待补完: {'；'.join(entity.unresolved_questions) if entity.unresolved_questions else '（无）'}",
                    "",
                ])
        else:
            lines.append("- （无）")
            lines.append("")
        lines.extend([
            "## 缺失项补完建议",
            *[f"- {x}" for x in (setting_pack.missing_items or ["（无）"])],
            "",
            "## 备注",
            setting_pack.author_notes or "（无）",
            "",
        ])
        md_path.write_text("\n".join(lines), encoding="utf-8")

        (self.setting_pack_dir / "theme.md").write_text(
            f"# 主旨命题\n\n{setting_pack.theme}\n\n## 深描\n\n{setting_pack.theme_longform or '（无）'}\n",
            encoding="utf-8",
        )
        (self.setting_pack_dir / "worldview.md").write_text(
            "# 世界观深描\n\n"
            + (setting_pack.worldview_longform or "（无）")
            + "\n\n## 世界规则\n\n"
            + "\n".join(f"- {x}" for x in (setting_pack.worldview_rules or ["（无）"])),
            encoding="utf-8",
        )
        (self.setting_pack_dir / "timeline.md").write_text(
            "# 关键事件时间线\n\n"
            + "\n".join(f"- {x}" for x in (setting_pack.core_events_timeline or ["（无）"])),
            encoding="utf-8",
        )
        if setting_pack.characters:
            char_lines = ["# 人物设定", ""]
            for c in setting_pack.characters:
                char_lines.extend([
                    f"## {c.name}",
                    f"- 角色定位: {c.role or '（无）'}",
                    f"- 性格特征: {'、'.join(c.personality_traits) if c.personality_traits else '（无）'}",
                    f"- 思考习惯: {'、'.join(c.inner_thinking_habits) if c.inner_thinking_habits else '（无）'}",
                    f"- 衣着打扮: {c.outfit_style or '（无）'}",
                    f"- 首次出场章节: 第{c.first_appearance_chapter}章",
                    f"- 首次出场时机: {c.first_appearance_moment or '（无）'}",
                    "",
                ])
            (self.setting_pack_dir / "characters.md").write_text("\n".join(char_lines), encoding="utf-8")
        if setting_pack.organizations:
            org_lines = ["# 组织设定", ""]
            for o in setting_pack.organizations:
                org_lines.extend([f"## {o.name}", o.description, ""])
            (self.setting_pack_dir / "organizations.md").write_text("\n".join(org_lines), encoding="utf-8")
        if setting_pack.items:
            item_lines = ["# 物品与技术设定", ""]
            for i in setting_pack.items:
                item_lines.extend([f"## {i.name}", i.description, ""])
            (self.setting_pack_dir / "items.md").write_text("\n".join(item_lines), encoding="utf-8")
        self._metadata["setting"]["generated"] = True
        self._save_metadata()
        return md_path

    def save_storyboard(self, storyboard: ChapterStoryboard) -> Path:
        """保存单章分镜。"""
        prefix = f"chapter_{storyboard.chapter_index:03d}"
        json_path = self.storyboards_dir / f"{prefix}_storyboard.json"
        md_path = self.storyboards_dir / f"{prefix}.md"
        self._write_json(json_path, storyboard.model_dump())

        md_lines = [
            f"# 第{storyboard.chapter_index}章 {storyboard.title} - 分镜",
            "",
            f"## 章节职责",
            storyboard.purpose,
            "",
            f"## 不可逆变化",
            storyboard.irreversible_change,
            "",
            f"## 必兑现线索",
            *[f"- {x}" for x in (storyboard.must_payoffs or ["（无）"])],
            "",
            "## 场景列表",
        ]
        for scene in storyboard.scenes:
            md_lines.extend([
                f"### 场景{scene.scene_index}: {scene.title} ({scene.scene_type})",
                f"- 目标: {scene.objective}",
                f"- 冲突: {scene.conflict_type or '（无）'}",
                f"- 地点: {scene.location or '（未指定）'}",
                f"- 角色: {'、'.join(scene.participating_characters) if scene.participating_characters else '（无）'}",
                f"- 因果: {scene.causal_from or '（无）'} -> {scene.causal_to or '（无）'}",
                f"- 信息增量: {scene.info_gain or '（无）'}",
                f"- 风格注记: {scene.style_notes or '（无）'}",
                f"- 节拍: {'、'.join(scene.expected_beats) if scene.expected_beats else '（无）'}",
                "",
            ])
        md_path.write_text("\n".join(md_lines), encoding="utf-8")
        return json_path

    def save_full_storyboards(self, storyboards: list[ChapterStoryboard]) -> Path:
        """保存全书分镜汇总。"""
        path = self.root / "full_storyboard.md"
        ordered = sorted(storyboards, key=lambda x: x.chapter_index)
        parts = [f"# {self.novel_title} - 全书分镜\n"]
        for sb in ordered:
            parts.append(
                "\n\n---\n\n"
                f"## 第{sb.chapter_index}章 {sb.title}\n\n"
                f"职责：{sb.purpose}\n\n"
                f"不可逆变化：{sb.irreversible_change}\n\n"
                f"必兑现线索：{'、'.join(sb.must_payoffs) if sb.must_payoffs else '（无）'}\n\n"
                f"场景数：{len(sb.scenes)}\n"
            )
        path.write_text("".join(parts), encoding="utf-8")
        self._metadata["storyboard"]["generated"] = True
        self._metadata["storyboard"]["generated_chapters"] = len(ordered)
        self._save_metadata()
        return path

    def mark_phase_approved(self, phase: str, approver: str = "manual") -> Path:
        """标记指定阶段已通过人工确认。"""
        approval_file = self.root / f"{phase}_approval.json"
        payload = {
            "phase": phase,
            "approved": True,
            "approved_by": approver,
            "approved_at": datetime.now().isoformat(),
        }
        self._write_json(approval_file, payload)
        if phase in self._metadata and isinstance(self._metadata[phase], dict):
            self._metadata[phase]["approved"] = True
        self._save_metadata()
        return approval_file

    def is_phase_approved(self, phase: str) -> bool:
        """检查指定阶段是否已人工确认。"""
        approval_file = self.root / f"{phase}_approval.json"
        if not approval_file.exists():
            return False
        try:
            data = json.loads(approval_file.read_text(encoding="utf-8"))
            return bool(data.get("approved"))
        except Exception:
            return False

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
        """将所有章节合并为完整小说。

        自动按 chapter_index 去重，只保留每章的最后一个版本。
        """
        filepath = self.root / "full_novel.md"

        # 按 chapter_index 去重：保留每章最后出现的版本
        seen: dict[int, Chapter] = {}
        for ch in chapters:
            seen[ch.chapter_index] = ch
        unique_chapters = sorted(seen.values(), key=lambda c: c.chapter_index)

        parts = [f"# {self.novel_title}\n"]
        for chapter in unique_chapters:
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
