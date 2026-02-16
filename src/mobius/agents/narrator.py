"""叙事生成 Agent：将角色行动和对话编译为流畅的小说文本。

升级功能：
- 潜台词文学化处理：利用 internal_monologue 创造言不由衷的深度
- 支线视角片段融合：在主线叙事中穿插非主角视角的观察
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from mobius.agents.utils import extract_response_text, invoke_with_retry
from mobius.engine.chaos_engine import ChaosEngine
from mobius.models.chapter import Chapter, ChapterStoryboard, Scene, SettingPack
from mobius.models.character import CharacterAction
from mobius.models.viewpoint import ViewpointFragment
from mobius.prompts import load_prompt
from mobius.state.novel_state import NovelState

logger = logging.getLogger(__name__)

# 从外部文件加载提示词
NARRATOR_SYSTEM_PROMPT = load_prompt("narrator_system")
NARRATION_ONLY_PROMPT = load_prompt("narrator_narration_only")
COMPILE_INSTRUCTIONS = load_prompt("narrator_compile_instructions")
EXPAND_OUTLINE_SYSTEM_PROMPT = load_prompt("narrator_expand_outline_system")
EXPAND_OUTLINE_INSTRUCTIONS = load_prompt("narrator_expand_outline_instructions")


def _extract_anchor_terms(text: str, limit: int = 4) -> list[str]:
    """为分镜构建可落地的锚词（用于正文命中）。"""
    if not text:
        return []
    terms: list[str] = []
    for t in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{3,}", text):
        t = t.strip()
        if len(t) < 2:
            continue
        if t not in terms:
            terms.append(t)
        if len(terms) >= limit:
            break
    return terms


def _extract_keywords_like_gate(text: str, limit: int = 12) -> list[str]:
    """与质量闸门一致的关键词提取策略（用于设定锚词）。"""
    if not text:
        return []
    tokens = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{3,}", text)
    stop = {"必须", "本章", "剧情", "角色", "变化", "系统", "章节", "推进", "线索"}
    uniq: list[str] = []
    for t in tokens:
        t = t.strip()
        if not t or t in stop:
            continue
        if t not in uniq:
            uniq.append(t)
        if len(uniq) >= limit:
            break
    return uniq


def _extract_setting_anchor_terms(setting_pack: SettingPack | None, limit: int = 10) -> list[str]:
    """从设定集中提取可命中的短锚词。"""
    if not setting_pack:
        return []
    seed_parts: list[str] = [setting_pack.theme or ""]
    seed_parts.extend(setting_pack.worldview_rules[:8] if setting_pack.worldview_rules else [])
    seed_parts.extend(setting_pack.core_events_timeline[:8] if setting_pack.core_events_timeline else [])
    merged = " ".join(str(x) for x in seed_parts if str(x).strip())
    return _extract_keywords_like_gate(merged, limit=limit)


def create_narration_node(model: BaseChatModel) -> Callable[[NovelState], dict[str, Any]]:
    """创建旁白/环境描写节点。用于不需要角色行动的纯叙事场景。"""

    def narration_node(state: NovelState) -> dict[str, Any]:
        scene: Scene | None = state.get("current_scene")
        if not scene:
            return {"next_action": "direct_scene"}

        chapter_plan = state.get("chapter_plan")
        chapter_title = chapter_plan.title if chapter_plan else ""

        # 环境状态注入
        env_ctx = ""
        env_state = state.get("environment")
        if env_state and env_state.variables:
            env_lines = [
                f"- {v.description or v.name}: {v.value:.0f}"
                for v in env_state.variables.values()
            ]
            env_ctx = f"\n\n环境氛围参考（不要直接提及数值）：\n" + "\n".join(env_lines)

        user_prompt = f"""# 章节：{chapter_title}

# 场景：{scene.title or scene.scene_id}
{scene.description}

地点：{scene.location or '未指定'}
氛围：{scene.mood or '未指定'}
导演指示：{scene.director_notes or '无'}
信息揭露目标：{scene.information_revelation or '无（本场景无需揭示新信息）'}{env_ctx}

请撰写 200-400 字的环境描写或旁白叙述。如果有信息揭露目标，必须将其自然融入叙事中。"""

        messages = [
            SystemMessage(content=NARRATION_ONLY_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        try:
            response = invoke_with_retry(model, messages, operation_name="narration")
            narration = extract_response_text(response).strip()
            logger.info("旁白生成完成: %d 字", len(narration))
            return {
                "narrative_buffer": [narration],
                "next_action": "update_state",
            }
        except Exception as e:
            logger.error("旁白生成失败: %s", e)
            return {
                "narrative_buffer": [f"（{scene.description}）"],
                "next_action": "update_state",
            }

    return narration_node


def create_compile_chapter_node(
    model: BaseChatModel,
    chaos_engine: ChaosEngine | None = None,
    chapter_min_words: int = 2500,
) -> Callable[[NovelState], dict[str, Any]]:
    """创建章节编译节点。

    将所有场景的行动记录、叙事整合为完整章节。
    新增：潜台词素材 + 支线视角片段。
    """

    def compile_chapter_node(state: NovelState) -> dict[str, Any]:
        chapter_plan = state.get("chapter_plan")
        if not chapter_plan:
            return {"error": "缺少章节规划", "next_action": "end"}

        narrative_buffer = state.get("narrative_buffer", [])
        scene_actions = state.get("scene_actions", [])
        viewpoint_fragments = state.get("viewpoint_fragments", [])

        # 整理所有素材
        raw_materials: list[str] = []

        # 添加叙事缓冲中的文本
        for narration in narrative_buffer:
            raw_materials.append(narration)

        # 添加角色行动记录（外显行为）
        if scene_actions:
            action_texts: list[str] = []
            for action in scene_actions:
                action_texts.append(
                    f"[{action.character_name} - {action.action_type}]\n{action.content}"
                )
            raw_materials.append("\n\n".join(action_texts))

        all_materials = "\n\n---\n\n".join(raw_materials)

        # ── 构建潜台词素材（内心独白）──
        subtext_materials = ""
        monologue_actions = [a for a in scene_actions if a.internal_monologue]
        if monologue_actions:
            subtext_parts = [
                "\n# 潜台词素材（角色真实内心活动——不要直白写出，要用文学手法暗示）",
                "以下是角色在行动时的真实内心想法。请在叙事中巧妙暗示这些想法，",
                "但不要直接说出。用环境描写、微表情、语气词等手法来传达'言不由衷'的感觉。\n",
            ]
            for action in monologue_actions:
                dissonance_hint = ""
                if action.cognitive_dissonance > 0.3:
                    dissonance_hint = "（此角色的外在表现与内心想法有显著差异——重要的文学张力点！）"
                subtext_parts.append(
                    f"[{action.character_name}的潜台词] {dissonance_hint}\n"
                    f"外显行为：{action.content[:100]}...\n"
                    f"真实想法：{action.internal_monologue[:300]}"
                )
            subtext_materials = "\n\n".join(subtext_parts)

        # ── 构建支线视角素材 ──
        viewpoint_materials = ""
        if viewpoint_fragments:
            vp_parts = [
                "\n# 支线视角素材（请在合适位置插入，作为插叙或画外音）",
                "以下是从非主角视角观察到的片段。请在合适的位置自然地穿插这些内容，",
                "增加叙事的空间感和多层次感。注意保持视角切换的自然过渡。\n",
            ]
            for frag in viewpoint_fragments:
                vp_parts.append(
                    f"[视角：{frag.viewpoint_name}]\n{frag.content}"
                )
            viewpoint_materials = "\n\n".join(vp_parts)

        # ── 构建上一章结尾上下文 ──
        prev_chapter_ctx = ""
        completed_chapters = state.get("chapters", [])
        if completed_chapters:
            last_chapter = completed_chapters[-1]
            ending_snippet = last_chapter.content[-300:]
            prev_chapter_ctx = f"""# 上一章结尾（第{last_chapter.chapter_index}章「{last_chapter.title}」）
以下是上一章的最后一段，你必须确保本章开头与之自然衔接，并且绝对不能重复以下内容中的描写、意象和用词：

---
{ending_snippet}
---

"""

        user_prompt = f"""{prev_chapter_ctx}# 章节规划
标题：第{chapter_plan.chapter_index}章 {chapter_plan.title}
概要：{chapter_plan.summary}
关键事件：{'、'.join(chapter_plan.key_events)}
情感基调：{chapter_plan.emotional_tone or '未指定'}

# 原始素材（角色行动记录和叙事片段）
{all_materials}

{subtext_materials}

{viewpoint_materials}

---

{COMPILE_INSTRUCTIONS}"""

        try:
            # 获取最低字数要求（由 NovelConfig 注入，默认 2500）
            min_words = max(int(chapter_min_words), 500)
            messages = [
                SystemMessage(content=NARRATOR_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"{user_prompt}\n\n"
                        f"# 强制字数要求\n"
                        f"本章最终正文必须不少于 {min_words} 字。"
                    )
                ),
            ]

            # 带字数不足重试的编译逻辑（最多重试 1 次）
            chapter_text = ""
            for compile_attempt in range(2):
                response = invoke_with_retry(model, messages, operation_name="compile_chapter")
                chapter_text = extract_response_text(response).strip()

                if len(chapter_text) >= min_words:
                    break

                if compile_attempt == 0:
                    logger.warning(
                        "章节编译字数不足 (%d < %d)，追加扩写指令重试",
                        len(chapter_text), min_words,
                    )
                    # 追加扩写指令重新请求
                    expand_msg = HumanMessage(
                        content=f"你刚才写的章节只有 {len(chapter_text)} 字，远低于要求的 {min_words} 字。"
                        f"请在以下内容的基础上**大幅扩写**，补充更多的环境描写、角色心理活动、"
                        f"对话细节和场景过渡。不要重写，而是在原文基础上丰富和展开。"
                        f"直接输出完整的扩写后章节正文。\n\n原始内容：\n{chapter_text}"
                    )
                    messages = [
                        SystemMessage(content=NARRATOR_SYSTEM_PROMPT),
                        expand_msg,
                    ]

            # 【v2.1】应用去AI味处理
            if chaos_engine:
                chapter_text = chaos_engine.process_text_humanization(
                    chapter_text,
                    context=f"章节{chapter_plan.chapter_index}: {chapter_plan.title}"
                )
                logger.info("章节文本已应用去AI味处理")

            word_count = len(chapter_text)

            chapter = Chapter(
                chapter_index=chapter_plan.chapter_index,
                title=chapter_plan.title,
                content=chapter_text,
                summary=chapter_plan.summary,
                word_count=word_count,
            )

            logger.info(
                "章节编译完成: 第%d章「%s」, %d字 (潜台词素材:%d条, 支线视角:%d条)",
                chapter.chapter_index,
                chapter.title,
                word_count,
                len(monologue_actions),
                len(viewpoint_fragments),
            )

            next_idx = chapter_plan.chapter_index + 1
            total = state.get("total_chapters", 10)
            is_finished = next_idx > total

            return {
                "chapters": [chapter],
                "current_chapter_index": next_idx,
                "scene_actions": [],
                "narrative_buffer": [],
                "viewpoint_fragments": [],
                "chapter_plan": None,
                "scene_queue": [],
                "current_scene": None,
                "next_action": "review_chapter" if not is_finished else "end",
            }
        except Exception as e:
            logger.error("章节编译失败: %s", e)
            return {"error": str(e), "next_action": "end"}

    return compile_chapter_node


def create_expand_storyboard_chapter_node(
    model: BaseChatModel,
    chaos_engine: ChaosEngine | None = None,
    chapter_min_words: int = 2500,
) -> Callable[[NovelState], dict[str, Any]]:
    """按章节分镜扩写正文（Phase4）。"""

    def expand_storyboard_chapter_node(state: NovelState) -> dict[str, Any]:
        storyboards: list[ChapterStoryboard] = state.get("chapter_storyboards", [])
        setting_pack: SettingPack | None = state.get("setting_pack")
        current_idx = state.get("current_chapter_index", 1)
        total = state.get("total_chapters", len(storyboards) or 0)
        guardrails = state.get("global_guardrails", [])
        metadata = dict(state.get("metadata", {}))
        rewrite_reason = str(metadata.get("chapter_rewrite_reason", "") or "")

        if not storyboards:
            return {"error": "缺少章节分镜", "next_action": "end"}
        if current_idx > len(storyboards):
            return {"next_action": "end"}

        storyboard = storyboards[current_idx - 1]
        min_words = max(int(chapter_min_words), 500)

        guardrails_text = "\n".join(f"- {g}" for g in guardrails) if guardrails else "（无）"
        setting_text = ""
        setting_anchor_terms = _extract_setting_anchor_terms(setting_pack, limit=10)
        setting_anchor_text = "、".join(setting_anchor_terms) if setting_anchor_terms else "（无）"
        if setting_pack:
            setting_text = (
                f"\n# 结构化设定锚点\n"
                f"主旨：{setting_pack.theme}\n"
                f"规则：{'；'.join(setting_pack.worldview_rules[:6]) if setting_pack.worldview_rules else '（无）'}\n"
                f"关键时间线：{'；'.join(setting_pack.core_events_timeline[:8]) if setting_pack.core_events_timeline else '（无）'}\n"
                f"设定锚词（正文至少自然命中2个）：{setting_anchor_text}\n"
            )
        rewrite_text = f"\n# 返工要求\n{rewrite_reason}\n" if rewrite_reason else ""
        scene_blocks = []
        for scene in storyboard.scenes:
            anchor_terms = _extract_anchor_terms(
                " ".join(
                    [
                        scene.title or "",
                        scene.objective or "",
                        scene.info_gain or "",
                        " ".join(scene.expected_beats or []),
                    ]
                ),
                limit=4,
            )
            anchor_text = "、".join(anchor_terms) if anchor_terms else "（无）"
            scene_blocks.append(
                f"- 场景{scene.scene_index} [{scene.scene_type}] {scene.title}\n"
                f"  目标: {scene.objective}\n"
                f"  冲突: {scene.conflict_type or '（无）'}\n"
                f"  因果: {scene.causal_from or '（无）'} -> {scene.causal_to or '（无）'}\n"
                f"  信息增量: {scene.info_gain or '（无）'}\n"
                f"  风格: {scene.style_notes or '（无）'}\n"
                f"  节拍: {'、'.join(scene.expected_beats) if scene.expected_beats else '（无）'}\n"
                f"  锚词: {anchor_text}"
            )
        scenes_text = "\n".join(scene_blocks)

        user_prompt = f"""# 扩写任务
你正在扩写第{storyboard.chapter_index}章（共{total}章）。

# 本章分镜（必须严格兑现）
标题：{storyboard.title}
章节职责：{storyboard.purpose}
不可逆变化：{storyboard.irreversible_change}
必兑现线索：{'、'.join(storyboard.must_payoffs) if storyboard.must_payoffs else '（无）'}

# 场景清单
{scenes_text}
{setting_text}

# 全书硬约束
{guardrails_text}
{rewrite_text}
---
{EXPAND_OUTLINE_INSTRUCTIONS}

# 执行清单（必须满足）
1) 至少覆盖全部场景中的 50%（每个被覆盖场景需体现目标+因果或信息增量）。
2) 优先自然命中场景锚词（不得堆砌关键词）。
3) 必兑现线索不可省略，且需在正文里落到具体动作/对话。
4) 至少自然命中 2 个“设定锚词”，并与当前冲突发生关联。
5) 禁止“总结式复述分镜”，必须写成连续的小说叙事。
6) 结尾最后 1-2 段必须写出“不可逆后果落地”，并至少出现以下触发词之一：从此/再也/无法/彻底/失去/被迫/已成定局。

# 强制字数要求
本章最终正文必须不少于 {min_words} 字。"""

        base_user_prompt = user_prompt
        messages = [
            SystemMessage(content=EXPAND_OUTLINE_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        try:
            chapter_text = ""
            for attempt in range(2):
                response = invoke_with_retry(model, messages, operation_name="expand_storyboard_chapter")
                chapter_text = extract_response_text(response).strip()
                if len(chapter_text) >= min_words:
                    break
                if attempt == 0:
                    logger.warning(
                        "扩写字数不足 (%d < %d)，追加扩写指令重试",
                        len(chapter_text), min_words,
                    )
                    messages = [
                        SystemMessage(content=EXPAND_OUTLINE_SYSTEM_PROMPT),
                        HumanMessage(
                            content=(
                                f"{base_user_prompt}\n\n"
                                f"# 二次扩写补救（必须执行）\n"
                                f"当前稿件仅 {len(chapter_text)} 字，低于 {min_words} 字。"
                                "请在保持既有情节顺序不变的前提下：\n"
                                "1) 补足字数到目标；\n"
                                "2) 明确补足必兑现线索落地动作；\n"
                                "3) 结尾补写不可逆后果句；\n"
                                "4) 自然补入至少2个设定锚词。\n"
                                f"\n\n当前草稿：\n{chapter_text}"
                            )
                        ),
                    ]

            if chaos_engine:
                chapter_text = chaos_engine.process_text_humanization(
                    chapter_text,
                    context=f"分镜扩写章节{storyboard.chapter_index}: {storyboard.title}",
                )

            chapter = Chapter(
                chapter_index=storyboard.chapter_index,
                title=storyboard.title,
                content=chapter_text,
                summary=storyboard.purpose,
                word_count=len(chapter_text),
            )
            logger.info("分镜扩写完成: 第%d章「%s」, %d字", chapter.chapter_index, chapter.title, chapter.word_count)
            return {
                "chapters": [chapter],
                "metadata": metadata,
                "next_action": "storyboard_quality_gate",
            }
        except Exception as e:
            logger.error("分镜扩写失败: %s", e)
            return {"error": str(e), "next_action": "end"}

    return expand_storyboard_chapter_node
