"""Mobius CLI 入口：小说创作多 Agent 系统。"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
from pathlib import Path

import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from mobius.agents.utils import extract_json, extract_response_text, invoke_with_retry
from mobius.config.settings import ModelConfig, NovelConfig
from mobius.engine.chaos_engine import ChaosEngine
from mobius.graph.novel_graph import (
    compile_expand_graph,
    compile_novel_graph,
    compile_outline_graph,
    compile_setting_pack_graph,
    compile_storyboard_graph,
    create_initial_state,
    load_setting_from_yaml,
)
from mobius.models.chapter import Chapter, ChapterOutline, ChapterStoryboard, SettingPack
from mobius.models.chaos_engine import ChaosEngineConfig
from mobius.output.manager import OutputManager

console = Console()
logger = logging.getLogger("mobius")


def _build_global_guardrails(worldview, plot_outline) -> list[str]:
    """构建全书硬约束（用于概要与扩写）。"""
    guardrails: list[str] = []
    if getattr(worldview, "rules", None):
        guardrails.extend([r.strip() for r in worldview.rules if str(r).strip()])
    if plot_outline.theme:
        guardrails.append(f"主命题必须持续围绕：{plot_outline.theme}")
    if not guardrails:
        guardrails = [
            "只允许信息逆流，禁止物质逆流",
            "存档点是共振锚点，不是重置按钮",
            "每章必须产生不可逆推进",
        ]
    return guardrails


def _load_outlines_from_dir(outlines_dir: Path) -> list[ChapterOutline]:
    """从 outlines 目录加载章节概要。"""
    files = sorted(outlines_dir.glob("chapter_*_outline.json"))
    outlines: list[ChapterOutline] = []
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        outlines.append(ChapterOutline.model_validate(data))
    outlines.sort(key=lambda x: x.chapter_index)
    return outlines


def _load_storyboards_from_dir(storyboards_dir: Path) -> list[ChapterStoryboard]:
    """从 storyboards 目录加载章节分镜。"""
    files = sorted(storyboards_dir.glob("chapter_*_storyboard.json"))
    storyboards: list[ChapterStoryboard] = []
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        storyboards.append(ChapterStoryboard.model_validate(data))
    storyboards.sort(key=lambda x: x.chapter_index)
    return storyboards


def _load_setting_pack(output_dir: Path) -> SettingPack | None:
    """从输出目录加载结构化设定集。"""
    path = output_dir / "setting_pack" / "setting_pack.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return SettingPack.model_validate(data)


def _sanitize_filename(name: str) -> str:
    """将标题转换为安全文件名。"""
    keep = []
    for ch in name:
        if ch.isalnum() or ch in {"-", "_"}:
            keep.append(ch)
        elif ch in {" ", "　"}:
            keep.append("_")
    cleaned = "".join(keep).strip("_")
    return cleaned or "startup"


def _normalize_markdown_payload(payload: dict, markdown_text: str, stem: str) -> dict:
    """规范化 LLM 产物，确保能被 YAML 设定加载器消费。"""
    title = (
        (payload.get("plot_outline") or {}).get("title")
        or payload.get("title")
        or stem
    )
    lines = markdown_text.strip().splitlines()
    first_line = (lines[0][:120] if lines else "") or ""
    synopsis = (
        (payload.get("plot_outline") or {}).get("synopsis")
        or first_line
        or f"{title} 的故事从一次异常事件展开。"
    )
    worldview = payload.get("worldview") if isinstance(payload.get("worldview"), dict) else {}
    plot_outline = payload.get("plot_outline") if isinstance(payload.get("plot_outline"), dict) else {}
    characters = payload.get("characters") if isinstance(payload.get("characters"), list) else []

    normalized_worldview = {
        "name": worldview.get("name") or f"{title}世界",
        "description": worldview.get("description") or "待补完世界观描述",
        "era": worldview.get("era", ""),
        "geography": worldview.get("geography", ""),
        "power_system": worldview.get("power_system", ""),
        "social_structure": worldview.get("social_structure", ""),
        "key_factions": worldview.get("key_factions", []),
        "rules": worldview.get("rules", []),
        "extra": worldview.get("extra", {}),
    }

    normalized_plot = {
        "title": plot_outline.get("title") or title,
        "genre": plot_outline.get("genre", ""),
        "theme": plot_outline.get("theme", ""),
        "synopsis": synopsis,
        "beginning": plot_outline.get("beginning") or "主角在异常环境中首次触发核心冲突。",
        "development": plot_outline.get("development") or "冲突升级，角色目标和代价逐步公开。",
        "twist": plot_outline.get("twist") or "关键真相揭示，迫使角色做不可逆选择。",
        "conclusion": plot_outline.get("conclusion") or "主命题在最终对抗后得到回应。",
        "total_chapters": int(plot_outline.get("total_chapters") or 20),
        "extra_plot_points": plot_outline.get("extra_plot_points", []),
    }

    normalized_characters = []
    for item in characters:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        normalized_characters.append(
            {
                "name": name,
                "role": item.get("role", "supporting"),
                "age": item.get("age", ""),
                "gender": item.get("gender", ""),
                "appearance": item.get("appearance", ""),
                "personality": item.get("personality") or "待补完",
                "background": item.get("background", ""),
                "abilities": item.get("abilities", []),
                "speech_style": item.get("speech_style", ""),
                "goals": item.get("goals", []),
                "weaknesses": item.get("weaknesses", []),
                "extra": item.get("extra", {}),
                "initial_state": item.get("initial_state", {}),
            }
        )

    if not normalized_characters:
        normalized_characters = [
            {
                "name": "主角",
                "role": "protagonist",
                "personality": "冷静、谨慎、强执行力",
                "appearance": "",
                "background": "待补完",
                "abilities": [],
                "speech_style": "",
                "goals": ["查明真相并守护关键关系"],
                "weaknesses": ["过度自责"],
                "extra": {},
                "initial_state": {},
            }
        ]

    novel_config = payload.get("novel_config") if isinstance(payload.get("novel_config"), dict) else {}
    secondary_viewpoints = (
        payload.get("secondary_viewpoints")
        if isinstance(payload.get("secondary_viewpoints"), list)
        else []
    )
    environment = payload.get("environment") if isinstance(payload.get("environment"), dict) else {}
    return {
        "worldview": normalized_worldview,
        "plot_outline": normalized_plot,
        "characters": normalized_characters,
        "secondary_viewpoints": secondary_viewpoints,
        "environment": environment,
        "novel_config": novel_config,
    }


def _translate_markdown_to_preset_yaml(markdown_path: Path, output_dir: Path, use_llm: bool = True) -> Path:
    """把启动 markdown 翻译为 preset YAML。"""
    markdown_text = markdown_path.read_text(encoding="utf-8")
    bootstrap_dir = output_dir / "bootstrap"
    bootstrap_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = bootstrap_dir / f"{_sanitize_filename(markdown_path.stem)}.preset.yaml"

    if yaml_path.exists() and yaml_path.stat().st_mtime >= markdown_path.stat().st_mtime:
        return yaml_path

    payload: dict
    try:
        if not use_llm:
            raise RuntimeError("offline-mode")

        cfg = NovelConfig().director_model
        cfg.model_name = os.environ.get("MOBIUS_MODEL", cfg.model_name)
        cfg.provider = os.environ.get("MOBIUS_PROVIDER", cfg.provider)
        cfg.temperature = float(os.environ.get("MOBIUS_TEMPERATURE", str(cfg.temperature)))
        model = _init_model(cfg)

        schema_hint = """{
  "worldview": {"name":"","description":"","era":"","geography":"","power_system":"","social_structure":"","key_factions":[],"rules":[],"extra":{}},
  "plot_outline": {"title":"","genre":"","theme":"","synopsis":"","beginning":"","development":"","twist":"","conclusion":"","total_chapters":20,"extra_plot_points":[]},
  "characters": [{"name":"","role":"protagonist|antagonist|supporting","appearance":"","personality":"","background":"","abilities":[],"speech_style":"","goals":[],"weaknesses":[],"initial_state":{}}],
  "secondary_viewpoints": [],
  "environment": {},
  "novel_config": {}
}"""
        prompt = (
            "你是小说工程助手。请把用户提供的 Markdown 启动文档转换为 Mobius 的 preset JSON。\n"
            "要求：\n"
            "1) 只输出 JSON，不要解释。\n"
            "2) 不要臆造无法支撑的细节，可留空字符串或空数组。\n"
            "3) `characters` 至少输出 1 个主角。\n"
            "4) `plot_outline.total_chapters` 默认 20，若文档明确给出则使用文档值。\n"
            "5) 若人物出场时机有描述，请放入角色 `extra.first_appearance`。\n\n"
            f"JSON 结构:\n{schema_hint}\n\n"
            f"Markdown 文档:\n{markdown_text}"
        )
        response = invoke_with_retry(
            model,
            [
                SystemMessage(content="你只输出严格 JSON。"),
                HumanMessage(content=prompt),
            ],
            operation_name="translate_markdown_to_preset",
        )
        raw = extract_json(extract_response_text(response))
        if not isinstance(raw, dict):
            raise ValueError("LLM 输出不是 JSON 对象")
        payload = _normalize_markdown_payload(raw, markdown_text, markdown_path.stem)
    except Exception as e:
        if str(e) != "offline-mode":
            console.print(f"[yellow]Markdown 转 preset 调用失败，使用兜底模板：{e}[/yellow]")
        payload = _normalize_markdown_payload({}, markdown_text, markdown_path.stem)

    yaml_path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return yaml_path


def _resolve_setting_yaml_path(setting_arg: str, output_dir: Path, use_llm: bool = True) -> Path:
    """解析 CLI 输入的设定源，支持 YAML 或 Markdown 启动文档。"""
    src = Path(setting_arg)
    if not src.exists():
        raise FileNotFoundError(f"文件不存在: {src}")
    suffix = src.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return src
    if suffix == ".md":
        yaml_path = _translate_markdown_to_preset_yaml(src, output_dir, use_llm=use_llm)
        console.print(f"[cyan]已从 Markdown 生成 preset: {yaml_path}[/cyan]")
        return yaml_path
    raise ValueError(f"不支持的设定文件类型: {src.suffix}（仅支持 .yaml/.yml/.md）")


def _load_setting_from_source(
    setting_arg: str,
    output_dir: Path,
    panel_title: str,
    use_llm: bool = True,
) -> tuple[Path, dict]:
    """统一加载设定：支持 YAML 和 Markdown 启动文档。"""
    try:
        setting_path = _resolve_setting_yaml_path(setting_arg, output_dir, use_llm=use_llm)
        console.print(Panel(f"正在加载设定集: [bold]{setting_path}[/bold]", title=panel_title))
        setting = load_setting_from_yaml(str(setting_path))
        return setting_path, setting
    except Exception as e:
        console.print(f"[red]加载设定集失败: {e}[/red]")
        sys.exit(1)


def _build_dry_run_outlines(plot_outline, total_chapters: int) -> list[ChapterOutline]:
    """离线模式下基于设定生成可审阅概要。"""
    points = list(getattr(plot_outline, "extra_plot_points", []) or [])
    outlines: list[ChapterOutline] = []
    stage_templates = [
        ("起势", "建立风险边界并抛出第一层矛盾"),
        ("承压", "把已知风险转化为行动代价"),
        ("分裂", "让角色立场出现公开冲突"),
        ("转折", "引入高压选择并触发不可逆后果"),
        ("回收", "回收旧线索并重构目标优先级"),
        ("逼近", "把局部冲突推向系统级对抗"),
        ("摊牌", "迫使关键角色公开底牌"),
        ("余震", "处理代价并埋入下一阶段承诺"),
    ]
    arc_blocks = [
        plot_outline.beginning or "开端阶段",
        plot_outline.development or "发展阶段",
        plot_outline.twist or "转折阶段",
        plot_outline.conclusion or "收束阶段",
    ]

    for idx in range(1, total_chapters + 1):
        point = points[idx - 1] if idx - 1 < len(points) else f"推进主线阶段 {idx}"
        stage_name, stage_goal = stage_templates[(idx - 1) % len(stage_templates)]
        arc_text = arc_blocks[min(3, (idx - 1) * 4 // max(total_chapters, 1))]
        purpose = f"[{stage_name}] 第{idx}章聚焦“{point}”，目标是{stage_goal}"
        core_plot = (
            f"第{idx}章进入“{arc_text}”轨道，主冲突围绕“{point}”展开。"
            "角色先通过线索核验确认风险，再在对抗场景中作出高代价选择，"
            "该选择会改变下一章的资源分配与关系格局。"
            f"章节尾部留出新的问题：第{idx+1 if idx < total_chapters else idx}章必须回应本章后果。"
        )
        outlines.append(
            ChapterOutline(
                chapter_index=idx,
                title=f"{stage_name}·第{idx}步",
                purpose=purpose,
                core_plot=core_plot,
                irreversible_change=f"第{idx}章后，围绕“{point}”的关系网络发生不可逆重排",
                character_arc_delta=f"关键角色在“{point}”议题上完成一次立场位移",
                must_payoffs=[f"兑现第{max(1, idx-1)}章围绕“{point}”遗留的问题"],
                new_promises=[f"埋设第{min(total_chapters, idx+1)}章将面对的新冲突门槛"],
            )
        )
    return outlines


def _build_dry_run_storyboards(outlines: list[ChapterOutline]) -> list[ChapterStoryboard]:
    """离线模式下从概要生成分镜草案。"""
    storyboards: list[ChapterStoryboard] = []
    for outline in outlines:
        scenes = [
            {
                "scene_index": 1,
                "scene_type": "plot_progress",
                "title": "冲突引爆",
                "objective": f"推进{outline.purpose}",
                "conflict_type": "立场冲突",
                "location": "核心场所",
                "participating_characters": ["主角"],
                "causal_from": "承接上章后果",
                "causal_to": "进入决策阶段",
                "info_gain": outline.core_plot[:80],
                "style_notes": "紧张节奏，动作优先",
                "expected_beats": ["冲突触发", "压力上升"],
            },
            {
                "scene_index": 2,
                "scene_type": "daily",
                "title": "降密呼吸",
                "objective": "缓冲节奏并展示人物日常侧面",
                "conflict_type": "弱冲突",
                "location": "生活空间",
                "participating_characters": ["主角", "配角"],
                "causal_from": "高压后的回落",
                "causal_to": "铺垫下一次对抗",
                "info_gain": "关系细节被揭示",
                "style_notes": "环境描写与心理描写",
                "expected_beats": ["沉默观察", "心境变化"],
            },
            {
                "scene_index": 3,
                "scene_type": "plot_progress",
                "title": "关键决策",
                "objective": "落实不可逆变化",
                "conflict_type": "价值抉择",
                "location": "决策现场",
                "participating_characters": ["主角", "对手"],
                "causal_from": "前序积压矛盾",
                "causal_to": "引出章节结尾后果",
                "info_gain": outline.irreversible_change,
                "style_notes": "对话与动作并重",
                "expected_beats": ["抉择", "后果落地"],
            },
            {
                "scene_index": 4,
                "scene_type": "narration",
                "title": "尾声钩子",
                "objective": "留下下一章承诺",
                "conflict_type": "悬念",
                "location": "转场空间",
                "participating_characters": ["叙事视角"],
                "causal_from": "决策后余波",
                "causal_to": "下一章冲突门槛",
                "info_gain": "新承诺被抛出",
                "style_notes": "克制收束，留白",
                "expected_beats": ["余震", "钩子"],
            },
        ]
        storyboards.append(
            ChapterStoryboard.model_validate(
                {
                    "chapter_index": outline.chapter_index,
                    "title": outline.title,
                    "purpose": outline.purpose,
                    "irreversible_change": outline.irreversible_change,
                    "must_payoffs": outline.must_payoffs,
                    "scenes": scenes,
                }
            )
        )
    return storyboards


def _dry_expand_storyboard(storyboard: ChapterStoryboard, min_words: int, guardrails: list[str]) -> str:
    """离线模式下将分镜扩展为正文占位稿。"""
    sections = [
        f"{storyboard.title}。",
        f"这一章的任务是：{storyboard.purpose}。",
        f"不可逆变化在本章必须落地：{storyboard.irreversible_change}。",
        f"必须兑现线索：{'、'.join(storyboard.must_payoffs) if storyboard.must_payoffs else '（无）'}。",
        f"全书硬约束：{'；'.join(guardrails[:5]) if guardrails else '（无）'}。",
    ]
    for scene in storyboard.scenes:
        sections.append(
            f"场景{scene.scene_index}《{scene.title}》：{scene.objective}。"
            f"冲突类型：{scene.conflict_type or '（无）'}；"
            f"因果：{scene.causal_from or '（无）'} -> {scene.causal_to or '（无）'}；"
            f"信息增量：{scene.info_gain or '（无）'}。"
        )
    paragraph = "\n\n".join(sections)
    body = paragraph
    while len(body) < min_words:
        body += (
            "\n\n角色在现场的每一次犹豫都会放大代价，信息被反复核验，"
            "决策者必须在集体安全与个体意志之间给出明确取舍。"
            "冲突不再停留在语言，而会体现为关系重组、资源失衡与结构后果。"
        )
    return body


def _init_model(model_config: ModelConfig):
    """根据配置初始化 LLM。"""
    provider = model_config.provider.lower()

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_config.model_name,
            temperature=model_config.temperature,
            max_output_tokens=model_config.max_tokens,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_config.model_name,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
        )
    elif provider == "minimax":
        from mobius.llm.minimax import ChatMiniMax

        return ChatMiniMax(
            api_key=os.environ.get("MINIMAX_API_KEY", model_config.api_key),
            model=model_config.model_name or "M2-her",
            temperature=model_config.temperature,
            max_completion_tokens=min(model_config.max_tokens, 2048),
        )
    else:
        # 通过 langchain 的通用接口
        from langchain.chat_models import init_chat_model

        return init_chat_model(
            f"{provider}:{model_config.model_name}",
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
        )


def cmd_generate(args: argparse.Namespace) -> None:
    """执行小说生成命令。"""
    interactive = args.interactive
    output_dir = Path(args.output)
    _, setting = _load_setting_from_source(args.setting, output_dir, "Mobius")

    worldview = setting["worldview"]
    plot_outline = setting["plot_outline"]
    character_profiles = setting["character_profiles"]
    character_states = setting["character_states"]
    environment = setting.get("environment")
    secondary_viewpoints = setting.get("secondary_viewpoints", [])

    # 显示设定摘要
    _print_setting_summary(worldview, plot_outline, character_profiles)

    # 配置（YAML 中可选 novel_config 覆盖，便于提速：如 max_interaction_rounds: 3）
    novel_config = NovelConfig()
    overrides = setting.get("novel_config_overrides") or {}
    if overrides:
        novel_config = NovelConfig(**(novel_config.model_dump() | overrides))

    # 可通过环境变量覆盖模型配置
    model_name = os.environ.get("MOBIUS_MODEL", novel_config.director_model.model_name)
    provider = os.environ.get("MOBIUS_PROVIDER", novel_config.director_model.provider)
    temperature = float(os.environ.get("MOBIUS_TEMPERATURE", str(novel_config.director_model.temperature)))

    for cfg in [novel_config.director_model, novel_config.character_model, novel_config.narrator_model]:
        cfg.model_name = model_name
        cfg.provider = provider
        cfg.temperature = temperature

    # 初始化 OutputManager（严格目录结构）
    output_mgr = OutputManager(output_dir, novel_title=plot_outline.title)
    console.print(f"\n产出物目录: [cyan]{output_dir}[/cyan]")
    console.print(f"  ├── chapters/    (逐章 Markdown)")
    console.print(f"  ├── events/      (世界事件/欲望提案)")
    console.print(f"  ├── reviews/     (章节评审)")
    console.print(f"  ├── memory/      (记忆蒸馏)")
    console.print(f"  ├── state/       (角色状态快照)")
    console.print(f"  └── metadata.json")

    # 初始化模型
    console.print(f"\n初始化模型: [cyan]{provider}:{model_name}[/cyan]")
    try:
        from mobius.utils.token_tracking_model import wrap_model_with_token_tracking
        from mobius.utils.token_tracker import reset_token_tracking

        # 重置token统计（新生成任务）
        reset_token_tracking()

        # 初始化基础模型
        director_model_raw = _init_model(novel_config.director_model)
        character_model_raw = _init_model(novel_config.character_model)
        narrator_model_raw = _init_model(novel_config.narrator_model)

        # 添加token跟踪包装器
        director_model = wrap_model_with_token_tracking(director_model_raw, "director")
        character_model = wrap_model_with_token_tracking(character_model_raw, "character")
        narrator_model = wrap_model_with_token_tracking(narrator_model_raw, "narrator")

        # 可选模型：角色扮演 (MiniMax M2-her)、评审、观察者
        roleplay_model = None
        if novel_config.character_roleplay_model:
            console.print(
                f"  角色扮演模型: [cyan]{novel_config.character_roleplay_model.provider}"
                f":{novel_config.character_roleplay_model.model_name}[/cyan]"
            )
            roleplay_model_raw = _init_model(novel_config.character_roleplay_model)
            roleplay_model = wrap_model_with_token_tracking(roleplay_model_raw, "roleplay")

        reviewer_model = None
        if novel_config.reviewer_model:
            reviewer_model_raw = _init_model(novel_config.reviewer_model)
            reviewer_model = wrap_model_with_token_tracking(reviewer_model_raw, "reviewer")

        observer_model = None
        if novel_config.observer_model:
            observer_model_raw = _init_model(novel_config.observer_model)
            observer_model = wrap_model_with_token_tracking(observer_model_raw, "observer")
    except Exception as e:
        console.print(f"[red]模型初始化失败: {e}[/red]")
        console.print("[yellow]请确保已设置正确的 API Key 环境变量。[/yellow]")
        sys.exit(1)

    # 【v2.1】创建失控引擎
    chaos_config = ChaosEngineConfig()  # 使用默认配置，可后续扩展为从YAML读取
    chaos_engine = ChaosEngine(chaos_config)
    console.print("🔥 [bold red]激活失控型叙事引擎 v2.1[/bold red] - 角色将带着偏见做错事")

    # 编译图（注入 OutputManager 和 ChaosEngine）
    graph = compile_novel_graph(
        director_model=director_model,
        character_model=character_model,
        narrator_model=narrator_model,
        character_profiles=character_profiles,
        config=novel_config,
        roleplay_model=roleplay_model,
        reviewer_model=reviewer_model or director_model,
        observer_model=observer_model or director_model,
        secondary_viewpoints=secondary_viewpoints or None,
        output_manager=output_mgr,
        chaos_engine=chaos_engine,
    )

    # 创建初始状态
    initial_state = create_initial_state(
        worldview=worldview,
        plot_outline=plot_outline,
        character_profiles=character_profiles,
        initial_character_states=character_states,
        environment=environment,
        secondary_viewpoints=secondary_viewpoints,
    )

    # ── 续写模式：从指定章节恢复状态 ──
    start_chapter = getattr(args, "start_chapter", 1)
    if start_chapter > 1:
        initial_state = _resume_from_chapter(
            initial_state, output_dir, start_chapter
        )

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    console.print(
        f"\n[green]开始生成小说: 「{plot_outline.title}」 "
        f"共 {plot_outline.total_chapters} 章[/green]\n"
    )

    if interactive:
        _run_interactive(graph, initial_state, config, output_mgr, plot_outline)
    else:
        _run_batch(graph, initial_state, config, output_mgr, plot_outline)


def cmd_setting_pack(args: argparse.Namespace) -> None:
    """执行结构化设定集生成（Layer1-A）。"""
    output_dir = Path(args.output)
    _, setting = _load_setting_from_source(
        args.setting,
        output_dir,
        "Mobius SettingPack",
        use_llm=not bool(getattr(args, "dry_run", False)),
    )

    worldview = setting["worldview"]
    plot_outline = setting["plot_outline"]
    character_profiles = setting["character_profiles"]
    character_states = setting["character_states"]
    environment = setting.get("environment")
    secondary_viewpoints = setting.get("secondary_viewpoints", [])

    _print_setting_summary(worldview, plot_outline, character_profiles)

    novel_config = NovelConfig()
    overrides = setting.get("novel_config_overrides") or {}
    if overrides:
        novel_config = NovelConfig(**(novel_config.model_dump() | overrides))

    model_name = os.environ.get("MOBIUS_MODEL", novel_config.director_model.model_name)
    provider = os.environ.get("MOBIUS_PROVIDER", novel_config.director_model.provider)
    temperature = float(os.environ.get("MOBIUS_TEMPERATURE", str(novel_config.director_model.temperature)))
    novel_config.director_model.model_name = model_name
    novel_config.director_model.provider = provider
    novel_config.director_model.temperature = temperature

    output_mgr = OutputManager(output_dir, novel_title=plot_outline.title)
    console.print(f"\n产出物目录: [cyan]{output_dir}[/cyan]")
    console.print("  ├── setting_pack/setting_pack.json")
    console.print("  ├── setting_pack/setting_pack.md")
    console.print("  └── metadata.json")

    outlines = _load_outlines_from_dir(output_mgr.outlines_dir)
    if not outlines:
        console.print("[red]未找到章节概要，请先执行 outline。[/red]")
        sys.exit(1)
    if not output_mgr.is_phase_approved("outline"):
        console.print("[red]未检测到概要审批标记，请先执行 approve-outline。[/red]")
        sys.exit(1)

    if args.dry_run:
        setting_pack = SettingPack(
            title=plot_outline.title,
            theme=plot_outline.theme or "（待补完）",
            theme_longform="（待补完）",
            worldview_longform="（待补完）",
            worldview_rules=list(getattr(worldview, "rules", []) or []),
            core_events_timeline=list(getattr(plot_outline, "extra_plot_points", []) or []),
            detailed_rules=[],
            timeline_events=[],
            entities=[],
            characters=[],
            organizations=[],
            items=[],
            missing_items=["补充关键组织结构", "补充角色关系图", "补充时间线细节"],
            author_notes="离线草案，仅用于流程验证",
        )
        output_mgr.save_setting_pack(setting_pack)
        console.print("[bold green]离线设定集生成完成[/bold green]")
        console.print(f"[cyan]请人工审阅: {output_mgr.root / 'setting_pack' / 'setting_pack.md'}[/cyan]")
        console.print(
            "[yellow]审阅通过后执行：mobius approve-setting --output "
            f"{output_mgr.root}[/yellow]"
        )
        return

    console.print(f"\n初始化模型: [cyan]{provider}:{model_name}[/cyan]")
    try:
        from mobius.utils.token_tracking_model import wrap_model_with_token_tracking
        from mobius.utils.token_tracker import reset_token_tracking

        reset_token_tracking()
        director_model_raw = _init_model(novel_config.director_model)
        director_model = wrap_model_with_token_tracking(director_model_raw, "director_outline")
    except Exception as e:
        console.print(f"[red]模型初始化失败: {e}[/red]")
        console.print("[yellow]请确保已设置正确的 API Key 环境变量。[/yellow]")
        sys.exit(1)

    graph = compile_setting_pack_graph(
        director_model=director_model,
        output_manager=output_mgr,
    )

    initial_state = create_initial_state(
        worldview=worldview,
        plot_outline=plot_outline,
        character_profiles=character_profiles,
        initial_character_states=character_states,
        environment=environment,
        secondary_viewpoints=secondary_viewpoints,
    )
    initial_state["chapter_outlines"] = outlines
    initial_state["outline_approved"] = True
    initial_state["global_guardrails"] = _build_global_guardrails(worldview, plot_outline)
    initial_state["next_action"] = "generate_setting_pack"

    run_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    console.print(f"\n[green]开始生成结构化设定集: 「{plot_outline.title}」[/green]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("正在生成结构化设定集...", total=None)
        try:
            result = graph.invoke(initial_state, run_config)
        except Exception as e:
            console.print(f"\n[red]设定集生成失败: {e}[/red]")
            return
        finally:
            progress.remove_task(task)

    setting_pack = result.get("setting_pack")
    if not setting_pack:
        console.print("[red]未生成结构化设定集。[/red]")
        return

    console.print("[bold green]结构化设定集生成完成[/bold green]")
    console.print(f"[cyan]请人工审阅: {output_mgr.root / 'setting_pack' / 'setting_pack.md'}[/cyan]")
    console.print(
        "[yellow]审阅通过后执行：mobius approve-setting --output "
        f"{output_mgr.root}[/yellow]"
    )


def cmd_outline(args: argparse.Namespace) -> None:
    """执行全书概要生成（Layer1-B）。"""
    output_dir = Path(args.output)
    _, setting = _load_setting_from_source(
        args.setting,
        output_dir,
        "Mobius Outline",
        use_llm=not bool(getattr(args, "dry_run", False)),
    )

    worldview = setting["worldview"]
    plot_outline = setting["plot_outline"]
    character_profiles = setting["character_profiles"]
    character_states = setting["character_states"]
    environment = setting.get("environment")
    secondary_viewpoints = setting.get("secondary_viewpoints", [])
    _print_setting_summary(worldview, plot_outline, character_profiles)

    output_mgr = OutputManager(output_dir, novel_title=plot_outline.title)
    setting_pack = _load_setting_pack(output_dir)
    if setting_pack is None:
        console.print("[yellow]未检测到 setting_pack，将先生成概要，后续再反向补完设定集。[/yellow]")

    end_chapter = min(int(args.end_chapter), int(plot_outline.total_chapters))
    if end_chapter < 1:
        console.print("[red]--end-chapter 必须 >= 1[/red]")
        sys.exit(1)

    if args.dry_run:
        outlines = _build_dry_run_outlines(plot_outline, end_chapter)
        for item in outlines:
            output_mgr.save_outline(item)
        output_mgr.save_full_outline(outlines)
        console.print(f"[bold green]离线概要生成完成：{len(outlines)} 章[/bold green]")
        console.print(f"[cyan]请人工审阅: {output_mgr.root / 'full_outline.md'}[/cyan]")
        console.print(
            "[yellow]审阅通过后执行：mobius approve-outline --output "
            f"{output_mgr.root}[/yellow]"
        )
        return

    novel_config = NovelConfig()
    overrides = setting.get("novel_config_overrides") or {}
    if overrides:
        novel_config = NovelConfig(**(novel_config.model_dump() | overrides))
    model_name = os.environ.get("MOBIUS_MODEL", novel_config.director_model.model_name)
    provider = os.environ.get("MOBIUS_PROVIDER", novel_config.director_model.provider)
    temperature = float(os.environ.get("MOBIUS_TEMPERATURE", str(novel_config.director_model.temperature)))
    novel_config.director_model.model_name = model_name
    novel_config.director_model.provider = provider
    novel_config.director_model.temperature = temperature

    console.print(f"\n初始化模型: [cyan]{provider}:{model_name}[/cyan]")
    try:
        from mobius.utils.token_tracking_model import wrap_model_with_token_tracking
        from mobius.utils.token_tracker import reset_token_tracking
        reset_token_tracking()
        director_model_raw = _init_model(novel_config.director_model)
        director_model = wrap_model_with_token_tracking(director_model_raw, "director_outline")
    except Exception as e:
        console.print(f"[red]模型初始化失败: {e}[/red]")
        sys.exit(1)

    graph = compile_outline_graph(
        director_model=director_model,
        character_profiles=character_profiles,
        config=novel_config,
        output_manager=output_mgr,
    )
    initial_state = create_initial_state(
        worldview=worldview,
        plot_outline=plot_outline,
        character_profiles=character_profiles,
        initial_character_states=character_states,
        environment=environment,
        secondary_viewpoints=secondary_viewpoints,
    )
    initial_state["total_chapters"] = end_chapter
    if setting_pack is not None:
        initial_state["setting_pack"] = setting_pack
        initial_state["setting_approved"] = output_mgr.is_phase_approved("setting")
    initial_state["global_guardrails"] = _build_global_guardrails(worldview, plot_outline)
    initial_state["next_action"] = "blueprint_refresh"

    run_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("正在生成章节概要...", total=None)
        result = graph.invoke(initial_state, run_config)
        progress.remove_task(task)

    outlines = result.get("chapter_outlines", [])
    if not outlines:
        console.print("[red]未生成任何章节概要。[/red]")
        return
    console.print(f"[bold green]概要生成完成：{len(outlines)} 章[/bold green]")
    console.print(f"[cyan]请人工审阅: {output_mgr.root / 'full_outline.md'}[/cyan]")
    console.print("[yellow]审阅通过后执行：mobius approve-outline --output "
                  f"{output_mgr.root}[/yellow]")


def cmd_storyboard(args: argparse.Namespace) -> None:
    """执行章节分镜生成（Layer2）。"""
    output_dir = Path(args.output)
    _, setting = _load_setting_from_source(
        args.setting,
        output_dir,
        "Mobius Storyboard",
        use_llm=not bool(getattr(args, "dry_run", False)),
    )
    outlines_dir = Path(args.from_outline) if args.from_outline else output_dir / "outlines"
    worldview = setting["worldview"]
    plot_outline = setting["plot_outline"]
    character_profiles = setting["character_profiles"]
    character_states = setting["character_states"]
    environment = setting.get("environment")
    secondary_viewpoints = setting.get("secondary_viewpoints", [])

    output_mgr = OutputManager(output_dir, novel_title=plot_outline.title)
    if not output_mgr.is_phase_approved("setting"):
        console.print("[red]未检测到设定集审批标记，请先 approve-setting。[/red]")
        sys.exit(1)
    if not output_mgr.is_phase_approved("outline"):
        console.print("[red]未检测到概要审批标记，请先 approve-outline。[/red]")
        sys.exit(1)

    setting_pack = _load_setting_pack(output_dir)
    if setting_pack is None:
        console.print("[red]缺少 setting_pack，请先执行 setting-pack。[/red]")
        sys.exit(1)
    outlines = _load_outlines_from_dir(outlines_dir)
    if not outlines:
        console.print(f"[red]概要目录中没有可用章节概要: {outlines_dir}[/red]")
        sys.exit(1)

    if args.dry_run:
        storyboards = _build_dry_run_storyboards(outlines)
        for sb in storyboards:
            output_mgr.save_storyboard(sb)
        output_mgr.save_full_storyboards(storyboards)
        console.print(f"[bold green]离线分镜生成完成：{len(storyboards)} 章[/bold green]")
        console.print(f"[cyan]请人工审阅: {output_mgr.root / 'full_storyboard.md'}[/cyan]")
        console.print("[yellow]审阅通过后执行：mobius approve-storyboard --output "
                      f"{output_mgr.root}[/yellow]")
        return

    novel_config = NovelConfig()
    model_name = os.environ.get("MOBIUS_MODEL", novel_config.director_model.model_name)
    provider = os.environ.get("MOBIUS_PROVIDER", novel_config.director_model.provider)
    novel_config.director_model.model_name = model_name
    novel_config.director_model.provider = provider

    from mobius.utils.token_tracking_model import wrap_model_with_token_tracking
    from mobius.utils.token_tracker import reset_token_tracking
    reset_token_tracking()
    director_model = wrap_model_with_token_tracking(_init_model(novel_config.director_model), "director_storyboard")
    graph = compile_storyboard_graph(director_model=director_model, output_manager=output_mgr)

    initial_state = create_initial_state(
        worldview=worldview,
        plot_outline=plot_outline,
        character_profiles=character_profiles,
        initial_character_states=character_states,
        environment=environment,
        secondary_viewpoints=secondary_viewpoints,
    )
    initial_state["setting_pack"] = setting_pack
    initial_state["chapter_outlines"] = outlines
    initial_state["global_guardrails"] = _build_global_guardrails(worldview, plot_outline)
    initial_state["next_action"] = "generate_storyboards"
    result = graph.invoke(initial_state, {"configurable": {"thread_id": str(uuid.uuid4())}})
    storyboards = result.get("chapter_storyboards", [])
    if not storyboards:
        console.print("[red]未生成章节分镜。[/red]")
        return
    console.print(f"[bold green]分镜生成完成：{len(storyboards)} 章[/bold green]")
    console.print(f"[cyan]请人工审阅: {output_mgr.root / 'full_storyboard.md'}[/cyan]")
    console.print("[yellow]审阅通过后执行：mobius approve-storyboard --output "
                  f"{output_mgr.root}[/yellow]")


def cmd_approve_phase(args: argparse.Namespace) -> None:
    """人工确认阶段产物。"""
    output_dir = Path(args.output)
    output_mgr = OutputManager(output_dir)
    phase = args.phase
    checks = {
        "setting": (output_mgr.setting_pack_dir / "setting_pack.json"),
        "outline": (output_mgr.root / "full_outline.md"),
        "storyboard": (output_mgr.root / "full_storyboard.md"),
    }
    target = checks.get(phase)
    if target is None or not target.exists():
        console.print(f"[red]未找到 {phase} 阶段产物: {target}[/red]")
        sys.exit(1)
    approval_file = output_mgr.mark_phase_approved(phase, approver="manual")
    console.print(f"[bold green]{phase} 已确认，写入: {approval_file}[/bold green]")


def cmd_expand(args: argparse.Namespace) -> None:
    """基于已确认分镜执行章节扩写（Layer3）。"""
    output_dir = Path(args.output)
    _, setting = _load_setting_from_source(
        args.setting,
        output_dir,
        "Mobius Expand",
        use_llm=not bool(getattr(args, "dry_run", False)),
    )
    storyboards_dir = Path(args.from_storyboard) if args.from_storyboard else output_dir / "storyboards"

    worldview = setting["worldview"]
    plot_outline = setting["plot_outline"]
    character_profiles = setting["character_profiles"]
    character_states = setting["character_states"]
    environment = setting.get("environment")
    secondary_viewpoints = setting.get("secondary_viewpoints", [])
    _print_setting_summary(worldview, plot_outline, character_profiles)

    output_mgr = OutputManager(output_dir, novel_title=plot_outline.title)
    for phase in ("setting", "outline", "storyboard"):
        if not output_mgr.is_phase_approved(phase):
            console.print(f"[red]未检测到 {phase} 审批标记，请先 approve-{phase}。[/red]")
            sys.exit(1)
    setting_pack = _load_setting_pack(output_dir)
    if setting_pack is None:
        console.print("[red]未检测到 setting_pack，请先执行 setting-pack。[/red]")
        sys.exit(1)

    if not storyboards_dir.exists():
        console.print(f"[red]分镜目录不存在: {storyboards_dir}[/red]")
        sys.exit(1)
    storyboards = _load_storyboards_from_dir(storyboards_dir)
    if not storyboards:
        console.print(f"[red]分镜目录中没有可用章节分镜: {storyboards_dir}[/red]")
        sys.exit(1)

    novel_config = NovelConfig()
    overrides = setting.get("novel_config_overrides") or {}
    if overrides:
        novel_config = NovelConfig(**(novel_config.model_dump() | overrides))

    end_chapter = min(int(args.end_chapter), len(storyboards))
    if end_chapter < 1:
        console.print("[red]--end-chapter 必须 >= 1[/red]")
        sys.exit(1)
    storyboards = storyboards[:end_chapter]
    guardrails = _build_global_guardrails(worldview, plot_outline)

    if args.dry_run:
        start_idx = max(1, int(args.start_chapter))
        generated: list[Chapter] = []
        for sb in storyboards:
            if sb.chapter_index < start_idx:
                continue
            text = _dry_expand_storyboard(sb, novel_config.chapter_min_words, guardrails)
            chapter = Chapter(
                chapter_index=sb.chapter_index,
                title=sb.title,
                content=text,
                summary=sb.purpose,
                word_count=len(text),
            )
            output_mgr.save_chapter(chapter)
            generated.append(chapter)
        if not generated:
            console.print("[red]离线扩写未产出章节（请检查 start/end 参数）[/red]")
            return
        full_path = output_mgr.save_full_novel(generated)
        console.print(f"[bold green]离线扩写完成，已输出 {len(generated)} 章[/bold green]")
        console.print(f"[bold green]完整小说: {full_path}[/bold green]")
        return

    model_name = os.environ.get("MOBIUS_MODEL", novel_config.narrator_model.model_name)
    provider = os.environ.get("MOBIUS_PROVIDER", novel_config.narrator_model.provider)
    temperature = float(os.environ.get("MOBIUS_TEMPERATURE", str(novel_config.narrator_model.temperature)))
    novel_config.narrator_model.model_name = model_name
    novel_config.narrator_model.provider = provider
    novel_config.narrator_model.temperature = temperature

    console.print(f"\n初始化模型: [cyan]{provider}:{model_name}[/cyan]")
    try:
        from mobius.utils.token_tracking_model import wrap_model_with_token_tracking
        from mobius.utils.token_tracker import reset_token_tracking

        reset_token_tracking()
        narrator_model_raw = _init_model(novel_config.narrator_model)
        narrator_model = wrap_model_with_token_tracking(narrator_model_raw, "expand_narrator")
    except Exception as e:
        console.print(f"[red]模型初始化失败: {e}[/red]")
        sys.exit(1)

    chaos_engine = ChaosEngine(ChaosEngineConfig())
    graph = compile_expand_graph(
        narrator_model=narrator_model,
        config=novel_config,
        output_manager=output_mgr,
        chaos_engine=chaos_engine,
    )

    initial_state = create_initial_state(
        worldview=worldview,
        plot_outline=plot_outline,
        character_profiles=character_profiles,
        initial_character_states=character_states,
        environment=environment,
        secondary_viewpoints=secondary_viewpoints,
    )
    initial_state["setting_pack"] = setting_pack
    initial_state["chapter_storyboards"] = storyboards
    initial_state["total_chapters"] = end_chapter
    initial_state["setting_approved"] = True
    initial_state["outline_approved"] = True
    initial_state["storyboard_approved"] = True
    initial_state["global_guardrails"] = guardrails
    initial_state["current_chapter_index"] = max(1, int(args.start_chapter))
    initial_state["next_action"] = "expand_storyboard_chapter"
    initial_state["metadata"] = {"chapter_rewrite_attempts": {}}

    run_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    console.print(
        f"\n[green]开始按分镜扩写: 共 {end_chapter} 章，从第 {initial_state['current_chapter_index']} 章开始[/green]\n"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("正在扩写章节...", total=None)
        try:
            result = graph.invoke(initial_state, run_config)
        except Exception as e:
            console.print(f"\n[red]扩写失败: {e}[/red]")
            return
        finally:
            progress.remove_task(task)

    chapters = result.get("chapters", [])
    if not chapters:
        console.print("[red]未生成任何章节正文。[/red]")
        return

    seen: dict[int, Chapter] = {}
    for ch in chapters:
        seen[ch.chapter_index] = ch
    unique_chapters = sorted(seen.values(), key=lambda c: c.chapter_index)
    full_path = output_mgr.save_full_novel(unique_chapters)
    console.print(f"[bold green]扩写完成，已输出 {len(unique_chapters)} 章[/bold green]")
    console.print(f"[bold green]完整小说: {full_path}[/bold green]")


def _resume_from_chapter(
    initial_state: dict,
    output_dir: Path,
    start_chapter: int,
) -> dict:
    """从指定章节恢复状态，用于续写。

    加载上一章的角色状态快照、已完成章节摘要，设置起始章节索引。
    """
    import json
    from mobius.models.character import CharacterDynamicState
    from mobius.models.chapter import Chapter

    prev_chapter = start_chapter - 1
    state_file = output_dir / "state" / f"chapter_{prev_chapter:03d}_state.json"

    # 1. 加载上一章角色状态
    if state_file.exists():
        console.print(f"[cyan]续写模式: 从第 {start_chapter} 章开始[/cyan]")
        console.print(f"  加载角色状态: {state_file.name}")
        with open(state_file, encoding="utf-8") as f:
            state_data = json.load(f)
        char_states_data = state_data.get("character_states", {})
        restored_states = {}
        for name, cs_data in char_states_data.items():
            restored_states[name] = CharacterDynamicState.model_validate(cs_data)
        initial_state["character_states"] = restored_states
    else:
        console.print(
            f"[yellow]警告: 未找到第 {prev_chapter} 章状态文件，使用初始状态续写[/yellow]"
        )

    # 2. 加载已完成章节的摘要（供导演参考前情）
    chapters_dir = output_dir / "chapters"
    existing_chapters: list[Chapter] = []
    for idx in range(1, start_chapter):
        ch_file = chapters_dir / f"chapter_{idx:03d}.md"
        if ch_file.exists():
            content = ch_file.read_text(encoding="utf-8")
            # 从 markdown 标题提取章节名
            first_line = content.split("\n", 1)[0]
            title = first_line.replace(f"# 第{idx}章 ", "").strip()
            # 取前 200 字作为摘要
            body = content.split("\n", 1)[1].strip() if "\n" in content else ""
            summary = body[:200] + "..." if len(body) > 200 else body
            existing_chapters.append(
                Chapter(
                    chapter_index=idx,
                    title=title,
                    content=body,
                    summary=summary,
                    word_count=len(body),
                )
            )
            console.print(f"  已加载第 {idx} 章「{title}」摘要")

    # 3. 设置续写起始点
    initial_state["current_chapter_index"] = start_chapter
    initial_state["chapters"] = existing_chapters

    console.print(
        f"  [green]已恢复 {len(existing_chapters)} 章上下文，"
        f"将从第 {start_chapter} 章继续生成[/green]\n"
    )
    return initial_state


def _run_batch(graph, initial_state, config, output_mgr: OutputManager, plot_outline) -> None:
    """批量模式：一次性生成所有章节。

    注意：逐章产出已由 graph 内的 persist_chapter 节点完成，
    此处只需等待完成后生成完整小说合并文件。
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"正在生成《{plot_outline.title}》...", total=None
        )

        try:
            result = graph.invoke(initial_state, config)
        except Exception as e:
            console.print(f"\n[red]生成过程中出错: {e}[/red]")
            # 尝试获取已完成的部分
            try:
                state = graph.get_state(config)
                result = state.values if state else {}
            except Exception:
                result = {}

        progress.remove_task(task)

    chapters = result.get("chapters", [])
    if not chapters:
        console.print("[red]未能生成任何章节。[/red]")
        return

    # 按 chapter_index 去重（防止 add reducer 导致的重复）
    seen: dict[int, Chapter] = {}
    for ch in chapters:
        seen[ch.chapter_index] = ch
    chapters = sorted(seen.values(), key=lambda c: c.chapter_index)

    # 打印各章节信息（文件已由 persist_chapter 节点写入）
    console.print("\n[bold]生成完成！各章节信息：[/bold]")
    for chapter in chapters:
        console.print(
            f"  [green]✓[/green] 第{chapter.chapter_index}章 「{chapter.title}」"
            f" ({chapter.word_count}字)"
        )

    # 合并为完整小说
    full_path = output_mgr.save_full_novel(chapters)
    total_words = sum(c.word_count for c in chapters)
    console.print(f"\n[bold green]完整小说已保存: {full_path}[/bold green]")
    console.print(f"[bold green]总字数: {total_words}[/bold green]")

    # 显示token消耗统计
    token_summary = output_mgr.get_token_summary()
    if "error" not in token_summary:
        console.print(f"\n[bold blue]Token消耗统计:[/bold blue]")
        console.print(f"  总调用次数: {token_summary['total_calls']}")
        console.print(f"  总Token数: {token_summary['total_tokens']:,}")
        console.print(f"  输入Token: {token_summary['total_input_tokens']:,}")
        console.print(f"  输出Token: {token_summary['total_output_tokens']:,}")
        console.print(f"  平均响应时间: {token_summary['total_duration_ms'] / max(token_summary['total_calls'], 1):.2f}ms")
        # 按operation统计
        if token_summary.get('by_operation'):
            console.print(f"\n  [dim]按操作分类:[/dim]")
            for op, stats in sorted(token_summary['by_operation'].items()):
                console.print(f"    {op}: {stats['tokens']:,} tokens ({stats['calls']}次)")
    else:
        console.print(f"[yellow]Token统计不可用: {token_summary['error']}[/yellow]")

    # 打印产出物目录概览
    _print_output_summary(output_mgr)


def _run_interactive(graph, initial_state, config, output_mgr: OutputManager, plot_outline) -> None:
    """交互模式：逐章生成，用户可在章节间介入。

    注意：逐章落盘已由 graph 内的 persist 节点完成。
    """
    current_state = initial_state
    total = plot_outline.total_chapters
    all_chapters: list[Chapter] = []

    for chapter_idx in range(1, total + 1):
        console.print(f"\n{'='*60}")
        console.print(
            f"[bold cyan]准备生成第 {chapter_idx}/{total} 章[/bold cyan]"
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"第{chapter_idx}章生成中...", total=None)

            try:
                result = graph.invoke(current_state, config)
            except Exception as e:
                console.print(f"\n[red]生成出错: {e}[/red]")
                break

            progress.remove_task(task)

        chapters = result.get("chapters", [])
        new_chapters = [c for c in chapters if c not in all_chapters]

        for chapter in new_chapters:
            all_chapters.append(chapter)
            console.print(
                f"\n[green]✓ 第{chapter.chapter_index}章 「{chapter.title}」"
                f" ({chapter.word_count}字) — 已写入磁盘[/green]"
            )
            # 显示预览
            preview = chapter.content[:300] + "..." if len(chapter.content) > 300 else chapter.content
            console.print(Panel(preview, title="章节预览", border_style="dim"))

        # 检查是否已完成
        if result.get("next_action") == "end" or chapter_idx >= total:
            break

        # 用户介入
        console.print("\n[yellow]选项:[/yellow]")
        console.print("  [c] 继续下一章")
        console.print("  [s] 停止生成")
        console.print("  [v] 查看当前角色状态")

        choice = input("\n请选择 [c/s/v]: ").strip().lower()
        if choice == "s":
            console.print("[yellow]用户停止生成。[/yellow]")
            break
        elif choice == "v":
            _print_character_states(result.get("character_states", {}))
            input("\n按回车继续生成...")

        # 更新状态以继续
        current_state = result

    if all_chapters:
        full_path = output_mgr.save_full_novel(all_chapters)
        total_words = sum(c.word_count for c in all_chapters)
        console.print(f"\n[bold green]完整小说已保存: {full_path}[/bold green]")
        console.print(f"[bold green]总字数: {total_words}[/bold green]")

        # 显示token消耗统计
        token_summary = output_mgr.get_token_summary()
        if "error" not in token_summary:
            console.print(f"\n[bold blue]Token消耗统计:[/bold blue]")
            console.print(f"  总调用次数: {token_summary['total_calls']}")
            console.print(f"  总Token数: {token_summary['total_tokens']:,}")
            console.print(f"  输入Token: {token_summary['total_input_tokens']:,}")
            console.print(f"  输出Token: {token_summary['total_output_tokens']:,}")
            console.print(f"  平均响应时间: {token_summary['total_duration_ms'] / max(token_summary['total_calls'], 1):.2f}ms")
        _print_output_summary(output_mgr)
    else:
        console.print("[red]未能生成任何章节。[/red]")


def _print_setting_summary(worldview, plot_outline, character_profiles) -> None:
    """打印设定集摘要。"""
    table = Table(title=f"小说设定: 《{plot_outline.title}》", show_lines=True)
    table.add_column("项目", style="cyan", width=12)
    table.add_column("内容", style="white")

    table.add_row("世界观", f"{worldview.name}\n{worldview.description[:100]}")
    table.add_row("类型", plot_outline.genre)
    table.add_row("主题", plot_outline.theme)
    table.add_row("概要", plot_outline.synopsis[:200])
    table.add_row("章节数", str(plot_outline.total_chapters))

    char_names = ", ".join(
        f"{name}({p.role})" for name, p in character_profiles.items()
    )
    table.add_row("角色", char_names)

    console.print(table)


def _print_character_states(char_states: dict) -> None:
    """打印角色状态表。"""
    table = Table(title="角色状态", show_lines=True)
    table.add_column("角色", style="cyan")
    table.add_column("情感", style="yellow")
    table.add_column("关系", style="green")
    table.add_column("属性", style="magenta")
    table.add_column("存活", style="red")

    for name, state in char_states.items():
        emotions = ", ".join(f"{k}:{v:.1f}" for k, v in state.emotional_state.items())
        rels = ", ".join(f"{k}:{v:+.1f}" for k, v in state.relationship_scores.items())
        attrs = ", ".join(f"{k}:{v:.0f}" for k, v in state.custom_attributes.items())
        table.add_row(name, emotions or "-", rels or "-", attrs or "-", "是" if state.is_alive else "否")

    console.print(table)


def _print_output_summary(output_mgr: OutputManager) -> None:
    """打印产出物目录概览。"""
    console.print("\n[bold]产出物目录结构:[/bold]")

    dirs = [
        ("chapters", output_mgr.chapters_dir),
        ("events", output_mgr.events_dir),
        ("reviews", output_mgr.reviews_dir),
        ("memory", output_mgr.memory_dir),
        ("state", output_mgr.state_dir),
    ]

    for label, d in dirs:
        files = sorted(d.glob("*"))
        count = len(files)
        console.print(f"  {label}/ — {count} 个文件")


def main() -> None:
    """CLI 主入口。"""
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(
        prog="mobius",
        description="Mobius - AI 小说创作多 Agent 系统",
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    setting_parser = subparsers.add_parser("setting-pack", help="基于已确认概要反向补完结构化设定集")
    setting_parser.add_argument("setting", help="设定源文件路径（YAML 或 Markdown 启动文档）")
    setting_parser.add_argument(
        "--output", "-o", default="output", help="输出目录（默认: output）"
    )
    setting_parser.add_argument(
        "--verbose", "-v", action="store_true", help="详细日志输出"
    )
    setting_parser.add_argument(
        "--dry-run", action="store_true", help="离线模式：不调用模型，生成设定集草案"
    )

    outline_parser = subparsers.add_parser("outline", help="基于设定集生成全书章节概要（Layer1-B）")
    outline_parser.add_argument("setting", help="设定源文件路径（YAML 或 Markdown 启动文档）")
    outline_parser.add_argument(
        "--output", "-o", default="output", help="输出目录（默认: output）"
    )
    outline_parser.add_argument(
        "--verbose", "-v", action="store_true", help="详细日志输出"
    )
    outline_parser.add_argument(
        "--dry-run", action="store_true", help="离线模式：不调用模型，生成可审阅概要草案"
    )
    outline_parser.add_argument(
        "--end-chapter", type=int, default=9999, help="概要生成到第几章结束（默认全部）"
    )

    approve_setting_parser = subparsers.add_parser("approve-setting", help="人工确认设定集")
    approve_setting_parser.add_argument(
        "--output", "-o", default="output", help="产出目录（需包含 outlines/）"
    )
    approve_setting_parser.set_defaults(phase="setting")

    approve_outline_parser = subparsers.add_parser("approve-outline", help="人工确认全书概要")
    approve_outline_parser.add_argument(
        "--output", "-o", default="output", help="产出目录（需包含 full_outline.md）"
    )
    approve_outline_parser.set_defaults(phase="outline")

    storyboard_parser = subparsers.add_parser("storyboard", help="基于概要生成章节分镜")
    storyboard_parser.add_argument("setting", help="设定源文件路径（YAML 或 Markdown 启动文档）")
    storyboard_parser.add_argument(
        "--from-outline", default="", help="概要目录路径（默认: <output>/outlines）"
    )
    storyboard_parser.add_argument(
        "--output", "-o", default="output", help="输出目录（默认: output）"
    )
    storyboard_parser.add_argument(
        "--verbose", "-v", action="store_true", help="详细日志输出"
    )
    storyboard_parser.add_argument(
        "--dry-run", action="store_true", help="离线模式：按概要生成分镜草案"
    )

    approve_storyboard_parser = subparsers.add_parser("approve-storyboard", help="人工确认全书分镜")
    approve_storyboard_parser.add_argument(
        "--output", "-o", default="output", help="产出目录（需包含 full_storyboard.md）"
    )
    approve_storyboard_parser.set_defaults(phase="storyboard")

    expand_parser = subparsers.add_parser("expand", help="基于已确认分镜扩写正文（Layer3）")
    expand_parser.add_argument("setting", help="设定源文件路径（YAML 或 Markdown 启动文档）")
    expand_parser.add_argument(
        "--from-storyboard", default="", help="分镜目录路径（默认: <output>/storyboards）"
    )
    expand_parser.add_argument(
        "--output", "-o", default="output", help="输出目录（默认: output）"
    )
    expand_parser.add_argument(
        "--start-chapter", type=int, default=1, help="从第几章开始扩写"
    )
    expand_parser.add_argument(
        "--end-chapter", type=int, default=9999, help="扩写到第几章结束（默认全部）"
    )
    expand_parser.add_argument(
        "--verbose", "-v", action="store_true", help="详细日志输出"
    )
    expand_parser.add_argument(
        "--dry-run", action="store_true", help="离线模式：按概要生成占位正文用于流程验证"
    )

    args = parser.parse_args()

    # 配置日志
    log_level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_time=False, show_path=False)],
    )

    if args.command == "setting-pack":
        cmd_setting_pack(args)
    elif args.command == "outline":
        cmd_outline(args)
    elif args.command == "storyboard":
        cmd_storyboard(args)
    elif args.command in {"approve-setting", "approve-outline", "approve-storyboard"}:
        cmd_approve_phase(args)
    elif args.command == "expand":
        cmd_expand(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
