"""Mobius CLI 入口：小说创作多 Agent 系统。"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import uuid
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from mobius.config.settings import ModelConfig, NovelConfig
from mobius.engine.chaos_engine import ChaosEngine
from mobius.graph.novel_graph import (
    compile_novel_graph,
    create_initial_state,
    load_setting_from_yaml,
)
from mobius.models.chapter import Chapter
from mobius.models.chaos_engine import ChaosEngineConfig
from mobius.output.manager import OutputManager

console = Console()
logger = logging.getLogger("mobius")


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
    setting_path = args.setting
    interactive = args.interactive
    output_dir = Path(args.output)

    # 加载设定集
    console.print(Panel(f"正在加载设定集: [bold]{setting_path}[/bold]", title="Mobius"))
    try:
        setting = load_setting_from_yaml(setting_path)
    except Exception as e:
        console.print(f"[red]加载设定集失败: {e}[/red]")
        sys.exit(1)

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
    parser = argparse.ArgumentParser(
        prog="mobius",
        description="Mobius - AI 小说创作多 Agent 系统",
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # generate 子命令
    gen_parser = subparsers.add_parser("generate", help="从设定集生成小说")
    gen_parser.add_argument("setting", help="设定集文件路径（YAML）")
    gen_parser.add_argument(
        "--interactive", "-i", action="store_true", help="交互模式（逐章生成）"
    )
    gen_parser.add_argument(
        "--output", "-o", default="output", help="输出目录（默认: output）"
    )
    gen_parser.add_argument(
        "--verbose", "-v", action="store_true", help="详细日志输出"
    )
    gen_parser.add_argument(
        "--start-chapter", type=int, default=1,
        help="从第几章开始生成（用于续写，会自动加载已有角色状态）",
    )

    args = parser.parse_args()

    # 配置日志
    log_level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_time=False, show_path=False)],
    )

    if args.command == "generate":
        cmd_generate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
