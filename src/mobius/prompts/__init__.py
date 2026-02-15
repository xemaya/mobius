"""提示词管理模块：将 agent 提示词从代码中分离。

所有提示词以 .txt 文件存放在本目录下，通过 load_prompt() 加载。
支持 {variable} 占位符，通过 format_prompt() 填充。
"""

from __future__ import annotations

import functools
from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent


@functools.lru_cache(maxsize=64)
def load_prompt(name: str) -> str:
    """加载指定名称的提示词文件。

    Args:
        name: 提示词文件名（不含 .txt 后缀亦可）。

    Returns:
        提示词文本内容。

    Raises:
        FileNotFoundError: 提示词文件不存在时。
    """
    filename = name if name.endswith(".txt") else f"{name}.txt"
    filepath = _PROMPTS_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"提示词文件不存在: {filepath}")
    return filepath.read_text(encoding="utf-8").strip()


def format_prompt(name: str, **kwargs: str) -> str:
    """加载并格式化提示词模板。

    Args:
        name: 提示词文件名。
        **kwargs: 填充模板的变量。

    Returns:
        格式化后的提示词。
    """
    template = load_prompt(name)
    return template.format(**kwargs)


__all__ = ["load_prompt", "format_prompt"]
