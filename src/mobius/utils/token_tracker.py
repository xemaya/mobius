"""Token消耗统计跟踪器。

提供全局的LLM调用token统计功能，支持：
- 按operation分类统计
- 按model分类统计
- 按chapter分类统计
- 总token消耗统计
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TokenUsage:
    """单次LLM调用的token使用情况。"""
    operation: str
    model_name: str
    chapter_index: Optional[int]
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenStats:
    """Token统计汇总。"""
    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_duration_ms: float = 0.0

    # 按operation统计
    by_operation: Dict[str, TokenStats] = field(default_factory=dict)

    # 按model统计
    by_model: Dict[str, TokenStats] = field(default_factory=dict)

    # 按chapter统计
    by_chapter: Dict[int, TokenStats] = field(default_factory=dict)


class TokenTracker:
    """全局Token消耗跟踪器（线程安全单例）。"""

    _instance: Optional[TokenTracker] = None
    _lock = threading.Lock()

    def __init__(self):
        self._usages: List[TokenUsage] = []
        self._stats = TokenStats()

    @classmethod
    def get_instance(cls) -> TokenTracker:
        """获取单例实例。"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def reset(self) -> None:
        """重置统计数据。"""
        with self._lock:
            self._usages.clear()
            self._stats = TokenStats()

    def record_usage(
        self,
        operation: str,
        model_name: str,
        chapter_index: Optional[int] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        duration_ms: float = 0.0,
        **metadata: Any
    ) -> None:
        """记录一次LLM调用的token消耗。"""
        usage = TokenUsage(
            operation=operation,
            model_name=model_name,
            chapter_index=chapter_index,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            duration_ms=duration_ms,
            metadata=metadata
        )

        with self._lock:
            self._usages.append(usage)
            self._update_stats(usage)

    def _update_stats(self, usage: TokenUsage) -> None:
        """更新统计数据。"""
        # 总计
        self._stats.total_calls += 1
        self._stats.total_input_tokens += usage.input_tokens
        self._stats.total_output_tokens += usage.output_tokens
        self._stats.total_tokens += usage.total_tokens
        self._stats.total_duration_ms += usage.duration_ms

        # 按operation
        if usage.operation not in self._stats.by_operation:
            self._stats.by_operation[usage.operation] = TokenStats()
        op_stats = self._stats.by_operation[usage.operation]
        op_stats.total_calls += 1
        op_stats.total_input_tokens += usage.input_tokens
        op_stats.total_output_tokens += usage.output_tokens
        op_stats.total_tokens += usage.total_tokens
        op_stats.total_duration_ms += usage.duration_ms

        # 按model
        if usage.model_name not in self._stats.by_model:
            self._stats.by_model[usage.model_name] = TokenStats()
        model_stats = self._stats.by_model[usage.model_name]
        model_stats.total_calls += 1
        model_stats.total_input_tokens += usage.input_tokens
        model_stats.total_output_tokens += usage.output_tokens
        model_stats.total_tokens += usage.total_tokens
        model_stats.total_duration_ms += usage.duration_ms

        # 按chapter
        if usage.chapter_index is not None:
            if usage.chapter_index not in self._stats.by_chapter:
                self._stats.by_chapter[usage.chapter_index] = TokenStats()
            ch_stats = self._stats.by_chapter[usage.chapter_index]
            ch_stats.total_calls += 1
            ch_stats.total_input_tokens += usage.input_tokens
            ch_stats.total_output_tokens += usage.output_tokens
            ch_stats.total_tokens += usage.total_tokens
            ch_stats.total_duration_ms += usage.duration_ms

    def get_stats(self) -> TokenStats:
        """获取当前统计数据。"""
        with self._lock:
            return self._stats

    def get_all_usages(self) -> List[TokenUsage]:
        """获取所有token使用记录。"""
        with self._lock:
            return self._usages.copy()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（用于JSON序列化）。"""
        stats = self.get_stats()
        usages = self.get_all_usages()

        return {
            "summary": {
                "total_calls": stats.total_calls,
                "total_input_tokens": stats.total_input_tokens,
                "total_output_tokens": stats.total_output_tokens,
                "total_tokens": stats.total_tokens,
                "total_duration_ms": stats.total_duration_ms,
                "avg_duration_ms": stats.total_duration_ms / max(stats.total_calls, 1),
            },
            "by_operation": {
                op: {
                    "calls": s.total_calls,
                    "input_tokens": s.total_input_tokens,
                    "output_tokens": s.total_output_tokens,
                    "total_tokens": s.total_tokens,
                    "duration_ms": s.total_duration_ms,
                }
                for op, s in stats.by_operation.items()
            },
            "by_model": {
                model: {
                    "calls": s.total_calls,
                    "input_tokens": s.total_input_tokens,
                    "output_tokens": s.total_output_tokens,
                    "total_tokens": s.total_tokens,
                    "duration_ms": s.total_duration_ms,
                }
                for model, s in stats.by_model.items()
            },
            "by_chapter": {
                ch: {
                    "calls": s.total_calls,
                    "input_tokens": s.total_input_tokens,
                    "output_tokens": s.total_output_tokens,
                    "total_tokens": s.total_tokens,
                    "duration_ms": s.total_duration_ms,
                }
                for ch, s in stats.by_chapter.items()
            },
            "detailed_usages": [
                {
                    "operation": u.operation,
                    "model_name": u.model_name,
                    "chapter_index": u.chapter_index,
                    "input_tokens": u.input_tokens,
                    "output_tokens": u.output_tokens,
                    "total_tokens": u.total_tokens,
                    "timestamp": u.timestamp,
                    "duration_ms": u.duration_ms,
                    "metadata": u.metadata,
                }
                for u in usages
            ]
        }


# 全局便捷函数
def track_token_usage(
    operation: str,
    model_name: str,
    chapter_index: Optional[int] = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    total_tokens: int = 0,
    duration_ms: float = 0.0,
    **metadata: Any
) -> None:
    """便捷函数：记录token消耗。"""
    tracker = TokenTracker.get_instance()
    tracker.record_usage(
        operation=operation,
        model_name=model_name,
        chapter_index=chapter_index,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        duration_ms=duration_ms,
        **metadata
    )


def get_token_stats() -> TokenStats:
    """获取当前token统计。"""
    tracker = TokenTracker.get_instance()
    return tracker.get_stats()


def reset_token_tracking() -> None:
    """重置token统计。"""
    tracker = TokenTracker.get_instance()
    tracker.reset()