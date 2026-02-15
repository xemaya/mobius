"""产出物管理模块：严格规划的输出目录结构。

目录结构：
output/<novel_name>/
├── chapters/                 # 逐章 Markdown 文件
│   ├── chapter_001.md
│   ├── chapter_002.md
│   └── ...
├── events/                   # 世界事件与冲突记录
│   ├── chapter_001_events.json      (世界事件)
│   ├── chapter_001_desires.json     (欲望提案)
│   └── ...
├── reviews/                  # 章节评审记录
│   ├── chapter_001_review.json
│   └── ...
├── memory/                   # 记忆蒸馏记录
│   ├── chapter_001_memory.json
│   └── ...
├── state/                    # 角色状态快照
│   ├── chapter_001_state.json
│   └── ...
├── full_novel.md             # 完整小说（全部章节合并）
└── metadata.json             # 生成元数据
"""

from mobius.output.manager import OutputManager

__all__ = ["OutputManager"]
