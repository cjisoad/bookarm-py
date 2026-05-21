"""选点后按目标 x 坐标分区执行抓书动作。

请在项目根目录运行：

    python scripts/grasp/grasp_book.py --port /dev/bookarm

    python scripts/grasp/grasp_book.py --execute

    干跑模式
    python scripts/grasp/grasp_book.py --dry-run
"""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from bookarm_control_py.task.grasp.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
