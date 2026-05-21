"""放书到圆盘流程的 ROS2 命令入口。"""

from __future__ import annotations

from pathlib import Path
import runpy


SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "grasp" / "place_book_on_disk.py"


def main() -> int:
    namespace = runpy.run_path(str(SCRIPT_PATH))
    script_main = namespace["main"]
    return int(script_main())


if __name__ == "__main__":
    raise SystemExit(main())
