"""ROS2 console entry point for the grasp_book workflow."""

from __future__ import annotations

from bookarm_control_py.task.grasp.cli import main as _main


def main() -> int:
    return _main()


if __name__ == "__main__":
    raise SystemExit(main())
