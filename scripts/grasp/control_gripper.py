"""交互式控制夹爪打开、闭合和持续闭合。

请在项目根目录运行：

    python scripts/grasp/control_gripper.py --port /dev/bookarm

运行后可输入单字母命令：

    o  打开夹爪
    c  闭合夹爪
    h  持续闭合夹爪
    r  读取夹爪反馈
    q  退出程序
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any, Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

if TYPE_CHECKING:
    from bookarm_control_py import BookArm


DEFAULT_PORT = "/dev/bookarm"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="交互式控制夹爪打开、闭合和持续闭合。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port", default=DEFAULT_PORT, help="夹爪串口号。")
    parser.add_argument(
        "--wait-response",
        action="store_true",
        help="发送夹爪命令后等待 ESP32 回包。",
    )
    parser.add_argument(
        "--response-timeout",
        type=float,
        default=None,
        help="等待回包或读取反馈的超时时间，单位秒。",
    )
    parser.add_argument(
        "--initial",
        choices=["none", "open", "close", "hold"],
        default="none",
        help="连接后自动执行的初始夹爪动作。",
    )
    return parser.parse_args(argv)


def print_help() -> None:
    print("\n命令：")
    print("  o  打开夹爪")
    print("  c  闭合夹爪")
    print("  h  持续闭合夹爪")
    print("  r  读取夹爪反馈")
    print("  q  退出程序")


def print_response(action: str, response: Any) -> None:
    print(f"\n{action}命令已发送。")
    if response is not None:
        print(response)


def run_gripper_action(
    action: str,
    command: Callable[..., Any],
    args: argparse.Namespace,
) -> None:
    response = command(
        wait_response=args.wait_response,
        response_timeout=args.response_timeout,
    )
    print_response(action, response)


def read_feedback(robot: BookArm, args: argparse.Namespace) -> None:
    feedback = robot.read_gripper_feedback(response_timeout=args.response_timeout)
    print("\n夹爪反馈：")
    print(feedback)


def run_initial_action(robot: BookArm, args: argparse.Namespace) -> None:
    if args.initial == "none":
        return
    if args.initial == "open":
        run_gripper_action("打开夹爪", robot.open_gripper, args)
        return
    if args.initial == "close":
        run_gripper_action("闭合夹爪", robot.close_gripper, args)
        return
    if args.initial == "hold":
        run_gripper_action("持续闭合夹爪", robot.hold_gripper_closed, args)


def command_loop(robot: BookArm, args: argparse.Namespace) -> None:
    print_help()

    while True:
        raw_command = input("\n> ").strip().lower()
        if raw_command in {"q", "quit", "exit"}:
            print("\n退出夹爪控制。")
            return
        if raw_command in {"o", "open"}:
            run_gripper_action("打开夹爪", robot.open_gripper, args)
            continue
        if raw_command in {"c", "close"}:
            run_gripper_action("闭合夹爪", robot.close_gripper, args)
            continue
        if raw_command in {"h", "hold"}:
            run_gripper_action("持续闭合夹爪", robot.hold_gripper_closed, args)
            continue
        if raw_command in {"r", "read"}:
            read_feedback(robot, args)
            continue
        if raw_command in {"", "?", "help"}:
            print_help()
            continue
        print(f"未知命令: {raw_command!r}。请输入 o、c、h、r 或 q。")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from bookarm_control_py import BookArm

    robot = BookArm()

    print("BookArm 交互式夹爪控制")
    print(f"串口号: {args.port}")

    try:
        robot.connect_serial_gripper(port=args.port)
        run_initial_action(robot, args)
        command_loop(robot, args)
    except KeyboardInterrupt:
        print("\n已收到停止请求。")
        return 130
    finally:
        robot.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
