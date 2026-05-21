"""交互式控制机械臂力矩开关，并持续读取当前关节角。

请在项目根目录运行：

    python scripts/grasp/manual_set_arm_configuration.py --port /dev/bookarm

脚本会持续读取当前关节角。第 1 关节使用固件反馈的连续多圈角度，
因此可以显示 -360 到 360 deg 范围内的多圈角度。运行中可反复输入：

    c  关闭力矩，手动移动机械臂
    o  打开力矩，锁住当前位置
    q  退出程序
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import threading
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from bookarm_control_py import ArmFeedback, BookArm


DEFAULT_PORT = "/dev/bookarm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="交互式控制机械臂力矩开关，并持续读取当前关节角。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port", default=DEFAULT_PORT, help="机械臂串口号。")
    parser.add_argument(
        "--input-unit",
        choices=["rad", "deg"],
        default="rad",
        help="ESP32 反馈 joints/q 这类无单位字段时使用的单位。",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="读取当前关节角的间隔，单位秒。",
    )
    parser.add_argument(
        "--wait-response",
        action="store_true",
        help="力矩开关命令发送后等待 ESP32 回包。",
    )
    parser.add_argument(
        "--response-timeout",
        type=float,
        default=None,
        help="读取反馈或等待力矩命令回包的超时时间，单位秒。",
    )
    return parser.parse_args()


def format_array(values: Sequence[float] | np.ndarray) -> str:
    return np.array2string(
        np.asarray(values, dtype=float),
        precision=3,
        suppress_small=True,
        separator=", ",
    )


def print_feedback(robot: BookArm, feedback: ArmFeedback) -> None:
    q_deg = np.rad2deg(feedback.q_rad)

    print("\n当前关节角:")
    for name, rad, deg in zip(robot.joint_names, feedback.q_rad, q_deg, strict=True):
        print(f"  {name}: {rad:9.6f} rad, {deg:9.3f} deg")
    print(f"  q_deg: {format_array(q_deg)}")
    print(f"  torque: {format_array(feedback.torque)}")


def monitor_feedback(
    robot: BookArm,
    args: argparse.Namespace,
    stop_event: threading.Event,
    transport_lock: threading.Lock,
) -> None:
    interval = max(float(args.interval), 0.0)

    while not stop_event.is_set():
        try:
            with transport_lock:
                feedback = robot.read_arm_feedback(
                    response_timeout=args.response_timeout,
                    input_unit=args.input_unit,
                )
        except Exception as exc:  # noqa: BLE001 - keep manual positioning usable.
            print(f"\n读取关节反馈失败: {exc}")
        else:
            print_feedback(robot, feedback)

        stop_event.wait(interval)


def set_torque(
    robot: BookArm,
    args: argparse.Namespace,
    transport_lock: threading.Lock,
    *,
    enabled: bool,
) -> None:
    action = "打开" if enabled else "关闭"
    with transport_lock:
        response = (
            robot.enable_torque(
                wait_response=args.wait_response,
                response_timeout=args.response_timeout,
            )
            if enabled
            else robot.disable_torque(
                wait_response=args.wait_response,
                response_timeout=args.response_timeout,
            )
        )
    print(f"\n{action}机械臂力矩命令已发送:")
    print(response)


def command_loop(
    robot: BookArm,
    args: argparse.Namespace,
    stop_event: threading.Event,
    transport_lock: threading.Lock,
) -> None:
    print("\n命令：c=关闭力矩，o=打开力矩，q=退出程序。")
    print("关闭力矩后可手动移动机械臂；移动到想要的位置后输入 o 锁住当前位置。")

    while not stop_event.is_set():
        raw_command = input("\n> ").strip().lower()
        if raw_command in {"q", "quit", "exit"}:
            stop_event.set()
            return
        if raw_command in {"c", "off", "disable", "close", "0"}:
            set_torque(robot, args, transport_lock, enabled=False)
            continue
        if raw_command in {"o", "on", "enable", "open", "1"}:
            set_torque(robot, args, transport_lock, enabled=True)
            continue
        if raw_command in {"", "help", "h", "?"}:
            print("命令：c=关闭力矩，o=打开力矩，q=退出程序。")
            continue
        print(f"未知命令: {raw_command!r}。请输入 c、o 或 q。")


def main() -> int:
    args = parse_args()

    robot = BookArm()
    stop_event = threading.Event()
    transport_lock = threading.Lock()
    monitor_thread: threading.Thread | None = None

    print("BookArm 交互式力矩控制")
    print(f"串口号: {args.port}")
    print(f"关节名称: {robot.joint_names}")

    try:
        robot.connect_serial_arm(port=args.port)

        monitor_thread = threading.Thread(
            target=monitor_feedback,
            args=(robot, args, stop_event, transport_lock),
            daemon=True,
        )
        monitor_thread.start()
        command_loop(robot, args, stop_event, transport_lock)
        print("\n已退出命令循环。")
    except KeyboardInterrupt:
        print("\n已收到停止请求。")
        return 130
    finally:
        stop_event.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=1.0)
        robot.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
