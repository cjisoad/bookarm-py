"""读取当前机械臂所有关节角度。

请在项目根目录运行：

    python scripts/arm/move/read_joint_angles.py --port /dev/bookarm

该脚本只读取机械臂反馈，不发送力矩开关或运动命令。
如果 ESP32 返回 joints/q 这类无法从字段名判断单位的数据，并且单位是角度制，
请使用 --input-unit deg。
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from bookarm_control_py import ArmFeedback, BookArm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="读取 BookArm 当前所有关节角度。")
    parser.add_argument("--port", default="COM8", help="串口号，例如 COM8 或 /dev/bookarm。")
    parser.add_argument(
        "--input-unit",
        choices=["rad", "deg"],
        default="rad",
        help="ESP32 反馈关节角使用的单位；仅在字段名无法判断单位时生效。",
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="打印 ESP32 原始反馈 JSON。",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.0,
        help="循环读取间隔，单位秒；0 表示只读取一次。",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="读取次数；配合 --interval 使用，0 表示一直读取直到 Ctrl+C。",
    )
    parser.add_argument(
        "--response-timeout",
        type=float,
        default=None,
        help="单次读取反馈的超时时间，单位秒。",
    )
    return parser.parse_args()


def format_array(values: np.ndarray) -> str:
    return np.array2string(values, precision=6, suppress_small=True, separator=", ")


def print_feedback(
    robot: BookArm,
    feedback: ArmFeedback,
    *,
    index: int,
    show_raw: bool,
) -> None:
    print(f"\n读取 {index}:")
    if show_raw:
        print("  原始反馈:")
        print(json.dumps(feedback.raw, ensure_ascii=False, indent=2))

    q_deg = np.rad2deg(feedback.q_rad)
    q_deg_j1_minus_360 = q_deg.copy()
    q_deg_j1_minus_360[0] -= 360.0
    print("  关节角:")
    for name, rad, deg in zip(robot.joint_names, feedback.q_rad, q_deg, strict=True):
        print(f"    {name}: {rad:.6f} rad, {deg:.3f} deg")

    print(f"  q_rad: {format_array(feedback.q_rad)}")
    print(f"  q_deg: {format_array(q_deg)}")
    print(f"  q_deg_j1_minus_360: {format_array(q_deg_j1_minus_360)}")
    print(f"  torque: {format_array(feedback.torque)}")


def should_continue(index: int, *, count: int) -> bool:
    return count == 0 or index < count


def main() -> None:
    args = parse_args()

    robot = BookArm()
    print("BookArm 当前关节角读取")
    print(f"串口号: {args.port}")
    print(f"关节名称: {robot.joint_names}")
    print("只读取反馈，不发送力矩或运动命令。")

    robot.connect_serial_arm(port=args.port)
    try:
        index = 0
        while True:
            index += 1
            feedback = robot.read_arm_feedback(
                response_timeout=args.response_timeout,
                input_unit=args.input_unit,
            )
            print_feedback(
                robot,
                feedback,
                index=index,
                show_raw=args.show_raw,
            )

            if not should_continue(index, count=args.count):
                break
            if args.interval > 0:
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n已停止读取。")
    finally:
        robot.close()


if __name__ == "__main__":
    main()
