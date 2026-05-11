"""按硬编码构型执行固定抓取流程。

请在项目根目录运行：

    python scripts/arm/run_record_sequence.py --port /dev/bookarm

流程：
1. 打开夹爪，并移动到构型 1。
2. 以速度 25、加速度 5 移动到构型 2。
3. 以最大速度、加速度 10 移动到构型 3，然后固定等待指定时间。
4. 持续关闭夹爪，等待 5 秒。
5. 以速度 35、加速度 8 移动到构型 5，并保持该构型。

关节角单位为度，脚本会转换为弧度后下发。
"""

from __future__ import annotations

import argparse
import time
from typing import Iterable

import numpy as np

from bookarm_control_py import BookArm


CONFIGURATIONS_DEG: dict[int, np.ndarray] = {
    1: np.array([0.0, -70.0, 60.0, 0.0, 0.0], dtype=float),
    2: np.array([  1.230469,  13.095703,  51.767578, -36.035156,  11.689453], dtype=float),
    3: np.array([  1.230469,  13.095703,  61.767578, -36.035156,  100.689453],dtype=float),
    5: np.array([0.0, -70.0, 60.0, 0.0, 76.289063], dtype=float),
}
MAX_ARM_SPEED = 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按硬编码构型执行固定流程。")
    parser.add_argument("--port", default="COM8", help="串口号，例如 COM8 或 /dev/bookarm。")
    parser.add_argument("--initial-speed", type=float, default=25.0, help="移动到构型 1 的速度。")
    parser.add_argument("--initial-acc", type=float, default=5.0, help="移动到构型 1 的加速度。")
    parser.add_argument("--gripper-wait", type=float, default=5.0, help="持续关闭夹爪后的等待时间。")
    parser.add_argument(
        "--config1-wait",
        type=float,
        default=5.0,
        help="移动到构型 1 后固定等待的时间，单位秒。",
    )
    parser.add_argument(
        "--config2-wait",
        type=float,
        default=5.0,
        help="移动到构型 2 后固定等待的时间，单位秒。",
    )
    parser.add_argument(
        "--config3-wait",
        type=float,
        default=4.0,
        help="移动到构型 3 后固定等待的时间，单位秒。",
    )
    parser.add_argument(
        "--config5-wait",
        type=float,
        default=0.0,
        help="移动到构型 5 后固定等待的时间，单位秒；默认保持构型后立即结束脚本。",
    )
    return parser.parse_args()


def format_array(values: np.ndarray) -> str:
    return np.array2string(values, precision=6, suppress_small=True)


def wait(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def move_to_configuration(
    robot: BookArm,
    q_deg: Iterable[float],
    *,
    label: str,
    speed: float,
    acc: float,
    wait_seconds: float,
) -> None:
    q_deg_array = np.asarray(q_deg, dtype=float)
    q_rad = robot.check_joint_angles(np.deg2rad(q_deg_array), context=f"{label} 构型")

    print(f"\n移动到{label}")
    print(f"  目标 q deg: {format_array(q_deg_array)}")
    print(f"  speed={speed:.3f}, acc={acc:.3f}")
    print(robot.move_joints_rad(q_rad, speed=speed, acceleration=acc))
    print(f"固定等待 {wait_seconds:.3f} 秒。")
    wait(wait_seconds)


def main() -> None:
    args = parse_args()
    robot = BookArm()
    configurations = CONFIGURATIONS_DEG

    print("BookArm 硬编码构型固定流程执行")
    print(f"串口号: {args.port}")
    print(f"关节名称: {robot.joint_names}")
    for index in (1, 2, 3, 5):
        print(f"构型 {index} deg: {format_array(configurations[index])}")

    robot.connect_serial(port=args.port)
    try:
        print("\n开启机械臂力矩。")
        print(robot.enable_torque())

        print("\n打开夹爪，并移动到构型 1。")
        print(robot.open_gripper())
        move_to_configuration(
            robot,
            configurations[1],
            label="构型 1",
            speed=args.initial_speed,
            acc=args.initial_acc,
            wait_seconds=args.config1_wait,
        )
        move_to_configuration(
            robot,
            configurations[2],
            label="构型 2",
            speed=25.0,
            acc=5.0,
            wait_seconds=args.config2_wait,
        )
        move_to_configuration(
            robot,
            configurations[3],
            label="构型 3",
            speed=MAX_ARM_SPEED,
            acc=10.0,
            wait_seconds=args.config3_wait,
        )

        print("\n持续关闭夹爪。")
        print(robot.hold_gripper_closed())
        print(f"等待 {args.gripper_wait:.3f} 秒。")
        wait(args.gripper_wait)

        move_to_configuration(
            robot,
            configurations[5],
            label="构型 5",
            speed=35.0,
            acc=8.0,
            wait_seconds=args.config5_wait,
        )

        print("\n流程完成：保持构型 5，未发送回零或关闭力矩命令。")
    finally:
        robot.close()


if __name__ == "__main__":
    main()
