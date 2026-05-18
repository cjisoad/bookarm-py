"""Run a fixed BookArm hold-grasp sequence.

Run from the project root:

    python scripts/arm/move/run_hold_grasp_sequence.py --port /dev/bookarm

Joint configurations are written in degrees and converted to radians before
sending to BookArm.
"""

from __future__ import annotations

import argparse
import time
from typing import Iterable

import numpy as np

from bookarm_control_py import BookArm


CONFIG_1_DEG = np.array([ -0.527344, -59.589844,  38.759766, -18.720703, -32.34375 ], dtype=float)
CONFIG_2_DEG = np.array([ -0.615234,   5.009766,  11.425781, -63.261719, -32.34375 ], dtype=float)
CONFIG_3_DEG = np.array([ -0.351562,  22.939453,  28.125   , -77.607422, -32.167969], dtype=float)
CONFIG_4_DEG = np.array([ -0.351562,  22.939453,  28.125   , -77.607422, -32.167969], dtype=float)

DEFAULT_PORT = "/dev/bookarm"
DEFAULT_SPEED = 25.0
DEFAULT_ACC = 8.0
DEFAULT_CONFIG1_WAIT = 5.0
DEFAULT_GRIPPER_HOLD_WAIT = 10.0
DEFAULT_BETWEEN_CONFIG_WAIT = 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="执行固定构型与持续夹紧流程。")
    parser.add_argument("--port", default=DEFAULT_PORT, help="串口号，例如 /dev/bookarm 或 COM8。")
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED, help="机械臂常规运动速度。")
    parser.add_argument("--acc", type=float, default=DEFAULT_ACC, help="机械臂常规运动加速度。")
    parser.add_argument("--config1-wait", type=float, default=DEFAULT_CONFIG1_WAIT, help="构型 1 后等待秒数。")
    parser.add_argument(
        "--between-config-wait",
        type=float,
        default=DEFAULT_BETWEEN_CONFIG_WAIT,
        help="构型 2、3、4 之间的短暂停顿秒数。",
    )
    parser.add_argument(
        "--gripper-hold-wait",
        type=float,
        default=DEFAULT_GRIPPER_HOLD_WAIT,
        help="发送持续闭合夹爪指令后等待秒数。",
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
) -> None:
    q_deg_array = np.asarray(q_deg, dtype=float)
    q_rad = robot.check_joint_angles(np.deg2rad(q_deg_array), context=f"{label} 构型")

    print(f"\n移动到 {label}")
    print(f"  q deg: {format_array(q_deg_array)}")
    print(f"  speed={speed:.3f}, acc={acc:.3f}")
    print(robot.move_joints_rad(q_rad, speed=speed, acceleration=acc))


def main() -> None:
    args = parse_args()
    robot = BookArm()

    print("BookArm 固定持续夹紧流程")
    print(f"串口号: {args.port}")
    print(f"关节名称: {robot.joint_names}")
    print(f"构型 1 deg: {format_array(CONFIG_1_DEG)}")
    print(f"构型 2 deg: {format_array(CONFIG_2_DEG)}")
    print(f"构型 3 deg: {format_array(CONFIG_3_DEG)}")
    print(f"构型 4 deg: {format_array(CONFIG_4_DEG)}")

    robot.connect_serial(port=args.port)
    try:
        print("\n开启机械臂力矩。")
        print(robot.enable_torque())

        move_to_configuration(robot, CONFIG_1_DEG, label="构型 1", speed=args.speed, acc=args.acc)
        print("\n打开夹爪。")
        print(robot.open_gripper())
        print(f"等待 {args.config1_wait:.3f} 秒。")
        wait(args.config1_wait)

        print("\n发送夹爪持续闭合指令。")
        print(robot.hold_gripper_closed())
        print(f"等待 {args.gripper_hold_wait:.3f} 秒。")
        wait(args.gripper_hold_wait)

        move_to_configuration(robot, CONFIG_2_DEG, label="构型 2", speed=args.speed, acc=args.acc)
        print(f"等待 {args.between_config_wait:.3f} 秒。")
        wait(args.between_config_wait)

        move_to_configuration(robot, CONFIG_3_DEG, label="构型 3", speed=args.speed, acc=args.acc)
        print(f"等待 {args.between_config_wait:.3f} 秒。")
        wait(args.between_config_wait)

        move_to_configuration(robot, CONFIG_4_DEG, label="构型 4", speed=args.speed, acc=args.acc)

        print("\n流程完成：未发送夹爪打开、回零或关闭力矩命令。")
    finally:
        robot.close()


if __name__ == "__main__":
    main()
