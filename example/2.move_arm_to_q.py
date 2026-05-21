"""将真实机械臂移动到指定关节构型。

请在项目根目录运行：

    python example/2.move_arm_to_q.py
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from bookarm_control_py import BookArm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 BookArm 移动到指定关节构型。"
    )
    parser.add_argument("--port", default="/dev/bookarm", help="串口号，例如 COM8。")
    return parser.parse_args()


def format_array(values: np.ndarray) -> str:
    return np.array2string(values, precision=6, suppress_small=True)


def main() -> None:
    args = parse_args()

    robot = BookArm()

    # 起始构型
    q_target_deg = np.array([0, -45,  -10 ,  45 ,  115], dtype=float)
    # q_target_deg = np.array([0.0, -70.0, 75.0, 0.0, 0.0], dtype=float)
    # q_target_deg = np.array([0.0, 0.0, 0.0, -20.0, 0.0], dtype=float)

    q_target = np.deg2rad(q_target_deg)
    q_target = robot.check_joint_angles(q_target, context="目标关节构型")

    print("BookArm 关节运动")
    print(f"串口号: {args.port}")
    print(f"关节名称: {robot.joint_names}")
    print(f"目标关节角 deg: {format_array(q_target_deg)}")

    robot.connect_serial_arm(port=args.port)
    try:
        command = robot.move_joints_rad(q_target)
        print("\n运动命令已发送:")
        print(command)

        time.sleep(5.0)
        feedback = robot.read_arm_feedback()
    finally:
        robot.close()

    error_deg = np.rad2deg(feedback.q_rad - q_target)

    print("\n运动后的反馈:")
    print(f"  关节角 deg: {format_array(np.rad2deg(feedback.q_rad))}")
    print(f"  关节角 rad: {format_array(feedback.q_rad)}")
    print(f"  误差 deg: {format_array(error_deg)}")
    print(f"  力矩: {format_array(feedback.torque)}")


if __name__ == "__main__":
    main()
