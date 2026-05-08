"""测试 BookArm 正逆运动学和真实机械臂运动链路。

请在项目根目录运行：

    conda run -n bookarm-beiyu python example/3.check_fk_ik.py --port /dev/bookarm
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from bookarm_control_py import BookArm, rpy_to_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="测试 BookArm 的正运动学、逆运动学和真实机械臂运动。"
    )
    parser.add_argument("--port", default="COM8", help="串口号，例如 COM8。")
    return parser.parse_args()


def format_array(values: np.ndarray) -> str:
    return np.array2string(values, precision=6, suppress_small=True)


def rotation_error_angle(target_rotation: np.ndarray, current_rotation: np.ndarray) -> float:
    relative_rotation = target_rotation.T @ current_rotation
    cos_angle = (np.trace(relative_rotation) - 1.0) / 2.0
    return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def main() -> None:
    args = parse_args()

    robot = BookArm()

    # 用一组默认关节角生成目标位姿，这样目标一定在机械臂可达空间内。
    # target_seed_q_deg = np.array([0.0, -75.0, 75.0, 0.0, 0.0], dtype=float)
    # target_seed_q = np.deg2rad(target_seed_q_deg)
    # target_seed_q = robot.check_joint_angles(target_seed_q, context="目标位姿生成构型")
    # target_pose = robot.fkine_dict(target_seed_q)

    target_position = np.array([0.2, 0.0, 0.15], dtype=float)
    target_rpy_deg = np.array([0.0, 0.0, 0.0], dtype=float)
    target_rotation = rpy_to_matrix(np.deg2rad(target_rpy_deg))
    target_pose = {
        "position": target_position,
        "rotation": target_rotation,
    }

    print("BookArm 正逆运动学测试")
    print(f"串口号: {args.port}")
    print(f"关节名称: {robot.joint_names}")
    print("目标位姿由 main() 中的 target_position 和 target_rotation 直接给定")
    print(f"目标位置 xyz: {format_array(target_pose['position'])}")
    print(f"目标欧拉角 rpy deg: {format_array(target_rpy_deg)}")
    print(f"目标旋转矩阵:\n{format_array(target_pose['rotation'])}")

    ik_result = robot.ikine(
        target_position=target_pose["position"],
        target_rotation=target_pose["rotation"],
        q0=np.zeros(robot.nq),
        max_iterations=500,
        tolerance=1e-1,
    )
    if not ik_result.success:
        raise RuntimeError(f"逆运动学求解失败，误差 {ik_result.error_norm:.6f}")

    print("\n逆运动学结果:")
    print(f"  迭代次数: {ik_result.iterations}")
    print(f"  求解误差: {ik_result.error_norm:.8f}")
    print(f"  q rad: {format_array(ik_result.q)}")
    print(f"  q deg: {format_array(np.rad2deg(ik_result.q))}")

    solved_pose = robot.fkine_dict(ik_result.q)
    solved_position_error = solved_pose["position"] - target_pose["position"]
    solved_rotation_error = rotation_error_angle(
        target_pose["rotation"],
        solved_pose["rotation"],
    )

    print("\n逆解 q 的正运动学误差:")
    print(f"  位置误差 xyz: {format_array(solved_position_error)}")
    print(f"  位置误差范数: {np.linalg.norm(solved_position_error):.8f}")
    print(f"  姿态角误差 deg: {np.rad2deg(solved_rotation_error):.8f}")

    robot.connect_serial_arm(port=args.port)
    try:
        command = robot.move_joints_rad(ik_result.q)
        print("\n运动命令已发送:")
        print(command)

        time.sleep(5.0)
        feedback = robot.read_arm_feedback()
    finally:
        robot.close()

    current_pose = robot.fkine_dict(feedback.q_rad)
    current_position_error = current_pose["position"] - target_pose["position"]
    current_rotation_error = rotation_error_angle(
        target_pose["rotation"],
        current_pose["rotation"],
    )
    joint_error_deg = np.rad2deg(feedback.q_rad - ik_result.q)

    print("\n真实机械臂反馈:")
    print(f"  当前 q deg: {format_array(np.rad2deg(feedback.q_rad))}")
    print(f"  目标 q deg: {format_array(np.rad2deg(ik_result.q))}")
    print(f"  关节误差 deg: {format_array(joint_error_deg)}")
    print(f"  力矩: {format_array(feedback.torque)}")

    print("\n当前位姿与目标位姿误差:")
    print(f"  当前位置 xyz: {format_array(current_pose['position'])}")
    print(f"  目标位置 xyz: {format_array(target_pose['position'])}")
    print(f"  位置误差 xyz: {format_array(current_position_error)}")
    print(f"  位置误差范数: {np.linalg.norm(current_position_error):.8f}")
    print(f"  姿态角误差 deg: {np.rad2deg(current_rotation_error):.8f}")


if __name__ == "__main__":
    main()
