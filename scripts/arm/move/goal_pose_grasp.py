"""BookArm 固定目标位姿抓取脚本。

请在项目根目录运行：

    python scripts/arm/move/manual_teach.py --port COM8

目标位姿在 main() 中定义：位置用 xyz，姿态用欧拉角 rpy。
欧拉角到旋转矩阵的转换调用 bookarm_control_py.math_utils.rpy_to_matrix。

逆解使用 BookArm.ikine_best_effort。它总会返回搜索过程中最接近目标的关节构型；
如果 success=False，说明误差没有达到 tolerance，目标位姿可能太难或不可达。
本抓取脚本默认会移动到 best-effort 返回的最近构型，并继续执行抓取流程。
请根据脚本打印的位置误差和姿态误差判断目标位姿是否需要调整。
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from bookarm_control_py import BookArm, rpy_to_matrix


START_Q_DEG = np.array([0.0, -60.0, 70.0, 0.0, -45.0], dtype=float)
DEFAULT_SPEED = 25.0
DEFAULT_RETURN_SPEED = 20.0
DEFAULT_ACC = 5.0
DEFAULT_ARM_WAIT = 5.0
DEFAULT_GRIPPER_WAIT = 1.0
DEFAULT_GRASP_HOLD_WAIT = 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="让 BookArm 从起始构型移动到固定目标位姿并执行抓取。"
    )
    parser.add_argument("--port", default="COM8", help="串口号，例如 COM8。")
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED, help="机械臂运动速度。")
    parser.add_argument("--return-speed", type=float, default=DEFAULT_RETURN_SPEED, help="抓取后返回起始构型的速度。")
    parser.add_argument("--acc", type=float, default=DEFAULT_ACC, help="机械臂运动加速度。")
    parser.add_argument("--arm-wait", type=float, default=DEFAULT_ARM_WAIT, help="机械臂运动后的等待秒数。")
    parser.add_argument("--gripper-wait", type=float, default=DEFAULT_GRIPPER_WAIT, help="夹爪动作后的等待秒数。")
    parser.add_argument("--grasp-hold-wait", type=float, default=DEFAULT_GRASP_HOLD_WAIT, help="执行抓取后、返回起始构型前的等待秒数。")
    parser.add_argument("--max-iterations", type=int, default=500, help="best-effort IK 最大迭代次数。")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="best-effort IK 收敛容差。")
    parser.add_argument("--damping", type=float, default=1e-6, help="阻尼最小二乘阻尼系数。")
    parser.add_argument("--step-size", type=float, default=0.4, help="IK 每步积分步长。")
    parser.add_argument(
        "--no-close",
        action="store_true",
        help="只移动到目标位姿，不闭合夹爪。",
    )
    return parser.parse_args()


def format_array(values: np.ndarray) -> str:
    return np.array2string(values, precision=6, suppress_small=True)


def wait(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def rotation_error_angle(target_rotation: np.ndarray, current_rotation: np.ndarray) -> float:
    relative_rotation = target_rotation.T @ current_rotation
    cos_angle = (np.trace(relative_rotation) - 1.0) / 2.0
    return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def main() -> None:
    args = parse_args()

    target_position = np.array([0.38, 0.0, 0.24], dtype=float)
    target_rpy_deg = np.array([0.0, -5.0, 0.0], dtype=float)
    target_rotation = rpy_to_matrix(np.deg2rad(target_rpy_deg))

    robot = BookArm()
    start_q = robot.check_joint_angles(
        np.deg2rad(START_Q_DEG),
        context="Start configuration",
    )

    ik_result = robot.ikine_best_effort(
        target_position=target_position,
        target_rotation=target_rotation,
        q0=start_q,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
        damping=args.damping,
        step_size=args.step_size,
        print_error=False,
    )

    goal_q = robot.check_joint_angles(ik_result.q, context="Goal configuration")

    print("BookArm goal pose grasp")
    print(f"serial port: {args.port}")
    print(f"start q deg: {format_array(START_Q_DEG)}")
    print(f"target position meter: {format_array(target_position)}")
    print(f"target rotation rpy deg: {format_array(target_rpy_deg)}")
    print(f"target rotation matrix:\n{format_array(target_rotation)}")
    print(f"goal q deg: {format_array(np.rad2deg(goal_q))}")
    print(f"best-effort IK success: {ik_result.success}")
    print(f"IK total error norm: {ik_result.error_norm:.8f}")
    print(f"IK position error norm: {ik_result.position_error_norm:.8f} m")
    print(f"IK rotation error deg: {np.rad2deg(ik_result.rotation_error_rad):.8f}")

    robot.connect_serial(port=args.port)
    try:
        print("\nOpen gripper.")
        print(robot.open_gripper())
        wait(args.gripper_wait)

        print("\nEnable arm torque.")
        print(robot.enable_torque())

        print("\nMove to start configuration.")
        print(robot.move_joints_rad(start_q, speed=args.speed, acceleration=args.acc))
        wait(args.arm_wait)

        print("\nMove to goal position.")
        print(robot.move_joints_rad(goal_q, speed=args.speed, acceleration=args.acc))
        wait(args.arm_wait)

        feedback = robot.read_arm_feedback()
        current_pose = robot.fkine_dict(feedback.q_rad)
        position_error = current_pose["position"] - target_position
        rotation_error = rotation_error_angle(target_rotation, current_pose["rotation"])

        print("\nFeedback after reaching goal:")
        print(f"  q deg: {format_array(np.rad2deg(feedback.q_rad))}")
        print(f"  current xyz meter: {format_array(current_pose['position'])}")
        print(f"  target xyz meter: {format_array(target_position)}")
        print(f"  position error xyz: {format_array(position_error)}")
        print(f"  position error norm: {np.linalg.norm(position_error):.8f}")
        print(f"  rotation error deg: {np.rad2deg(rotation_error):.8f}")
        print(f"  current rotation matrix:\n{format_array(current_pose['rotation'])}")

        if not args.no_close:
            print("\nHold gripper closed.")
            print(robot.hold_gripper_closed())
            wait(args.gripper_wait)

            print(f"\nWait {args.grasp_hold_wait:.1f}s before returning to start.")
            wait(args.grasp_hold_wait)
        
        print(f"\nMove to start position at speed {args.return_speed:.1f}.")
        print(robot.move_joints_rad(start_q, speed=args.return_speed, acceleration=args.acc))
        wait(args.arm_wait)

    finally:
        robot.close()


if __name__ == "__main__":
    main()
