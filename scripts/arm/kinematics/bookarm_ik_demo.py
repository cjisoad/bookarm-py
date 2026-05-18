"""Lightweight inverse kinematics demo for BookArm.

Run from the project root:

    conda run -n bookarm-beiyu python scripts/arm/bookarm_ik_demo.py

Edit TARGET_POSITION_M and INITIAL_Q_RAD below to test different IK targets.
"""

from __future__ import annotations

import numpy as np

from bookarm_control_py import BookArm


# 目标末端位置，单位是米，顺序是 x, y, z。
# 这个目标来自 FK demo 中 [0.2, -0.3, 0.4, 0.1, 0.0] 的末端位置，
# 因此默认情况下应该是可达的。
TARGET_POSITION_M = np.array([0.266179, 0.053958, 0.316738], dtype=float)

# 逆解初始构型，单位是弧度。
# 不需要手动调用限位检查；BookArm.ikine 会在内部检查。
INITIAL_Q_RAD = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)

# 当前真实末端位姿使用 link5。
END_EFFECTOR_LINK = "link5"


def format_array(values: np.ndarray) -> str:
    return np.array2string(values, precision=6, suppress_small=True)


def main() -> None:
    robot = BookArm(end_effector_link=END_EFFECTOR_LINK)

    result = robot.ikine(
        target_position=TARGET_POSITION_M,
        q0=INITIAL_Q_RAD,
        max_iterations=200,
        tolerance=1e-4,
    )

    if not result.success:
        raise RuntimeError(f"IK failed with error norm {result.error_norm:.8f}")

    solved_pose = robot.fkine_dict(result.q)
    position_error = np.linalg.norm(TARGET_POSITION_M - solved_pose["position"])

    print("BookArm inverse kinematics demo")
    print(f"URDF: {robot.urdf_path}")
    print(f"end effector: {END_EFFECTOR_LINK}")
    print(f"joint names: {robot.joint_names}")
    print(f"target position xyz meter: {format_array(TARGET_POSITION_M)}")
    print(f"initial q rad: {format_array(INITIAL_Q_RAD)}")
    print("\nIK result:")
    print(f"success: {result.success}")
    print(f"iterations: {result.iterations}")
    print(f"solver error norm: {result.error_norm:.8f}")
    print(f"solved q rad: {format_array(result.q)}")
    print(f"solved q deg: {format_array(np.rad2deg(result.q))}")
    print("\nFK check from solved q:")
    print(f"position xyz meter: {format_array(solved_pose['position'])}")
    print(f"position error meter: {position_error:.8f}")


if __name__ == "__main__":
    main()
