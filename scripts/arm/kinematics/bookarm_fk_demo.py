"""Lightweight forward kinematics demo for BookArm.

Run from the project root:

    conda run -n bookarm-beiyu python scripts/arm/bookarm_fk_demo.py

Edit JOINT_ANGLES_RAD below to test different arm poses.
"""

from __future__ import annotations

import numpy as np

from bookarm_control_py import BookArm




# The real end-effector pose currently uses link5.
END_EFFECTOR_LINK = "link5"


def format_array(values: np.ndarray) -> str:
    return np.array2string(values, precision=6, suppress_small=True)


def main() -> None:
    robot = BookArm(end_effector_link=END_EFFECTOR_LINK)
    
    # Joint angles in radians, ordered as:
    # joint1_base, joint2_shoulder, joint3_elbow, joint4_wrist, joint_pole
    q = np.array([0.2, -0.3, 0.4, 0.1, 0.0], dtype=float)
    fk = robot.fkine_dict(q)

    print("BookArm forward kinematics demo")
    print(f"URDF: {robot.urdf_path}")
    print(f"end effector: {END_EFFECTOR_LINK}")
    print(f"joint names: {robot.joint_names}")
    print(f"q rad: {format_array(q)}")
    print(f"q deg: {format_array(np.rad2deg(q))}")
    print("\nFK result:")
    print(f"position xyz meter: {format_array(fk['position'])}")
    print(f"rotation matrix:\n{format_array(fk['rotation'])}")
    print(f"transform matrix:\n{format_array(fk['transform'])}")


if __name__ == "__main__":
    main()
