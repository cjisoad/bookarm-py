"""Reusable helpers for the book-on-disk scripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, TYPE_CHECKING

import numpy as np

from bookarm_control_py.task.grasp.config import ACTION_REGULAR_ACC, ACTION_REGULAR_SPEED
from bookarm_control_py.task.grasp.utils import format_array, wait

if TYPE_CHECKING:
    from bookarm_control_py import BookArm


BOUNDARY_TOLERANCE_RAD = 1e-5


@dataclass(frozen=True)
class MotionStep:
    label: str
    q_deg: np.ndarray
    speed: float
    acc: float
    wait_seconds: float

    @classmethod
    def from_mapping(
        cls,
        label: str,
        payload: Mapping[str, Any],
        *,
        default_speed: float = ACTION_REGULAR_SPEED,
        default_acc: float = ACTION_REGULAR_ACC,
    ) -> "MotionStep":
        try:
            q_deg = np.asarray(payload["q_deg"], dtype=float)
        except KeyError as exc:
            raise ValueError(f"{label} 缺少 q_deg") from exc

        return cls(
            label=label,
            q_deg=q_deg,
            speed=float(payload.get("speed", default_speed)),
            acc=float(payload.get("acc", default_acc)),
            wait_seconds=float(payload.get("wait", 0.0)),
        )


def create_bookarm() -> "BookArm":
    try:
        from bookarm_control_py import BookArm
    except ImportError as exc:
        raise RuntimeError(
            "BookArm 需要 pinocchio。请激活 bookarm-beiyu 环境，"
            "或从 conda-forge 安装 pinocchio。"
        ) from exc
    return BookArm()


def snap_to_joint_limits(robot: "BookArm", q_rad: np.ndarray) -> np.ndarray:
    q = np.asarray(q_rad, dtype=float).copy()
    lower = robot.model.lowerPositionLimit
    upper = robot.model.upperPositionLimit
    near_lower = (q < lower) & np.isclose(q, lower, atol=BOUNDARY_TOLERANCE_RAD)
    near_upper = (q > upper) & np.isclose(q, upper, atol=BOUNDARY_TOLERANCE_RAD)
    q[near_lower] = lower[near_lower]
    q[near_upper] = upper[near_upper]
    return q


def checked_joint_radians(robot: "BookArm", q_deg: Sequence[float], *, label: str) -> np.ndarray:
    q_deg_array = np.asarray(q_deg, dtype=float)
    if q_deg_array.shape != (robot.nq,):
        raise ValueError(f"{label} 需要 {robot.nq} 个关节角，收到 {q_deg_array.shape[0]} 个。")

    q_rad = snap_to_joint_limits(robot, np.deg2rad(q_deg_array))
    return robot.check_joint_angles(q_rad, context=label)


def move_to_configuration(
    robot: "BookArm",
    step: MotionStep,
    *,
    execute: bool,
) -> None:
    q_rad = checked_joint_radians(robot, step.q_deg, label=step.label)

    print(f"\n{step.label}")
    print(f"  q deg: {format_array(np.rad2deg(q_rad))}")
    print(f"  speed={step.speed:.3f}, acc={step.acc:.3f}")

    if execute:
        print(robot.move_joints_rad(q_rad, speed=step.speed, acceleration=step.acc))
    else:
        print("  干跑模式：未发送机械臂运动命令。")

    actual_wait = max(float(step.wait_seconds), 0.0)
    print(f"  等待 {actual_wait:.3f} 秒。")
    wait(actual_wait)


def release_gripper(robot: "BookArm", *, execute: bool, wait_seconds: float) -> None:
    print("\n释放夹爪。")
    print("  机械臂保持当前构型不动，仅打开夹爪。")
    actual_wait = max(float(wait_seconds), 0.0)
    print(f"  打开前等待 {actual_wait:.3f} 秒。")
    wait(actual_wait)

    if execute:
        print(robot.open_gripper())
    else:
        print("  干跑模式：未发送夹爪释放命令。")

    print(f"  打开后等待 {actual_wait:.3f} 秒。")
    wait(actual_wait)


def open_gripper(robot: "BookArm", *, execute: bool, wait_seconds: float) -> None:
    print("\n打开夹爪。")
    if execute:
        print(robot.open_gripper())
    else:
        print("  干跑模式：未发送夹爪打开命令。")
    actual_wait = max(float(wait_seconds), 0.0)
    print(f"  等待 {actual_wait:.3f} 秒。")
    wait(actual_wait)


def close_gripper(robot: "BookArm", *, execute: bool, wait_seconds: float) -> None:
    print("\n持续闭合夹爪。")
    if execute:
        print(robot.hold_gripper_closed())
    else:
        print("  干跑模式：未发送夹爪持续闭合命令。")
    actual_wait = max(float(wait_seconds), 0.0)
    print(f"  等待 {actual_wait:.3f} 秒。")
    wait(actual_wait)
