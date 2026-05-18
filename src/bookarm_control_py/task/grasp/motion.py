"""Arm motion helpers for grasp tasks."""

from __future__ import annotations

import argparse
from typing import Sequence

import numpy as np

from bookarm_control_py.task.grasp.config import START_Q_DEG
from bookarm_control_py.task.grasp.types import ActionContext
from bookarm_control_py.task.grasp.utils import ask_yes_no, format_array, wait


def create_bookarm():
    try:
        from bookarm_control_py import BookArm
    except ImportError as exc:
        raise RuntimeError(
            "BookArm 需要 pinocchio。请激活 bookarm-beiyu 环境，"
            "或从 conda-forge 安装 pinocchio。"
        ) from exc
    return BookArm()


def move_arm_to_startup_default(robot, args: argparse.Namespace) -> bool:
    start_q = robot.check_joint_angles(np.deg2rad(START_Q_DEG), context="起始构型")
    print("\n相机采集前起始构型")
    print(f"  q deg: {format_array(START_Q_DEG)}")

    if not args.execute:
        print("  干跑模式：未发送起始构型命令。")
        return True

    robot.connect_serial(port=args.port)
    print("  打开夹爪。")
    print(robot.open_gripper())
    wait(args.gripper_wait)

    print("  开启机械臂力矩。")
    print(robot.enable_torque())

    print("  移动到起始构型。")
    print(robot.move_joints_rad(start_q, speed=args.speed, acceleration=args.acc))
    wait(args.arm_wait)

    if not args.arm_test or args.yes:
        return True
    if ask_yes_no("机械臂是否已正常回到起始构型？"):
        return True
    print("起始构型动作未确认，停止相机采集。")
    return False


def prepare_robot_if_startup_skipped(robot: object, args: argparse.Namespace) -> None:
    if not args.execute or not args.skip_startup:
        return

    print("\n跳过起始构型流程，准备真实动作串口。")
    robot.connect_serial(port=args.port)
    print("开启机械臂力矩。")
    print(robot.enable_torque())
    print("打开夹爪。")
    print(robot.open_gripper())
    wait(args.gripper_wait)


def move_to_joint_configuration(
    context: ActionContext,
    q_deg: Sequence[float],
    *,
    label: str,
    speed: float,
    acc: float,
) -> None:
    q_deg_array = np.asarray(q_deg, dtype=float)
    q_rad = context.robot.check_joint_angles(np.deg2rad(q_deg_array), context=label)

    print(f"\n{context.action.label}: 移动到{label}")
    print(f"  q deg: {format_array(q_deg_array)}")
    print(f"  speed={speed:.3f}, acc={acc:.3f}")
    print(context.robot.move_joints_rad(q_rad, speed=speed, acceleration=acc))
    wait(context.args.arm_wait)
