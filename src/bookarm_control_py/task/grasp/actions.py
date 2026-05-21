"""X-range grasp action table for book grasping."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

import numpy as np

from bookarm_control_py.task.grasp.config import (
    ACTION_REGULAR_ACC,
    ACTION_REGULAR_SPEED,
    ACTION_X_MAX_EXCLUSIVE_CM,
    ACTION_X_MIN_CM,
)
from bookarm_control_py.task.grasp.motion import move_to_joint_configuration
from bookarm_control_py.task.grasp.types import ActionContext, XRangeAction
from bookarm_control_py.task.grasp.utils import wait


@dataclass(frozen=True)
class BookGraspSequence:
    first_q_deg: np.ndarray
    second_q_deg: np.ndarray
    final_q_deg: np.ndarray
    third_q_deg: np.ndarray | None = None
    gripper_close_after: Literal["first", "second", "third", "final"] = "second"
    first_speed: float = ACTION_REGULAR_SPEED
    first_acc: float = ACTION_REGULAR_ACC
    second_speed: float = 35.0
    second_acc: float = 10.0
    third_speed: float = ACTION_REGULAR_SPEED
    third_acc: float = ACTION_REGULAR_ACC
    final_speed: float = ACTION_REGULAR_SPEED
    final_acc: float = ACTION_REGULAR_ACC

    @classmethod
    def from_degrees(
        cls,
        first_q_deg: Sequence[float],
        second_q_deg: Sequence[float],
        final_q_deg: Sequence[float],
        third_q_deg: Sequence[float] | None = None,
        gripper_close_after: Literal["first", "second", "third", "final"] = "second",
        **kwargs,
    ) -> "BookGraspSequence":
        return cls(
            first_q_deg=np.asarray(first_q_deg, dtype=float),
            second_q_deg=np.asarray(second_q_deg, dtype=float),
            final_q_deg=np.asarray(final_q_deg, dtype=float),
            third_q_deg=None if third_q_deg is None else np.asarray(third_q_deg, dtype=float),
            gripper_close_after=gripper_close_after,
            **kwargs,
        )


BOOK_GRASP_SEQUENCES: dict[int, BookGraspSequence] = {
    35: BookGraspSequence.from_degrees(
        first_q_deg=[  2.197266,  26.630859,  42.978516, -41.396484,   1.845703],
        second_q_deg=[2.197266,  32.630859,  42.978516, -60.396484,   115],
        third_q_deg=[  2.548828,  -9.404297,  38.232422, -32.958984,  115],
        final_q_deg=[0.0, -70.0, 60.0, 0.0, 115.0],
        gripper_close_after="second",
    ),
    36: BookGraspSequence.from_degrees(
        first_q_deg=[  2.197266,  26.630859,  42.978516, -41.396484,   1.845703],
        second_q_deg=[2.197266,  32.630859,  42.978516, -60.396484,   115],
        third_q_deg=[  2.548828,  -9.404297,  38.232422, -32.958984,  115],
        final_q_deg=[0.0, -70.0, 60.0, 0.0, 115.0],
        gripper_close_after="second",
    ),
    37: BookGraspSequence.from_degrees(
        first_q_deg=[0,  28.476562,  40.605469, -30.585937, -11.25],
        second_q_deg=[0,  32.476562,  40.605469, -65.585937, 115],
        third_q_deg=[0,  -10.630859,  38.978516, -32.396484,   115],
        final_q_deg=[0.0, -70.0, 60.0, 0.0, 115.0],
        gripper_close_after="second",
    ),
    38: BookGraspSequence.from_degrees(
        first_q_deg=[0,  28.476562,  40.605469, -30.585937, -11.25],
        second_q_deg=[0,  32.476562,  40.605469, -65.585937, 115],
        third_q_deg=[0,  -10.630859,  38.978516, -32.396484,   115],
        final_q_deg=[0.0, -70.0, 60.0, 0.0, 115.0],
        gripper_close_after="second",
    ),
    50: BookGraspSequence.from_degrees(
        first_q_deg=[  3.691406,  50.537109,  -1.40625 , -23.818359,  10.371094],
        second_q_deg=[  3.691406,  50.537109,  -6.40625 , -40.818359,  115],
        third_q_deg=[  3.779297,  -0.527344,  24.960938, -28.652344,  115],
        final_q_deg=[0.0, -70.0, 60.0, 0.0, 115.0],
        gripper_close_after="second",
    ),
}


def get_x_range_action(x_m: float) -> tuple[float, XRangeAction]:
    x_cm = float(x_m) * 100.0
    if not (ACTION_X_MIN_CM <= x_cm < ACTION_X_MAX_EXCLUSIVE_CM):
        raise RuntimeError(
            "目标点 x 不在可执行范围内："
            f"x={x_cm:.6f} cm，需要 {ACTION_X_MIN_CM}-{ACTION_X_MAX_EXCLUSIVE_CM} cm。"
        )

    start_cm = int(math.floor(x_cm))
    action = ACTION_HANDLERS.get(start_cm)
    if action is None:
        raise RuntimeError(f"目标点 x={x_cm:.6f} cm 未找到对应动作区间。")
    return x_cm, action


def run_reserved_action(context: ActionContext) -> None:
    print(f"\n进入动作区间 {context.action.label}，目标 x={context.x_cm:.6f} cm。")
    print("该区间动作入口已预留，尚未填写具体关节构型/运动序列。")
    print("当前不会发送机械臂运动命令。")
    print("后续可以在 BOOK_GRASP_SEQUENCES 中为该区间填写 first/second/final 构型。")


def run_book_grasp_sequence(context: ActionContext, sequence: BookGraspSequence) -> None:
    def maybe_close_gripper(stage: str) -> None:
        if sequence.gripper_close_after != stage:
            return
        print(f"\n{context.action.label}: 持续关闭夹爪。")
        print(context.robot.hold_gripper_closed())
        wait(context.args.gripper_wait)

    move_to_joint_configuration(
        context,
        sequence.first_q_deg,
        label=f"{context.action.label} 第一构型",
        speed=sequence.first_speed,
        acc=sequence.first_acc,
    )
    maybe_close_gripper("first")

    move_to_joint_configuration(
        context,
        sequence.second_q_deg,
        label=f"{context.action.label} 第二构型",
        speed=sequence.second_speed,
        acc=sequence.second_acc,
    )
    maybe_close_gripper("second")

    if sequence.third_q_deg is not None:
        move_to_joint_configuration(
            context,
            sequence.third_q_deg,
            label=f"{context.action.label} 第三构型",
            speed=sequence.third_speed,
            acc=sequence.third_acc,
        )
        maybe_close_gripper("third")

    move_to_joint_configuration(
        context,
        sequence.final_q_deg,
        label=f"{context.action.label} 最终构型",
        speed=sequence.final_speed,
        acc=sequence.final_acc,
    )
    maybe_close_gripper("final")


def run_configured_or_reserved_action(context: ActionContext) -> None:
    sequence = BOOK_GRASP_SEQUENCES.get(context.action.start_cm)
    if sequence is None:
        run_reserved_action(context)
        return
    run_book_grasp_sequence(context, sequence)


ACTION_HANDLERS: dict[int, XRangeAction] = {
    start_cm: XRangeAction(start_cm, start_cm + 1, run_configured_or_reserved_action)
    for start_cm in range(ACTION_X_MIN_CM, ACTION_X_MAX_EXCLUSIVE_CM)
}


def print_action_table() -> None:
    print("\n已预留 x 区间动作：")
    for action in ACTION_HANDLERS.values():
        print(f"  {action.label}: {action.handler.__name__}")
