"""抓取放置在转盘上的书。

请在项目根目录运行：

    python scripts/place/grasp_book_from_disk.py --dry-run
    python scripts/place/grasp_book_from_disk.py --port /dev/bookarm --execute

主流程保留在本脚本中；复用的运动执行和夹爪动作来自 src 包。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from bookarm_control_py.task.grasp.config import ACTION_REGULAR_ACC, ACTION_REGULAR_SPEED
from bookarm_control_py.task.grasp.place_book_on_disk import (
    MotionStep,
    close_gripper,
    create_bookarm,
    move_to_configuration,
    open_gripper,
    release_gripper,
)
from bookarm_control_py.task.grasp.utils import ask_yes_no, format_array


DEFAULT_PORT = "/dev/bookarm"
ARM_WAIT_SECONDS = 4.0
GRIPPER_WAIT_SECONDS = 1.0

# 抓取开始前的安全姿态。脚本启动后夹爪会先打开。
INITIAL_Q_DEG = np.array([0.0, -70.0, 60.0, 0.0, 115.0], dtype=float)
BOOKSHELF_FLOW_CONFIGURED = True

# 从初始侧转到转盘侧。
TURNAROUND_STEPS: list[dict[str, Any]] = [
    {
        "label": "转身 1：抬臂避让",
        "q_deg": [0.0, -45.0, -10.0, 45.0, 115.0],
        "speed": 35,
        "acc": 10,
        "wait": 4,
    },
    {
        "label": "转身 2：第 1 关节转到圆盘右侧",
        "q_deg": [-90.0, -45.0, -10.0, 45.0, 115.0],
        "speed": 35.0,
        "acc": 10.0,
        "wait": 3,
    },
    {
        "label": "转身 3：第 1 关节转到圆盘右后侧",
        "q_deg": [-180.0, -45.0, -10.0, 45.0, 115.0],
        "speed": 35.0,
        "acc": 10.0,
        "wait": 3,
    },
]

# 抓取转盘上书本的构型组。到达 grab_after_group 指定组后持续闭合夹爪。
GRASP_CONFIG_GROUPS: dict[int, dict[str, Any]] = {
    1: {
        "q_deg": [-191.602, -9.58, 22.061, 6.68, 115.0],
        "speed": ACTION_REGULAR_SPEED,
        "acc": ACTION_REGULAR_ACC,
        "wait": 3.0,
    },
    2: {
        "q_deg": [-193.35,  -6.328125,  61.259766, -58.095703, 110.302734],
        "speed": 25.0,
        "acc": 8.0,
        "wait": 3.0,
    },
    3: {
        "q_deg": [-194.24,  49.570312,  43.417969, -95.712891, 110.302734],
        "speed": 25.0,
        "acc": 8.0,
        "wait": 3.0,
    },
    4: {
        "q_deg": [-194.24,  56.25,  38.144531, -96.240234, 110.302734],
        "speed": 25.0,
        "acc": 8.0,
        "wait": 3.0,
    },
}

GRASP_PLAN: dict[str, Any] = {
    "group_ids": [1, 2, 3, 4],
    "grab_after_group": 4,
}

# 抓取完成并夹紧书本后，先抬起离开转盘。
LIFT_STEPS: list[dict[str, Any]] = [
    {
        "label": "抬升 1：回到圆盘右后侧安全姿态",
        "q_deg": [-193.35, 27.421875, 53.085938, -73.828125, 110.302734],
        "speed": 25.0,
        "acc": 8.0,
        "wait": 3,
    },
    {
        "label": "抬升 2：第 1 关节保持圆盘侧",
        "q_deg": [-193.35,  -6.328125,  61.259766, -58.095703, 110.302734],
        "speed": 25.0,
        "acc": 8.0,
        "wait": 2,
    },
    {
        "label": "抬升 3：准备转身返回",
        "q_deg": [-193.35,  -6.328125,  61.259766, -58.095703, 110.302734],
        "speed": ACTION_REGULAR_SPEED,
        "acc": ACTION_REGULAR_ACC,
        "wait": 2,
    },
]

# 从初始侧将已抓取的书移动到目标书缝处。
BOOKSHELF_MOVE_TO_SLOT_STEPS: list[dict[str, Any]] = [
    {
        "label": "书架放回 1：抬臂准备前往目标书缝",
        "q_deg": [  3.955078, -24.082031,   5.712891,  20.390625, 110.390625],
        "speed": ACTION_REGULAR_SPEED,
        "acc": ACTION_REGULAR_ACC,
        "wait": ARM_WAIT_SECONDS,
    },
    {
        "label": "书架放回 2：移动到目标书缝前",
        "q_deg": [  2.109375, -19.511719,  37.96875 , -30.498047, 110.390625],
        "speed": 25.0,
        "acc": 8.0,
        "wait": 3,
    },
    {
        "label": "书架放回 3：将书送入目标书缝",
        "q_deg": [  2.021484,  24.861328,  19.708984, -75.322266, 110.390625],
        "speed": 20.0,
        "acc": 6.0,
        "wait": ARM_WAIT_SECONDS,
    },
    # {
    #     "label": "书架放回 3：将书送入目标书缝",
    #     "q_deg": [  2.021484,  29.267578,  20.126953, -75.673828, 110.390625],
    #     "speed": 20.0,
    #     "acc": 6.0,
    #     "wait": ARM_WAIT_SECONDS,
    # },

]

# 到达目标书缝后释放书本。release_gripper 会自动包含打开前和打开后的等待。
BOOKSHELF_RELEASE_ACTIONS: list[dict[str, Any]] = [
    {
        "label": "书架释放 1：打开夹爪释放书本",
        "wait": GRIPPER_WAIT_SECONDS,
    },
]

# 释放书本后，先推书回到原处，再回到初始构型。
BOOKSHELF_RETURN_STEPS: list[dict[str, Any]] = [
    {
        "label": "推书 1：贴近书本",
        "q_deg": [ -1.318359, -22.060547,  37.96875 , -18.984375, 110.302734],
        "speed": 20.0,
        "acc": 6.0,
        "wait": ARM_WAIT_SECONDS,
    },
    {
        "label": "推书 2：向内推回",
        "q_deg": [ -0.263672,  33.925781,  51.416016, -73.476563, 110.302734],
        "speed": 20.0,
        "acc": 6.0,
        "wait": ARM_WAIT_SECONDS,
    },
    {
        "label": "推书 3：推书完成",
        "q_deg": [ -0.527344,  55.810547,  25.224609, -72.773437, 110.478516],
        "speed": 20.0,
        "acc": 6.0,
        "wait": ARM_WAIT_SECONDS,
    },
    {
        "label": "书架返回 1：回到初始前过渡构型",
        "q_deg": [ -0.263672,  33.925781,  51.416016, -73.476563, 110.302734],
        "speed": ACTION_REGULAR_SPEED,
        "acc": ACTION_REGULAR_ACC,
        "wait": ARM_WAIT_SECONDS,
    },
    {
        "label": "书架返回 2：回到初始构型",
        "q_deg": INITIAL_Q_DEG.tolist(),
        "speed": ACTION_REGULAR_SPEED,
        "acc": ACTION_REGULAR_ACC,
        "wait": ARM_WAIT_SECONDS,
    },
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="抓取放置在转盘上的书。默认干跑，不发送机械臂命令。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port", default=DEFAULT_PORT, help="机械臂和夹爪共用串口。")
    parser.add_argument("--execute", dest="execute", action="store_true", help="真实连接串口并执行动作。")
    parser.add_argument("--dry-run", dest="execute", action="store_false", help="只打印流程和命令，不发送动作。")
    parser.set_defaults(execute=False)
    parser.add_argument("--yes", action="store_true", help="跳过真实运动前的确认提示。")
    parser.add_argument("--start-at-initial", action="store_true", help="流程开始前先移动到初始构型。")
    parser.add_argument("--return-speed", type=float, default=ACTION_REGULAR_SPEED, help="回初始构型的速度。")
    parser.add_argument("--return-acc", type=float, default=ACTION_REGULAR_ACC, help="回初始构型的加速度。")
    return parser.parse_args(argv)


def get_turnaround_steps() -> list[MotionStep]:
    steps: list[MotionStep] = []
    for index, payload in enumerate(TURNAROUND_STEPS, start=1):
        label = str(payload.get("label", f"转身动作 {index}"))
        steps.append(MotionStep.from_mapping(label, payload))
    return steps


def get_grasp_steps() -> list[tuple[int, MotionStep]]:
    raw_group_ids = GRASP_PLAN.get("group_ids", [])
    if not isinstance(raw_group_ids, Sequence) or isinstance(raw_group_ids, (str, bytes)):
        raise ValueError("GRASP_PLAN['group_ids'] 必须是构型组编号列表。")

    steps: list[tuple[int, MotionStep]] = []
    for raw_group_id in raw_group_ids:
        group_id = int(raw_group_id)
        payload = GRASP_CONFIG_GROUPS.get(group_id)
        if payload is None:
            raise ValueError(f"GRASP_PLAN 选择了不存在的构型组 {group_id}。")
        steps.append((group_id, MotionStep.from_mapping(f"执行抓取构型组 {group_id}", payload)))
    return steps


def get_lift_steps() -> list[MotionStep]:
    steps: list[MotionStep] = []
    for index, payload in enumerate(LIFT_STEPS, start=1):
        label = str(payload.get("label", f"抬升动作 {index}"))
        steps.append(MotionStep.from_mapping(label, payload))
    return steps


def get_bookshelf_move_to_slot_steps() -> list[MotionStep]:
    steps: list[MotionStep] = []
    for index, payload in enumerate(BOOKSHELF_MOVE_TO_SLOT_STEPS, start=1):
        label = str(payload.get("label", f"书架移动动作 {index}"))
        steps.append(MotionStep.from_mapping(label, payload))
    return steps


def get_bookshelf_return_steps() -> list[MotionStep]:
    steps: list[MotionStep] = []
    for index, payload in enumerate(BOOKSHELF_RETURN_STEPS, start=1):
        label = str(payload.get("label", f"书架返回动作 {index}"))
        steps.append(MotionStep.from_mapping(label, payload))
    return steps


def get_return_steps(turnaround_steps: Sequence[MotionStep]) -> list[MotionStep]:
    return [
        MotionStep(
            label=f"返回 {index}：反向经过 {step.label}",
            q_deg=step.q_deg,
            speed=step.speed,
            acc=step.acc,
            wait_seconds=step.wait_seconds,
        )
        for index, step in enumerate(reversed(turnaround_steps), start=1)
    ]


def wait_for_next_flow(*, execute: bool) -> bool:
    if not execute:
        print("\n干跑模式：自动继续打印书架放回流程。")
        return True

    try:
        answer = input("\n转盘取书流程完成。按回车执行放回书架流程，输入 q 后回车结束程序: ").strip().lower()
    except EOFError:
        print("\n未收到输入，结束程序。")
        return False
    return answer not in {"q", "quit", "exit"}


def ensure_bookshelf_flow_configured(*, execute: bool) -> None:
    if execute and not BOOKSHELF_FLOW_CONFIGURED:
        raise ValueError(
            "书架放回流程仍是占位构型。请先调整 BOOKSHELF_MOVE_TO_SLOT_STEPS "
            "和 BOOKSHELF_RETURN_STEPS，并将 BOOKSHELF_FLOW_CONFIGURED 改为 True。"
        )


def print_grasp_from_disk_plan(
    turnaround_steps: Sequence[MotionStep],
    grasp_steps: Sequence[tuple[int, MotionStep]],
    lift_steps: Sequence[MotionStep],
    return_steps: Sequence[MotionStep],
) -> None:
    grab_after = GRASP_PLAN.get("grab_after_group")
    grab_group = None if grab_after is None else int(grab_after)

    print("\n转盘取书流程")
    print(f"初始构型 deg: {format_array(INITIAL_Q_DEG)}")
    print(f"夹爪启动状态: 打开；到达抓取组 {grab_group} 后持续闭合。")
    print("转身动作：")
    for step in turnaround_steps:
        print(
            f"  {step.label}: q deg={format_array(step.q_deg)}, "
            f"speed={step.speed:.3f}, acc={step.acc:.3f}, wait={step.wait_seconds:.3f}"
        )
    print("抓取构型组：")
    for group_id, step in grasp_steps:
        suffix = "，该组后持续闭合夹爪" if grab_group == group_id else ""
        print(
            f"  组 {group_id}: q deg={format_array(step.q_deg)}, "
            f"speed={step.speed:.3f}, acc={step.acc:.3f}, wait={step.wait_seconds:.3f}{suffix}"
        )
    print("抓取后抬升动作：")
    for step in lift_steps:
        print(
            f"  {step.label}: q deg={format_array(step.q_deg)}, "
            f"speed={step.speed:.3f}, acc={step.acc:.3f}, wait={step.wait_seconds:.3f}"
        )
    print("返回动作（TURNAROUND_STEPS 倒序）：")
    for step in return_steps:
        print(
            f"  {step.label}: q deg={format_array(step.q_deg)}, "
            f"speed={step.speed:.3f}, acc={step.acc:.3f}, wait={step.wait_seconds:.3f}"
        )


def print_bookshelf_return_plan(
    move_to_slot_steps: Sequence[MotionStep],
    return_steps: Sequence[MotionStep],
) -> None:
    print("\n放回书架流程")
    print("阶段 1：将书本移动到目标书缝处")
    for step in move_to_slot_steps:
        print(
            f"  {step.label}: q deg={format_array(step.q_deg)}, "
            f"speed={step.speed:.3f}, acc={step.acc:.3f}, wait={step.wait_seconds:.3f}"
        )
    print("阶段 2：打开夹爪释放书本")
    for index, payload in enumerate(BOOKSHELF_RELEASE_ACTIONS, start=1):
        label = str(payload.get("label", f"书架释放动作 {index}"))
        wait_seconds = float(payload.get("wait", GRIPPER_WAIT_SECONDS))
        print(f"  {label}: wait={wait_seconds:.3f}")
    print("阶段 3：机械臂返回")
    for step in return_steps:
        print(
            f"  {step.label}: q deg={format_array(step.q_deg)}, "
            f"speed={step.speed:.3f}, acc={step.acc:.3f}, wait={step.wait_seconds:.3f}"
        )


def run_grasp_from_disk_sequence(
    robot: object,
    args: argparse.Namespace,
    turnaround_steps: Sequence[MotionStep],
    grasp_steps: Sequence[tuple[int, MotionStep]],
    lift_steps: Sequence[MotionStep],
    return_steps: Sequence[MotionStep],
) -> bool:
    grab_after = GRASP_PLAN.get("grab_after_group")
    grab_group = None if grab_after is None else int(grab_after)

    print_grasp_from_disk_plan(turnaround_steps, grasp_steps, lift_steps, return_steps)
    print(f"串口号: {args.port}")
    print(f"关节名称: {robot.joint_names}")

    if args.execute and not args.yes:
        if not ask_yes_no("确认执行转盘取书流程？"):
            print("已取消，未发送运动流程。")
            return False

    if args.execute:
        robot.connect_serial(port=args.port)
        print("\n开启机械臂力矩。")
        print(robot.enable_torque())

    open_gripper(robot, execute=args.execute, wait_seconds=GRIPPER_WAIT_SECONDS)

    if args.start_at_initial:
        initial_step = MotionStep(
            label="流程开始前到达初始构型",
            q_deg=INITIAL_Q_DEG,
            speed=float(args.return_speed),
            acc=float(args.return_acc),
            wait_seconds=ARM_WAIT_SECONDS,
        )
        move_to_configuration(robot, initial_step, execute=args.execute)

    for step in turnaround_steps:
        move_to_configuration(robot, step, execute=args.execute)

    for group_id, step in grasp_steps:
        move_to_configuration(robot, step, execute=args.execute)
        if grab_group == group_id:
            close_gripper(robot, execute=args.execute, wait_seconds=GRIPPER_WAIT_SECONDS)

    for step in lift_steps:
        move_to_configuration(robot, step, execute=args.execute)

    for step in return_steps:
        move_to_configuration(robot, step, execute=args.execute)

    return_step = MotionStep(
        label="回到初始构型",
        q_deg=INITIAL_Q_DEG,
        speed=float(args.return_speed),
        acc=float(args.return_acc),
        wait_seconds=ARM_WAIT_SECONDS,
    )
    move_to_configuration(robot, return_step, execute=args.execute)

    print("\n转盘取书流程完成。")
    return True


def run_bookshelf_return_sequence(
    robot: object,
    args: argparse.Namespace,
    move_to_slot_steps: Sequence[MotionStep],
    return_steps: Sequence[MotionStep],
) -> None:
    ensure_bookshelf_flow_configured(execute=args.execute)
    print_bookshelf_return_plan(move_to_slot_steps, return_steps)

    for step in move_to_slot_steps:
        move_to_configuration(robot, step, execute=args.execute)

    for index, payload in enumerate(BOOKSHELF_RELEASE_ACTIONS, start=1):
        label = str(payload.get("label", f"书架释放动作 {index}"))
        wait_seconds = float(payload.get("wait", GRIPPER_WAIT_SECONDS))
        print(f"\n{label}")
        release_gripper(robot, execute=args.execute, wait_seconds=wait_seconds)

    for step in return_steps:
        move_to_configuration(robot, step, execute=args.execute)

    print("\n放回书架流程完成。")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    robot = None
    try:
        turnaround_steps = get_turnaround_steps()
        grasp_steps = get_grasp_steps()
        lift_steps = get_lift_steps()
        return_steps = get_return_steps(turnaround_steps)
        bookshelf_move_to_slot_steps = get_bookshelf_move_to_slot_steps()
        bookshelf_return_steps = get_bookshelf_return_steps()
        robot = create_bookarm()
        first_flow_done = run_grasp_from_disk_sequence(
            robot,
            args,
            turnaround_steps,
            grasp_steps,
            lift_steps,
            return_steps,
        )
        if first_flow_done and wait_for_next_flow(execute=args.execute):
            run_bookshelf_return_sequence(
                robot,
                args,
                bookshelf_move_to_slot_steps,
                bookshelf_return_steps,
            )
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted by user", file=sys.stderr)
        return 130
    finally:
        if robot is not None:
            robot.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
