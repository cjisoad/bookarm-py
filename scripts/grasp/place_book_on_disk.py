"""抓取书本后放置到圆盘上的固定流程。

请在项目根目录运行：

    python scripts/grasp/place_book_on_disk.py --dry-run
    python scripts/grasp/place_book_on_disk.py --port /dev/bookarm --execute
    python scripts/grasp/place_book_on_disk.py --port /dev/bookarm --execute --start-at-initial

关节构型使用角度制填写。第 1 关节支持多圈角度，脚本会直接下发配置中的角度。
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

from bookarm_control_py.task.grasp.config import (
    ACTION_REGULAR_ACC,
    ACTION_REGULAR_SPEED,
    ARM_WAIT_SECONDS,
)
from bookarm_control_py.task.grasp.place_book_on_disk import (
    MotionStep,
    close_gripper,
    create_bookarm,
    move_to_configuration,
    open_gripper,
    release_gripper,
)
from bookarm_control_py.task.grasp.utils import ask_yes_no, flush_pending_terminal_input, format_array


DEFAULT_PORT = "/dev/bookarm"
TURN_WAIT_SECONDS = 4.0
GRIPPER_WAIT_SECONDS = 1.0

# 脚本启动前，机械臂应已处于该初始构型。脚本启动时不会先下发该构型，
# 但会在流程结束时用它作为回到初始姿态的目标。
INITIAL_Q_DEG = np.array([0.0, -70.0, 60.0, 0.0, 115.0], dtype=float)

# 抓取书本的动作序列。每个构型都可以分别配置 speed/acc/wait。
GRASP_STAGES: tuple[str, ...] = ("zero", "first", "second", "third", "final")
BOOK_GRASP_PLAN: dict[str, Any] = {
    "gripper_close_after": "second",
    "steps": [
        {
            "stage": "zero",
            "label": "初始构型",
            "q_deg": [0.0, -70.0, 60.0, 0.0, 0.0],
            "speed": 35,
            "acc": 10,
            "wait": 5,
        },
        {
            "stage": "first",
            "label": "抓取 1：靠近书本",
            "q_deg": [ -1.669922,  31.113281,  41.660156, -40.253906,   0.966797],
            "speed": ACTION_REGULAR_SPEED,
            "acc": ACTION_REGULAR_ACC,
            "wait": 6,
        },
        {
            "stage": "second",
            "label": "抓取 2：插入并准备夹紧",
            "q_deg": [ -1.669922,  31.113281,  41.660156, -60.253906,   115],
            "speed": 50.0,
            "acc": 30.0,
            "wait": 5,
        },
        {
            "stage": "third",
            "label": "抓取 3：夹紧后抬离",
            "q_deg": [  1.933,  -7.822266,  39.726562, -33.310547, 115],
            "speed": ACTION_REGULAR_SPEED,
            "acc": ACTION_REGULAR_ACC,
            "wait": 5,
        },
        {
            "stage": "final",
            "label": "抓取 4：回到放置起始构型",
            "q_deg": INITIAL_Q_DEG.tolist(),
            "speed": ACTION_REGULAR_SPEED,
            "acc": ACTION_REGULAR_ACC,
            "wait": 5,
        },
    ],
}

# 转身动作序列。这里取代旧版的单个构型 1 流程。
# 按执行顺序填写多个构型，让机械臂逐步转身到放置书本前的姿态。
TURNAROUND_STEPS: list[dict[str, Any]] = [
    {
        "label": "转身 1：抬臂避让",
        "q_deg": [0.0, -45.0, -10.0, 45.0, 115.0],
        "speed": ACTION_REGULAR_SPEED,
        "acc": ACTION_REGULAR_ACC,
        "wait": TURN_WAIT_SECONDS,
    },
    {
        "label": "转身 2：第 1 关节转到圆盘右侧",
        "q_deg": [-90, -45.0, -10.0, 45.0, 115.0],
        "speed": 25.0,
        "acc": 8.0,
        "wait": TURN_WAIT_SECONDS,
    },
    {
        "label": "转身 3：第 1 关节转到圆盘右后侧",
        "q_deg": [-180, -45.0, -10.0, 45.0, 115.0],
        "speed": 25.0,
        "acc": 8.0,
        "wait": TURN_WAIT_SECONDS,
    },
]

# 修改这个字典即可配置放置流程的构型组。这里的角度会原样用于模型校验和硬件下发。
PLACE_CONFIG_GROUPS: dict[int, dict[str, Any]] = {
    1: {
        "q_deg": [-179.648437, -23.554687, 8.701172, 6.152344, 111.269531],
        "speed": ACTION_REGULAR_SPEED,
        "acc": ACTION_REGULAR_ACC,
        "wait": 3.0,
    },
    2: {
        "q_deg": [-181.318359, -17.753906, 37.880859, -28.564453, 111.269531],
        "speed": ACTION_REGULAR_SPEED,
        "acc": ACTION_REGULAR_ACC,
        "wait": 3.0,
    },
    3: {
        "q_deg": [-192.392578, -5.361328, 38.671875, -27.158203, 111.269531],
        "speed": ACTION_REGULAR_SPEED,
        "acc": ACTION_REGULAR_ACC,
        "wait": 6.0,
    },
    4: {
        "q_deg": [-192.480469, 15.884766, 27.685547, -23.027344, 111.269531],
        "speed": ACTION_REGULAR_SPEED,
        "acc": ACTION_REGULAR_ACC,
        "wait": 3.0,
    },
    5: {
        "q_deg": [-192.744141, 23.291016, 33.486328, -42.890625, 111.269531],
        "speed": ACTION_REGULAR_SPEED,
        "acc": ACTION_REGULAR_ACC,
        "wait": 3.0,
    },
    6: {
        "q_deg": [-192.568359, 26.806641, 38.232422, -62.050781, 111.269531],
        "speed": ACTION_REGULAR_SPEED,
        "acc": ACTION_REGULAR_ACC,
        "wait": 3.0,
    },
    7: {
        "q_deg": [-192.656250, 39.638672, 46.494141, -97.382813, 111.269531],
        "speed": ACTION_REGULAR_SPEED,
        "acc": ACTION_REGULAR_ACC,
        "wait": 3.0,
    },
    8: {
        "q_deg": [-195.732422, 49.921875, 41.835937, -113.906250, 111.269531],
        "speed": ACTION_REGULAR_SPEED,
        "acc": ACTION_REGULAR_ACC,
        "wait": 3.0,
    },
    9: {
        "q_deg": [-195.644531, 64.687500, 30.410156, -115.224609, 111.269531],
        "speed": ACTION_REGULAR_SPEED,
        "acc": ACTION_REGULAR_ACC,
        "wait": 3.0,
    },
    10: {
        "q_deg": [-195.029297, 64.423828, 31.201172, -115.488281, 111.269531],
        "speed": ACTION_REGULAR_SPEED,
        "acc": ACTION_REGULAR_ACC,
        "wait": 3.0,
    },
    11: {
        "q_deg": [-191.777344, 58.095703, 41.220703, -114.697266, 111.269531],
        "speed": ACTION_REGULAR_SPEED,
        "acc": ACTION_REGULAR_ACC,
        "wait": 3.0,
    },
    12: {
        "q_deg": [-190.986328, 9.052734, 56.425781, -52.031250, 111.269531],
        "speed": ACTION_REGULAR_SPEED,
        "acc": ACTION_REGULAR_ACC,
        "wait": 3.0,
    },
}

PLACE_PLAN: dict[str, Any] = {
    # 修改这个列表即可选择要运行多少组构型，以及它们的执行顺序。
    "group_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    # 如果不需要释放夹爪，将这里设为 None。
    "release_gripper_after_group": 9,
}

# 放置并释放书本后，先抬起离开圆盘，再执行转回动作。
PLACE_LIFT_STEPS: list[dict[str, Any]] = [
    {
        "label": "放置后抬升 1：离开圆盘放置点",
        "q_deg": [-194.16, 10.283203, 52.382813, -67.060547, 110.478516],
        "speed": 25.0,
        "acc": 8.0,
        "wait": 2.0,
    },
    {
        "label": "放置后抬升 2：回到圆盘右后侧安全姿态",
        "q_deg": [-180.36, -39.111328, 60.820312, -18.193359, 110.478516],
        "speed": ACTION_REGULAR_SPEED,
        "acc": ACTION_REGULAR_ACC,
        "wait": 3.0,
    },
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="先抓取书本，再按键确认是否放置到圆盘上。默认干跑，不发送机械臂命令。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port", default=DEFAULT_PORT, help="机械臂和夹爪共用串口。")
    parser.add_argument("--execute", dest="execute", action="store_true", help="真实连接串口并执行动作。")
    parser.add_argument("--dry-run", dest="execute", action="store_false", help="只打印流程和命令，不发送动作。")
    parser.set_defaults(execute=False)
    parser.add_argument("--yes", action="store_true", help="跳过真实运动前的确认提示。")
    parser.add_argument(
        "--start-at-initial",
        action="store_true",
        help="流程开始前先移动到初始构型。",
    )
    parser.add_argument("--return-speed", type=float, default=ACTION_REGULAR_SPEED, help="回初始构型的速度。")
    parser.add_argument("--return-acc", type=float, default=ACTION_REGULAR_ACC, help="回初始构型的加速度。")
    return parser.parse_args(argv)


def get_grasp_steps() -> list[tuple[str, MotionStep]]:
    raw_steps = BOOK_GRASP_PLAN.get("steps", [])
    if not isinstance(raw_steps, Sequence) or isinstance(raw_steps, (str, bytes)):
        raise ValueError("BOOK_GRASP_PLAN['steps'] 必须是抓取构型列表。")

    steps: list[tuple[str, MotionStep]] = []
    seen_stages: set[str] = set()
    for index, payload in enumerate(raw_steps, start=1):
        if not isinstance(payload, dict):
            raise ValueError(f"BOOK_GRASP_PLAN['steps'][{index}] 必须是字典。")
        stage = str(payload.get("stage", "")).strip().lower()
        if stage not in GRASP_STAGES:
            raise ValueError(f"抓取构型 {index} 的 stage 必须是 {GRASP_STAGES} 之一。")
        if stage in seen_stages:
            raise ValueError(f"抓取构型 stage={stage!r} 重复。")
        seen_stages.add(stage)
        label = str(payload.get("label", f"抓取 {index}：{stage} 构型"))
        steps.append((stage, MotionStep.from_mapping(label, payload)))

    close_after = str(BOOK_GRASP_PLAN.get("gripper_close_after", "")).strip().lower()
    if close_after not in seen_stages:
        raise ValueError("BOOK_GRASP_PLAN['gripper_close_after'] 必须指向已配置的抓取 stage。")
    return steps


def get_turnaround_steps() -> list[MotionStep]:
    steps: list[MotionStep] = []
    for index, payload in enumerate(TURNAROUND_STEPS, start=1):
        label = str(payload.get("label", f"转身动作 {index}"))
        steps.append(MotionStep.from_mapping(label, payload))
    return steps


def get_place_steps() -> list[tuple[int, MotionStep]]:
    raw_group_ids = PLACE_PLAN.get("group_ids", [])
    if not isinstance(raw_group_ids, Sequence) or isinstance(raw_group_ids, (str, bytes)):
        raise ValueError("PLACE_PLAN['group_ids'] 必须是构型组编号列表。")

    steps: list[tuple[int, MotionStep]] = []
    for raw_group_id in raw_group_ids:
        group_id = int(raw_group_id)
        payload = PLACE_CONFIG_GROUPS.get(group_id)
        if payload is None:
            raise ValueError(f"PLACE_PLAN 选择了不存在的构型组 {group_id}。")
        steps.append(
            (
                group_id,
                MotionStep.from_mapping(f"执行放置构型组 {group_id}", payload),
            )
        )
    return steps


def get_place_lift_steps() -> list[MotionStep]:
    steps: list[MotionStep] = []
    for index, payload in enumerate(PLACE_LIFT_STEPS, start=1):
        label = str(payload.get("label", f"放置后抬升动作 {index}"))
        steps.append(MotionStep.from_mapping(label, payload))
    return steps


def get_turnback_steps(turnaround_steps: Sequence[MotionStep]) -> list[MotionStep]:
    turnback_steps: list[MotionStep] = []
    for index, step in enumerate(reversed(turnaround_steps), start=1):
        turnback_steps.append(
            MotionStep(
                label=f"转回 {index}：反向经过 {step.label}",
                q_deg=step.q_deg,
                speed=step.speed,
                acc=step.acc,
                wait_seconds=step.wait_seconds,
            )
        )
    return turnback_steps


def print_plan(
    grasp_steps: Sequence[tuple[str, MotionStep]],
    turnaround_steps: Sequence[MotionStep],
    place_steps: Sequence[tuple[int, MotionStep]],
    place_lift_steps: Sequence[MotionStep],
    turnback_steps: Sequence[MotionStep],
) -> None:
    close_after = str(BOOK_GRASP_PLAN.get("gripper_close_after", "")).strip().lower()
    release_after = PLACE_PLAN.get("release_gripper_after_group")
    release_group = None if release_after is None else int(release_after)

    print("\n抓取书本并放置到圆盘流程")
    print(f"初始构型 deg: {format_array(INITIAL_Q_DEG)}")
    print("如需流程开始前先到达初始构型，请传入 --start-at-initial。")
    print("抓取动作：")
    for stage, step in grasp_steps:
        suffix = "，该构型后持续闭合夹爪" if close_after == stage else ""
        print(
            f"  {step.label}: stage={stage}, q deg={format_array(step.q_deg)}, "
            f"speed={step.speed:.3f}, acc={step.acc:.3f}, wait={step.wait_seconds:.3f}{suffix}"
        )
    print("抓取 final 构型完成后，会等待键盘输入确认是否继续执行放置流程。")
    print("转身动作：")
    for step in turnaround_steps:
        print(
            f"  {step.label}: q deg={format_array(step.q_deg)}, "
            f"speed={step.speed:.3f}, acc={step.acc:.3f}, wait={step.wait_seconds:.3f}"
        )
    print("放置构型组：")
    for group_id, step in place_steps:
        suffix = "，该组后保持机械臂不动并释放夹爪" if release_group == group_id else ""
        print(
            f"  组 {group_id}: q deg={format_array(step.q_deg)}, "
            f"speed={step.speed:.3f}, acc={step.acc:.3f}, wait={step.wait_seconds:.3f}{suffix}"
        )
    print("放置后抬升动作：")
    for step in place_lift_steps:
        print(
            f"  {step.label}: q deg={format_array(step.q_deg)}, "
            f"speed={step.speed:.3f}, acc={step.acc:.3f}, wait={step.wait_seconds:.3f}"
        )
    print("转回动作（转身动作倒序）：")
    for step in turnback_steps:
        print(
            f"  {step.label}: q deg={format_array(step.q_deg)}, "
            f"speed={step.speed:.3f}, acc={step.acc:.3f}, wait={step.wait_seconds:.3f}"
        )


def wait_for_place_flow(*, execute: bool) -> bool:
    if not execute:
        print("\n干跑模式：自动继续打印放置书本到圆盘流程。")
        return True

    flush_pending_terminal_input()
    try:
        answer = input("\n抓取 final 构型完成。按回车继续执行放置到圆盘流程，输入 q 后回车结束程序: ").strip().lower()
    except EOFError:
        print("\n未收到输入，结束程序。")
        return False
    return answer not in {"q", "quit", "exit"}


def run_grasp_sequence(
    robot: object,
    args: argparse.Namespace,
    grasp_steps: Sequence[tuple[str, MotionStep]],
) -> bool:
    close_after = str(BOOK_GRASP_PLAN.get("gripper_close_after", "")).strip().lower()

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

    for stage, step in grasp_steps:
        move_to_configuration(robot, step, execute=args.execute)
        if close_after == stage:
            close_gripper(robot, execute=args.execute, wait_seconds=GRIPPER_WAIT_SECONDS)

    print("\n抓取流程完成。")
    return wait_for_place_flow(execute=args.execute)


def run_sequence(
    robot: object,
    args: argparse.Namespace,
    grasp_steps: Sequence[tuple[str, MotionStep]],
    turnaround_steps: Sequence[MotionStep],
    place_steps: Sequence[tuple[int, MotionStep]],
    place_lift_steps: Sequence[MotionStep],
    turnback_steps: Sequence[MotionStep],
) -> None:
    release_after = PLACE_PLAN.get("release_gripper_after_group")
    release_group = None if release_after is None else int(release_after)

    print_plan(grasp_steps, turnaround_steps, place_steps, place_lift_steps, turnback_steps)
    print(f"串口号: {args.port}")
    print(f"关节名称: {robot.joint_names}")

    if args.execute and not args.yes:
        if not ask_yes_no("确认执行抓取书本并放置到圆盘流程？"):
            print("已取消，未发送运动流程。")
            return

    if args.execute:
        robot.connect_serial(port=args.port)
        print("\n开启机械臂力矩。")
        print(robot.enable_torque())

    if not run_grasp_sequence(robot, args, grasp_steps):
        return

    for step in turnaround_steps:
        move_to_configuration(robot, step, execute=args.execute)

    for group_id, step in place_steps:
        move_to_configuration(robot, step, execute=args.execute)
        if release_group == group_id:
            release_gripper(robot, execute=args.execute, wait_seconds=GRIPPER_WAIT_SECONDS)

    for step in place_lift_steps:
        move_to_configuration(robot, step, execute=args.execute)

    for step in turnback_steps:
        move_to_configuration(robot, step, execute=args.execute)

    return_step = MotionStep(
        label="回到初始构型",
        q_deg=INITIAL_Q_DEG,
        speed=float(args.return_speed),
        acc=float(args.return_acc),
        wait_seconds=ARM_WAIT_SECONDS,
    )
    move_to_configuration(robot, return_step, execute=args.execute)

    print("\n放置流程完成。")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    robot = None
    try:
        grasp_steps = get_grasp_steps()
        turnaround_steps = get_turnaround_steps()
        place_steps = get_place_steps()
        place_lift_steps = get_place_lift_steps()
        turnback_steps = get_turnback_steps(turnaround_steps)
        robot = create_bookarm()
        run_sequence(
            robot,
            args,
            grasp_steps,
            turnaround_steps,
            place_steps,
            place_lift_steps,
            turnback_steps,
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
