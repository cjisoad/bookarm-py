"""BookArm 机械臂手动示教与轨迹回放工具。

请在项目根目录运行：

    conda run -n bookarm-beiyu python scripts/arm/move/manual_teach.py --port /dev/bookarm

力矩打开时的按键：
    t  关闭机械臂力矩，开始或继续示教
    o  立即打开夹爪
    c  立即关闭夹爪
    p  回放上一次保存的轨迹文件
    q  结束程序；如果本次有记录，会保存、回初始构型并回放

断力矩示教时的按键：
    e  打开机械臂力矩，暂停示教
    o  立即打开夹爪
    c  立即关闭夹爪
    q  结束示教，保存轨迹，回初始构型并回放

示教过程中如果夹爪处于关闭状态，脚本会持续发送关闭夹爪命令，
直到用户按 o 打开夹爪。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import select
import sys
import time
from typing import Any, Literal

import numpy as np

from bookarm_control_py import BookArm


START_Q_DEG = np.array([0.0, -70.0, 60.0, 0.0, 0.0], dtype=float)
DEFAULT_TRAJECTORY_PATH = Path("recordings/manual_teach_trajectory.json")
DEFAULT_RECORD_HZ = 10.0
DEFAULT_CLOSE_REPEAT_HZ = 2.0
DEFAULT_MOVE_SPEED = 25.0
DEFAULT_MOVE_ACC = 5.0
DEFAULT_GRIPPER_WAIT = 1.0
DEFAULT_HOME_WAIT = 5.0
DEFAULT_TORQUE_ENABLE_WAIT = 0.2

GripperState = Literal["open", "closed"]


@dataclass
class Sample:
    t: float
    q_rad: list[float]
    gripper: GripperState

    def to_json(self) -> dict[str, Any]:
        return {
            "t": self.t,
            "q_rad": self.q_rad,
            "gripper": self.gripper,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BookArm 机械臂手动示教与轨迹回放。")
    parser.add_argument("--port", default="COM8", help="串口号，例如 COM8。")
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=DEFAULT_TRAJECTORY_PATH,
        help="用于保存和加载示教轨迹的 JSON 文件。",
    )
    parser.add_argument("--record-hz", type=float, default=DEFAULT_RECORD_HZ, help="示教记录频率。")
    parser.add_argument(
        "--close-repeat-hz",
        type=float,
        default=DEFAULT_CLOSE_REPEAT_HZ,
        help="示教时夹爪关闭状态下重复发送关闭命令的频率。",
    )
    parser.add_argument("--speed", type=float, default=DEFAULT_MOVE_SPEED, help="轨迹回放速度。")
    parser.add_argument("--acc", type=float, default=DEFAULT_MOVE_ACC, help="轨迹回放加速度。")
    parser.add_argument("--gripper-wait", type=float, default=DEFAULT_GRIPPER_WAIT, help="夹爪动作等待时间。")
    parser.add_argument("--home-wait", type=float, default=DEFAULT_HOME_WAIT, help="回到初始构型后的等待时间。")
    parser.add_argument(
        "--torque-enable-wait",
        type=float,
        default=DEFAULT_TORQUE_ENABLE_WAIT,
        help="发送开力矩命令后的等待时间。",
    )
    parser.add_argument(
        "--play-last",
        action="store_true",
        help="回到初始构型并回放已有轨迹文件，然后退出。",
    )
    return parser.parse_args()


def format_array(values: np.ndarray) -> str:
    return np.array2string(values, precision=4, suppress_small=True)


def read_key() -> str | None:
    if sys.platform == "win32":
        try:
            import msvcrt
        except ImportError:
            return None
        if not msvcrt.kbhit():
            return None
        return msvcrt.getwch().lower()

    readable, _, _ = select.select([sys.stdin], [], [], 0)
    if not readable:
        return None
    return sys.stdin.read(1).lower()


def wait_with_keys(seconds: float) -> None:
    deadline = time.monotonic() + max(0.0, seconds)
    while time.monotonic() < deadline:
        time.sleep(min(0.1, deadline - time.monotonic()))


def save_trajectory(path: Path, samples: list[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "start_q_deg": START_Q_DEG.tolist(),
        "joint_order": [
            "joint1_base",
            "joint2_shoulder",
            "joint3_elbow",
            "joint4_wrist",
            "joint_pole",
        ],
        "samples": [sample.to_json() for sample in samples],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_trajectory(path: Path) -> list[Sample]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        Sample(
            t=float(item["t"]),
            q_rad=[float(value) for value in item["q_rad"]],
            gripper="closed" if item.get("gripper") == "closed" else "open",
        )
        for item in payload.get("samples", [])
    ]


def ensure_arm_torque_enabled(robot: BookArm, *, wait: float) -> None:
    print(robot.enable_torque())
    if wait > 0:
        time.sleep(wait)


def move_home(
    robot: BookArm,
    start_q: np.ndarray,
    *,
    speed: float,
    acc: float,
    wait: float,
    torque_wait: float,
) -> None:
    ensure_arm_torque_enabled(robot, wait=torque_wait)
    print("\n机械臂移动到初始构型...")
    print(robot.move_joints_rad(start_q, speed=speed, acceleration=acc))
    wait_with_keys(wait)


def set_gripper(robot: BookArm, state: GripperState, *, wait: float) -> None:
    if state == "closed":
        print(robot.close_gripper())
    else:
        print(robot.open_gripper())
    wait_with_keys(wait)


def replay_trajectory(
    robot: BookArm,
    samples: list[Sample],
    *,
    speed: float,
    acc: float,
    gripper_wait: float,
    torque_wait: float,
) -> None:
    if not samples:
        print("没有可回放的轨迹点。")
        return

    print("\n确认机械臂力矩已打开，准备回放。")
    ensure_arm_torque_enabled(robot, wait=torque_wait)

    print(f"\n开始回放 {len(samples)} 个轨迹点...")
    last_t = samples[0].t
    last_gripper: GripperState | None = None

    for index, sample in enumerate(samples, start=1):
        delay = max(0.0, sample.t - last_t)
        if delay > 0:
            wait_with_keys(delay)
        last_t = sample.t

        if sample.gripper != last_gripper:
            set_gripper(robot, sample.gripper, wait=gripper_wait)
            last_gripper = sample.gripper

        ensure_arm_torque_enabled(robot, wait=torque_wait)
        command = robot.move_joints_rad(sample.q_rad, speed=speed, acceleration=acc)
        print(f"回放 {index}/{len(samples)}: {format_array(np.rad2deg(sample.q_rad))} deg")
        print(command)

    print("轨迹回放完成。")


def record_teaching(
    robot: BookArm,
    samples: list[Sample],
    *,
    record_hz: float,
    close_repeat_hz: float,
    gripper_state: GripperState,
) -> tuple[GripperState, bool]:
    print("\n进入示教模式：机械臂力矩已关闭。")
    print("按键：e=打开力矩暂停示教，o=打开夹爪，c=关闭夹爪，q=结束示教。")

    period = 1.0 / max(record_hz, 0.1)
    close_period = 1.0 / max(close_repeat_hz, 0.1)
    start = time.monotonic()
    next_record = start
    next_close = start

    while True:
        now = time.monotonic()
        key = read_key()

        if key == "e":
            print("\n打开机械臂力矩，暂停示教。")
            print(robot.enable_torque())
            return gripper_state, False
        if key == "q":
            print("\n结束示教。")
            print(robot.enable_torque())
            return gripper_state, True
        if key == "o":
            gripper_state = "open"
            print("\n打开夹爪。")
            print(robot.open_gripper())
        elif key == "c":
            gripper_state = "closed"
            print("\n关闭夹爪。")
            print(robot.close_gripper())
            next_close = now + close_period

        if gripper_state == "closed" and now >= next_close:
            robot.close_gripper()
            next_close = now + close_period

        if now >= next_record:
            try:
                feedback = robot.read_arm_feedback(response_timeout=0.2)
            except Exception as exc:  # pragma: no cover - 真实硬件路径
                print(f"读取反馈失败: {exc}")
            else:
                sample = Sample(
                    t=now - start if not samples else samples[-1].t + period,
                    q_rad=feedback.q_rad.tolist(),
                    gripper=gripper_state,
                )
                samples.append(sample)
                print(
                    "已记录 "
                    f"{len(samples)}: {format_array(np.rad2deg(feedback.q_rad))} deg, "
                    f"夹爪={gripper_state}"
                )
            next_record = now + period

        time.sleep(0.01)


def interactive_loop(
    robot: BookArm,
    start_q: np.ndarray,
    *,
    trajectory_path: Path,
    speed: float,
    acc: float,
    gripper_wait: float,
    home_wait: float,
    record_hz: float,
    close_repeat_hz: float,
    torque_wait: float,
) -> None:
    samples: list[Sample] = []
    gripper_state: GripperState = "open"

    print("\n准备就绪。")
    print("按键：t=断力矩示教，o=打开夹爪，c=关闭夹爪，p=回放已保存轨迹，q=结束。")
    if trajectory_path.exists():
        print(f"检测到已保存轨迹: {trajectory_path}")

    while True:
        key = read_key()
        if key is None:
            time.sleep(0.05)
            continue

        if key == "t":
            print("\n关闭机械臂力矩，进入手动示教。")
            print(robot.disable_torque())
            gripper_state, finished = record_teaching(
                robot,
                samples,
                record_hz=record_hz,
                close_repeat_hz=close_repeat_hz,
                gripper_state=gripper_state,
            )
            if finished:
                break
            print("\n力矩已打开。按键：t=示教，o=打开，c=关闭，p=回放，q=结束。")
        elif key == "o":
            gripper_state = "open"
            print("\n打开夹爪。")
            set_gripper(robot, "open", wait=gripper_wait)
        elif key == "c":
            gripper_state = "closed"
            print("\n关闭夹爪。")
            set_gripper(robot, "closed", wait=gripper_wait)
        elif key == "p":
            if not trajectory_path.exists():
                print(f"没有找到已保存轨迹文件: {trajectory_path}")
                continue
            saved_samples = load_trajectory(trajectory_path)
            move_home(
                robot,
                start_q,
                speed=speed,
                acc=acc,
                wait=home_wait,
                torque_wait=torque_wait,
            )
            replay_trajectory(
                robot,
                saved_samples,
                speed=speed,
                acc=acc,
                gripper_wait=gripper_wait,
                torque_wait=torque_wait,
            )
        elif key == "q":
            break

    if samples:
        save_trajectory(trajectory_path, samples)
        print(f"\n已保存 {len(samples)} 个轨迹点到 {trajectory_path}")
        move_home(
            robot,
            start_q,
            speed=speed,
            acc=acc,
            wait=home_wait,
            torque_wait=torque_wait,
        )
        replay_trajectory(
            robot,
            samples,
            speed=speed,
            acc=acc,
            gripper_wait=gripper_wait,
            torque_wait=torque_wait,
        )
    else:
        print("\n本次没有记录新的轨迹点。")


def main() -> None:
    args = parse_args()
    robot = BookArm()
    start_q = robot.check_joint_angles(np.deg2rad(START_Q_DEG), context="初始构型")

    print("BookArm 机械臂手动示教")
    print(f"串口号: {args.port}")
    print(f"初始构型 deg: {format_array(START_Q_DEG)}")
    print(f"轨迹文件: {args.trajectory}")

    robot.connect_serial(port=args.port)
    try:
        print("\n打开夹爪并开启机械臂力矩。")
        set_gripper(robot, "open", wait=args.gripper_wait)
        ensure_arm_torque_enabled(robot, wait=args.torque_enable_wait)
        move_home(
            robot,
            start_q,
            speed=args.speed,
            acc=args.acc,
            wait=args.home_wait,
            torque_wait=args.torque_enable_wait,
        )

        if args.play_last:
            samples = load_trajectory(args.trajectory)
            replay_trajectory(
                robot,
                samples,
                speed=args.speed,
                acc=args.acc,
                gripper_wait=args.gripper_wait,
                torque_wait=args.torque_enable_wait,
            )
            return

        interactive_loop(
            robot,
            start_q,
            trajectory_path=args.trajectory,
            speed=args.speed,
            acc=args.acc,
            gripper_wait=args.gripper_wait,
            home_wait=args.home_wait,
            record_hz=args.record_hz,
            close_repeat_hz=args.close_repeat_hz,
            torque_wait=args.torque_enable_wait,
        )
    finally:
        robot.close()


if __name__ == "__main__":
    main()
