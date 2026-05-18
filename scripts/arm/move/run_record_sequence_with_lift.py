"""先升降台上升 15 cm，再按硬编码构型执行固定抓取流程。

请在项目根目录运行：

    python scripts/arm/move/run_record_sequence_with_lift.py --port /dev/bookarm --lift-port /dev/lift_port

流程：
0. 升降台上升指定距离，默认 15 cm。
1. 打开夹爪，并移动到构型 1。
2. 以速度 25、加速度 5 移动到构型 2。
3. 以最大速度、加速度 10 移动到构型 3，然后固定等待指定时间。
4. 持续关闭夹爪，等待 5 秒。
5. 以速度 35、加速度 8 移动到构型 5，并保持该构型。

关节角单位为度，脚本会转换为弧度后下发。
升降台距离到脉冲数需要按实际机构标定；默认按 4580 pulse/cm 计算。
"""

from __future__ import annotations

import argparse
import time
from typing import Iterable

import numpy as np

from bookarm_control_py import BookArm
from bookarm_control_py.actuator.lift_platform import (
    DEFAULT_LIFT_ACCELERATION,
    DEFAULT_LIFT_BAUDRATE,
    DEFAULT_LIFT_PORT,
    DEFAULT_LIFT_PULSES_PER_CM,
    DEFAULT_LIFT_SLAVE_ID,
    DEFAULT_LIFT_SPEED_RPM,
    LiftPlatformActuator,
)


CONFIGURATIONS_DEG: dict[int, np.ndarray] = {
    1: np.array([0.0, -70.0, 60.0, 0.0, 0.0], dtype=float),
    2: np.array([1.230469, 13.095703, 51.767578, -36.035156, 11.689453], dtype=float),
    3: np.array([1.230469, 13.095703, 61.767578, -36.035156, 100.689453], dtype=float),
    5: np.array([0.0, -70.0, 60.0, 0.0, 76.289063], dtype=float),
}
MAX_ARM_SPEED = 100.0
DEFAULT_LIFT_DISTANCE_CM = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="先升降台上升，再按硬编码构型执行固定流程。")
    parser.add_argument("--port", default="COM8", help="机械臂串口号，例如 COM8 或 /dev/bookarm。")
    parser.add_argument("--initial-speed", type=float, default=25.0, help="移动到构型 1 的速度。")
    parser.add_argument("--initial-acc", type=float, default=5.0, help="移动到构型 1 的加速度。")
    parser.add_argument("--gripper-wait", type=float, default=5.0, help="持续关闭夹爪后的等待时间。")
    parser.add_argument(
        "--config1-wait",
        type=float,
        default=5.0,
        help="移动到构型 1 后固定等待的时间，单位秒。",
    )
    parser.add_argument(
        "--config2-wait",
        type=float,
        default=5.0,
        help="移动到构型 2 后固定等待的时间，单位秒。",
    )
    parser.add_argument(
        "--config3-wait",
        type=float,
        default=4.0,
        help="移动到构型 3 后固定等待的时间，单位秒。",
    )
    parser.add_argument(
        "--config5-wait",
        type=float,
        default=0.0,
        help="移动到构型 5 后固定等待的时间，单位秒；默认保持构型后立即结束脚本。",
    )
    parser.add_argument("--lift-port", default=DEFAULT_LIFT_PORT, help=f"升降台串口号，默认 {DEFAULT_LIFT_PORT}。")
    parser.add_argument("--lift-baudrate", type=int, default=DEFAULT_LIFT_BAUDRATE, help="升降台串口波特率，默认 19200。")
    parser.add_argument("--lift-slave-id", type=int, default=DEFAULT_LIFT_SLAVE_ID, help="升降台 Modbus 从站地址，默认 1。")
    parser.add_argument(
        "--lift-speed-rpm",
        type=int,
        default=DEFAULT_LIFT_SPEED_RPM,
        help=f"升降台目标速度 RPM，默认 {DEFAULT_LIFT_SPEED_RPM}；调小会更慢。",
    )
    parser.add_argument(
        "--lift-acceleration",
        type=int,
        default=DEFAULT_LIFT_ACCELERATION,
        help=f"升降台加速度，默认 {DEFAULT_LIFT_ACCELERATION}；调小会更柔和。",
    )
    parser.add_argument(
        "--lift-distance-cm",
        type=float,
        default=DEFAULT_LIFT_DISTANCE_CM,
        help="升降台上升距离，单位 cm，默认 15。",
    )
    parser.add_argument(
        "--lift-pulses-per-cm",
        type=float,
        default=DEFAULT_LIFT_PULSES_PER_CM,
        help="升降台每厘米对应脉冲数，默认 4580；请按实际机构标定。",
    )
    parser.add_argument(
        "--lift-pulses",
        type=int,
        default=None,
        help="直接指定升降台增量脉冲数；设置后会覆盖 distance 和 pulses-per-cm。",
    )
    parser.add_argument(
        "--lift-direction",
        type=int,
        choices=(-1, 1),
        default=1,
        help="升降台上升方向，默认 1；如果实际向下，请改为 -1。",
    )
    parser.add_argument(
        "--lift-wait",
        type=float,
        default=8.0,
        help="发送升降台上升指令后的等待时间，单位秒，默认 8。",
    )
    return parser.parse_args()


def format_array(values: np.ndarray) -> str:
    return np.array2string(values, precision=6, suppress_small=True)


def wait(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def move_lift_platform_up(args: argparse.Namespace) -> None:
    if args.lift_pulses is None:
        pulses = int(round(args.lift_distance_cm * args.lift_pulses_per_cm)) * args.lift_direction
    else:
        pulses = int(args.lift_pulses)

    if pulses == 0:
        print("\n升降台上升脉冲数为 0，跳过升降台运动。")
        return

    print("\n升降台开始上升。")
    print(f"  串口号: {args.lift_port}")
    print(f"  目标距离: {args.lift_distance_cm:.3f} cm")
    print(f"  速度: {args.lift_speed_rpm} RPM")
    print(f"  加速度: {args.lift_acceleration}")
    print(f"  脉冲数: {pulses}")
    print(f"  指令后等待: {args.lift_wait:.3f} 秒")

    try:
        motor = LiftPlatformActuator(
            port=args.lift_port,
            baudrate=args.lift_baudrate,
            slave_id=args.lift_slave_id,
            speed_rpm=args.lift_speed_rpm,
            acceleration=args.lift_acceleration,
        ).open()
    except Exception as exc:
        raise RuntimeError(f"升降台初始化失败，请检查串口连接: {exc}") from exc

    if not motor.connected:
        raise RuntimeError("升降台初始化失败，请检查串口连接。")

    try:
        motor.move_incremental(pulses)
        wait(args.lift_wait)
    finally:
        motor.cleanup()

    print("升降台上升完成，准备开始机械臂流程。")


def move_to_configuration(
    robot: BookArm,
    q_deg: Iterable[float],
    *,
    label: str,
    speed: float,
    acc: float,
    wait_seconds: float,
) -> None:
    q_deg_array = np.asarray(q_deg, dtype=float)
    q_rad = robot.check_joint_angles(np.deg2rad(q_deg_array), context=f"{label} 构型")

    print(f"\n移动到{label}")
    print(f"  目标 q deg: {format_array(q_deg_array)}")
    print(f"  speed={speed:.3f}, acc={acc:.3f}")
    print(robot.move_joints_rad(q_rad, speed=speed, acceleration=acc))
    print(f"固定等待 {wait_seconds:.3f} 秒。")
    wait(wait_seconds)


def main() -> None:
    args = parse_args()
    robot = BookArm()
    configurations = CONFIGURATIONS_DEG

    print("BookArm 硬编码构型固定流程执行（升降台先上升）")
    print(f"机械臂串口号: {args.port}")
    print(f"升降台串口号: {args.lift_port}")
    print(f"关节名称: {robot.joint_names}")
    for index in (1, 2, 3, 5):
        print(f"构型 {index} deg: {format_array(configurations[index])}")

    move_lift_platform_up(args)

    robot.connect_serial(port=args.port)
    try:
        print("\n开启机械臂力矩。")
        print(robot.enable_torque())

        print("\n打开夹爪，并移动到构型 1。")
        print(robot.open_gripper())
        move_to_configuration(
            robot,
            configurations[1],
            label="构型 1",
            speed=args.initial_speed,
            acc=args.initial_acc,
            wait_seconds=args.config1_wait,
        )
        move_to_configuration(
            robot,
            configurations[2],
            label="构型 2",
            speed=25.0,
            acc=5.0,
            wait_seconds=args.config2_wait,
        )
        move_to_configuration(
            robot,
            configurations[3],
            label="构型 3",
            speed=MAX_ARM_SPEED,
            acc=10.0,
            wait_seconds=args.config3_wait,
        )

        print("\n持续关闭夹爪。")
        print(robot.hold_gripper_closed())
        print(f"等待 {args.gripper_wait:.3f} 秒。")
        wait(args.gripper_wait)

        move_to_configuration(
            robot,
            configurations[5],
            label="构型 5",
            speed=35.0,
            acc=8.0,
            wait_seconds=args.config5_wait,
        )

        print("\n流程完成：保持构型 5，未发送回零或关闭力矩命令。")
    finally:
        robot.close()


if __name__ == "__main__":
    main()
