"""
升降台固定距离移动脚本

启动方式：
    python3 scripts/lift_platform/lift_platform_move_distance.py --distance-cm 5 --speed-rpm 200 --acceleration 100

    python3 scripts/lift_platform/lift_platform_move_distance.py --distance-cm -30
    python3 scripts/lift_platform/lift_platform_move_distance.py --distance-cm 30

说明：
    距离到脉冲数需要按实际机构标定，默认按 4580 pulse/cm 计算。
    --distance-cm 为正数时按上升方向移动，为负数时按下降方向移动。
"""

import argparse
from pathlib import Path
import sys
import time

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bookarm_control_py.actuator.lift_platform import (
    DEFAULT_LIFT_ACCELERATION,
    DEFAULT_LIFT_PORT,
    DEFAULT_LIFT_PULSES_PER_CM,
    DEFAULT_LIFT_SPEED_RPM,
    LiftPlatformActuator,
)


DEFAULT_WAIT_SECONDS = 1.5


def nonzero_float(value: str) -> float:
    parsed = float(value)
    if parsed == 0:
        raise argparse.ArgumentTypeError("必须是非 0 数值；正数上升，负数下降。")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须是大于 0 的数值。")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="控制升降台移动固定距离。")
    parser.add_argument("--port", default=DEFAULT_LIFT_PORT, help=f"串口设备，默认 {DEFAULT_LIFT_PORT}")
    parser.add_argument("--baudrate", type=int, default=19200, help="串口波特率，默认 19200")
    parser.add_argument("--slave-id", type=int, default=1, help="Modbus 从站地址，默认 1")
    parser.add_argument(
        "--speed-rpm",
        type=int,
        default=DEFAULT_LIFT_SPEED_RPM,
        help=f"升降台目标速度 RPM，默认 {DEFAULT_LIFT_SPEED_RPM}；调小会更慢。",
    )
    parser.add_argument(
        "--acceleration",
        type=int,
        default=DEFAULT_LIFT_ACCELERATION,
        help=f"升降台加速度，默认 {DEFAULT_LIFT_ACCELERATION}；调小会更柔和。",
    )
    parser.add_argument(
        "--distance-cm",
        type=nonzero_float,
        default=None,
        help="移动距离，单位 cm；正数上升，负数下降；不指定 --pulses 时必须提供。",
    )
    parser.add_argument(
        "--pulses-per-cm",
        type=positive_float,
        default=DEFAULT_LIFT_PULSES_PER_CM,
        help="每厘米对应脉冲数，默认 4580，请按实际机构标定。",
    )
    parser.add_argument(
        "--pulses",
        type=int,
        default=None,
        help="直接指定增量脉冲数；设置后会覆盖 --distance-cm。",
    )
    parser.add_argument(
        "--up-direction",
        type=int,
        choices=(-1, 1),
        default=1,
        help="上升对应的脉冲方向，默认 1；如果实际方向相反，设为 -1。",
    )
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=DEFAULT_WAIT_SECONDS,
        help="发送指令后的等待时间，单位秒；应大于实际运动时间，默认 8。",
    )
    return parser.parse_args()


def calculate_pulses(args: argparse.Namespace) -> int:
    if args.pulses is not None:
        return args.pulses

    if args.distance_cm is None:
        raise ValueError("请提供 --distance-cm，或用 --pulses 直接指定脉冲数。")

    return int(round(args.distance_cm * args.pulses_per_cm)) * args.up_direction


def main() -> int:
    args = parse_args()

    try:
        pulses = calculate_pulses(args)
    except ValueError as exc:
        print(f"参数错误: {exc}")
        return 2

    if pulses == 0:
        print("目标脉冲数为 0，跳过运动。")
        return 0

    print("升降台固定距离移动")
    print(f"  串口: {args.port}")
    print(f"  速度: {args.speed_rpm} RPM")
    print(f"  加速度: {args.acceleration}")
    if args.distance_cm is not None:
        print(f"  方向: {'up' if args.distance_cm > 0 else 'down'}")
        print(f"  距离: {args.distance_cm:.3f} cm")
        print(f"  标定: {args.pulses_per_cm:.3f} pulse/cm")
    print(f"  脉冲: {pulses}")
    print(f"  等待: {args.wait_seconds:.3f} s")

    try:
        motor = LiftPlatformActuator(
            port=args.port,
            baudrate=args.baudrate,
            slave_id=args.slave_id,
            speed_rpm=args.speed_rpm,
            acceleration=args.acceleration,
        ).open()
    except Exception as exc:
        print(f"升降台初始化失败，请检查串口连接: {exc}")
        return 1

    if not motor.connected:
        print("升降台初始化失败，请检查串口连接。")
        return 1

    try:
        if args.pulses is not None:
            motor.move_incremental(pulses)
        else:
            motor.move_lift_distance_cm(
                args.distance_cm,
                speed_rpm=args.speed_rpm,
                acceleration=args.acceleration,
                pulses_per_cm=args.pulses_per_cm,
            )
        time.sleep(max(args.wait_seconds, 0.0))
    finally:
        motor.close()

    print("升降台固定距离移动完成，力矩保持未被关闭。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
