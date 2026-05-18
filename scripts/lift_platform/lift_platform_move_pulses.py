"""
升降台按脉冲数移动脚本

启动方式：
    python scripts/lift_platform/lift_platform_move_pulses.py --pulses 1000

说明：
    正脉冲表示一个方向，负脉冲表示反方向。方向含义取决于实际接线和机构安装。
    退出时默认只关闭串口，不发送断力矩命令。
"""

import argparse
from pathlib import Path
import sys
import time

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bookarm_control_py.actuator.lift_platform import DEFAULT_LIFT_PORT, LiftPlatformActuator


DEFAULT_WAIT_SECONDS = 8.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="控制升降台按脉冲数移动。")
    parser.add_argument("--port", default=DEFAULT_LIFT_PORT, help=f"串口设备，默认 {DEFAULT_LIFT_PORT}")
    parser.add_argument("--baudrate", type=int, default=19200, help="串口波特率，默认 19200")
    parser.add_argument("--slave-id", type=int, default=1, help="Modbus 从站地址，默认 1")
    parser.add_argument(
        "--speed-rpm",
        type=int,
        default=200,
        help="升降台目标速度 RPM，默认 200；调小会更慢。",
    )
    parser.add_argument(
        "--acceleration",
        type=int,
        default=100,
        help="升降台加速度，默认 100；调小会更柔和。",
    )
    parser.add_argument(
        "--pulses",
        type=int,
        required=True,
        help="要发送的增量脉冲数；正数和负数分别对应两个方向。",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="重复发送次数，默认 1；用于模拟连续发送增量脉冲。",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.01,
        help="重复发送时每次之间的间隔，单位秒，默认 0.01。",
    )
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=DEFAULT_WAIT_SECONDS,
        help="发送指令后的等待时间，单位秒；应大于实际运动时间，默认 8。",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印 Modbus 发送和接收字节，便于排查通信问题。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.pulses == 0:
        print("脉冲数为 0，跳过运动。")
        return 0
    if args.repeat <= 0:
        print("repeat 必须大于 0。")
        return 2

    print("升降台按脉冲移动")
    print(f"  串口: {args.port}")
    print(f"  速度: {args.speed_rpm} RPM")
    print(f"  加速度: {args.acceleration}")
    print(f"  脉冲: {args.pulses}")
    print(f"  重复: {args.repeat}")
    print(f"  间隔: {args.interval:.3f} s")
    print(f"  等待: {args.wait_seconds:.3f} s")

    try:
        motor = LiftPlatformActuator(
            port=args.port,
            baudrate=args.baudrate,
            slave_id=args.slave_id,
            speed_rpm=args.speed_rpm,
            acceleration=args.acceleration,
            verbose=args.verbose,
        ).open()
    except Exception as exc:
        print(f"升降台初始化失败，请检查串口连接: {exc}")
        return 1

    if not motor.connected:
        print("升降台初始化失败，请检查串口连接。")
        return 1

    try:
        for index in range(args.repeat):
            response = motor.move_incremental(args.pulses)
            if args.verbose:
                print(f"move {index + 1}/{args.repeat} response length: {len(response or b'')}")
            if index + 1 < args.repeat:
                time.sleep(max(args.interval, 0.0))
        time.sleep(max(args.wait_seconds, 0.0))
    finally:
        motor.close()

    print("升降台按脉冲移动完成，力矩保持未被关闭。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
