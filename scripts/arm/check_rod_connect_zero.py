"""诊断杆电机是否在连接串口时自动回零。

这个脚本默认只做一件事：打开 BookArm 串口，然后暂停等待人工观察。
如果仅执行 connect_serial 后杆电机就回零，说明问题发生在串口打开或固件
初始化阶段，不是 IK 或 move_joints_rad 导致。

推荐先运行：

    python scripts/arm/check_rod_connect_zero.py --port /dev/bookarm

可选地继续测试使能力矩和发送起始构型：

    python scripts/arm/check_rod_connect_zero.py --port /dev/bookarm --enable-torque --move-start
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from bookarm_control_py import BookArm


START_Q_DEG = np.array([0.0, -60.0, 70.0, 0.0, -45.0], dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="分阶段测试 connect_serial / enable_torque / 起始构型是否导致杆电机回零。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port", default="/dev/bookarm", help="机械臂串口。")
    parser.add_argument("--enable-torque", action="store_true", help="连接后继续测试 enable_torque。")
    parser.add_argument("--move-start", action="store_true", help="连接后继续发送 START_Q_DEG 起始构型。")
    parser.add_argument("--speed", type=float, default=25.0, help="发送起始构型时的速度。")
    parser.add_argument("--acc", type=float, default=5.0, help="发送起始构型时的加速度。")
    parser.add_argument("--wait", type=float, default=2.0, help="每个动作后的等待时间，单位秒。")
    return parser.parse_args()


def wait_for_enter(message: str) -> None:
    if sys.stdin.isatty():
        input(message)
    else:
        print(message)


def main() -> int:
    args = parse_args()
    robot: BookArm | None = None

    print("杆电机回零诊断")
    print(f"  port: {args.port}")
    print(f"  START_Q_DEG: {START_Q_DEG}")
    print("\n步骤 0：此时还没有连接串口。请先观察杆电机当前位置。")
    wait_for_enter("确认后按回车，下一步只执行 robot.connect_serial(...)。")

    try:
        robot = BookArm()

        print("\n步骤 1：打开串口，共享连接机械臂和夹爪。")
        print("执行：robot.connect_serial(port=args.port)")
        robot.connect_serial(port=args.port)
        print("connect_serial 已完成。")
        time.sleep(args.wait)
        wait_for_enter("请观察杆电机是否已经回零；记录现象后按回车继续。")

        if args.enable_torque:
            print("\n步骤 2：测试 enable_torque。")
            print("执行：robot.enable_torque()")
            print(robot.enable_torque())
            time.sleep(args.wait)
            wait_for_enter("请观察杆电机是否在 enable_torque 后回零；按回车继续。")

        if args.move_start:
            start_q = np.deg2rad(START_Q_DEG)
            print("\n步骤 3：发送起始构型。")
            print(f"执行：robot.move_joints_rad({START_Q_DEG} deg)")
            print(robot.move_joints_rad(start_q, speed=args.speed, acceleration=args.acc))
            time.sleep(args.wait)
            wait_for_enter("请观察杆电机是否移动到 -45 deg；按回车结束。")

        print("\n测试结束。")
        print("判断：如果步骤 1 后杆电机就回零，原因在串口打开/固件初始化阶段。")
        print("如果步骤 1 不动、步骤 2 后回零，原因在 enable_torque 对应固件命令。")
        print("如果只有步骤 3 才动，说明 Python 运动命令在控制该关节。")
        return 0
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted by user", file=sys.stderr)
        return 130
    finally:
        if robot is not None:
            robot.close()


if __name__ == "__main__":
    raise SystemExit(main())
