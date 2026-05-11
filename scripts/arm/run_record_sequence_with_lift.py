"""先升降台上升 15 cm，再按硬编码构型执行固定抓取流程。

请在项目根目录运行：

    python scripts/arm/run_record_sequence_with_lift.py --port /dev/bookarm --lift-port /dev/lift_port

流程：
0. 升降台上升指定距离，默认 15 cm。
1. 打开夹爪，并移动到构型 1。
2. 以速度 25、加速度 5 移动到构型 2。
3. 以最大速度、加速度 10 移动到构型 3，然后固定等待指定时间。
4. 持续关闭夹爪，等待 5 秒。
5. 以速度 35、加速度 8 移动到构型 5，并保持该构型。

关节角单位为度，脚本会转换为弧度后下发。
升降台距离到脉冲数需要按实际机构标定；默认按 500 pulse/cm 计算。
"""

from __future__ import annotations

import argparse
import os
import struct
import time
from typing import Iterable

import numpy as np
import serial

from bookarm_control_py import BookArm


CONFIGURATIONS_DEG: dict[int, np.ndarray] = {
    1: np.array([0.0, -70.0, 60.0, 0.0, 0.0], dtype=float),
    2: np.array([1.230469, 13.095703, 51.767578, -36.035156, 11.689453], dtype=float),
    3: np.array([1.230469, 13.095703, 61.767578, -36.035156, 100.689453], dtype=float),
    5: np.array([0.0, -70.0, 60.0, 0.0, 76.289063], dtype=float),
}
MAX_ARM_SPEED = 100.0
DEFAULT_LIFT_PORT = "COM9" if os.name == "nt" else "/dev/lift_port"
DEFAULT_LIFT_DISTANCE_CM = 5.0
DEFAULT_LIFT_PULSES_PER_CM = 500.0


class YZAIM_Motor:
    def __init__(self, port=DEFAULT_LIFT_PORT, baudrate=19200, slave_id=1):
        """
        初始化电机通信。
        根据手册默认通信参数为: 19200, 8, N, 1。
        """
        self.port = port
        self.slave_id = slave_id
        self.ser = None

        try:
            self.ser = serial.Serial(
                port=port,
                baudrate=baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1,
            )
            if self.ser.is_open:
                print(f"成功连接电机于端口 {port}")
            else:
                print("串口打开失败")
                return
        except Exception as e:
            print(f"打开串口时发生错误: {e}")
            self.ser = None
            return

        self.test_speed = 2000
        self.test_accel = 10000
        self._initial_setup()

    def _crc16(self, data: bytes):
        """CRC16-Modbus 校验计算。"""
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x0001:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc = crc >> 1
        return crc.to_bytes(2, byteorder="little")

    def _send_modbus_command(self, function_code, register_address, data_value=None, is_write_single=True):
        """构造并发送 Modbus 命令。"""
        if not self.ser:
            return None

        if is_write_single and data_value is not None:
            frame = bytearray()
            frame.append(self.slave_id)
            frame.append(function_code)
            frame.extend(register_address.to_bytes(2, byteorder="big"))
            frame.extend(data_value.to_bytes(2, byteorder="big"))
        elif not is_write_single and data_value is not None and function_code == 0x10:
            frame = bytearray()
            frame.append(self.slave_id)
            frame.append(0x10)
            frame.extend(struct.pack(">H", 0x000C))
            frame.extend(struct.pack(">H", 0x0002))
            frame.append(0x04)
            if data_value < 0:
                data_value = 0xFFFFFFFF + data_value + 1
            pu_8_15 = (data_value >> 8) & 0xFF
            pu_0_7 = data_value & 0xFF
            pu_24_31 = (data_value >> 24) & 0xFF
            pu_16_23 = (data_value >> 16) & 0xFF
            frame.append(pu_8_15)
            frame.append(pu_0_7)
            frame.append(pu_24_31)
            frame.append(pu_16_23)
        else:
            return None

        frame.extend(self._crc16(frame))
        self.ser.write(frame)
        time.sleep(0.01)
        return self.ser.read(8)

    def _initial_setup(self):
        """使驱动器进入准备状态。"""
        if not self.ser:
            return

        print("正在进行电机初始化设置...")
        print("  步骤1: 使能Modbus控制...")
        self._send_modbus_command(0x06, 0x0000, 1)
        time.sleep(0.05)

        print(f"  步骤2: 设置加速度为 {self.test_accel} (r/min)/s...")
        self._send_modbus_command(0x06, 0x0003, self.test_accel)
        time.sleep(0.05)

        print(f"  步骤3: 设置目标速度为 {self.test_speed} RPM...")
        self._send_modbus_command(0x06, 0x0002, self.test_speed)
        time.sleep(0.05)

        print("  步骤4: 使能驱动器输出...")
        self._send_modbus_command(0x06, 0x0001, 1)
        time.sleep(0.1)

        print("初始化设置完成。电机已准备就绪。")

    def move_incremental(self, pulses):
        """发送增量位置指令，使电机移动指定的脉冲数。"""
        self._send_modbus_command(0x10, 0x000C, pulses, is_write_single=False)

    def stop_motor(self):
        """停止电机。"""
        self.move_incremental(0)

    def cleanup(self):
        """程序结束前清理，停止电机并关闭串口。"""
        print("正在停止电机并关闭串口...")
        if self.ser and self.ser.is_open:
            self.stop_motor()
            time.sleep(0.1)
            self._send_modbus_command(0x06, 0x0001, 0)
            self.ser.close()
        print("程序退出。")


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
    parser.add_argument("--lift-baudrate", type=int, default=19200, help="升降台串口波特率，默认 19200。")
    parser.add_argument("--lift-slave-id", type=int, default=1, help="升降台 Modbus 从站地址，默认 1。")
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
        help="升降台每厘米对应脉冲数，默认 500；请按实际机构标定。",
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
    print(f"  脉冲数: {pulses}")
    print(f"  指令后等待: {args.lift_wait:.3f} 秒")

    motor = YZAIM_Motor(
        port=args.lift_port,
        baudrate=args.lift_baudrate,
        slave_id=args.lift_slave_id,
    )
    if not motor.ser:
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
