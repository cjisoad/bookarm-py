"""Command-line entry point for the book grasp workflow."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from bookarm_control_py.task.grasp import (
    confirm_action,
    create_bookarm,
    execute_action,
    get_x_range_action,
    prepare_robot_if_startup_skipped,
    save_selected_target,
    select_target,
)
from bookarm_control_py.task.grasp.config import (
    ARM_ACCELERATION,
    ARM_SPEED,
    ARM_TEST,
    ARM_WAIT_SECONDS,
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_SERIAL,
    CAMERA_TIMEOUT_MS,
    CAMERA_TO_BASE_RPY_DEG,
    CAMERA_TO_BASE_XYZ_M,
    CAMERA_WARMUP_FRAMES,
    CAMERA_WIDTH,
    DEFAULT_AXIS_MAP,
    DEFAULT_CALIBRATION_PATH,
    DEFAULT_OFFSET_MM,
    DEFAULT_TARGET_SAVE_PATH,
    DEPTH_MAX_M,
    DEPTH_MIN_M,
    FLIP_VIEW,
    GRIPPER_WAIT_SECONDS,
    PICK_POINT_SIZE,
    POINT_CLOUD_STRIDE,
    POINT_INDEX,
    SAVE_PLY_PATH,
    TARGET_OFFSET_BASE_M,
    TRANSFORM_MATRIX_PATH,
    USE_CALIBRATION_FILE,
    VOXEL_SIZE_M,
    WORKSPACE_X_MAX_MM,
    WORKSPACE_X_MIN_MM,
    WORKSPACE_XY_RADIUS_MAX_MM,
    WORKSPACE_XY_RADIUS_MIN_MM,
    WORKSPACE_Y_MAX_MM,
    WORKSPACE_Y_MIN_MM,
    WORKSPACE_Z_MAX_MM,
    WORKSPACE_Z_MIN_MM,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="先选取目标点，再根据目标 x 坐标选择预留抓书动作。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    arm = parser.add_argument_group("arm")
    arm.add_argument("--port", default="/dev/bookarm", help="机械臂和夹爪共用串口。")
    arm.add_argument("--execute", dest="execute", action="store_true", help="真实连接串口并执行已实现动作。")
    arm.add_argument("--dry-run", dest="execute", action="store_false", help="只选点、保存和分区，不发送动作。")
    arm.set_defaults(execute=False)
    arm.add_argument("--yes", action="store_true", help="跳过真实运动前的确认提示。")
    arm.add_argument("--skip-startup", action="store_true", help="跳过选点前回到起始构型。")

    target = parser.add_argument_group("target")
    target.add_argument("--target-save", type=Path, default=DEFAULT_TARGET_SAVE_PATH, help="保存选点结果的 JSON 路径。")
    target.add_argument("--no-save-target", action="store_true", help="不保存选点结果。")
    target.add_argument("--point-index", type=int, default=POINT_INDEX, help="调试用：跳过窗口选点，直接使用点云索引。")
    target.add_argument("--ply", type=Path, default=SAVE_PLY_PATH, help="可选：保存本次点云为 PLY。")
    target.add_argument("--clamp-workspace", action="store_true", help="将目标点裁剪到工作空间范围。")

    parser.set_defaults(
        serial=CAMERA_SERIAL,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        fps=CAMERA_FPS,
        warmup=CAMERA_WARMUP_FRAMES,
        timeout_ms=CAMERA_TIMEOUT_MS,
        depth_min=DEPTH_MIN_M,
        depth_max=DEPTH_MAX_M,
        stride=POINT_CLOUD_STRIDE,
        voxel_size=VOXEL_SIZE_M,
        point_size=PICK_POINT_SIZE,
        flip_view=FLIP_VIEW,
        axis_map=DEFAULT_AXIS_MAP,
        offset_mm=DEFAULT_OFFSET_MM,
        transform_matrix=TRANSFORM_MATRIX_PATH,
        calibration=DEFAULT_CALIBRATION_PATH,
        no_calibration=not USE_CALIBRATION_FILE,
        camera_to_base_xyz=CAMERA_TO_BASE_XYZ_M,
        camera_to_base_rpy_deg=CAMERA_TO_BASE_RPY_DEG,
        target_offset_base=TARGET_OFFSET_BASE_M,
        x_min=WORKSPACE_X_MIN_MM,
        x_max=WORKSPACE_X_MAX_MM,
        y_min=WORKSPACE_Y_MIN_MM,
        y_max=WORKSPACE_Y_MAX_MM,
        z_min=WORKSPACE_Z_MIN_MM,
        z_max=WORKSPACE_Z_MAX_MM,
        xy_radius_min=WORKSPACE_XY_RADIUS_MIN_MM,
        xy_radius_max=WORKSPACE_XY_RADIUS_MAX_MM,
        arm_test=ARM_TEST,
        speed=ARM_SPEED,
        acc=ARM_ACCELERATION,
        arm_wait=ARM_WAIT_SECONDS,
        gripper_wait=GRIPPER_WAIT_SECONDS,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    robot = None
    try:
        print("BookArm 选点分区抓书流程")

        robot = create_bookarm()
        selected = select_target(robot, args)
        x_cm, action = get_x_range_action(float(selected.target_position_m[0]))

        print(f"\n目标 x={x_cm:.6f} cm，匹配动作区间: {action.label}")
        if not args.no_save_target:
            save_selected_target(selected, action, x_cm, args.target_save)

        if not args.execute:
            print("\n干跑结束：已完成选点、保存和 x 区间分类，没有发送机械臂动作。")
            return 0

        prepare_robot_if_startup_skipped(robot, args)
        if not confirm_action(args, action, x_cm):
            print("已取消，未进入动作入口。")
            return 0

        execute_action(robot, args, selected, x_cm, action)
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted by user", file=sys.stderr)
        return 130
    finally:
        if args.execute and robot is not None:
            robot.close()

    return 0
