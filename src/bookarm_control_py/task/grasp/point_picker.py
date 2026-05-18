"""RealSense/Open3D target-picking helpers for grasp tasks."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np

from bookarm_control_py.task.grasp.target import (
    clamp_arm_point_to_workspace,
    load_camera_to_base_transform,
    load_transform_matrix,
    make_workspace,
    print_target_summary,
    transform_camera_point_to_base_m,
)
from bookarm_control_py.task.grasp.types import SelectedTarget
from bookarm_control_py.task.grasp.utils import format_array


def import_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("缺少 open3d，无法打开点云选点窗口。请先安装 open3d。") from exc
    return o3d


def import_realsense_d435():
    try:
        from bookarm_control_py.camera.realsense import RealSenseD435
    except ImportError as exc:
        raise RuntimeError(
            "RealSense 相机需要 pyrealsense2。请确认已安装 Intel RealSense SDK "
            "和 Python 依赖。"
        ) from exc
    return RealSenseD435


def capture_rgbd_frame(args: argparse.Namespace):
    RealSenseD435 = import_realsense_d435()
    with RealSenseD435(
        width=args.width,
        height=args.height,
        fps=args.fps,
        serial=args.serial,
        align_depth_to_color=True,
    ) as camera:
        print(f"D435 已启动，depth scale = {camera.depth_scale:.8f} m/unit")
        camera.warmup(frame_count=args.warmup, timeout_ms=args.timeout_ms)
        return camera.get_frame(timeout_ms=args.timeout_ms)


def make_point_cloud(frame, args: argparse.Namespace):
    if args.depth_min < 0:
        raise RuntimeError("--depth-min must be non-negative.")
    if args.depth_max <= args.depth_min:
        raise RuntimeError("--depth-max must be greater than --depth-min.")
    if args.stride < 1:
        raise RuntimeError("--stride must be >= 1.")

    o3d = import_open3d()
    cloud_data = frame.to_point_cloud(
        max_depth_m=args.depth_max,
        stride=args.stride,
        include_color=True,
    )
    points = np.asarray(cloud_data.points_xyz_m, dtype=np.float64)
    colors_rgb = cloud_data.colors_rgb

    if args.depth_min > 0 and points.size:
        keep = points[:, 2] >= args.depth_min
        points = points[keep]
        if colors_rgb is not None:
            colors_rgb = colors_rgb[keep]

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    if colors_rgb is not None and len(colors_rgb) == len(points):
        colors = np.asarray(colors_rgb, dtype=np.float64) / 255.0
        cloud.colors = o3d.utility.Vector3dVector(colors)

    if args.voxel_size > 0:
        cloud = cloud.voxel_down_sample(args.voxel_size)
    return cloud


def save_point_cloud(point_cloud, path: Path) -> None:
    o3d = import_open3d()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(path), point_cloud):
        raise RuntimeError(f"保存点云失败：{path}")
    print(f"已保存点云：{path}")


def prepare_view_cloud(point_cloud, flip_view: bool):
    if not flip_view:
        return point_cloud

    view_cloud = copy.deepcopy(point_cloud)
    view_cloud.transform(
        [
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ]
    )
    return view_cloud


def pick_point_index(point_cloud, point_size: float, flip_view: bool) -> int:
    o3d = import_open3d()
    if len(point_cloud.points) == 0:
        raise RuntimeError("点云为空，请调整深度范围或相机视角。")

    print("Open3D 选点操作：")
    print("  Shift + 左键：选择一个点")
    print("  Shift + 右键：撤销上一次选择")
    print("  Q 或 Esc：结束选点")

    view_cloud = prepare_view_cloud(point_cloud, flip_view)
    visualizer = o3d.visualization.VisualizerWithEditing()
    if not visualizer.create_window("Pick Arm Target Point", width=1280, height=720):
        raise RuntimeError("创建 Open3D 选点窗口失败。")

    try:
        visualizer.add_geometry(view_cloud)
        render_options = visualizer.get_render_option()
        render_options.point_size = point_size
        render_options.background_color = np.asarray([0.02, 0.02, 0.02])
        visualizer.run()
        picked_points = visualizer.get_picked_points()
    finally:
        visualizer.destroy_window()

    if not picked_points:
        raise RuntimeError("没有选择任何点。")
    if len(picked_points) > 1:
        print(f"选中了 {len(picked_points)} 个点，将使用第一个。")
    return int(picked_points[0])


def select_target(robot: object, args: argparse.Namespace) -> SelectedTarget:
    from bookarm_control_py.task.grasp.motion import move_arm_to_startup_default

    import_open3d()
    workspace = make_workspace(args)
    matrix = load_transform_matrix(args.transform_matrix) if args.transform_matrix else None
    camera_to_base = load_camera_to_base_transform(args)

    if camera_to_base is not None:
        print("使用相机到基座外参矩阵:")
        print(camera_to_base.matrix)
    elif matrix is not None:
        print("使用 4x4 相机到机械臂矩阵，单位 mm:")
        print(matrix)
    else:
        print(f"使用轴映射: {args.axis_map}, offset-mm: {format_array(args.offset_mm)}")
    print(f"执行模式: {'真实运动' if args.execute else '干跑，不连接机械臂'}")

    if not args.skip_startup:
        if not move_arm_to_startup_default(robot, args):
            raise RuntimeError("起始构型动作未确认，停止选点。")

    frame = capture_rgbd_frame(args)
    point_cloud = make_point_cloud(frame, args)
    point_count = len(point_cloud.points)
    print(f"point count: {point_count}")
    if point_count == 0:
        raise RuntimeError("点云为空，请调整深度范围或相机视角。")

    if args.ply is not None:
        save_point_cloud(point_cloud, args.ply)

    if args.point_index is None:
        picked_index = pick_point_index(point_cloud, args.point_size, args.flip_view)
    else:
        picked_index = args.point_index
    if picked_index < 0 or picked_index >= point_count:
        raise RuntimeError(f"点索引 {picked_index} 超出范围；点云共有 {point_count} 个点。")

    camera_point_m = np.asarray(point_cloud.points)[picked_index]
    base_point_m = transform_camera_point_to_base_m(
        camera_point_m,
        args,
        matrix=matrix,
        camera_to_base=camera_to_base,
    )
    print(f"Picked point #{picked_index} 机械臂坐标系 xyz m: {format_array(base_point_m)}")
    print(f"Picked point #{picked_index} 机械臂坐标系 xyz cm: {format_array(base_point_m * 100.0)}")
    raw_target_point_m = base_point_m + np.asarray(args.target_offset_base, dtype=float)

    if args.clamp_workspace:
        final_target_point_mm, workspace_notes = clamp_arm_point_to_workspace(
            raw_target_point_m * 1000.0,
            workspace,
        )
        final_target_point_m = final_target_point_mm / 1000.0
    else:
        final_target_point_m = raw_target_point_m
        workspace_notes = ()

    selected = SelectedTarget(
        picked_index=picked_index,
        camera_point_m=np.asarray(camera_point_m, dtype=float),
        base_point_m=np.asarray(base_point_m, dtype=float),
        raw_target_point_m=np.asarray(raw_target_point_m, dtype=float),
        final_target_point_m=np.asarray(final_target_point_m, dtype=float),
        target_position_m=np.asarray(final_target_point_m, dtype=float),
        workspace_notes=workspace_notes,
    )
    print_target_summary(selected)
    return selected
