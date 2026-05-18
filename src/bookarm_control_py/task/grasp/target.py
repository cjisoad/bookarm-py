"""Target geometry and workspace helpers for grasp tasks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from bookarm_control_py.camera.geometry import RigidTransform
from bookarm_control_py.task.grasp.config import DEFAULT_CALIBRATION_PATH
from bookarm_control_py.task.grasp.types import SelectedTarget, Workspace
from bookarm_control_py.task.grasp.utils import format_array


def make_workspace(args: argparse.Namespace) -> Workspace:
    if args.x_min > args.x_max:
        raise RuntimeError("--x-min must be <= --x-max.")
    if args.y_min > args.y_max:
        raise RuntimeError("--y-min must be <= --y-max.")
    if args.z_min > args.z_max:
        raise RuntimeError("--z-min must be <= --z-max.")
    if args.xy_radius_min < 0 or args.xy_radius_max < 0:
        raise RuntimeError("XY radius limits must be non-negative.")
    if args.xy_radius_min > args.xy_radius_max:
        raise RuntimeError("--xy-radius-min must be <= --xy-radius-max.")
    return Workspace(
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        z_min=args.z_min,
        z_max=args.z_max,
        xy_radius_min=args.xy_radius_min,
        xy_radius_max=args.xy_radius_max,
    )


def load_camera_to_base_transform(args: argparse.Namespace) -> RigidTransform | None:
    explicit_xyz_rpy = args.camera_to_base_xyz is not None or args.camera_to_base_rpy_deg is not None

    if args.transform_matrix is not None:
        if explicit_xyz_rpy:
            raise RuntimeError("--transform-matrix cannot be combined with --camera-to-base-*.")
        return None

    if explicit_xyz_rpy:
        if args.camera_to_base_xyz is None or args.camera_to_base_rpy_deg is None:
            raise RuntimeError("--camera-to-base-xyz 和 --camera-to-base-rpy-deg 必须同时提供。")
        return RigidTransform.from_xyz_rpy_deg(
            args.camera_to_base_xyz,
            args.camera_to_base_rpy_deg,
        )

    if args.no_calibration or args.calibration is None:
        return None

    if not args.calibration.exists():
        if args.calibration == DEFAULT_CALIBRATION_PATH:
            return None
        raise RuntimeError(f"标定文件不存在：{args.calibration}")
    return RigidTransform.from_json(args.calibration)


def parse_axis_component(component: str) -> tuple[int, float]:
    names = {"x": 0, "y": 1, "z": 2}
    component = component.strip().lower()
    if not component:
        raise RuntimeError("Axis map contains an empty component.")

    sign = 1.0
    if component[0] in ("+", "-"):
        if component[0] == "-":
            sign = -1.0
        component = component[1:]

    if component not in names:
        raise RuntimeError(
            f"Invalid axis map component '{component}'. Use x, y, z, -x, -y, or -z."
        )
    return names[component], sign


def parse_axis_map(axis_map: str) -> list[tuple[int, float]]:
    parts = [part.strip() for part in axis_map.split(",")]
    if len(parts) != 3:
        raise RuntimeError("--axis-map must contain exactly three comma-separated components.")
    return [parse_axis_component(part) for part in parts]


def load_transform_matrix(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        matrix = np.load(path)
    else:
        text = path.read_text(encoding="utf-8")
        try:
            matrix = np.asarray(json.loads(text), dtype=float)
        except json.JSONDecodeError:
            values = [float(token) for token in text.replace(",", " ").split()]
            matrix = np.asarray(values, dtype=float)

    if matrix.size == 16:
        matrix = matrix.reshape(4, 4)
    if matrix.shape != (4, 4):
        raise RuntimeError(f"Transform matrix must be 4x4 or contain 16 values: {path}")
    return matrix


def transform_camera_point_to_base_m(
    camera_point_m: np.ndarray,
    args: argparse.Namespace,
    matrix: np.ndarray | None,
    camera_to_base: RigidTransform | None,
) -> np.ndarray:
    if camera_to_base is not None:
        return camera_to_base.transform_point(camera_point_m)

    camera_point_mm = np.asarray(camera_point_m, dtype=float) * 1000.0
    if matrix is not None:
        homogeneous = np.asarray(
            [camera_point_mm[0], camera_point_mm[1], camera_point_mm[2], 1.0],
            dtype=float,
        )
        arm_point = matrix @ homogeneous
        if abs(float(arm_point[3])) > 1e-9:
            arm_point = arm_point / arm_point[3]
        return arm_point[:3] / 1000.0

    parsed = parse_axis_map(args.axis_map)
    mapped = np.asarray(
        [sign * camera_point_mm[index] for index, sign in parsed],
        dtype=float,
    )
    return (mapped + np.asarray(args.offset_mm, dtype=float)) / 1000.0


def clamp_arm_point_to_workspace(
    arm_point_mm: np.ndarray,
    workspace: Workspace,
) -> tuple[np.ndarray, tuple[str, ...]]:
    projected = np.asarray(arm_point_mm, dtype=float).copy()
    notes: list[str] = []

    limits = (
        ("x", workspace.x_min, workspace.x_max),
        ("y", workspace.y_min, workspace.y_max),
        ("z", workspace.z_min, workspace.z_max),
    )
    for index, (name, lower, upper) in enumerate(limits):
        before = float(projected[index])
        after = min(max(before, lower), upper)
        projected[index] = after
        if not math.isclose(before, after, abs_tol=1e-6):
            notes.append(f"{name}: {before:.2f} -> {after:.2f} mm")

    x = float(projected[0])
    y = float(projected[1])
    radius = math.hypot(x, y)
    target_radius = radius
    if radius > workspace.xy_radius_max:
        target_radius = workspace.xy_radius_max
    elif radius < workspace.xy_radius_min:
        target_radius = workspace.xy_radius_min

    if not math.isclose(radius, target_radius, abs_tol=1e-6):
        before_x = float(projected[0])
        before_y = float(projected[1])
        if radius > 1e-9:
            scale = target_radius / radius
            projected[0] *= scale
            projected[1] *= scale
        else:
            projected[0] = target_radius
            projected[1] = 0.0
        notes.append(
            "xy radius: "
            f"{radius:.2f} -> {target_radius:.2f} mm "
            f"({before_x:.2f}, {before_y:.2f}) -> "
            f"({projected[0]:.2f}, {projected[1]:.2f}) mm"
        )

    for index, (name, lower, upper) in enumerate(limits[:2]):
        before = float(projected[index])
        after = min(max(before, lower), upper)
        projected[index] = after
        if not math.isclose(before, after, abs_tol=1e-6):
            notes.append(f"{name} after radius clamp: {before:.2f} -> {after:.2f} mm")

    return projected, tuple(notes)


def print_target_summary(selected: SelectedTarget) -> None:
    print("\n选点结果")
    print(f"  picked point index: {selected.picked_index}")
    print(f"  camera point xyz m: {format_array(selected.camera_point_m)}")
    print(f"  选中点-机械臂坐标系 xyz m: {format_array(selected.base_point_m)}")
    print(f"  选中点-机械臂坐标系 xyz cm: {format_array(selected.base_point_m * 100.0)}")
    print(f"  偏移后目标点 xyz m: {format_array(selected.raw_target_point_m)}")
    if selected.workspace_notes:
        print("  workspace clamp applied:")
        for note in selected.workspace_notes:
            print(f"    {note}")
    else:
        print("  workspace clamp: disabled/not needed")
    print(f"  最终目标点-机械臂坐标系 xyz m: {format_array(selected.target_position_m)}")
    print(f"  最终目标点-机械臂坐标系 xyz cm: {format_array(selected.target_position_m * 100.0)}")
