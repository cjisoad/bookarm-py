"""用点云选点测试 BookArm link6 的真实到达准度。

请在项目根目录运行：

    conda run -n bookarm-beiyu python example/7.check_link6_pointcloud_accuracy.py --port /dev/bookarm

流程：

1. 连接机械臂并移动到起始构型。
2. 采集 RealSense D435 点云，并在 Open3D 中选取目标点。
3. 将目标点转换到机械臂基座坐标系，根据脚本顶部变量选择强制末端姿态或位置优先姿态偏好。
4. 机械臂移动到目标点，读取真实关节反馈，并打印 link6 位姿误差。
5. 根据键盘输入决定回到起始构型，或保持当前位置退出。
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from typing import Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_PATH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bookarm_control_py import BookArm, matrix_to_rpy, rpy_to_matrix


# 目标末端：本脚本专门测试 link6 位姿到达准度。
END_EFFECTOR_LINK = "link6"

# 机械臂起始构型，单位为度；每次测试前先回到这个构型。
START_Q_DEG = np.array([0.0, -70.0, 60.0, 0.0, -45.0], dtype=float)

# 是否强制传递末端姿态；False 时只强制位置，并让姿态尽量接近 PREFERRED_RPY_DEG。
USE_TARGET_ROTATION = False

# 强制传递末端姿态时使用的 link6 目标姿态，单位为度，顺序为滚转、俯仰、偏航。
TARGET_RPY_DEG = np.array([0.0, 20.0, 0.0], dtype=float)

# 不强制末端姿态时使用的姿态偏好，单位为度；位置优先，姿态只作为次要优化目标。
PREFERRED_RPY_DEG = np.array([0.0, 45.0, 0.0], dtype=float)

# link6 测试默认不加抓取用的回退和重力补偿，避免影响准度判断。
TARGET_X_BACKOFF_M = 0.02
TARGET_Z_GRAVITY_COMPENSATION_M = 0.1

# 串口默认值；通常运行时只需要按实际设备改 --port。
DEFAULT_PORT = "/dev/bookarm"

# 机械臂运动参数。
ARM_SPEED = 35.0
ARM_RETURN_SPEED = 35.0
ARM_ACCELERATION = 5.0
ARM_WAIT_SECONDS = 5.0
ARM_REACH_TIMEOUT_SECONDS = 6.0
ARM_REACH_TOLERANCE_DEG = 2.0
ARM_REACH_POLL_SECONDS = 0.2
ARM_FEEDBACK_TIMEOUT_SECONDS = 1.0

# RealSense D435 采集参数。
CAMERA_SERIAL = None
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_WARMUP_FRAMES = 30
CAMERA_TIMEOUT_MS = 5000

# 点云过滤和显示参数。
DEPTH_MIN_M = 0.0
DEPTH_MAX_M = 4.0
POINT_CLOUD_STRIDE = 1
VOXEL_SIZE_M = 0.0
PICK_POINT_SIZE = 1.0
POINT_INDEX = None
FLIP_VIEW = True

# 相机到机械臂基座的坐标转换参数；默认优先读取标定文件。
CALIBRATION_PATH = REPO_ROOT / "calibration" / "camera_to_base.json"
USE_CALIBRATION_FILE = True
AXIS_MAP = "z,-x,-y"
OFFSET_MM = (300.0, -100.0, 0.0)
TRANSFORM_MATRIX_PATH = None
CAMERA_TO_BASE_XYZ_M = None
CAMERA_TO_BASE_RPY_DEG = None
TARGET_OFFSET_BASE_M = [0.0, 0.0, 0.0]

# 工作空间裁剪参数；默认不裁剪，只在需要保守限制目标点时打开。
CLAMP_WORKSPACE = False
WORKSPACE_X_MIN_MM = 0.0
WORKSPACE_X_MAX_MM = 450.0
WORKSPACE_Y_MIN_MM = -300.0
WORKSPACE_Y_MAX_MM = 300.0
WORKSPACE_Z_MIN_MM = 20.0
WORKSPACE_Z_MAX_MM = 400.0
WORKSPACE_XY_RADIUS_MIN_MM = 40.0
WORKSPACE_XY_RADIUS_MAX_MM = 450.0

# link6 位姿逆解参数。
IK_MAX_ITERATIONS = 500
IK_TOLERANCE = 1e-4
IK_DAMPING = 1e-6
IK_STEP_SIZE = 0.4
IK_POSITION_PRIORITY_TOLERANCE_M = 0.005
IK_ORIENTATION_PREFERENCE_WEIGHT = 0.15

# 逆解结果安全阈值；超过阈值说明该目标位姿不可用，不再下发机械臂运动。
MAX_PLANNED_POSITION_ERROR_M = 0.08
MAX_PLANNED_ROTATION_ERROR_DEG = 15.0


@dataclass(frozen=True)
class Workspace:
    """机械臂工作空间范围，单位为毫米。"""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    xy_radius_min: float
    xy_radius_max: float


@dataclass(frozen=True)
class SelectedTarget:
    """一次点云选点得到的目标点信息。"""

    picked_index: int
    camera_point_m: np.ndarray
    base_point_m: np.ndarray
    raw_target_point_m: np.ndarray
    final_target_point_m: np.ndarray
    target_position_m: np.ndarray
    workspace_notes: tuple[str, ...]


@dataclass(frozen=True)
class CameraToBaseTransform:
    """从 RealSense 相机坐标系到机械臂基座坐标系的刚体变换。"""

    rotation: np.ndarray
    translation: np.ndarray

    @classmethod
    def from_xyz_rpy_deg(
        cls,
        xyz_m: Sequence[float],
        rpy_deg: Sequence[float],
    ) -> "CameraToBaseTransform":
        return cls(
            rotation=rpy_to_matrix(np.deg2rad(np.asarray(rpy_deg, dtype=float))),
            translation=np.asarray(xyz_m, dtype=float).reshape(3),
        )

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> "CameraToBaseTransform":
        transform = np.asarray(matrix, dtype=float).reshape(4, 4)
        return cls(rotation=transform[:3, :3], translation=transform[:3, 3])

    @classmethod
    def from_json(cls, path: Path) -> "CameraToBaseTransform":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "xyz_m" in payload and "rpy_deg" in payload:
            return cls.from_xyz_rpy_deg(payload["xyz_m"], payload["rpy_deg"])
        if "matrix" in payload:
            return cls.from_matrix(np.asarray(payload["matrix"], dtype=float))
        raise RuntimeError("标定文件必须包含 xyz_m 和 rpy_deg，或包含 matrix。")

    @property
    def matrix(self) -> np.ndarray:
        transform = np.eye(4)
        transform[:3, :3] = self.rotation
        transform[:3, 3] = self.translation
        return transform

    def transform_point(self, point: np.ndarray) -> np.ndarray:
        return self.rotation @ np.asarray(point, dtype=float).reshape(3) + self.translation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="采集点云选点，并测试 BookArm link6 到达该点的真实误差。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    arm = parser.add_argument_group("常用参数")
    arm.add_argument("--port", default=DEFAULT_PORT, help="机械臂串口，例如 /dev/bookarm。")

    parser.set_defaults(
        speed=ARM_SPEED,
        return_speed=ARM_RETURN_SPEED,
        acc=ARM_ACCELERATION,
        arm_wait=ARM_WAIT_SECONDS,
        arm_reach_timeout=ARM_REACH_TIMEOUT_SECONDS,
        arm_reach_tolerance_deg=ARM_REACH_TOLERANCE_DEG,
        arm_reach_poll=ARM_REACH_POLL_SECONDS,
        arm_feedback_timeout=ARM_FEEDBACK_TIMEOUT_SECONDS,
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
        point_index=POINT_INDEX,
        flip_view=FLIP_VIEW,
        calibration=CALIBRATION_PATH,
        no_calibration=not USE_CALIBRATION_FILE,
        axis_map=AXIS_MAP,
        offset_mm=OFFSET_MM,
        transform_matrix=TRANSFORM_MATRIX_PATH,
        camera_to_base_xyz=CAMERA_TO_BASE_XYZ_M,
        camera_to_base_rpy_deg=CAMERA_TO_BASE_RPY_DEG,
        target_offset_base=TARGET_OFFSET_BASE_M,
        target_x_backoff=TARGET_X_BACKOFF_M,
        target_z_gravity_compensation=TARGET_Z_GRAVITY_COMPENSATION_M,
        clamp_workspace=CLAMP_WORKSPACE,
        x_min=WORKSPACE_X_MIN_MM,
        x_max=WORKSPACE_X_MAX_MM,
        y_min=WORKSPACE_Y_MIN_MM,
        y_max=WORKSPACE_Y_MAX_MM,
        z_min=WORKSPACE_Z_MIN_MM,
        z_max=WORKSPACE_Z_MAX_MM,
        xy_radius_min=WORKSPACE_XY_RADIUS_MIN_MM,
        xy_radius_max=WORKSPACE_XY_RADIUS_MAX_MM,
        max_iterations=IK_MAX_ITERATIONS,
        tolerance=IK_TOLERANCE,
        damping=IK_DAMPING,
        step_size=IK_STEP_SIZE,
        position_priority_tolerance=IK_POSITION_PRIORITY_TOLERANCE_M,
        orientation_preference_weight=IK_ORIENTATION_PREFERENCE_WEIGHT,
    )
    return parser.parse_args()


def format_array(values: np.ndarray | Sequence[float]) -> str:
    return np.array2string(np.asarray(values), precision=6, suppress_small=True)


def wait(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def rotation_error_angle(target_rotation: np.ndarray, current_rotation: np.ndarray) -> float:
    relative_rotation = target_rotation.T @ current_rotation
    cos_angle = (np.trace(relative_rotation) - 1.0) / 2.0
    return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def validate_args(args: argparse.Namespace) -> None:
    if args.target_x_backoff < 0:
        raise RuntimeError("TARGET_X_BACKOFF_M 必须大于等于 0。")
    if args.target_z_gravity_compensation < 0:
        raise RuntimeError("TARGET_Z_GRAVITY_COMPENSATION_M 必须大于等于 0。")
    if args.depth_min < 0:
        raise RuntimeError("DEPTH_MIN_M 必须大于等于 0。")
    if args.depth_max <= args.depth_min:
        raise RuntimeError("DEPTH_MAX_M 必须大于 DEPTH_MIN_M。")
    if args.stride < 1:
        raise RuntimeError("POINT_CLOUD_STRIDE 必须大于等于 1。")


def import_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("缺少 open3d，无法打开点云选点窗口。请先安装 open3d。") from exc
    return o3d


def import_realsense_d435():
    module_name = "_bookarm_control_py_realsense_local"
    module_path = SRC_PATH / "bookarm_control_py" / "camera" / "realsense.py"
    if module_name in sys.modules:
        return sys.modules[module_name].RealSenseD435
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 RealSense 模块：{module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        raise RuntimeError(
            "RealSense 相机需要 pyrealsense2。请确认已安装 Intel RealSense SDK 和 Python 依赖。"
        ) from exc
    return module.RealSenseD435


def capture_rgbd_frame(args: argparse.Namespace):
    RealSenseD435 = import_realsense_d435()
    with RealSenseD435(
        width=args.width,
        height=args.height,
        fps=args.fps,
        serial=args.serial,
        align_depth_to_color=True,
    ) as camera:
        print(f"D435 已启动，深度比例 = {camera.depth_scale:.8f} m/unit")
        camera.warmup(frame_count=args.warmup, timeout_ms=args.timeout_ms)
        return camera.get_frame(timeout_ms=args.timeout_ms)


def make_point_cloud(frame, args: argparse.Namespace):
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
    if not visualizer.create_window("选择 link6 目标点", width=1280, height=720):
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


def parse_axis_component(component: str) -> tuple[int, float]:
    names = {"x": 0, "y": 1, "z": 2}
    component = component.strip().lower()
    if not component:
        raise RuntimeError("AXIS_MAP 中存在空的轴分量。")

    sign = 1.0
    if component[0] in ("+", "-"):
        if component[0] == "-":
            sign = -1.0
        component = component[1:]

    if component not in names:
        raise RuntimeError(f"AXIS_MAP 分量 {component!r} 无效，只能使用 x、y、z、-x、-y、-z。")
    return names[component], sign


def parse_axis_map(axis_map: str) -> list[tuple[int, float]]:
    parts = [part.strip() for part in axis_map.split(",")]
    if len(parts) != 3:
        raise RuntimeError("AXIS_MAP 必须包含 3 个用逗号分隔的轴分量。")
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
        raise RuntimeError(f"变换矩阵必须是 4x4，或包含 16 个数值：{path}")
    return matrix


def load_camera_to_base_transform(args: argparse.Namespace) -> CameraToBaseTransform | None:
    explicit_xyz_rpy = args.camera_to_base_xyz is not None or args.camera_to_base_rpy_deg is not None

    if args.transform_matrix is not None:
        if explicit_xyz_rpy:
            raise RuntimeError("TRANSFORM_MATRIX_PATH 不能和 CAMERA_TO_BASE_XYZ_M/CAMERA_TO_BASE_RPY_DEG 同时使用。")
        return None

    if explicit_xyz_rpy:
        if args.camera_to_base_xyz is None or args.camera_to_base_rpy_deg is None:
            raise RuntimeError("CAMERA_TO_BASE_XYZ_M 和 CAMERA_TO_BASE_RPY_DEG 必须同时设置。")
        return CameraToBaseTransform.from_xyz_rpy_deg(
            args.camera_to_base_xyz,
            args.camera_to_base_rpy_deg,
        )

    if args.no_calibration or args.calibration is None:
        return None

    if not args.calibration.exists():
        if args.calibration == CALIBRATION_PATH:
            return None
        raise RuntimeError(f"标定文件不存在：{args.calibration}")
    return CameraToBaseTransform.from_json(args.calibration)


def transform_camera_point_to_base_m(
    camera_point_m: np.ndarray,
    args: argparse.Namespace,
    matrix: np.ndarray | None,
    camera_to_base: CameraToBaseTransform | None,
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


def make_workspace(args: argparse.Namespace) -> Workspace:
    if args.x_min > args.x_max:
        raise RuntimeError("WORKSPACE_X_MIN_MM 必须小于等于 WORKSPACE_X_MAX_MM。")
    if args.y_min > args.y_max:
        raise RuntimeError("WORKSPACE_Y_MIN_MM 必须小于等于 WORKSPACE_Y_MAX_MM。")
    if args.z_min > args.z_max:
        raise RuntimeError("WORKSPACE_Z_MIN_MM 必须小于等于 WORKSPACE_Z_MAX_MM。")
    if args.xy_radius_min < 0 or args.xy_radius_max < 0:
        raise RuntimeError("工作空间 XY 半径限制必须大于等于 0。")
    if args.xy_radius_min > args.xy_radius_max:
        raise RuntimeError("WORKSPACE_XY_RADIUS_MIN_MM 必须小于等于 WORKSPACE_XY_RADIUS_MAX_MM。")
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
            "xy 半径: "
            f"{radius:.2f} -> {target_radius:.2f} mm "
            f"({before_x:.2f}, {before_y:.2f}) -> "
            f"({projected[0]:.2f}, {projected[1]:.2f}) mm"
        )

    for index, (name, lower, upper) in enumerate(limits[:2]):
        before = float(projected[index])
        after = min(max(before, lower), upper)
        projected[index] = after
        if not math.isclose(before, after, abs_tol=1e-6):
            notes.append(f"{name} 半径裁剪后: {before:.2f} -> {after:.2f} mm")

    return projected, tuple(notes)


def print_target_summary(selected: SelectedTarget) -> None:
    print("\n选点结果")
    print(f"  选中点索引: {selected.picked_index}")
    print(f"  相机坐标 xyz m: {format_array(selected.camera_point_m)}")
    print(f"  外参转换后的基座坐标 xyz m: {format_array(selected.base_point_m)}")
    print(f"  加偏移后的目标坐标 xyz m: {format_array(selected.raw_target_point_m)}")
    if selected.workspace_notes:
        print("  工作空间裁剪:")
        for note in selected.workspace_notes:
            print(f"    {note}")
    else:
        print("  工作空间裁剪: 未启用或无需裁剪")
    print(f"  最终机械臂目标坐标 xyz m: {format_array(selected.target_position_m)}")


def wait_until_arm_reaches(
    robot: BookArm,
    target_q: np.ndarray,
    args: argparse.Namespace,
    *,
    reached_message: str = "继续执行后续动作",
) -> np.ndarray:
    """循环读取关节反馈，直到机械臂到达目标构型或超时。"""

    tolerance_rad = np.deg2rad(float(args.arm_reach_tolerance_deg))
    timeout = float(args.arm_reach_timeout)
    poll_interval = float(args.arm_reach_poll)
    deadline = time.monotonic() + timeout
    last_error_deg: np.ndarray | None = None
    last_feedback_q: np.ndarray | None = None
    last_feedback_error: Exception | None = None

    print(
        "\n等待机械臂到达目标构型"
        f"（超时 {timeout:.1f}s，关节误差阈值 {args.arm_reach_tolerance_deg:.2f} deg）..."
    )

    while time.monotonic() < deadline:
        try:
            feedback = robot.read_arm_feedback(response_timeout=args.arm_feedback_timeout)
        except (RuntimeError, ValueError) as exc:
            last_feedback_error = exc
            wait(poll_interval)
            continue

        last_feedback_q = feedback.q_rad
        error = feedback.q_rad - target_q
        last_error_deg = np.rad2deg(error)
        max_abs_error_deg = float(np.max(np.abs(last_error_deg)))
        print(
            "  当前 q deg: "
            f"{format_array(np.rad2deg(feedback.q_rad))}, "
            f"最大关节误差: {max_abs_error_deg:.3f} deg"
        )

        if np.all(np.abs(error) <= tolerance_rad):
            print(f"机械臂已到达目标构型，{reached_message}。")
            return feedback.q_rad

        wait(poll_interval)

    detail = f"机械臂未在 {timeout:.1f}s 内到达目标构型"
    if last_feedback_q is not None and last_error_deg is not None:
        detail += (
            f"；最后反馈 q deg: {format_array(np.rad2deg(last_feedback_q))}"
            f"；最后关节误差 deg: {format_array(last_error_deg)}"
        )
    elif last_feedback_error is not None:
        detail += f"；最后一次读取反馈失败: {last_feedback_error}"
    print(f"警告: {detail}")
    print(f"继续执行后续动作：{reached_message}。")
    if last_feedback_q is not None:
        return last_feedback_q
    return target_q


def move_to_start(robot: BookArm, start_q: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    print("\n移动到起始构型")
    print(f"  起始关节角 deg: {format_array(START_Q_DEG)}")
    print(robot.enable_torque())
    print(robot.move_joints_rad(start_q, speed=args.speed, acceleration=args.acc))

    wait(args.arm_wait)
    feedback = robot.read_arm_feedback(response_timeout=args.arm_feedback_timeout)
    start_feedback_q = robot.check_joint_angles(feedback.q_rad, context="起始构型反馈")
    print("到达起始构型后的反馈:")
    print(f"  当前 q deg: {format_array(np.rad2deg(start_feedback_q))}")
    print(f"  力矩: {format_array(feedback.torque)}")
    return start_feedback_q


def pick_link6_target(args: argparse.Namespace) -> SelectedTarget:
    workspace = make_workspace(args)
    matrix = load_transform_matrix(args.transform_matrix) if args.transform_matrix else None
    camera_to_base = load_camera_to_base_transform(args)

    if camera_to_base is not None:
        print("\n使用相机到基座外参矩阵:")
        print(camera_to_base.matrix)
    elif matrix is not None:
        print("\n使用 4x4 相机到机械臂矩阵，单位 mm:")
        print(matrix)
    else:
        print(f"\n使用轴映射: {args.axis_map}, offset-mm: {format_array(args.offset_mm)}")
    print(f"目标末端 link: {END_EFFECTOR_LINK}")
    print(f"目标点 x 回退: {args.target_x_backoff:.3f} m")
    print(f"目标点 z 重力补偿: +{args.target_z_gravity_compensation:.3f} m")

    frame = capture_rgbd_frame(args)
    point_cloud = make_point_cloud(frame, args)
    point_count = len(point_cloud.points)
    print(f"点云点数: {point_count}")
    if point_count == 0:
        raise RuntimeError("点云为空，请调整脚本顶部的 DEPTH_MIN_M/DEPTH_MAX_M 或相机视角。")

    picked_index = (
        pick_point_index(point_cloud, args.point_size, args.flip_view)
        if args.point_index is None
        else args.point_index
    )
    if picked_index < 0 or picked_index >= point_count:
        raise RuntimeError(f"点索引 {picked_index} 超出范围；点云共有 {point_count} 个点。")

    camera_point_m = np.asarray(point_cloud.points)[picked_index]
    base_point_m = transform_camera_point_to_base_m(
        camera_point_m,
        args,
        matrix=matrix,
        camera_to_base=camera_to_base,
    )
    raw_target_point_m = base_point_m + np.asarray(args.target_offset_base, dtype=float)
    raw_target_point_m[0] -= args.target_x_backoff
    raw_target_point_m[2] += args.target_z_gravity_compensation

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


def solve_link6_target(
    robot: BookArm,
    start_q: np.ndarray,
    target_position: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    requested_rpy_deg = TARGET_RPY_DEG if USE_TARGET_ROTATION else PREFERRED_RPY_DEG
    requested_rotation = rpy_to_matrix(np.deg2rad(requested_rpy_deg))
    ik_result = robot.ikine_link6_best_effort(
        target_x=float(target_position[0]),
        target_y=float(target_position[1]),
        target_z=float(target_position[2]),
        target_rotation=requested_rotation if USE_TARGET_ROTATION else None,
        preferred_rotation=None if USE_TARGET_ROTATION else requested_rotation,
        consider_rotation=USE_TARGET_ROTATION,
        q0=start_q,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
        damping=args.damping,
        step_size=args.step_size,
        orientation_weight=args.orientation_preference_weight,
        position_priority_tolerance=args.position_priority_tolerance,
        print_error=False,
    )
    mode_description = "强制末端姿态" if USE_TARGET_ROTATION else "不考虑姿态，优先保证位置"
    rotation_threshold_deg = MAX_PLANNED_ROTATION_ERROR_DEG if USE_TARGET_ROTATION else float("inf")
    goal_q = robot.check_joint_angles(ik_result.q, context="link6 目标构型")
    planned_pose = robot.fkine_dict(goal_q, end_effector_link=END_EFFECTOR_LINK)
    planned_position_error = planned_pose["position"] - target_position
    planned_rotation_error = rotation_error_angle(requested_rotation, planned_pose["rotation"])
    planned_position_error_norm = float(np.linalg.norm(planned_position_error))
    planned_rotation_error_deg = float(np.rad2deg(planned_rotation_error))
    planned_rpy_deg = np.rad2deg(matrix_to_rpy(planned_pose["rotation"]))

    print("\nlink6 位姿逆解结果")
    print(f"  求解模式: {mode_description}")
    print(f"  目标 xyz m: {format_array(target_position)}")
    if USE_TARGET_ROTATION:
        print(f"  强制目标滚转/俯仰/偏航 deg: {format_array(TARGET_RPY_DEG)}")
    else:
        print(f"  姿态偏好滚转/俯仰/偏航 deg: {format_array(PREFERRED_RPY_DEG)}")
    print(f"  目标关节角 deg: {format_array(np.rad2deg(goal_q))}")
    print(f"  规划 link6 xyz m: {format_array(planned_pose['position'])}")
    print(f"  规划 link6 滚转/俯仰/偏航 deg: {format_array(planned_rpy_deg)}")
    print(f"  规划位置误差 xyz m: {format_array(planned_position_error)}")
    print(f"  规划位置误差模长 m: {planned_position_error_norm:.8f}")
    print(f"  相对请求姿态/偏好姿态的误差 deg: {planned_rotation_error_deg:.8f}")
    print(f"  逆解是否达到严格容差: {ik_result.success}")
    print(f"  逆解总误差模长: {ik_result.error_norm:.8f}")
    print(f"  逆解位置误差模长 m: {ik_result.position_error_norm:.8f}")
    print(f"  逆解姿态/偏好姿态误差 deg: {np.rad2deg(ik_result.rotation_error_rad):.8f}")
    if planned_position_error_norm > MAX_PLANNED_POSITION_ERROR_M or planned_rotation_error_deg > rotation_threshold_deg:
        raise RuntimeError(
            "link6 目标逆解不可用，已停止下发机械臂运动。"
            f"规划位置误差 {planned_position_error_norm * 1000.0:.1f} mm "
            f"（阈值 {MAX_PLANNED_POSITION_ERROR_M * 1000.0:.1f} mm），"
            f"规划姿态误差 {planned_rotation_error_deg:.1f} deg "
            f"（阈值 {'不限制' if not USE_TARGET_ROTATION else f'{rotation_threshold_deg:.1f} deg'}）。"
            "请调整目标点位置，或在 USE_TARGET_ROTATION=True 时调整 TARGET_RPY_DEG。"
        )
    return goal_q, requested_rotation


def print_reach_error(
    robot: BookArm,
    reached_q: np.ndarray,
    goal_q: np.ndarray,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
) -> None:
    current_pose = robot.fkine_dict(reached_q, end_effector_link=END_EFFECTOR_LINK)
    position_error = current_pose["position"] - target_position
    rotation_error = rotation_error_angle(target_rotation, current_pose["rotation"])
    joint_error_deg = np.rad2deg(reached_q - goal_q)
    current_rpy_deg = np.rad2deg(matrix_to_rpy(current_pose["rotation"]))
    target_rpy_deg = np.rad2deg(matrix_to_rpy(target_rotation))

    print("\n真实位姿误差")
    print(f"  反馈关节角 deg: {format_array(np.rad2deg(reached_q))}")
    print(f"  目标关节角 deg: {format_array(np.rad2deg(goal_q))}")
    print(f"  关节角误差 deg: {format_array(joint_error_deg)}")
    print(f"  当前 link6 xyz m: {format_array(current_pose['position'])}")
    print(f"  目标 link6 xyz m: {format_array(target_position)}")
    print(f"  当前 link6 滚转/俯仰/偏航 deg: {format_array(current_rpy_deg)}")
    print(f"  目标/偏好 link6 滚转/俯仰/偏航 deg: {format_array(target_rpy_deg)}")
    print(f"  位置误差 xyz m: {format_array(position_error)}")
    print(f"  位置误差 xyz mm: {format_array(position_error * 1000.0)}")
    print(f"  位置误差模长 m: {np.linalg.norm(position_error):.8f}")
    print(f"  位置误差模长 mm: {np.linalg.norm(position_error) * 1000.0:.3f}")
    print(f"  姿态误差 deg: {np.rad2deg(rotation_error):.8f}")
    print(f"  当前 link6 旋转矩阵:\n{format_array(current_pose['rotation'])}")


def prompt_return_action() -> str:
    while True:
        choice = input("\n输入 r 回到起始构型，输入 q 保持当前位置并退出: ").strip().lower()
        if choice in {"r", "q"}:
            return choice
        print("请输入 r 或 q。")


def main() -> int:
    args = parse_args()
    robot: BookArm | None = None
    try:
        validate_args(args)
        import_open3d()

        robot = BookArm(end_effector_link=END_EFFECTOR_LINK)
        start_q = robot.check_joint_angles(np.deg2rad(START_Q_DEG), context="起始构型")

        print("BookArm link6 点云目标到达准度测试")
        print(f"串口号: {args.port}")
        print(f"目标末端 link: {END_EFFECTOR_LINK}")
        print(f"关节名称: {robot.joint_names}")

        robot.connect_serial_arm(port=args.port)
        start_feedback_q = move_to_start(robot, start_q, args)

        selected = pick_link6_target(args)
        goal_q, target_rotation = solve_link6_target(
            robot,
            start_feedback_q,
            selected.target_position_m,
            args,
        )

        print(f"\n移动 link6 到点云目标点（{END_EFFECTOR_LINK}）。")
        print(robot.move_joints_rad(goal_q, speed=args.speed, acceleration=args.acc))
        reached_q = wait_until_arm_reaches(
            robot,
            goal_q,
            args,
            reached_message="读取 link6 到达误差",
        )

        print_reach_error(
            robot,
            reached_q,
            goal_q,
            selected.target_position_m,
            target_rotation,
        )

        if prompt_return_action() == "r":
            print(f"\n返回起始构型，速度 {args.return_speed:.1f}。")
            print(robot.move_joints_rad(start_q, speed=args.return_speed, acceleration=args.acc))
            wait(args.arm_wait)
        else:
            print("\n保持当前位置，程序退出。")
        return 0
    except (RuntimeError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n用户中断", file=sys.stderr)
        return 130
    finally:
        if robot is not None:
            robot.close()


if __name__ == "__main__":
    raise SystemExit(main())
