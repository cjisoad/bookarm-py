"""采集 RealSense D435 点云，在 Open3D 中选点，并驱动 BookArm 的 link6 移动到目标点。

1. 可选：先让机械臂回到起始关节构型。
2. 采集一帧 D435 RGB-D，并生成彩色点云。
3. 在 Open3D 窗口中 Shift + 左键选中目标点，按 Q 或 Esc 结束选点。
4. 把相机坐标转换到机械臂基座坐标；转换后的目标点作为 link6 的目标点。
5. 使用当前项目的 `BookArm.ikine_best_effort()` 以 link6 为末端求目标关节角；
   显式加 --execute 时，从当前构型直接移动到目标构型，闭合夹爪后上抬并返回起始构型。

运行前请在项目根目录执行命令，并确认已激活包含 pyrealsense2、open3d、
pinocchio 的环境。Open3D 选点窗口打开后：

- Shift + 鼠标左键：选择目标点。
- Shift + 鼠标右键：撤销上一次选择。
- Q 或 Esc：结束选点并继续后续流程。

真实执行前的终端确认会直接读取单个按键：按 y 确认，按 n、Esc 或回车取消。

推荐先干跑检查坐标转换和 IK，不连接机械臂：

    python scripts/camera_arm/click_point_grasp_link6.py

默认会自动读取 `calibration/camera_to_base.json`。当前保存的外参表示：

    base_point = [camera_z, -camera_x, -camera_y] + [-0.02, -0.15, 0.145] m

相机位置或朝向改变后，请先更新 `calibration/camera_to_base.json` 中的
`xyz_m` 和 `rpy_deg`。该文件里写有详细测量、修改和验证步骤。

如果临时不用标定文件，而使用参考项目那种简单轴映射和偏移，可以这样运行：

    python scripts/camera_arm/click_point_grasp_link6.py --no-calibration --axis-map z,-x,-y --offset-mm 300 -100 0

如果已经完成外参标定，推荐使用当前项目的标定文件：

    python scripts/camera_arm/click_point_grasp_link6.py --calibration calibration/camera_to_base.json

确认终端打印的相机点、机械臂目标点和 IK 结果正确后，再显式执行真实目标移动：

    python scripts/camera_arm/click_point_grasp_link6.py --port /dev/bookarm --execute

真实执行时仍可指定串口：

    python scripts/camera_arm/click_point_grasp_link6.py --port /dev/bookarm --execute

相机、点云、标定、工作空间、IK 和动作等待时间等不常改的参数，统一在
脚本顶部的常量区修改。
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

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from bookarm_control_py.math_utils import matrix_to_rpy, rpy_to_matrix


END_EFFECTOR_LINK = "link6"
START_Q_DEG = np.array([0.0, -70.0, 60.0, 0.0, -45.0], dtype=float)
DEFAULT_AXIS_MAP = "z,-x,-y"
DEFAULT_OFFSET_MM = (300.0, -100.0, 0.0)
DEFAULT_CALIBRATION_PATH = Path("calibration/camera_to_base.json")
# 机械臂目标点的默认后退距离
DEFAULT_TARGET_X_BACKOFF_M = 0.02
# 重力导致末端下垂时，目标点沿基座 +Z 方向额外抬高的补偿量
DEFAULT_TARGET_Z_GRAVITY_COMPENSATION_M = 0.05
DEFAULT_SPEED = 25.0
DEFAULT_RETURN_SPEED = 20.0
DEFAULT_ACC = 5.0
DEFAULT_ARM_WAIT = 5.0
DEFAULT_GRIPPER_WAIT = 1.0
DEFAULT_GRASP_HOLD_WAIT = 3.0

# Less frequently changed settings live here instead of the command line.
CAMERA_SERIAL = None
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_WARMUP_FRAMES = 30
CAMERA_TIMEOUT_MS = 5000
DEPTH_MIN_M = 0.0
DEPTH_MAX_M = 4.0
POINT_CLOUD_STRIDE = 1
VOXEL_SIZE_M = 0.0

PICK_POINT_SIZE = 1.0
POINT_INDEX = None
SAVE_PLY_PATH = None
FLIP_VIEW = True

TRANSFORM_MATRIX_PATH = None
USE_CALIBRATION_FILE = True
CAMERA_TO_BASE_XYZ_M = None
CAMERA_TO_BASE_RPY_DEG = None
TARGET_OFFSET_BASE_M = [0.0, 0.0, 0.0]
TARGET_X_BACKOFF_M = DEFAULT_TARGET_X_BACKOFF_M
TARGET_Z_GRAVITY_COMPENSATION_M = DEFAULT_TARGET_Z_GRAVITY_COMPENSATION_M

WORKSPACE_X_MIN_MM = 0.0
WORKSPACE_X_MAX_MM = 450.0
WORKSPACE_Y_MIN_MM = -300.0
WORKSPACE_Y_MAX_MM = 300.0
WORKSPACE_Z_MIN_MM = 20.0
WORKSPACE_Z_MAX_MM = 400.0
WORKSPACE_XY_RADIUS_MIN_MM = 40.0
WORKSPACE_XY_RADIUS_MAX_MM = 450.0
CLAMP_WORKSPACE = False

SKIP_STARTUP = False
ARM_TEST = False
ARM_SPEED = DEFAULT_SPEED
ARM_RETURN_SPEED = 35.0
ARM_ACCELERATION = DEFAULT_ACC
ARM_WAIT_SECONDS = DEFAULT_ARM_WAIT
GRIPPER_WAIT_SECONDS = DEFAULT_GRIPPER_WAIT
GRASP_HOLD_WAIT_SECONDS = DEFAULT_GRASP_HOLD_WAIT
LIFT_AFTER_GRASP_Z_M = 0.1
ARM_REACH_TIMEOUT_SECONDS = 6.0
ARM_REACH_TOLERANCE_DEG = 2.0
ARM_REACH_POLL_SECONDS = 0.2
ARM_FEEDBACK_TIMEOUT_SECONDS = 1.0
TARGET_RPY_DEG = [0.0, 0.0, 0.0]
IK_MAX_ITERATIONS = 500
IK_TOLERANCE = 1e-4
IK_DAMPING = 1e-6
IK_STEP_SIZE = 0.4


@dataclass(frozen=True)
class Workspace:
    """Simple arm workspace model in millimeters."""

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
    picked_index: int
    camera_point_m: np.ndarray
    base_point_m: np.ndarray
    raw_target_point_m: np.ndarray
    final_target_point_m: np.ndarray
    target_position_m: np.ndarray
    workspace_notes: tuple[str, ...]


def import_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("缺少 open3d，无法打开点云选点窗口。请先安装 open3d。") from exc
    return o3d


def create_bookarm():
    try:
        from bookarm_control_py.bookarm import BookArm
    except ImportError as exc:
        raise RuntimeError(
            "BookArm IK 需要 pinocchio。请激活 bookarm-beiyu 环境，"
            "或从 conda-forge 安装 pinocchio。"
        ) from exc
    return BookArm(end_effector_link=END_EFFECTOR_LINK)


@dataclass(frozen=True)
class CameraToBaseTransform:
    """Rigid transform from RealSense camera coordinates to arm base coordinates."""

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
        description="采集 D435 点云，在 Open3D 中选点，并让 BookArm 抓取该点。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    arm = parser.add_argument_group("arm")
    arm.add_argument("--port", default="/dev/bookarm", help="机械臂和夹爪共用串口，例如 /dev/bookarm。")
    arm.add_argument("--execute", dest="execute", action="store_true", help="真正连接串口并执行机械臂运动。")
    arm.add_argument("--dry-run", dest="execute", action="store_false", help="只打印目标、IK 和命令，不连接机械臂。")
    arm.set_defaults(execute=False)
    arm.add_argument("--yes", action="store_true", help="跳过真实运动前的确认提示。")

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
        point_index=POINT_INDEX,
        ply=SAVE_PLY_PATH,
        flip_view=FLIP_VIEW,
        axis_map=DEFAULT_AXIS_MAP,
        offset_mm=DEFAULT_OFFSET_MM,
        transform_matrix=TRANSFORM_MATRIX_PATH,
        calibration=DEFAULT_CALIBRATION_PATH,
        no_calibration=not USE_CALIBRATION_FILE,
        camera_to_base_xyz=CAMERA_TO_BASE_XYZ_M,
        camera_to_base_rpy_deg=CAMERA_TO_BASE_RPY_DEG,
        target_offset_base=TARGET_OFFSET_BASE_M,
        target_x_backoff=TARGET_X_BACKOFF_M,
        target_z_gravity_compensation=TARGET_Z_GRAVITY_COMPENSATION_M,
        x_min=WORKSPACE_X_MIN_MM,
        x_max=WORKSPACE_X_MAX_MM,
        y_min=WORKSPACE_Y_MIN_MM,
        y_max=WORKSPACE_Y_MAX_MM,
        z_min=WORKSPACE_Z_MIN_MM,
        z_max=WORKSPACE_Z_MAX_MM,
        xy_radius_min=WORKSPACE_XY_RADIUS_MIN_MM,
        xy_radius_max=WORKSPACE_XY_RADIUS_MAX_MM,
        clamp_workspace=CLAMP_WORKSPACE,
        skip_startup=SKIP_STARTUP,
        arm_test=ARM_TEST,
        speed=ARM_SPEED,
        return_speed=ARM_RETURN_SPEED,
        acc=ARM_ACCELERATION,
        arm_wait=ARM_WAIT_SECONDS,
        gripper_wait=GRIPPER_WAIT_SECONDS,
        grasp_hold_wait=GRASP_HOLD_WAIT_SECONDS,
        lift_after_grasp_z=LIFT_AFTER_GRASP_Z_M,
        arm_reach_timeout=ARM_REACH_TIMEOUT_SECONDS,
        arm_reach_tolerance_deg=ARM_REACH_TOLERANCE_DEG,
        arm_reach_poll=ARM_REACH_POLL_SECONDS,
        arm_feedback_timeout=ARM_FEEDBACK_TIMEOUT_SECONDS,
        target_rpy_deg=TARGET_RPY_DEG,
        max_iterations=IK_MAX_ITERATIONS,
        tolerance=IK_TOLERANCE,
        damping=IK_DAMPING,
        step_size=IK_STEP_SIZE,
    )
    return parser.parse_args()


def format_array(values: np.ndarray | Sequence[float]) -> str:
    return np.array2string(np.asarray(values), precision=6, suppress_small=True)


def wait(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def flush_pending_terminal_input() -> None:
    """Drop stale keys left by Open3D/window-focus changes before prompting."""

    try:
        if sys.platform.startswith("win"):
            import msvcrt

            while msvcrt.kbhit():
                msvcrt.getwch()
            return

        import select

        if not sys.stdin.isatty():
            return
        while True:
            readable, _, _ = select.select([sys.stdin], [], [], 0)
            if not readable:
                break
            sys.stdin.read(1)
    except (ImportError, OSError):
        return


def ask_yes_no(prompt: str, *, default: bool = False) -> bool:
    """Ask for a yes/no confirmation without getting stuck on stale input."""

    flush_pending_terminal_input()
    suffix = " [Y/n]: " if default else " [y/N]: "
    full_prompt = prompt.rstrip()
    if full_prompt.endswith(("[y/N]:", "[Y/n]:")):
        display_prompt = full_prompt + " "
    else:
        display_prompt = full_prompt + suffix

    if not sys.stdin.isatty():
        print(f"{display_prompt}{'yes' if default else 'no'}")
        print("当前不是交互式终端，已使用默认选择。")
        return default

    if sys.platform.startswith("win"):
        try:
            import msvcrt

            print(display_prompt, end="", flush=True)
            while True:
                key = msvcrt.getwch().lower()
                if key in {"\x00", "\xe0"}:
                    if msvcrt.kbhit():
                        msvcrt.getwch()
                    continue
                if key in {"\r", "\n"}:
                    print()
                    return default
                if key in {"y", "n"}:
                    print(key)
                    return key == "y"
                if key == "\x03":
                    raise KeyboardInterrupt
                if key == "\x1b":
                    print()
                    return False
        except OSError:
            print()

    try:
        answer = input(display_prompt).strip().lower()
    except EOFError:
        print()
        return default
    if not answer:
        return default
    return answer in {"y", "yes"}


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


def load_camera_to_base_transform(args: argparse.Namespace) -> CameraToBaseTransform | None:
    explicit_xyz_rpy = args.camera_to_base_xyz is not None or args.camera_to_base_rpy_deg is not None

    if args.transform_matrix is not None:
        if explicit_xyz_rpy:
            raise RuntimeError("--transform-matrix cannot be combined with --camera-to-base-*.")
        return None

    if explicit_xyz_rpy:
        if args.camera_to_base_xyz is None or args.camera_to_base_rpy_deg is None:
            raise RuntimeError("--camera-to-base-xyz 和 --camera-to-base-rpy-deg 必须同时提供。")
        return CameraToBaseTransform.from_xyz_rpy_deg(
            args.camera_to_base_xyz,
            args.camera_to_base_rpy_deg,
        )

    if args.no_calibration or args.calibration is None:
        return None

    if not args.calibration.exists():
        if args.calibration == DEFAULT_CALIBRATION_PATH:
            return None
        raise RuntimeError(f"标定文件不存在：{args.calibration}")
    return CameraToBaseTransform.from_json(args.calibration)

    return None


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
            "RealSense 相机需要 pyrealsense2。请确认已安装 Intel RealSense SDK "
            "和 Python 依赖。"
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


def move_arm_to_startup_default(robot, args: argparse.Namespace) -> bool:
    start_q = robot.check_joint_angles(np.deg2rad(START_Q_DEG), context="起始构型")
    print("\n相机采集前起始构型")
    print(f"  q deg: {format_array(START_Q_DEG)}")

    if not args.execute:
        print("  干跑模式：未发送起始构型命令。")
        return True

    robot.connect_serial(port=args.port)
    print("  打开夹爪。")
    print(robot.open_gripper())
    wait(args.gripper_wait)

    print("  开启机械臂力矩。")
    print(robot.enable_torque())

    print("  移动到起始构型。")
    print(robot.move_joints_rad(start_q, speed=args.speed, acceleration=args.acc))
    wait(args.arm_wait)

    if not args.arm_test or args.yes:
        return True
    if ask_yes_no("机械臂是否已正常回到起始构型？"):
        return True
    print("起始构型动作未确认，停止相机采集。")
    return False


def solve_target_configuration(
    robot,
    target_x_m: float,
    target_y_m: float,
    target_z_m: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, object]:
    start_q = robot.check_joint_angles(
        np.deg2rad(START_Q_DEG),
        context="起始构型",
    )
    target_position = np.array([target_x_m, target_y_m, target_z_m], dtype=float)
    target_rotation = rpy_to_matrix(np.deg2rad(np.asarray(args.target_rpy_deg, dtype=float)))

    ik_result = robot.ikine_best_effort(
        target_position=target_position,
        target_rotation=target_rotation,
        q0=start_q,
        end_effector_link=END_EFFECTOR_LINK,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
        damping=args.damping,
        step_size=args.step_size,
        print_error=False,
    )
    goal_q = robot.check_joint_angles(ik_result.q, context="目标构型")
    reached_pose = robot.fkine_dict(goal_q, end_effector_link=END_EFFECTOR_LINK)
    reached_position = reached_pose["position"]
    reached_rotation = reached_pose["rotation"]
    reached_rpy_deg = np.rad2deg(matrix_to_rpy(reached_rotation))
    position_error_xyz = reached_position - target_position

    print("\nIK 结果")
    print(f"  目标末端 link: {END_EFFECTOR_LINK}")
    print(f"  起始构型 deg: {format_array(START_Q_DEG)}")
    print(f"  目标位置 xyz m: {format_array(target_position)}")
    print(f"  目标姿态 rpy deg: {format_array(np.asarray(args.target_rpy_deg, dtype=float))}")
    print(f"  目标构型 deg: {format_array(np.rad2deg(goal_q))}")
    print(f"  实际将到达位置 xyz m: {format_array(reached_position)}")
    print(f"  实际将到达姿态 rpy deg: {format_array(reached_rpy_deg)}")
    print(f"  实际将到达旋转矩阵:\n{format_array(reached_rotation)}")
    print(f"  位置误差 xyz m: {format_array(position_error_xyz)}")
    print(f"  best-effort success: {ik_result.success}")
    print(f"  总误差: {ik_result.error_norm:.8f}")
    print(f"  位置误差: {ik_result.position_error_norm:.8f} m")
    print(f"  姿态误差: {np.rad2deg(ik_result.rotation_error_rad):.8f} deg")
    return goal_q, ik_result


def wait_until_arm_reaches(
    robot,
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
    print(f"warning: {detail}")
    print(f"继续执行后续动作：{reached_message}。")
    if last_feedback_q is not None:
        return last_feedback_q
    return target_q


def solve_lift_after_grasp_configuration(
    robot,
    current_q: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    """根据当前末端位姿，生成沿基座 +Z 方向上抬后的目标构型。"""

    current_pose = robot.fkine_dict(current_q, end_effector_link=END_EFFECTOR_LINK)
    lift_position = current_pose["position"].copy()
    lift_position[2] += float(args.lift_after_grasp_z)
    lift_rotation = current_pose["rotation"]

    ik_result = robot.ikine_best_effort(
        target_position=lift_position,
        target_rotation=lift_rotation,
        q0=current_q,
        end_effector_link=END_EFFECTOR_LINK,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
        damping=args.damping,
        step_size=args.step_size,
        print_error=False,
    )
    lift_q = robot.check_joint_angles(ik_result.q, context="抓取后上抬构型")
    reached_pose = robot.fkine_dict(lift_q, end_effector_link=END_EFFECTOR_LINK)

    print("\n抓取后上抬 IK 结果")
    print(f"  目标末端 link: {END_EFFECTOR_LINK}")
    print(f"  当前末端位置 xyz m: {format_array(current_pose['position'])}")
    print(f"  上抬目标位置 xyz m: {format_array(lift_position)}")
    print(f"  上抬目标构型 deg: {format_array(np.rad2deg(lift_q))}")
    print(f"  实际将到达位置 xyz m: {format_array(reached_pose['position'])}")
    print(f"  best-effort success: {ik_result.success}")
    print(f"  位置误差: {ik_result.position_error_norm:.8f} m")
    print(f"  姿态误差: {np.rad2deg(ik_result.rotation_error_rad):.8f} deg")
    return lift_q


def execute_grasp_lift_and_return(
    robot,
    reached_q: np.ndarray,
    args: argparse.Namespace,
) -> None:
    """执行到达目标点后的动作：闭合夹爪、上抬、返回起始构型、可选打开夹爪。"""

    start_q = robot.check_joint_angles(np.deg2rad(START_Q_DEG), context="起始构型")

    print("\n闭合夹爪，执行抓取。")
    # print(robot.hold_gripper_closed())
    # wait(args.grasp_hold_wait)

    lift_q = solve_lift_after_grasp_configuration(robot, reached_q, args)
    print(f"\n抓取完成，沿基座 +Z 方向上抬 {args.lift_after_grasp_z:.3f} m。")
    print(robot.move_joints_rad(lift_q, speed=args.speed, acceleration=args.acc))
    wait_until_arm_reaches(
        robot,
        lift_q,
        args,
        reached_message="继续返回起始构型",
    )

    print(f"\n返回起始构型，速度 {args.return_speed:.1f}。")
    print(robot.move_joints_rad(start_q, speed=args.return_speed, acceleration=args.acc))
    wait(args.arm_wait)

    if ask_yes_no("\n已返回起始构型，是否打开夹爪？", default=False):
        print("打开夹爪。")
        print(robot.open_gripper())
        wait(args.gripper_wait)
    else:
        print("保持夹爪当前状态，结束程序。")


def execute_target_move(robot, goal_q: np.ndarray, args: argparse.Namespace) -> None:
    print(f"\n从当前构型直接移动到点云目标对应构型（目标末端 {END_EFFECTOR_LINK}）。")
    print(robot.move_joints_rad(goal_q, speed=args.speed, acceleration=args.acc))
    reached_q = wait_until_arm_reaches(
        robot,
        goal_q,
        args,
        reached_message="继续执行抓取",
    )
    execute_grasp_lift_and_return(robot, reached_q, args)


def print_target_summary(selected: SelectedTarget) -> None:
    print("\n选点结果")
    print(f"  picked point index: {selected.picked_index}")
    print(f"  camera point xyz m: {format_array(selected.camera_point_m)}")
    print(f"  base point from extrinsic xyz m: {format_array(selected.base_point_m)}")
    print(f"  target after offset/backoff xyz m: {format_array(selected.raw_target_point_m)}")
    if selected.workspace_notes:
        print("  workspace clamp applied:")
        for note in selected.workspace_notes:
            print(f"    {note}")
    else:
        print("  workspace clamp: disabled/not needed")
    print(f"  final arm target xyz m: {format_array(selected.target_position_m)}")


def confirm_move(args: argparse.Namespace) -> bool:
    if args.yes:
        return True
    return ask_yes_no("确认从当前构型移动到该点，闭合夹爪后返回起始构型？")


def main() -> int:
    args = parse_args()
    robot = None
    try:
        # 在访问相机或机械臂硬件前，先检查脚本配置是否合理。
        if args.target_x_backoff < 0:
            raise RuntimeError("--target-x-backoff 必须大于等于 0。")
        if args.target_z_gravity_compensation < 0:
            raise RuntimeError("TARGET_Z_GRAVITY_COMPENSATION_M 必须大于等于 0。")

        # 准备 Open3D、工作空间范围，以及相机到机械臂基座的坐标变换。
        import_open3d()
        workspace = make_workspace(args)
        matrix = load_transform_matrix(args.transform_matrix) if args.transform_matrix else None
        camera_to_base = load_camera_to_base_transform(args)

        # 打印当前坐标转换方式，方便在执行前发现外参或轴映射配置问题。
        if camera_to_base is not None:
            print("使用相机到基座外参矩阵:")
            print(camera_to_base.matrix)
        elif matrix is not None:
            print("使用 4x4 相机到机械臂矩阵，单位 mm:")
            print(matrix)
        else:
            print(f"使用轴映射: {args.axis_map}, offset-mm: {format_array(args.offset_mm)}")
        print(f"执行模式: {'真实运动' if args.execute else '干跑，不连接机械臂'}")
        print(f"目标末端 link: {END_EFFECTOR_LINK}")
        print(f"目标点 x 回退: {args.target_x_backoff:.3f} m")
        print(f"目标点 z 重力补偿: +{args.target_z_gravity_compensation:.3f} m")

        # 创建 BookArm 模型；真实执行时可先让机械臂回到相机采集前的起始构型。
        robot = create_bookarm()
        if not args.skip_startup:
            if not move_arm_to_startup_default(robot, args):
                return 0

        # 采集一帧 RGB-D 数据，并转换为后续选点用的点云。
        frame = capture_rgbd_frame(args)
        point_cloud = make_point_cloud(frame, args)
        point_count = len(point_cloud.points)
        print(f"point count: {point_count}")
        if point_count == 0:
            raise RuntimeError("点云为空，请调整 --depth-min/--depth-max 或相机视角。")

        if args.ply is not None:
            save_point_cloud(point_cloud, args.ply)

        # 通过 Open3D 交互选点，或使用固定点索引进行调试。
        if args.point_index is None:
            picked_index = pick_point_index(point_cloud, args.point_size, args.flip_view)
        else:
            picked_index = args.point_index
        if picked_index < 0 or picked_index >= point_count:
            raise RuntimeError(f"点索引 {picked_index} 超出范围；点云共有 {point_count} 个点。")

        # 将相机坐标系中的选中点转换到机械臂基座坐标系，并叠加目标偏移。
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

        # 如有需要，将目标点裁剪到保守工作空间范围内。
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

        # 求解目标构型，并打印目标位姿以及该构型正解得到的实际到达位姿。
        goal_q, _ik_result = solve_target_configuration(
            robot,
            float(selected.target_position_m[0]),
            float(selected.target_position_m[1]),
            # 0.0,
            float(selected.target_position_m[2]),
            args,
        )

        # 干跑模式只做选点、坐标转换和 IK 计算，不发送真实机械臂动作。
        if not args.execute:
            print("\n干跑结束：已完成点云选点、坐标转换和 IK 计算，没有连接机械臂。")
            print("确认坐标无误后，加 --execute 运行真实目标移动。")
            return 0

        # 发送目标运动命令前，再向用户做一次最终确认。
        if not confirm_move(args):
            print("已取消，未发送目标运动命令。")
            return 0

        # 执行目标移动，并把到达后的抓取、上抬、返回动作交给独立函数，便于后续调整。
        execute_target_move(robot, goal_q, args)
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted by user", file=sys.stderr)
        return 130
    finally:
        # 如果本次运行连接了真实硬件，结束前关闭共享串口连接。
        if args.execute and robot is not None:
            robot.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
