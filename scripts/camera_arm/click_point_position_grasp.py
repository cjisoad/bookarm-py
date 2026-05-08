"""采集 RealSense D435 点云，在 Open3D 中选点，并驱动 BookArm 移动到目标点。

1. 可选：先让机械臂回到起始关节构型。
2. 采集一帧 D435 RGB-D，并生成彩色点云。
3. 在 Open3D 窗口中 Shift + 左键选中目标点，按 Q 或 Esc 结束选点。
4. 把相机坐标转换到机械臂基座坐标；默认不改动转换后的目标点。
5. 使用当前项目的 `BookArm.ikine_position_best_effort()` 只按目标位置求
   关节角；显式加 --execute 时，从当前构型直接移动到目标构型，闭合夹爪
   后返回起始构型。

运行前请在项目根目录执行命令，并确认已激活包含 pyrealsense2、open3d、
pinocchio 的环境。Open3D 选点窗口打开后：

- Shift + 鼠标左键：选择目标点。
- Shift + 鼠标右键：撤销上一次选择。
- Q 或 Esc：结束选点并继续后续流程。

真实执行前的终端确认会直接读取单个按键：按 y 确认，按 n、Esc 或回车取消。

推荐先干跑检查坐标转换和 IK，不连接机械臂：

    python scripts/camera_arm/click_point_position_grasp.py

默认会自动读取 `calibration/camera_to_base.json`。当前保存的外参表示：

    base_point = [camera_z, -camera_x, -camera_y] + [-0.02, -0.15, 0.145] m

相机位置或朝向改变后，请先更新 `calibration/camera_to_base.json` 中的
`xyz_m` 和 `rpy_deg`。该文件里写有详细测量、修改和验证步骤。

如果临时不用标定文件，而使用参考项目那种简单轴映射和偏移，可以这样运行：

    python scripts/camera_arm/click_point_position_grasp.py --no-calibration --axis-map z,-x,-y --offset-mm 300 -100 0

如果已经完成外参标定，推荐使用当前项目的标定文件：

    python scripts/camera_arm/click_point_position_grasp.py --calibration calibration/camera_to_base.json

确认终端打印的相机点、机械臂目标点和 IK 结果正确后，再显式执行真实目标移动：

    python scripts/camera_arm/click_point_position_grasp.py --port /dev/bookarm --execute

真实执行时仍可加坐标转换参数，例如：

    python scripts/camera_arm/click_point_position_grasp.py --port /dev/bookarm --calibration calibration/camera_to_base.json --execute

常用调试参数：

    --depth-min 0.1 --depth-max 1.5     限制点云深度范围
    --stride 2                          降低点云密度，加快显示
    --voxel-size 0.003                  用 Open3D 体素降采样
    --point-index N                     直接使用点云索引，不打开选点窗口
    --clamp-workspace                   显式启用工作空间裁剪
    --skip-startup                      跳过相机采集前的起始构型动作
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

from bookarm_control_py.math_utils import rpy_to_matrix


START_Q_DEG = np.array([0.0, -60.0, 70.0, 0.0, -45.0], dtype=float)
DEFAULT_AXIS_MAP = "z,-x,-y"
DEFAULT_OFFSET_MM = (300.0, -100.0, 0.0)
DEFAULT_CALIBRATION_PATH = Path("calibration/camera_to_base.json")
# 机械臂目标点的默认后退距离
DEFAULT_TARGET_X_BACKOFF_M = 0.02
DEFAULT_SPEED = 25.0
DEFAULT_RETURN_SPEED = 20.0
DEFAULT_ACC = 5.0
DEFAULT_ARM_WAIT = 5.0
DEFAULT_GRIPPER_WAIT = 1.0
DEFAULT_GRASP_HOLD_WAIT = 3.0


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
    return BookArm()


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
        description="采集 D435 点云，在 Open3D 中选点，并让 BookArm 只按位置抓取该点。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    camera = parser.add_argument_group("camera")
    camera.add_argument("--serial", default=None, help="指定 D435 序列号；只有一台相机时可不填。")
    camera.add_argument("--width", type=int, default=640, help="彩色和深度流宽度。")
    camera.add_argument("--height", type=int, default=480, help="彩色和深度流高度。")
    camera.add_argument("--fps", type=int, default=30, help="采集帧率。")
    camera.add_argument("--warmup", type=int, default=30, help="丢弃前 N 帧等待自动曝光稳定。")
    camera.add_argument("--timeout-ms", type=int, default=5000, help="等待相机帧的超时时间。")
    camera.add_argument("--depth-min", type=float, default=0.0, help="点云保留的最小深度，单位米。")
    camera.add_argument("--depth-max", "--max-depth", dest="depth_max", type=float, default=4.0, help="点云保留的最大深度，单位米。")
    camera.add_argument("--stride", type=int, default=1, help="点云采样步长；1 表示保留所有有效深度点。")
    camera.add_argument("--voxel-size", type=float, default=0.0, help="Open3D 体素降采样尺寸，单位米；0 表示不降采样。")

    picker = parser.add_argument_group("point picking")
    picker.add_argument("--point-size", type=float, default=1.0, help="Open3D 选点窗口中的点大小。")
    picker.add_argument("--point-index", type=int, default=None, help="直接使用点云索引，不打开选点窗口。")
    picker.add_argument("--ply", type=Path, default=None, help="可选：保存采集到的点云 PLY。")
    picker.add_argument("--no-flip-view", dest="flip_view", action="store_false", default=True, help="Open3D 窗口保留 RealSense 原始坐标朝向。")

    transform = parser.add_argument_group("camera to arm transform")
    transform.add_argument("--axis-map", default=DEFAULT_AXIS_MAP, help="相机坐标到机械臂坐标的轴映射，例如 z,-x,-y。")
    transform.add_argument("--offset-mm", type=float, nargs=3, default=DEFAULT_OFFSET_MM, metavar=("X", "Y", "Z"), help="轴映射后叠加的机械臂坐标偏移，单位毫米。")
    transform.add_argument("--transform-matrix", type=Path, default=None, help="4x4 齐次矩阵文件，输入/输出单位均为毫米；会覆盖 --axis-map 和 --offset-mm。")
    transform.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION_PATH, help="当前项目格式的外参 JSON，输入/输出单位为米；默认自动读取 calibration/camera_to_base.json。")
    transform.add_argument("--no-calibration", action="store_true", help="不读取默认标定文件，改用 --axis-map/--offset-mm 或 --transform-matrix。")
    transform.add_argument("--camera-to-base-xyz", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"), help="直接指定相机原点在基座坐标系下的位置，单位米。")
    transform.add_argument("--camera-to-base-rpy-deg", type=float, nargs=3, default=None, metavar=("ROLL", "PITCH", "YAW"), help="直接指定相机到基座的欧拉角，单位度。")
    transform.add_argument("--target-offset-base", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("DX", "DY", "DZ"), help="转换到基座坐标后额外叠加的末端偏移，单位米。")
    transform.add_argument("--target-x-backoff", type=float, default=DEFAULT_TARGET_X_BACKOFF_M, help="在机械臂基坐标系下，让目标点相对点云选取点沿 x 负方向回退的距离，单位米。")

    workspace = parser.add_argument_group("arm workspace")
    workspace.add_argument("--x-min", type=float, default=0.0, help="机械臂目标 x 最小值，单位毫米。")
    workspace.add_argument("--x-max", type=float, default=450.0, help="机械臂目标 x 最大值，单位毫米。")
    workspace.add_argument("--y-min", type=float, default=-300.0, help="机械臂目标 y 最小值，单位毫米。")
    workspace.add_argument("--y-max", type=float, default=300.0, help="机械臂目标 y 最大值，单位毫米。")
    workspace.add_argument("--z-min", type=float, default=20.0, help="机械臂目标 z 最小值，单位毫米。")
    workspace.add_argument("--z-max", type=float, default=400.0, help="机械臂目标 z 最大值，单位毫米。")
    workspace.add_argument("--xy-radius-min", type=float, default=40.0, help="机械臂水平半径最小值，单位毫米。")
    workspace.add_argument("--xy-radius-max", type=float, default=450.0, help="机械臂水平半径最大值，单位毫米。")
    workspace.add_argument("--clamp-workspace", action="store_true", help="显式启用工作空间裁剪；默认不裁剪外参转换后的目标点。")

    arm = parser.add_argument_group("arm")
    arm.add_argument("--port", default="/dev/bookarm", help="机械臂和夹爪共用串口，例如 /dev/bookarm。")
    arm.add_argument("--execute", dest="execute", action="store_true", help="真正连接串口并执行机械臂运动。")
    arm.add_argument("--dry-run", dest="execute", action="store_false", help="只打印目标、IK 和命令，不连接机械臂。")
    arm.set_defaults(execute=False)
    arm.add_argument("--yes", action="store_true", help="跳过真实运动前的确认提示。")
    arm.add_argument("--skip-startup", action="store_true", help="跳过相机采集前的起始构型运动。")
    arm.add_argument("--arm-test", action="store_true", help="起始构型运动后要求人工确认机械臂动作正常。")
    arm.add_argument("--speed", type=float, default=DEFAULT_SPEED, help="机械臂运动速度。")
    arm.add_argument("--return-speed", type=float, default=35, help="抓取后返回起始构型速度。")
    arm.add_argument("--acc", type=float, default=DEFAULT_ACC, help="机械臂运动加速度。")
    arm.add_argument("--arm-wait", type=float, default=DEFAULT_ARM_WAIT, help="机械臂运动后的等待秒数。")
    arm.add_argument("--gripper-wait", type=float, default=DEFAULT_GRIPPER_WAIT, help="夹爪动作后的等待秒数。")
    arm.add_argument("--grasp-hold-wait", type=float, default=DEFAULT_GRASP_HOLD_WAIT, help="发送夹紧指令后、返回起始构型前的等待秒数。")
    arm.add_argument("--grasp-repeat-hz", type=float, default=None, help=argparse.SUPPRESS)
    arm.add_argument("--no-grasp", action="store_true", help=argparse.SUPPRESS)
    arm.add_argument("--max-iterations", type=int, default=500, help="位置 best-effort IK 最大迭代次数。")
    arm.add_argument("--tolerance", type=float, default=1e-4, help="位置 best-effort IK 收敛容差。")
    arm.add_argument("--damping", type=float, default=1e-6, help="阻尼最小二乘阻尼系数。")
    arm.add_argument("--step-size", type=float, default=0.4, help="IK 每步积分步长。")
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

    # print("  开启机械臂力矩。")
    # print(robot.enable_torque())

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

    ik_result = robot.ikine_position_best_effort(
        target_position=target_position,
        q0=start_q,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
        damping=args.damping,
        step_size=args.step_size,
        print_error=False,
    )
    goal_q = robot.check_joint_angles(ik_result.q, context="目标构型")

    print("\nIK 结果")
    print(f"  起始构型 deg: {format_array(START_Q_DEG)}")
    print(f"  目标位置 xyz m: {format_array(target_position)}")
    print(f"  目标构型 deg: {format_array(np.rad2deg(goal_q))}")
    print(f"  best-effort success: {ik_result.success}")
    print(f"  位置误差: {ik_result.position_error_norm:.8f} m")
    return goal_q, ik_result


def execute_target_move(robot, goal_q: np.ndarray, args: argparse.Namespace) -> None:
    start_q = robot.check_joint_angles(np.deg2rad(START_Q_DEG), context="起始构型")

    print("\n从当前构型直接移动到点云目标对应构型。")
    print(robot.move_joints_rad(goal_q, speed=args.speed, acceleration=args.acc))
    wait(args.arm_wait)

    print("\n闭合夹爪，执行抓取。")
    print(robot.hold_gripper_closed())
    wait(args.grasp_hold_wait)

    print(f"\n返回起始构型，速度 {args.return_speed:.1f}。")
    print(robot.move_joints_rad(start_q, speed=args.return_speed, acceleration=args.acc))
    wait(args.arm_wait)

    if ask_yes_no("\n已返回起始构型，是否打开夹爪？", default=False):
        print("打开夹爪。")
        print(robot.open_gripper())
        wait(args.gripper_wait)
    else:
        print("保持夹爪当前状态，结束程序。")


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
        if args.target_x_backoff < 0:
            raise RuntimeError("--target-x-backoff 必须大于等于 0。")

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
        print(f"目标点 x 回退: {args.target_x_backoff:.3f} m")

        robot = create_bookarm()
        if not args.skip_startup:
            if not move_arm_to_startup_default(robot, args):
                return 0

        frame = capture_rgbd_frame(args)
        point_cloud = make_point_cloud(frame, args)
        point_count = len(point_cloud.points)
        print(f"point count: {point_count}")
        if point_count == 0:
            raise RuntimeError("点云为空，请调整 --depth-min/--depth-max 或相机视角。")

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
        raw_target_point_m = base_point_m + np.asarray(args.target_offset_base, dtype=float)
        raw_target_point_m[0] -= args.target_x_backoff
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

        goal_q, _ik_result = solve_target_configuration(
            robot,
            float(selected.target_position_m[0]),
            float(selected.target_position_m[1]),
            # 0.0,
            float(selected.target_position_m[2]),
            args,
        )

        if not args.execute:
            print("\n干跑结束：已完成点云选点、坐标转换和 IK 计算，没有连接机械臂。")
            print("确认坐标无误后，加 --execute 运行真实目标移动。")
            return 0

        if not confirm_move(args):
            print("已取消，未发送目标运动命令。")
            return 0

        execute_target_move(robot, goal_q, args)
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


if __name__ == "__main__":
    raise SystemExit(main())
