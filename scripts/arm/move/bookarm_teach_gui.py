"""Manual teaching GUI for BookArm.

Run on Ubuntu with:

    python3 scripts/arm/move/bookarm_teach_gui.py

The script works with the local `bookarm_control_py` package when its
dependencies are available. If the package cannot be imported, the GUI still
starts in a small simulation mode so the interface and JSON export can be
exercised.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import tkinter as tk
from tkinter import font as tkfont
from tkinter import filedialog, messagebox, ttk


def _bootstrap_src_path() -> None:
    here = Path(__file__).resolve()
    for base in (here.parent, *here.parents):
        src = base / "src"
        if src.is_dir():
            src_str = str(src)
            if src_str not in sys.path:
                sys.path.insert(0, src_str)
            return


_bootstrap_src_path()

try:  # pragma: no cover - import availability depends on the user machine
    from serial.tools import list_ports
except Exception:  # pragma: no cover - optional helper
    list_ports = None

try:  # pragma: no cover - depends on external package installation
    from bookarm_control_py import BookArm
except Exception as exc:  # pragma: no cover - handled at runtime
    BookArm = None
    IMPORT_ERROR = exc
else:  # pragma: no cover - handled at runtime
    IMPORT_ERROR = None


DEFAULT_JOINT_SPEED = 25.0
DEFAULT_JOINT_ACCEL = 5.0
DEFAULT_RAW_SPEED = 300.0
DEFAULT_RAW_ACCEL = 20.0
DEFAULT_SETTLE_SECONDS = 3.0
DEFAULT_RESPONSE_TIMEOUT = 5.0
DEFAULT_STATUS_INTERVAL = 1.0
DEFAULT_SERIAL_PORT = "/dev/bookarm"
DEFAULT_BAUDRATE = 921600
DEFAULT_DTR = True
DEFAULT_RTS = False
DEFAULT_UI_SCALE = 1.35
DEFAULT_FONT_SIZE = 13


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def to_float_list(values: Any) -> list[float] | None:
    if values is None:
        return None
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [float(x) for x in values]


def to_int_list(values: Any) -> list[int] | None:
    if values is None:
        return None
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [int(x) for x in values]


def to_bool_list(values: Any) -> list[bool] | None:
    if values is None:
        return None
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [bool(x) for x in values]


def safe_float(text: str, field_name: str) -> float:
    try:
        return float(text.strip())
    except Exception as exc:
        raise ValueError(f"{field_name} 不是有效数字：{text!r}") from exc


def safe_port(text: str) -> str:
    port = text.strip()
    if not port:
        raise ValueError("串口不能为空")
    if port.startswith("dev/"):
        port = "/" + port
    elif port.startswith(("ttyUSB", "ttyACM", "ttyS")):
        port = "/dev/" + port
    return port


def display_joint_values(values: list[float] | None) -> str:
    if not values:
        return "-"
    return ", ".join(f"{v:.4f}" for v in values)


def display_int_values(values: list[int] | None) -> str:
    if not values:
        return "-"
    return ", ".join(str(v) for v in values)


@dataclass
class TeachPoint:
    name: str
    q_rad: list[float]
    created_at: str
    gripper_closed: bool | None = None
    torque: list[float] | None = None
    raw_pos: list[int] | None = None
    online: list[bool] | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "q_rad": self.q_rad,
            "created_at": self.created_at,
        }
        if self.gripper_closed is not None:
            data["gripper_closed"] = self.gripper_closed
        if self.torque is not None:
            data["torque"] = self.torque
        if self.raw_pos is not None:
            data["raw_pos"] = self.raw_pos
        if self.online is not None:
            data["online"] = self.online
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any], index: int) -> "TeachPoint":
        name = str(data.get("name") or f"P{index + 1:02d}")
        q_raw = data.get("q_rad")
        if q_raw is None:
            raise ValueError(f"点位 {name} 缺少 q_rad")
        q_rad = [float(x) for x in q_raw]
        created_at = str(data.get("created_at") or now_iso())
        gripper_closed = data.get("gripper_closed")
        torque = data.get("torque")
        raw_pos = data.get("raw_pos")
        online = data.get("online")
        return cls(
            name=name,
            q_rad=q_rad,
            created_at=created_at,
            gripper_closed=None if gripper_closed is None else bool(gripper_closed),
            torque=[float(x) for x in torque] if torque is not None else None,
            raw_pos=[int(x) for x in raw_pos] if raw_pos is not None else None,
            online=[bool(x) for x in online] if online is not None else None,
        )


@dataclass(frozen=True)
class GuiSettings:
    joint_speed: float
    joint_acceleration: float
    raw_speed: float
    raw_acceleration: float
    settle_seconds: float
    response_timeout: float
    playback_mode: str


class DummyArm:
    """Tiny fallback backend for UI testing without the real robot package."""

    def __init__(self) -> None:
        self.port = ""
        self.connected = False
        self.torque_enabled = False
        self.gripper_closed = False
        self.q_rad = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.joint_names = ("joint1_base", "joint2_shoulder", "joint3_elbow", "joint4_wrist", "joint_pole")

    def connect_serial(self, *, port: str) -> "DummyArm":
        self.port = port
        self.connected = True
        return self

    def close(self) -> None:
        self.connected = False

    def enable_torque(self, *, wait_response: bool = False, response_timeout: float | None = None) -> dict[str, Any]:
        self.torque_enabled = True
        return {"ok": True, "simulated": True, "torque": "enabled"}

    def disable_torque(self, *, wait_response: bool = False, response_timeout: float | None = None) -> dict[str, Any]:
        self.torque_enabled = False
        return {"ok": True, "simulated": True, "torque": "disabled"}

    def read_arm_feedback(self, *, response_timeout: float | None = None, validate_limits: bool = False) -> SimpleNamespace:
        torque = [0.0, 0.0, 0.0, 0.0, 0.0]
        return SimpleNamespace(
            raw={"simulated": True},
            q_rad=list(self.q_rad),
            torque=torque,
            raw_pos=None,
            online=None,
        )

    def move_joints_rad(
        self,
        joint_angles_rad: list[float],
        *,
        speed: float = DEFAULT_JOINT_SPEED,
        acceleration: float = DEFAULT_JOINT_ACCEL,
        wait_response: bool = False,
        response_timeout: float | None = None,
    ) -> dict[str, Any]:
        self.q_rad = [float(x) for x in joint_angles_rad]
        return {"ok": True, "simulated": True, "mode": "joint", "q_rad": self.q_rad}

    def move_raw_positions(
        self,
        positions: list[int],
        *,
        speed: float = DEFAULT_RAW_SPEED,
        acceleration: float = DEFAULT_RAW_ACCEL,
        wait_response: bool = False,
        response_timeout: float | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError("BookArm 后端不支持原始位置回放，请使用“关节角”模式。")

    def set_gripper_closed(
        self,
        closed: bool,
        *,
        speed: float = 100.0,
        acceleration: float = 10.0,
        torque: float = 1000.0,
        wait_response: bool = False,
        response_timeout: float | None = None,
    ) -> dict[str, Any]:
        self.gripper_closed = bool(closed)
        return {"ok": True, "simulated": True, "gripper_closed": self.gripper_closed}


class RealArm:
    """Thin wrapper around the package API."""

    def __init__(self) -> None:
        self.robot = None

    @property
    def connected(self) -> bool:
        return self.robot is not None

    @property
    def joint_names(self) -> tuple[str, ...]:
        if self.robot is not None:
            return tuple(self.robot.joint_names)
        return ("joint1_base", "joint2_shoulder", "joint3_elbow", "joint4_wrist", "joint_pole")

    def connect_serial(self, *, port: str) -> "RealArm":
        if BookArm is None:
            raise RuntimeError(f"BookArm 控制库不可用：{IMPORT_ERROR}")
        self.robot = BookArm().connect_serial(port=port)
        return self

    def close(self) -> None:
        if self.robot is not None:
            self.robot.close()
            self.robot = None

    def enable_torque(self, *, wait_response: bool = False, response_timeout: float | None = None) -> Any:
        return self.robot.enable_torque(wait_response=False, response_timeout=response_timeout)

    def disable_torque(self, *, wait_response: bool = False, response_timeout: float | None = None) -> Any:
        return self.robot.disable_torque(wait_response=False, response_timeout=response_timeout)

    def read_arm_feedback(self, *, response_timeout: float | None = None, validate_limits: bool = False) -> Any:
        return self.robot.read_arm_feedback(response_timeout=response_timeout)

    def move_joints_rad(
        self,
        joint_angles_rad: list[float],
        *,
        speed: float = DEFAULT_JOINT_SPEED,
        acceleration: float = DEFAULT_JOINT_ACCEL,
        wait_response: bool = False,
        response_timeout: float | None = None,
    ) -> Any:
        return self.robot.move_joints_rad(
            joint_angles_rad,
            speed=speed,
            acceleration=acceleration,
            wait_response=False,
            response_timeout=response_timeout,
        )

    def move_raw_positions(
        self,
        positions: list[int],
        *,
        speed: float = DEFAULT_RAW_SPEED,
        acceleration: float = DEFAULT_RAW_ACCEL,
        wait_response: bool = False,
        response_timeout: float | None = None,
    ) -> Any:
        raise NotImplementedError("BookArm 控制库不支持原始位置回放，请使用“关节角”模式。")

    def set_gripper_closed(
        self,
        closed: bool,
        *,
        speed: float = 100.0,
        acceleration: float = 10.0,
        torque: float = 1000.0,
        wait_response: bool = False,
        response_timeout: float | None = None,
    ) -> Any:
        if closed:
            return self.robot.hold_gripper_closed(
                wait_response=False,
                response_timeout=response_timeout,
            )
        return self.robot.open_gripper(
            wait_response=False,
            response_timeout=response_timeout,
        )


class ArmTeachGUI:
    def __init__(self, root: tk.Tk, simulate: bool = False, ui_scale: float = DEFAULT_UI_SCALE) -> None:
        self.root = root
        self.simulate_requested = simulate
        self.ui_scale = max(1.0, float(ui_scale))
        self.backend: Any | None = None
        self.points: list[TeachPoint] = []
        self.task_busy = False
        self.task_lock = threading.Lock()
        self.io_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.status_stop_event = threading.Event()
        self.status_thread: threading.Thread | None = None
        self.gripper_commanded_closed: bool | None = None
        self.events: "queue.Queue[tuple[str, Any]]" = queue.Queue()

        self.port_var = tk.StringVar(value=DEFAULT_SERIAL_PORT)
        self.baudrate_var = tk.StringVar(value=str(DEFAULT_BAUDRATE))
        self.dtr_var = tk.BooleanVar(value=DEFAULT_DTR)
        self.rts_var = tk.BooleanVar(value=DEFAULT_RTS)
        self.backend_mode_var = tk.StringVar(value=self._default_backend_label())
        self.point_name_var = tk.StringVar(value="P01")
        self.playback_mode_var = tk.StringVar(value="关节角")
        self.joint_speed_var = tk.StringVar(value=str(DEFAULT_JOINT_SPEED))
        self.joint_acc_var = tk.StringVar(value=str(DEFAULT_JOINT_ACCEL))
        self.raw_speed_var = tk.StringVar(value=str(DEFAULT_RAW_SPEED))
        self.raw_acc_var = tk.StringVar(value=str(DEFAULT_RAW_ACCEL))
        self.settle_var = tk.StringVar(value=str(DEFAULT_SETTLE_SECONDS))
        self.timeout_var = tk.StringVar(value=str(DEFAULT_RESPONSE_TIMEOUT))
        self.status_interval_var = tk.StringVar(value=str(DEFAULT_STATUS_INTERVAL))
        self.ui_scale_var = tk.StringVar(value=f"{self.ui_scale:.2f}")
        self.gripper_closed_var = tk.BooleanVar(value=False)
        self.gripper_status_var = tk.StringVar(value="爪夹: 开")
        self.status_var = tk.StringVar(value="未连接")
        self.feedback_var = tk.StringVar(value="q=-")
        self.torque_var = tk.StringVar(value="扭矩: -")
        self.realtime_var = tk.StringVar(value="实时: 停止")
        self.last_update_var = tk.StringVar(value="更新时间: -")
        self.online_var = tk.StringVar(value="在线: -")
        self.mode_var = tk.StringVar(value=self._default_backend_label())

        self._configure_ui_scale()
        self._build_ui()
        self._refresh_ports()
        self._poll_events()

    def _default_backend_label(self) -> str:
        if self.simulate_requested or BookArm is None:
            return "模拟"
        return "控制库"

    def _configure_ui_scale(self) -> None:
        try:
            self.root.tk.call("tk", "scaling", self.ui_scale)
        except tk.TclError:
            pass

        base_size = max(DEFAULT_FONT_SIZE, int(round(DEFAULT_FONT_SIZE * self.ui_scale / DEFAULT_UI_SCALE)))
        text_size = base_size
        heading_size = base_size + 1

        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkFixedFont"):
            try:
                tkfont.nametofont(name).configure(size=text_size)
            except tk.TclError:
                pass
        for name in ("TkHeadingFont", "TkCaptionFont"):
            try:
                tkfont.nametofont(name).configure(size=heading_size)
            except tk.TclError:
                pass

        style = ttk.Style(self.root)
        style.configure(".", font=("TkDefaultFont", text_size))
        style.configure("TButton", padding=(10, 7))
        style.configure("TEntry", padding=(4, 4))
        style.configure("TCombobox", padding=(4, 4))
        style.configure("Treeview", rowheight=int(30 * self.ui_scale))
        style.configure("Treeview.Heading", font=("TkHeadingFont", heading_size))

    def _build_ui(self) -> None:
        self.root.title("BookArm 机械臂手动示教")
        width = int(1380 * self.ui_scale / DEFAULT_UI_SCALE)
        height = int(900 * self.ui_scale / DEFAULT_UI_SCALE)
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(int(1180 * self.ui_scale / DEFAULT_UI_SCALE), int(760 * self.ui_scale / DEFAULT_UI_SCALE))

        outer = ttk.Frame(self.root, padding=10)
        outer.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(2, weight=1)

        conn = ttk.Frame(outer)
        conn.grid(row=0, column=0, sticky="ew")
        conn.columnconfigure(17, weight=1)

        ttk.Label(conn, text="串口").grid(row=0, column=0, padx=(0, 6), pady=2, sticky="w")
        self.port_combo = ttk.Combobox(conn, textvariable=self.port_var, width=18, state="normal")
        self.port_combo.grid(row=0, column=1, padx=(0, 6), pady=2, sticky="w")
        ttk.Button(conn, text="刷新", command=self._refresh_ports).grid(row=0, column=2, padx=(0, 10), pady=2)
        ttk.Label(conn, text="波特率").grid(row=0, column=3, padx=(0, 6), pady=2, sticky="w")
        ttk.Entry(conn, textvariable=self.baudrate_var, width=10).grid(row=0, column=4, padx=(0, 10), pady=2, sticky="w")
        ttk.Checkbutton(conn, text="DTR", variable=self.dtr_var).grid(row=0, column=5, padx=(0, 6), pady=2, sticky="w")
        ttk.Checkbutton(conn, text="RTS", variable=self.rts_var).grid(row=0, column=6, padx=(0, 10), pady=2, sticky="w")
        ttk.Label(conn, text="后端").grid(row=0, column=7, padx=(0, 6), pady=2, sticky="w")
        self.backend_combo = ttk.Combobox(
            conn,
            textvariable=self.backend_mode_var,
            values=("控制库", "模拟"),
            state="readonly",
            width=10,
        )
        self.backend_combo.grid(row=0, column=8, padx=(0, 10), pady=2, sticky="w")
        ttk.Button(conn, text="连接", command=self.on_connect).grid(row=0, column=9, padx=(0, 6), pady=2)
        ttk.Button(conn, text="断开", command=self.on_disconnect).grid(row=0, column=10, padx=(0, 6), pady=2)
        ttk.Button(conn, text="通讯自检", command=self.on_probe_connection).grid(row=0, column=11, padx=(0, 14), pady=2)

        ttk.Label(conn, text="模式").grid(row=0, column=12, padx=(0, 6), pady=2, sticky="w")
        ttk.Label(conn, textvariable=self.mode_var, width=10).grid(row=0, column=13, padx=(0, 14), pady=2, sticky="w")

        ttk.Label(conn, text="状态").grid(row=0, column=14, padx=(0, 6), pady=2, sticky="w")
        ttk.Label(conn, textvariable=self.status_var).grid(row=0, column=15, padx=(0, 14), pady=2, sticky="w")

        ttk.Label(conn, textvariable=self.torque_var).grid(row=0, column=16, padx=(0, 14), pady=2, sticky="w")
        ttk.Label(conn, textvariable=self.feedback_var).grid(row=0, column=17, padx=(0, 0), pady=2, sticky="w")

        settings = ttk.Frame(outer)
        settings.grid(row=1, column=0, sticky="ew", pady=(8, 8))
        for col in range(16):
            settings.columnconfigure(col, weight=0)
        settings.columnconfigure(15, weight=1)

        ttk.Label(settings, text="点名").grid(row=0, column=0, padx=(0, 6), pady=2, sticky="w")
        ttk.Entry(settings, textvariable=self.point_name_var, width=16).grid(row=0, column=1, padx=(0, 12), pady=2, sticky="w")

        ttk.Label(settings, text="回放").grid(row=0, column=2, padx=(0, 6), pady=2, sticky="w")
        self.mode_combo = ttk.Combobox(settings, textvariable=self.playback_mode_var, values=("关节角", "原始位置"), state="readonly", width=10)
        self.mode_combo.grid(row=0, column=3, padx=(0, 12), pady=2, sticky="w")

        ttk.Label(settings, text="关节速").grid(row=0, column=4, padx=(0, 6), pady=2, sticky="w")
        ttk.Entry(settings, textvariable=self.joint_speed_var, width=8).grid(row=0, column=5, padx=(0, 12), pady=2, sticky="w")
        ttk.Label(settings, text="关节加").grid(row=0, column=6, padx=(0, 6), pady=2, sticky="w")
        ttk.Entry(settings, textvariable=self.joint_acc_var, width=8).grid(row=0, column=7, padx=(0, 12), pady=2, sticky="w")

        ttk.Label(settings, text="原始速").grid(row=0, column=8, padx=(0, 6), pady=2, sticky="w")
        ttk.Entry(settings, textvariable=self.raw_speed_var, width=8).grid(row=0, column=9, padx=(0, 12), pady=2, sticky="w")
        ttk.Label(settings, text="原始加").grid(row=0, column=10, padx=(0, 6), pady=2, sticky="w")
        ttk.Entry(settings, textvariable=self.raw_acc_var, width=8).grid(row=0, column=11, padx=(0, 12), pady=2, sticky="w")

        ttk.Label(settings, text="停顿").grid(row=0, column=12, padx=(0, 6), pady=2, sticky="w")
        ttk.Entry(settings, textvariable=self.settle_var, width=8).grid(row=0, column=13, padx=(0, 12), pady=2, sticky="w")
        ttk.Label(settings, text="超时").grid(row=0, column=14, padx=(0, 6), pady=2, sticky="w")
        ttk.Entry(settings, textvariable=self.timeout_var, width=8).grid(row=0, column=15, padx=(0, 0), pady=2, sticky="w")

        ttk.Label(settings, text="状态刷新").grid(row=1, column=0, padx=(0, 6), pady=2, sticky="w")
        ttk.Entry(settings, textvariable=self.status_interval_var, width=8).grid(row=1, column=1, padx=(0, 12), pady=2, sticky="w")
        ttk.Button(settings, text="开始实时", command=self.on_start_realtime).grid(row=1, column=2, padx=(0, 6), pady=2, sticky="w")
        ttk.Button(settings, text="停止实时", command=self.on_stop_realtime).grid(row=1, column=3, padx=(0, 12), pady=2, sticky="w")
        ttk.Label(settings, text="缩放").grid(row=1, column=4, padx=(0, 6), pady=2, sticky="w")
        ttk.Entry(settings, textvariable=self.ui_scale_var, width=8).grid(row=1, column=5, padx=(0, 6), pady=2, sticky="w")
        ttk.Button(settings, text="应用缩放", command=self.on_apply_ui_scale).grid(row=1, column=6, padx=(0, 12), pady=2, sticky="w")
        ttk.Label(settings, textvariable=self.realtime_var).grid(row=1, column=7, columnspan=2, padx=(0, 12), pady=2, sticky="w")
        ttk.Label(settings, textvariable=self.last_update_var).grid(row=1, column=9, columnspan=4, padx=(0, 12), pady=2, sticky="w")
        ttk.Label(settings, textvariable=self.online_var).grid(row=1, column=13, columnspan=3, padx=(0, 0), pady=2, sticky="w")

        actions = ttk.Frame(outer)
        actions.grid(row=2, column=0, sticky="nsew")
        actions.columnconfigure(0, weight=2)
        actions.columnconfigure(1, weight=1)
        actions.rowconfigure(0, weight=1)

        left = ttk.Frame(actions)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)

        tree_wrap = ttk.Frame(left)
        tree_wrap.grid(row=0, column=0, sticky="nsew")
        tree_wrap.columnconfigure(0, weight=1)
        tree_wrap.rowconfigure(0, weight=1)

        columns = ("name", "gripper", "q0", "q1", "q2", "q3", "q4", "created")
        self.tree = ttk.Treeview(tree_wrap, columns=columns, show="headings", height=16, selectmode="extended")
        self.tree.heading("name", text="点名")
        self.tree.heading("gripper", text="爪夹")
        self.tree.heading("q0", text="q0")
        self.tree.heading("q1", text="q1")
        self.tree.heading("q2", text="q2")
        self.tree.heading("q3", text="q3")
        self.tree.heading("q4", text="q4")
        self.tree.heading("created", text="时间")
        self.tree.column("name", width=110, anchor="w")
        self.tree.column("gripper", width=70, anchor="center")
        for col in ("q0", "q1", "q2", "q3", "q4"):
            self.tree.column(col, width=105, anchor="e")
        self.tree.column("created", width=170, anchor="w")
        self.tree.grid(row=0, column=0, sticky="nsew")
        tree_scroll = ttk.Scrollbar(tree_wrap, orient="vertical", command=self.tree.yview)
        tree_scroll.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=tree_scroll.set)

        point_buttons = ttk.Frame(left)
        point_buttons.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        for i in range(7):
            point_buttons.columnconfigure(i, weight=0)
        point_buttons.columnconfigure(7, weight=1)

        ttk.Button(point_buttons, text="读取", command=self.on_read_feedback).grid(row=0, column=0, padx=(0, 6), pady=2)
        ttk.Button(point_buttons, text="记录", command=self.on_record_point).grid(row=0, column=1, padx=(0, 6), pady=2)
        ttk.Button(point_buttons, text="回放选中", command=self.on_play_selected).grid(row=0, column=2, padx=(0, 6), pady=2)
        ttk.Button(point_buttons, text="回放全部", command=self.on_play_all).grid(row=0, column=3, padx=(0, 6), pady=2)
        ttk.Button(point_buttons, text="停止回放", command=self.on_stop_playback).grid(row=0, column=4, padx=(0, 6), pady=2)
        ttk.Button(point_buttons, text="上移", command=self.on_move_up).grid(row=0, column=5, padx=(0, 6), pady=2)
        ttk.Button(point_buttons, text="下移", command=self.on_move_down).grid(row=0, column=6, padx=(0, 6), pady=2)
        ttk.Button(point_buttons, text="删除选中", command=self.on_delete_selected).grid(row=0, column=7, padx=(0, 6), pady=2, sticky="e")
        ttk.Button(point_buttons, text="清空", command=self.on_clear_points).grid(row=0, column=8, padx=(0, 6), pady=2, sticky="e")

        right = ttk.Frame(actions)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)

        torque_box = ttk.Frame(right)
        torque_box.grid(row=0, column=0, sticky="ew")
        ttk.Button(torque_box, text="锁定扭矩", command=self.on_enable_torque).grid(row=0, column=0, padx=(0, 6), pady=2, sticky="ew")
        ttk.Button(torque_box, text="释放扭矩", command=self.on_disable_torque).grid(row=0, column=1, padx=(0, 6), pady=2, sticky="ew")
        ttk.Checkbutton(
            torque_box,
            text="爪夹闭合",
            variable=self.gripper_closed_var,
            command=self.on_gripper_toggle,
        ).grid(row=1, column=0, padx=(0, 6), pady=6, sticky="ew")
        ttk.Label(torque_box, textvariable=self.gripper_status_var).grid(row=1, column=1, padx=(0, 6), pady=6, sticky="w")
        ttk.Button(torque_box, text="导入JSON", command=self.on_import_json).grid(row=2, column=0, padx=(0, 6), pady=6, sticky="ew")
        ttk.Button(torque_box, text="导出JSON", command=self.on_export_json).grid(row=2, column=1, padx=(0, 6), pady=6, sticky="ew")

        info = ttk.Frame(right)
        info.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        info.columnconfigure(1, weight=1)
        ttk.Label(info, text="关节").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=2)
        self.joint_feedback_var = tk.StringVar(value="-")
        ttk.Label(info, textvariable=self.joint_feedback_var).grid(row=0, column=1, sticky="w", pady=2)
        ttk.Label(info, text="原始").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=2)
        self.raw_feedback_var = tk.StringVar(value="-")
        ttk.Label(info, textvariable=self.raw_feedback_var).grid(row=1, column=1, sticky="w", pady=2)
        ttk.Label(info, text="扭矩").grid(row=2, column=0, sticky="w", padx=(0, 6), pady=2)
        self.torque_feedback_var = tk.StringVar(value="-")
        ttk.Label(info, textvariable=self.torque_feedback_var).grid(row=2, column=1, sticky="w", pady=2)
        ttk.Label(info, text="在线").grid(row=3, column=0, sticky="w", padx=(0, 6), pady=2)
        self.online_feedback_var = tk.StringVar(value="-")
        ttk.Label(info, textvariable=self.online_feedback_var).grid(row=3, column=1, sticky="w", pady=2)
        ttk.Label(info, text="更新").grid(row=4, column=0, sticky="w", padx=(0, 6), pady=2)
        self.update_feedback_var = tk.StringVar(value="-")
        ttk.Label(info, textvariable=self.update_feedback_var).grid(row=4, column=1, sticky="w", pady=2)

        log_box = ttk.Frame(outer)
        log_box.grid(row=3, column=0, sticky="nsew", pady=(8, 0))
        log_box.columnconfigure(0, weight=1)
        log_box.rowconfigure(1, weight=1)

        ttk.Label(log_box, text="日志").grid(row=0, column=0, sticky="w", pady=(0, 4))
        self.log_text = tk.Text(log_box, height=10, wrap="word", state="disabled")
        self.log_text.grid(row=1, column=0, sticky="nsew")
        log_scroll = ttk.Scrollbar(log_box, orient="vertical", command=self.log_text.yview)
        log_scroll.grid(row=1, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=log_scroll.set)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._set_buttons_busy(False)
        self._log("GUI 已启动")
        if BookArm is None and not self.simulate_requested:
            self._log(f"控制库导入失败，当前自动进入模拟模式：{IMPORT_ERROR}")

    def _refresh_ports(self) -> None:
        ports: list[str] = []
        if list_ports is not None:
            ports = [item.device for item in list_ports.comports()]
        if DEFAULT_SERIAL_PORT not in ports:
            ports.append(DEFAULT_SERIAL_PORT)
        self.port_combo["values"] = ports
        current_port = self.port_var.get().strip()
        if current_port:
            self.port_var.set(safe_port(current_port))
        elif ports:
            self.port_var.set(ports[0])
        if ports:
            self._log(f"已刷新串口：{', '.join(ports)}")
        else:
            self._log("未发现串口")

    def _current_backend(self) -> Any:
        if self.backend is None:
            raise RuntimeError("尚未连接")
        return self.backend

    def _make_backend(self, *, mode: str, baudrate: int, dtr: bool, rts: bool) -> Any:
        if self.simulate_requested or mode == "模拟":
            return DummyArm()
        if mode == "控制库":
            if BookArm is None:
                raise RuntimeError(f"控制库不可用：{IMPORT_ERROR}")
            return RealArm()
        raise RuntimeError(f"未知后端：{mode}")

    def _log(self, message: str) -> None:
        stamp = dt.datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {message}\n"
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _post(self, kind: str, payload: Any = None) -> None:
        self.events.put((kind, payload))

    def _poll_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                self._handle_event(kind, payload)
        except queue.Empty:
            pass
        self.root.after(80, self._poll_events)

    def _handle_event(self, kind: str, payload: Any) -> None:
        if kind == "log":
            self._log(str(payload))
        elif kind == "status":
            self.status_var.set(str(payload))
        elif kind == "status_mode":
            self.mode_var.set(str(payload))
        elif kind == "torque":
            self.torque_var.set(str(payload))
        elif kind == "gripper":
            closed = bool(payload)
            self.gripper_closed_var.set(closed)
            self.gripper_status_var.set("爪夹: 闭" if closed else "爪夹: 开")
        elif kind == "feedback":
            fb = payload
            q_rad = to_float_list(getattr(fb, "q_rad", None))
            torque = to_float_list(getattr(fb, "torque", None))
            raw_pos = to_int_list(getattr(fb, "raw_pos", None))
            online = to_bool_list(getattr(fb, "online", None))
            updated_at = dt.datetime.now().strftime("%H:%M:%S")
            self.feedback_var.set(f"q={display_joint_values(q_rad)}")
            self.joint_feedback_var.set(display_joint_values(q_rad))
            self.raw_feedback_var.set(display_int_values(raw_pos))
            self.torque_feedback_var.set(display_joint_values(torque))
            online_text = "-" if online is None else ", ".join("1" if item else "0" for item in online)
            self.online_feedback_var.set(online_text)
            self.update_feedback_var.set(updated_at)
            self.last_update_var.set(f"更新时间: {updated_at}")
            self.online_var.set(f"在线: {online_text}")
        elif kind == "realtime":
            self.realtime_var.set(str(payload))
        elif kind == "refresh_points":
            self._refresh_tree()
        elif kind == "busy":
            self._set_buttons_busy(bool(payload))
        elif kind == "select_point":
            self._select_tree_item(int(payload))
        elif kind == "error":
            title, message = payload
            messagebox.showerror(title, message)
            self._log(f"{title}: {message}")
        elif kind == "info":
            title, message = payload
            messagebox.showinfo(title, message)
        elif kind == "warn":
            title, message = payload
            messagebox.showwarning(title, message)
        elif kind == "port_refresh":
            self._refresh_ports()
        elif kind == "start_realtime":
            interval, timeout = payload
            self._start_realtime_worker(interval=interval, timeout=timeout)
        elif kind == "apply_scale":
            self._apply_ui_scale(float(payload))

    def _set_buttons_busy(self, busy: bool) -> None:
        self.task_busy = busy
        for widget in (
            self.port_combo,
            self.mode_combo,
            self.backend_combo,
        ):
            try:
                readonly = widget in (self.mode_combo, self.backend_combo)
                widget.configure(state="disabled" if busy else "readonly" if readonly else "normal")
            except tk.TclError:
                pass

    def _run_task(self, label: str, func: Any) -> None:
        with self.task_lock:
            if self.task_busy:
                self._log("有任务正在运行")
                return
            self.task_busy = True
        self._post("busy", True)
        self._post("log", f"{label} 开始")

        def worker() -> None:
            try:
                func()
            except Exception as exc:
                self._post("error", (label, str(exc)))
            else:
                self._post("log", f"{label} 完成")
            finally:
                self._post("busy", False)

        threading.Thread(target=worker, daemon=True).start()

    def _parse_settings(self) -> GuiSettings:
        joint_speed = safe_float(self.joint_speed_var.get(), "关节速度")
        joint_acc = safe_float(self.joint_acc_var.get(), "关节加速度")
        raw_speed = safe_float(self.raw_speed_var.get(), "原始速度")
        raw_acc = safe_float(self.raw_acc_var.get(), "原始加速度")
        settle = safe_float(self.settle_var.get(), "停顿时间")
        timeout = safe_float(self.timeout_var.get(), "超时时间")
        mode = self.playback_mode_var.get()
        if mode not in {"关节角", "原始位置"}:
            raise ValueError(f"未知回放模式：{mode}")
        return GuiSettings(
            joint_speed=joint_speed,
            joint_acceleration=joint_acc,
            raw_speed=raw_speed,
            raw_acceleration=raw_acc,
            settle_seconds=settle,
            response_timeout=timeout,
            playback_mode=mode,
        )

    def _parse_status_interval(self) -> float:
        interval = safe_float(self.status_interval_var.get(), "状态刷新间隔")
        if interval < 0.1:
            raise ValueError("状态刷新间隔不能小于 0.1 秒")
        return interval

    def _parse_baudrate(self) -> int:
        baudrate = int(safe_float(self.baudrate_var.get(), "波特率"))
        if baudrate <= 0:
            raise ValueError("波特率必须大于 0")
        return baudrate

    def _parse_ui_scale(self) -> float:
        scale = safe_float(self.ui_scale_var.get(), "界面缩放")
        if scale < 1.0 or scale > 2.4:
            raise ValueError("界面缩放建议在 1.0 到 2.4 之间")
        return scale

    def _apply_ui_scale(self, scale: float) -> None:
        self.ui_scale = max(1.0, float(scale))
        self.ui_scale_var.set(f"{self.ui_scale:.2f}")
        self._configure_ui_scale()
        self.root.update_idletasks()

    def on_apply_ui_scale(self) -> None:
        try:
            scale = self._parse_ui_scale()
        except Exception as exc:
            messagebox.showerror("应用缩放", str(exc))
            return
        self._apply_ui_scale(scale)
        self._log(f"界面缩放已调整为 {scale:.2f}")

    def _ensure_backend(self) -> Any:
        if self.backend is None:
            raise RuntimeError("尚未连接机械臂")
        return self.backend

    def _read_feedback_locked(self, backend: Any, timeout: float) -> Any:
        with self.io_lock:
            return backend.read_arm_feedback(response_timeout=timeout, validate_limits=False)

    def _set_gripper_closed_once(self, backend: Any, closed: bool, *, timeout: float) -> bool:
        if self.gripper_commanded_closed is closed:
            return False
        with self.io_lock:
            backend.set_gripper_closed(
                closed,
                wait_response=True,
                response_timeout=timeout,
            )
        self.gripper_commanded_closed = closed
        return True

    def _selected_indices(self) -> list[int]:
        selected = self.tree.selection()
        indices: list[int] = []
        for item in selected:
            try:
                indices.append(int(self.tree.item(item, "tags")[0]))
            except Exception:
                continue
        return indices

    def _selected_point(self) -> tuple[int, TeachPoint] | None:
        selected = self.tree.selection()
        if not selected:
            return None
        item = selected[0]
        try:
            index = int(self.tree.item(item, "tags")[0])
        except Exception:
            return None
        if index < 0 or index >= len(self.points):
            return None
        return index, self.points[index]

    def _select_tree_item(self, index: int) -> None:
        children = self.tree.get_children()
        if 0 <= index < len(children):
            self.tree.selection_set(children[index])
            self.tree.see(children[index])

    def _refresh_tree(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        for index, point in enumerate(self.points):
            values = (
                point.name,
                self._gripper_text(point.gripper_closed),
                f"{point.q_rad[0]:.4f}" if len(point.q_rad) > 0 else "-",
                f"{point.q_rad[1]:.4f}" if len(point.q_rad) > 1 else "-",
                f"{point.q_rad[2]:.4f}" if len(point.q_rad) > 2 else "-",
                f"{point.q_rad[3]:.4f}" if len(point.q_rad) > 3 else "-",
                f"{point.q_rad[4]:.4f}" if len(point.q_rad) > 4 else "-",
                point.created_at,
            )
            self.tree.insert("", "end", values=values, tags=(str(index),))
        self.point_name_var.set(f"P{len(self.points) + 1:02d}")

    def _gripper_text(self, gripper_closed: bool | None) -> str:
        if gripper_closed is None:
            return "-"
        return "闭" if gripper_closed else "开"

    def on_refresh_ports(self) -> None:
        self._refresh_ports()

    def on_connect(self) -> None:
        try:
            port = safe_port(self.port_var.get())
            mode = self.backend_mode_var.get()
            baudrate = self._parse_baudrate()
            dtr = self.dtr_var.get()
            rts = self.rts_var.get()
            interval = self._parse_status_interval()
            timeout = self._parse_settings().response_timeout
        except Exception as exc:
            messagebox.showerror("连接", str(exc))
            return

        def task() -> None:
            backend = self._make_backend(mode=mode, baudrate=baudrate, dtr=dtr, rts=rts)
            with self.io_lock:
                backend.connect_serial(port=port)
            self.backend = backend
            if isinstance(backend, DummyArm):
                label = "模拟"
            else:
                label = "控制库"
            self._post("status", f"已连接 {port} / {label}")
            self._post("status_mode", label)
            self._post("torque", "扭矩: 未知")
            self._post("log", f"连接成功：{port}")
            if isinstance(backend, RealArm):
                self._post("log", "连接库后端成功")
            self._post("start_realtime", (interval, timeout))
            try:
                fb = self._read_feedback_locked(backend, max(1.0, timeout))
                self._post("feedback", fb)
            except Exception:
                pass

        self._run_task("连接", task)

    def on_probe_connection(self) -> None:
        try:
            settings = self._parse_settings()
        except Exception as exc:
            messagebox.showerror("通讯自检", str(exc))
            return

        def task() -> None:
            backend = self._ensure_backend()
            fb = self._read_feedback_locked(backend, settings.response_timeout)
            self._post("feedback", fb)
            self._post("log", "通讯自检通过：读取反馈成功")

        self._run_task("通讯自检", task)

    def on_disconnect(self) -> None:
        self._stop_realtime_worker()

        def task() -> None:
            backend = self.backend
            if backend is not None:
                try:
                    with self.io_lock:
                        backend.close()
                finally:
                    self.backend = None
                    self.gripper_commanded_closed = None
            self._post("status", "未连接")
            self._post("torque", "扭矩: -")
            self._post("log", "已断开")

        self._run_task("断开", task)

    def on_enable_torque(self) -> None:
        try:
            settings = self._parse_settings()
        except Exception as exc:
            messagebox.showerror("锁定扭矩", str(exc))
            return

        def task() -> None:
            backend = self._ensure_backend()
            with self.io_lock:
                backend.enable_torque(wait_response=True, response_timeout=settings.response_timeout)
            self._post("torque", "扭矩: 锁定")

        self._run_task("锁定扭矩", task)

    def on_disable_torque(self) -> None:
        try:
            settings = self._parse_settings()
        except Exception as exc:
            messagebox.showerror("释放扭矩", str(exc))
            return

        def task() -> None:
            backend = self._ensure_backend()
            with self.io_lock:
                backend.disable_torque(wait_response=True, response_timeout=settings.response_timeout)
            self._post("torque", "扭矩: 释放")

        self._run_task("释放扭矩", task)

    def on_gripper_toggle(self) -> None:
        closed = self.gripper_closed_var.get()
        try:
            settings = self._parse_settings()
        except Exception as exc:
            messagebox.showerror("爪夹控制", str(exc))
            self.gripper_closed_var.set(not closed)
            return

        def task() -> None:
            backend = self._ensure_backend()
            sent = self._set_gripper_closed_once(
                backend,
                closed,
                timeout=settings.response_timeout,
            )
            self._post("gripper", closed)
            if sent:
                self._post("log", "爪夹已进入持续闭合" if closed else "爪夹已打开")
            else:
                self._post("log", "爪夹已保持闭合状态，未重复发送指令" if closed else "爪夹已保持打开状态，未重复发送指令")

        self._run_task("爪夹控制", task)

    def on_read_feedback(self) -> None:
        try:
            settings = self._parse_settings()
        except Exception as exc:
            messagebox.showerror("读取反馈", str(exc))
            return

        def task() -> None:
            backend = self._ensure_backend()
            fb = self._read_feedback_locked(backend, settings.response_timeout)
            self._post("feedback", fb)
            q_rad = to_float_list(getattr(fb, "q_rad", None))
            torque = to_float_list(getattr(fb, "torque", None))
            raw_pos = to_int_list(getattr(fb, "raw_pos", None))
            self._post("log", f"读取反馈：q={display_joint_values(q_rad)} raw={display_int_values(raw_pos)}")
            if torque is not None:
                self._post("torque", f"扭矩: {display_joint_values(torque)}")

        self._run_task("读取反馈", task)

    def on_record_point(self) -> None:
        try:
            settings = self._parse_settings()
        except Exception as exc:
            messagebox.showerror("记录点位", str(exc))
            return
        name = self.point_name_var.get().strip() or f"P{len(self.points) + 1:02d}"
        gripper_closed = self.gripper_closed_var.get()

        def task() -> None:
            backend = self._ensure_backend()
            fb = self._read_feedback_locked(backend, settings.response_timeout)
            q_rad = to_float_list(getattr(fb, "q_rad", None))
            torque = to_float_list(getattr(fb, "torque", None))
            raw_pos = to_int_list(getattr(fb, "raw_pos", None))
            online = to_bool_list(getattr(fb, "online", None))
            if q_rad is None or len(q_rad) != 5:
                raise ValueError("机械臂反馈中没有 5 个关节角")
            point = TeachPoint(
                name=name,
                q_rad=q_rad,
                created_at=now_iso(),
                gripper_closed=gripper_closed,
                torque=torque,
                raw_pos=raw_pos,
                online=online,
            )
            self.points.append(point)
            self._post("refresh_points", None)
            self._post("feedback", fb)
            self._post("log", f"已记录点位：{name}，爪夹={self._gripper_text(gripper_closed)}")

        self._run_task("记录点位", task)

    def _play_points(self, points: list[TeachPoint], settings: GuiSettings) -> None:
        backend = self._ensure_backend()
        if not points:
            raise ValueError("没有可回放的点位")

        for seq, point in enumerate(points, start=1):
            if self.stop_event.is_set():
                self._post("log", "回放已停止")
                return
            self._post("log", f"回放 {seq}/{len(points)}：{point.name}")
            if settings.playback_mode == "关节角":
                with self.io_lock:
                    backend.move_joints_rad(
                        point.q_rad,
                        speed=settings.joint_speed,
                        acceleration=settings.joint_acceleration,
                        wait_response=True,
                        response_timeout=settings.response_timeout,
                    )
            else:
                if point.raw_pos is None:
                    raise ValueError(f"点位 {point.name} 没有 raw_pos，无法按原始位置回放")
                with self.io_lock:
                    backend.move_raw_positions(
                        point.raw_pos,
                        speed=settings.raw_speed,
                        acceleration=settings.raw_acceleration,
                        wait_response=True,
                        response_timeout=settings.response_timeout,
                    )
            if point.gripper_closed is not None:
                sent = self._set_gripper_closed_once(
                    backend,
                    point.gripper_closed,
                    timeout=settings.response_timeout,
                )
                self._post("gripper", point.gripper_closed)
                if sent:
                    self._post("log", f"点位 {point.name} 爪夹={self._gripper_text(point.gripper_closed)}")
                else:
                    self._post("log", f"点位 {point.name} 爪夹状态未变化，未重复发送指令")
            end = time.monotonic() + max(0.0, settings.settle_seconds)
            while time.monotonic() < end:
                if self.stop_event.is_set():
                    self._post("log", "回放已停止")
                    return
                time.sleep(0.03)

        self._post("log", "回放完成")

    def on_play_selected(self) -> None:
        selected = self._selected_indices()
        if not selected:
            messagebox.showwarning("回放选中", "先选择一个点位")
            return
        try:
            settings = self._parse_settings()
        except Exception as exc:
            messagebox.showerror("回放选中", str(exc))
            return
        points = [self.points[index] for index in selected if 0 <= index < len(self.points)]

        def task() -> None:
            self.stop_event.clear()
            self._play_points(points, settings)

        self._run_task("回放选中", task)

    def on_play_all(self) -> None:
        if not self.points:
            messagebox.showwarning("回放全部", "没有点位")
            return
        try:
            settings = self._parse_settings()
        except Exception as exc:
            messagebox.showerror("回放全部", str(exc))
            return
        points = list(self.points)

        def task() -> None:
            self.stop_event.clear()
            self._play_points(points, settings)

        self._run_task("回放全部", task)

    def on_start_realtime(self) -> None:
        try:
            interval = self._parse_status_interval()
            timeout = self._parse_settings().response_timeout
            self._ensure_backend()
        except Exception as exc:
            messagebox.showerror("开始实时", str(exc))
            return
        self._start_realtime_worker(interval=interval, timeout=timeout)

    def on_stop_realtime(self) -> None:
        self._stop_realtime_worker()
        self._log("实时状态已停止")

    def _start_realtime_worker(self, *, interval: float, timeout: float) -> None:
        self._stop_realtime_worker()
        self.status_stop_event.clear()
        self._post("realtime", f"实时: {interval:.2f}s")

        def worker() -> None:
            self._post("log", "实时状态已启动")
            while not self.status_stop_event.is_set():
                backend = self.backend
                if backend is None:
                    break
                try:
                    fb = self._read_feedback_locked(backend, timeout)
                except Exception as exc:
                    self._post("log", f"实时状态读取失败：{exc}")
                    self.status_stop_event.wait(max(interval, 1.0))
                    continue
                self._post("feedback", fb)
                self.status_stop_event.wait(interval)
            self._post("realtime", "实时: 停止")

        self.status_thread = threading.Thread(target=worker, daemon=True)
        self.status_thread.start()

    def _stop_realtime_worker(self) -> None:
        self.status_stop_event.set()
        thread = self.status_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.8)
        self.status_thread = None
        self.realtime_var.set("实时: 停止")

    def on_stop_playback(self) -> None:
        self.stop_event.set()
        self._log("已请求停止回放")

    def on_delete_selected(self) -> None:
        selected = self._selected_indices()
        if not selected:
            messagebox.showwarning("删除选中", "先选择一个点位")
            return
        for index in sorted(set(selected), reverse=True):
            if 0 <= index < len(self.points):
                del self.points[index]
        self._refresh_tree()
        self._log("已删除选中点位")

    def on_clear_points(self) -> None:
        if not self.points:
            return
        if not messagebox.askyesno("清空", "清空所有点位？"):
            return
        self.points.clear()
        self._refresh_tree()
        self._log("已清空点位")

    def on_move_up(self) -> None:
        selected = self._selected_indices()
        if len(selected) != 1:
            messagebox.showwarning("上移", "先选择一个点位")
            return
        index = selected[0]
        if index <= 0:
            return
        self.points[index - 1], self.points[index] = self.points[index], self.points[index - 1]
        self._refresh_tree()
        self._select_tree_item(index - 1)

    def on_move_down(self) -> None:
        selected = self._selected_indices()
        if len(selected) != 1:
            messagebox.showwarning("下移", "先选择一个点位")
            return
        index = selected[0]
        if index >= len(self.points) - 1:
            return
        self.points[index + 1], self.points[index] = self.points[index], self.points[index + 1]
        self._refresh_tree()
        self._select_tree_item(index + 1)

    def on_export_json(self) -> None:
        if not self.points:
            messagebox.showwarning("导出JSON", "没有点位")
            return
        default_name = f"bookarm_points_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = filedialog.asksaveasfilename(
            title="导出 JSON",
            defaultextension=".json",
            initialfile=default_name,
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        try:
            settings = self._parse_settings()
            status_interval = self._parse_status_interval()
        except Exception as exc:
            messagebox.showerror("导出JSON", str(exc))
            return
        backend = self.backend
        payload = {
            "schema": "bookarm.manual_teach.v1",
            "exported_at": now_iso(),
            "backend": "simulation" if isinstance(backend, DummyArm) else "real",
            "port": self.port_var.get().strip(),
            "playback_mode": settings.playback_mode,
            "settings": {
                "joint_speed": settings.joint_speed,
                "joint_acceleration": settings.joint_acceleration,
                "raw_speed": settings.raw_speed,
                "raw_acceleration": settings.raw_acceleration,
                "settle_seconds": settings.settle_seconds,
                "response_timeout": settings.response_timeout,
                "status_interval": status_interval,
            },
            "joint_names": list(getattr(backend, "joint_names", ("base", "shoulder", "elbow", "wrist"))),
            "points": [point.to_dict() for point in self.points],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        self._log(f"已导出：{path}")

    def on_import_json(self) -> None:
        path = filedialog.askopenfilename(
            title="导入 JSON",
            filetypes=[("JSON", "*.json"), ("All Files", "*")],
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            if isinstance(payload, dict) and "points" in payload:
                raw_points = payload["points"]
                settings = payload.get("settings", {})
                if isinstance(settings, dict):
                    if "joint_speed" in settings:
                        self.joint_speed_var.set(str(settings["joint_speed"]))
                    if "joint_acceleration" in settings:
                        self.joint_acc_var.set(str(settings["joint_acceleration"]))
                    if "raw_speed" in settings:
                        self.raw_speed_var.set(str(settings["raw_speed"]))
                    if "raw_acceleration" in settings:
                        self.raw_acc_var.set(str(settings["raw_acceleration"]))
                    if "settle_seconds" in settings:
                        self.settle_var.set(str(settings["settle_seconds"]))
                    if "response_timeout" in settings:
                        self.timeout_var.set(str(settings["response_timeout"]))
                    if "status_interval" in settings:
                        self.status_interval_var.set(str(settings["status_interval"]))
                if "playback_mode" in payload:
                    self.playback_mode_var.set(str(payload["playback_mode"]))
            elif isinstance(payload, list):
                raw_points = payload
            else:
                raise ValueError("JSON 格式不对，找不到 points")
            self.points = [TeachPoint.from_dict(item, i) for i, item in enumerate(raw_points)]
            self._refresh_tree()
            self._log(f"已导入：{path}")
        except Exception as exc:
            messagebox.showerror("导入JSON", str(exc))

    def on_close(self) -> None:
        self.stop_event.set()
        self._stop_realtime_worker()
        try:
            if self.backend is not None:
                with self.io_lock:
                    self.backend.close()
        except Exception:
            pass
        self.root.destroy()


def main() -> int:
    parser = argparse.ArgumentParser(description="BookArm manual teaching GUI")
    parser.add_argument("--simulate", action="store_true", help="force simulation mode")
    parser.add_argument(
        "--ui-scale",
        type=float,
        default=DEFAULT_UI_SCALE,
        help=f"Tk UI scaling factor, default {DEFAULT_UI_SCALE}",
    )
    args = parser.parse_args()

    root = tk.Tk()
    app = ArmTeachGUI(root, simulate=args.simulate, ui_scale=args.ui_scale)
    if app.simulate_requested and BookArm is not None:
        app._log("已强制进入模拟模式")
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
