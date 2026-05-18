"""Workflow helpers for x-range book grasping."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

from bookarm_control_py.task.grasp.types import ActionContext, SelectedTarget, XRangeAction
from bookarm_control_py.task.grasp.utils import array_to_list, ask_yes_no


def save_selected_target(
    selected: SelectedTarget,
    action: XRangeAction,
    x_cm: float,
    path: Path,
) -> None:
    payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "picked_index": selected.picked_index,
        "x_cm": x_cm,
        "x_range_cm": [action.start_cm, action.end_cm],
        "camera_point_m": array_to_list(selected.camera_point_m),
        "base_point_m": array_to_list(selected.base_point_m),
        "raw_target_point_m": array_to_list(selected.raw_target_point_m),
        "final_target_point_m": array_to_list(selected.final_target_point_m),
        "target_position_m": array_to_list(selected.target_position_m),
        "workspace_notes": list(selected.workspace_notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存目标点数据: {path}")


def confirm_action(args: argparse.Namespace, action: XRangeAction, x_cm: float) -> bool:
    if args.yes:
        return True
    return ask_yes_no(
        f"目标 x={x_cm:.6f} cm，将进入 {action.label} 动作入口，确认继续？"
    )


def execute_action(
    robot: object,
    args: argparse.Namespace,
    selected: SelectedTarget,
    x_cm: float,
    action: XRangeAction,
) -> None:
    context = ActionContext(
        robot=robot,
        args=args,
        selected=selected,
        x_cm=x_cm,
        action=action,
    )
    action.handler(context)
