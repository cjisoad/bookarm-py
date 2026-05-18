"""Shared data types for grasp task workflows."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Workspace:
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


@dataclass(frozen=True)
class XRangeAction:
    start_cm: int
    end_cm: int
    handler: Callable[["ActionContext"], None]

    @property
    def label(self) -> str:
        return f"{self.start_cm}-{self.end_cm} cm"


@dataclass(frozen=True)
class ActionContext:
    robot: object
    args: argparse.Namespace
    selected: SelectedTarget
    x_cm: float
    action: XRangeAction
