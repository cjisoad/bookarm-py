"""Small math helpers shared by BookArm scripts and examples."""

from __future__ import annotations

import math

import numpy as np


def rpy_to_matrix(rpy_rad: np.ndarray) -> np.ndarray:
    """Convert roll, pitch, yaw radians to a rotation matrix.

    The convention is ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``.
    """

    roll, pitch, yaw = np.asarray(rpy_rad, dtype=float).reshape(3)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rotation_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cr, -sr],
            [0.0, sr, cr],
        ],
        dtype=float,
    )
    rotation_y = np.array(
        [
            [cp, 0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ],
        dtype=float,
    )
    rotation_z = np.array(
        [
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return rotation_z @ rotation_y @ rotation_x


def matrix_to_rpy(rotation: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to roll, pitch, yaw radians.

    This is the inverse convention of :func:`rpy_to_matrix`, using
    ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``.
    """

    matrix = np.asarray(rotation, dtype=float).reshape(3, 3)
    pitch = math.atan2(
        -float(matrix[2, 0]),
        math.hypot(float(matrix[0, 0]), float(matrix[1, 0])),
    )
    roll = math.atan2(float(matrix[2, 1]), float(matrix[2, 2]))
    yaw = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
    return np.array([roll, pitch, yaw], dtype=float)


__all__ = ["matrix_to_rpy", "rpy_to_matrix"]
