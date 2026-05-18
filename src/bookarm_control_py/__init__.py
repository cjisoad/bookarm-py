"""Book arm robot control package."""

from __future__ import annotations

from bookarm_control_py.math_utils import matrix_to_rpy, rpy_to_matrix

__version__ = "0.1.0"

__all__ = [
    "ArmFeedback",
    "BestEffortIKResult",
    "BestEffortIKSolver",
    "BookArm",
    "IKResult",
    "matrix_to_rpy",
    "rpy_to_matrix",
    "__version__",
]


def __getattr__(name: str):
    if name in {
        "ArmFeedback",
        "BestEffortIKResult",
        "BestEffortIKSolver",
        "BookArm",
        "IKResult",
    }:
        from bookarm_control_py import bookarm

        return getattr(bookarm, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
