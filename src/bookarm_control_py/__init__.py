"""Book arm robot control package."""

from bookarm_control_py.bookarm import (
    ArmFeedback,
    BestEffortIKResult,
    BestEffortIKSolver,
    BookArm,
    IKResult,
)
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
