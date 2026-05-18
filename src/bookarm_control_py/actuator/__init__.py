"""底层执行器指令转换模块。"""

from __future__ import annotations

__all__ = ["ArmActuator", "GripperActuator", "LiftPlatformActuator"]


def __getattr__(name: str):
    if name == "ArmActuator":
        from bookarm_control_py.actuator.arm import ArmActuator

        return ArmActuator
    if name == "GripperActuator":
        from bookarm_control_py.actuator.gripper import GripperActuator

        return GripperActuator
    if name == "LiftPlatformActuator":
        from bookarm_control_py.actuator.lift_platform import LiftPlatformActuator

        return LiftPlatformActuator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
