"""High-level BookArm model.

This module owns high-level robot intent, Pinocchio-based kinematics, joint
limits, and gripper intent. Low-level ESP32 JSON construction stays in the
actuator layer.

Hardware execution is delegated to actuator objects:

    BookArm -> ArmActuator / GripperActuator -> ESP32 JSON transport
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Literal, TYPE_CHECKING

import numpy as np

try:
    import pinocchio as pin
except ImportError as exc:  # pragma: no cover - depends on local conda env
    raise ImportError(
        "BookArm kinematics requires pinocchio. Activate the bookarm-beiyu "
        "environment or install pinocchio from conda-forge."
    ) from exc

if TYPE_CHECKING:
    from bookarm_control_py.actuator.arm import ArmActuator
    from bookarm_control_py.actuator.gripper import GripperActuator


DEFAULT_URDF_PATH = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "bookarm_urdf"
    / "urdf"
    / "bookarm_urdf.urdf"
)
DEFAULT_MOVE_SPEED = 25.0
DEFAULT_MOVE_ACCELERATION = 5.0
DEFAULT_LINK6_ANGLE_RAD = float(np.deg2rad(-45.0))
LINK6_JOINT_NAME = "joint_pole"


@dataclass(frozen=True)
class IKResult:
    """Inverse kinematics result."""

    success: bool
    q: np.ndarray
    error_norm: float
    iterations: int


@dataclass(frozen=True)
class BestEffortIKResult:
    """Best-effort IK result.

    ``success`` indicates whether the requested tolerance was reached. Even when
    it is false, ``q`` is the best configuration found during the search.
    """

    success: bool
    q: np.ndarray
    error_norm: float
    iterations: int
    position_error_norm: float
    rotation_error_rad: float


@dataclass(frozen=True)
class ArmFeedback:
    """Joint feedback reported by the arm firmware."""

    raw: dict[str, Any]
    q_rad: np.ndarray
    torque: np.ndarray


class BookArm:
    """High-level BookArm robot model based on Pinocchio.

    Parameters
    ----------
    end_effector_link:
        End-effector link used by FK/IK. In this project, ``link5`` is the real
        end-effector pose.
    arm_actuator:
        Optional low-level arm actuator. BookArm only calls its public methods.
    gripper_actuator:
        Optional low-level gripper actuator. BookArm only calls its public
        methods.
    """

    def __init__(
        self,
        end_effector_link: str = "link5",
        arm_actuator: "ArmActuator | None" = None,
        gripper_actuator: "GripperActuator | None" = None,
    ) -> None:
        self.urdf_path = DEFAULT_URDF_PATH
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"URDF file does not exist: {self.urdf_path}")

        self.model = self._build_model_from_assets_urdf()
        self.data = self.model.createData()
        self.end_effector_link = end_effector_link
        self.end_effector_frame_id = self._get_frame_id(end_effector_link)

        self.joint_names = self._get_actuated_joint_names()
        self.neutral_q = self.check_joint_angles(
            pin.neutral(self.model),
            context="Pinocchio neutral configuration",
        )
        self.arm_actuator = arm_actuator
        self.gripper_actuator = gripper_actuator

    @property
    def nq(self) -> int:
        return self.model.nq

    @property
    def nv(self) -> int:
        return self.model.nv

    @property
    def arm(self) -> "ArmActuator":
        if self.arm_actuator is None:
            raise RuntimeError("No arm actuator is connected.")
        return self.arm_actuator

    @property
    def gripper(self) -> "GripperActuator":
        if self.gripper_actuator is None:
            raise RuntimeError("No gripper actuator is connected.")
        return self.gripper_actuator

    def connect_serial_arm(
        self,
        *,
        port: str,
    ) -> "BookArm":
        """Connect the arm actuator through its serial interface."""

        from bookarm_control_py.actuator.arm import ArmActuator

        kwargs: dict[str, Any] = {
            "joint_count": self.nq,
            "port": port,
        }

        self.arm_actuator = ArmActuator(**kwargs).open()
        return self

    def connect_serial_gripper(
        self,
        *,
        port: str,
    ) -> "BookArm":
        """Connect the gripper actuator through its serial interface."""

        from bookarm_control_py.actuator.gripper import GripperActuator

        self.gripper_actuator = GripperActuator(port=port).open()
        return self

    def connect_serial(
        self,
        *,
        port: str,
    ) -> "BookArm":
        """Connect arm and gripper through one shared serial transport."""

        from bookarm_control_py.actuator.arm import ArmActuator
        from bookarm_control_py.actuator.gripper import GripperActuator
        from bookarm_control_py.protocol.esp32 import JsonSerialTransport

        transport = JsonSerialTransport(port=port).open()
        self.arm_actuator = ArmActuator(joint_count=self.nq, transport=transport)
        self.gripper_actuator = GripperActuator(transport=transport)
        return self

    def close(self) -> None:
        """Close connected hardware interfaces."""

        closed: set[int] = set()
        for actuator in (self.arm_actuator, self.gripper_actuator):
            if actuator is None or id(actuator) in closed:
                continue
            close = getattr(actuator, "close", None)
            if callable(close):
                close()
            closed.add(id(actuator))

    def check_joint_angles(
        self,
        q: Iterable[float],
        tolerance: float = 1e-9,
        *,
        context: str = "Joint angle",
    ) -> np.ndarray:
        """Validate joint angles against URDF limits.

        Unlike clipping, this method never changes the user's command. If any
        joint is outside its allowed range, it raises ValueError.
        """

        q_array = self._as_configuration(q)
        lower = self.model.lowerPositionLimit
        upper = self.model.upperPositionLimit
        invalid = np.where((q_array < lower - tolerance) | (q_array > upper + tolerance))[0]

        if invalid.size:
            details = []
            for index in invalid:
                details.append(
                    f"{self.joint_names[index]}={q_array[index]:.6f} "
                    f"not in [{lower[index]:.6f}, {upper[index]:.6f}]"
                )
            raise ValueError(f"{context} out of range: " + "; ".join(details))

        return q_array

    def fkine(
        self,
        q: Iterable[float],
        end_effector_link: str | None = None,
    ) -> pin.SE3:
        """Compute end-effector pose from joint angles in radians."""

        q_array = self.check_joint_angles(q, context="FK configuration")
        frame_id = self._get_frame_id(end_effector_link or self.end_effector_link)

        pin.forwardKinematics(self.model, self.data, q_array)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[frame_id].copy()

    def fkine_dict(
        self,
        q: Iterable[float],
        end_effector_link: str | None = None,
    ) -> dict[str, np.ndarray]:
        """Return FK as position, rotation matrix, and homogeneous transform."""

        pose = self.fkine(q, end_effector_link=end_effector_link)
        transform = np.eye(4)
        transform[:3, :3] = pose.rotation
        transform[:3, 3] = pose.translation

        return {
            "position": pose.translation.copy(),
            "rotation": pose.rotation.copy(),
            "transform": transform,
        }

    def ikine(
        self,
        target_position: Iterable[float],
        target_rotation: np.ndarray | None = None,
        q0: Iterable[float] | None = None,
        end_effector_link: str | None = None,
        max_iterations: int = 200,
        tolerance: float = 1e-4,
        damping: float = 1e-6,
        step_size: float = 0.4,
        link6_angle_rad: float = DEFAULT_LINK6_ANGLE_RAD,
    ) -> IKResult:
        """Solve numerical IK while enforcing URDF joint limits."""

        target_translation = np.asarray(target_position, dtype=float).reshape(3)
        q = self._as_configuration(q0) if q0 is not None else self.neutral_q.copy()
        q = self._with_link6_angle(
            q,
            link6_angle_rad,
            context="IK link6 angle",
        )
        q = self.check_joint_angles(q, context="IK initial configuration")
        frame_id = self._get_frame_id(end_effector_link or self.end_effector_link)

        if target_rotation is None:
            return self._ikine_position_only(
                target_translation=target_translation,
                q=q,
                frame_id=frame_id,
                max_iterations=max_iterations,
                tolerance=tolerance,
                damping=damping,
                step_size=step_size,
                link6_angle_rad=link6_angle_rad,
            )

        target_pose = pin.SE3(
            np.asarray(target_rotation, dtype=float).reshape(3, 3),
            target_translation,
        )
        return self._ikine_pose(
            target_pose=target_pose,
            q=q,
            frame_id=frame_id,
            max_iterations=max_iterations,
            tolerance=tolerance,
            damping=damping,
            step_size=step_size,
            link6_angle_rad=link6_angle_rad,
        )

    def ikine_best_effort(
        self,
        target_position: Iterable[float],
        target_rotation: np.ndarray,
        q0: Iterable[float] | None = None,
        end_effector_link: str | None = None,
        max_iterations: int = 500,
        tolerance: float = 1e-4,
        damping: float = 1e-6,
        step_size: float = 0.4,
        print_error: bool = True,
        link6_angle_rad: float = DEFAULT_LINK6_ANGLE_RAD,
    ) -> BestEffortIKResult:
        """Solve pose IK and return the closest configuration if exact IK fails.

        The initial configuration is evaluated first. Each accepted update is
        then evaluated, so the final update is never skipped when the iteration
        budget is exhausted.
        """

        if max_iterations < 0:
            raise ValueError("max_iterations must be greater than or equal to 0")
        if tolerance <= 0.0:
            raise ValueError("tolerance must be greater than 0")
        if damping <= 0.0:
            raise ValueError("damping must be greater than 0")
        if step_size <= 0.0:
            raise ValueError("step_size must be greater than 0")

        target_translation = np.asarray(target_position, dtype=float).reshape(3)
        target_pose = pin.SE3(
            np.asarray(target_rotation, dtype=float).reshape(3, 3),
            target_translation,
        )
        q = self._as_configuration(q0) if q0 is not None else self.neutral_q.copy()
        q = self._with_link6_angle(
            q,
            link6_angle_rad,
            context="Best-effort IK link6 angle",
        )
        q = self.check_joint_angles(q, context="Best-effort IK initial configuration")
        frame_id = self._get_frame_id(end_effector_link or self.end_effector_link)

        best_result: BestEffortIKResult | None = None

        for iteration in range(max_iterations + 1):
            q = self._clip_configuration_to_limits(q)
            q = self._with_link6_angle(
                q,
                link6_angle_rad,
                context=f"Best-effort IK iteration {iteration} link6 angle",
            )
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)

            current_pose = self.data.oMf[frame_id]
            frame_error = current_pose.actInv(target_pose)
            error = pin.log(frame_error).vector
            error_norm = float(np.linalg.norm(error))
            position_error_norm = float(np.linalg.norm(target_translation - current_pose.translation))
            rotation_error_rad = float(np.linalg.norm(pin.log3(current_pose.rotation.T @ target_pose.rotation)))

            result = BestEffortIKResult(
                success=error_norm < tolerance,
                q=q.copy(),
                error_norm=error_norm,
                iterations=iteration,
                position_error_norm=position_error_norm,
                rotation_error_rad=rotation_error_rad,
            )
            if best_result is None or result.error_norm < best_result.error_norm:
                best_result = result

            if result.success:
                if print_error:
                    self._print_best_effort_ik_error(result)
                return result

            if iteration == max_iterations:
                break

            jacobian = pin.computeFrameJacobian(
                self.model,
                self.data,
                q,
                frame_id,
                pin.ReferenceFrame.LOCAL,
            )
            jacobian = -pin.Jlog6(frame_error.inverse()) @ jacobian
            velocity = -self._damped_least_squares(jacobian, error, damping)
            next_q = self._clip_configuration_to_limits(
                pin.integrate(self.model, q, step_size * velocity)
            )
            next_q = self._with_link6_angle(
                next_q,
                link6_angle_rad,
                context=f"Best-effort IK iteration {iteration} proposed link6 angle",
            )
            if np.linalg.norm(next_q - q) < 1e-12:
                break
            q = next_q

        if best_result is None:  # pragma: no cover - guarded by max_iterations validation.
            raise RuntimeError("Best-effort IK did not evaluate any configuration")
        result = BestEffortIKResult(
            success=False,
            q=best_result.q,
            error_norm=best_result.error_norm,
            iterations=best_result.iterations,
            position_error_norm=best_result.position_error_norm,
            rotation_error_rad=best_result.rotation_error_rad,
        )
        if print_error:
            self._print_best_effort_ik_error(result)
        return result

    def ikine_link6_best_effort(
        self,
        target_position: Iterable[float] | None = None,
        target_rotation: np.ndarray | None = None,
        q0: Iterable[float] | None = None,
        max_iterations: int = 500,
        tolerance: float = 1e-4,
        damping: float = 1e-6,
        step_size: float = 0.4,
        print_error: bool = True,
        consider_rotation: bool = True,
        preferred_rotation: np.ndarray | None = None,
        orientation_weight: float = 0.15,
        position_priority_tolerance: float | None = None,
        target_x: float | None = None,
        target_y: float | None = None,
        target_z: float | None = None,
    ) -> BestEffortIKResult:
        """Solve link6 IK with optional orientation consideration."""

        if max_iterations < 0:
            raise ValueError("max_iterations must be greater than or equal to 0")
        if tolerance <= 0.0:
            raise ValueError("tolerance must be greater than 0")
        if damping <= 0.0:
            raise ValueError("damping must be greater than 0")
        if step_size <= 0.0:
            raise ValueError("step_size must be greater than 0")
        target_translation = self._resolve_target_translation(
            target_position=target_position,
            target_x=target_x,
            target_y=target_y,
            target_z=target_z,
        )
        if consider_rotation and target_rotation is None:
            raise ValueError("target_rotation is required when consider_rotation is True")
        if not consider_rotation and preferred_rotation is None:
            preferred_rotation = pin.rpy.rpyToMatrix(0.0, np.deg2rad(45.0), 0.0)

        if consider_rotation:
            target_pose = pin.SE3(
                np.asarray(target_rotation, dtype=float).reshape(3, 3),
                target_translation,
            )
            q = self._as_configuration(q0) if q0 is not None else self.neutral_q.copy()
            q = self.check_joint_angles(q, context="Link6 best-effort IK initial configuration")
            frame_id = self._get_frame_id("link6")

            best_result: BestEffortIKResult | None = None

            for iteration in range(max_iterations + 1):
                q = self._clip_configuration_to_limits(q)
                pin.forwardKinematics(self.model, self.data, q)
                pin.updateFramePlacements(self.model, self.data)

                current_pose = self.data.oMf[frame_id]
                frame_error = current_pose.actInv(target_pose)
                error = pin.log(frame_error).vector
                error_norm = float(np.linalg.norm(error))
                position_error_norm = float(np.linalg.norm(target_translation - current_pose.translation))
                rotation_error_rad = float(np.linalg.norm(pin.log3(current_pose.rotation.T @ target_pose.rotation)))

                result = BestEffortIKResult(
                    success=error_norm < tolerance,
                    q=q.copy(),
                    error_norm=error_norm,
                    iterations=iteration,
                    position_error_norm=position_error_norm,
                    rotation_error_rad=rotation_error_rad,
                )
                if best_result is None or result.error_norm < best_result.error_norm:
                    best_result = result

                if result.success:
                    if print_error:
                        self._print_best_effort_ik_error(result)
                    return result

                if iteration == max_iterations:
                    break

                jacobian = pin.computeFrameJacobian(
                    self.model,
                    self.data,
                    q,
                    frame_id,
                    pin.ReferenceFrame.LOCAL,
                )
                jacobian = -pin.Jlog6(frame_error.inverse()) @ jacobian
                velocity = -self._damped_least_squares(jacobian, error, damping)
                next_q = self._clip_configuration_to_limits(
                    pin.integrate(self.model, q, step_size * velocity)
                )
                next_q = self.check_joint_angles(
                    next_q,
                    context=f"Link6 best-effort IK iteration {iteration} proposed configuration",
                )
                if np.linalg.norm(next_q - q) < 1e-12:
                    break
                q = next_q

            if best_result is None:  # pragma: no cover - guarded by max_iterations validation.
                raise RuntimeError("Link6 best-effort IK did not evaluate any configuration")
            result = BestEffortIKResult(
                success=False,
                q=best_result.q,
                error_norm=best_result.error_norm,
                iterations=best_result.iterations,
                position_error_norm=best_result.position_error_norm,
                rotation_error_rad=best_result.rotation_error_rad,
            )
            if print_error:
                self._print_best_effort_ik_error(result)
            return result

        position_priority_tolerance = tolerance if position_priority_tolerance is None else position_priority_tolerance
        if position_priority_tolerance <= 0.0:
            raise ValueError("position_priority_tolerance must be greater than 0")
        if orientation_weight < 0.0:
            raise ValueError("orientation_weight must be greater than or equal to 0")
        preferred_rotation_matrix = np.asarray(preferred_rotation, dtype=float).reshape(3, 3)
        q = self._as_configuration(q0) if q0 is not None else self.neutral_q.copy()
        q = self.check_joint_angles(
            q,
            context="Link6 position-preferred IK initial configuration",
        )
        frame_id = self._get_frame_id("link6")

        best_result: BestEffortIKResult | None = None

        def is_better(
            candidate: BestEffortIKResult,
            current_best: BestEffortIKResult | None,
        ) -> bool:
            if current_best is None:
                return True
            candidate_position_ok = candidate.position_error_norm <= position_priority_tolerance
            current_position_ok = current_best.position_error_norm <= position_priority_tolerance
            if candidate_position_ok and current_position_ok:
                if not np.isclose(candidate.rotation_error_rad, current_best.rotation_error_rad):
                    return candidate.rotation_error_rad < current_best.rotation_error_rad
                return candidate.position_error_norm < current_best.position_error_norm
            if candidate_position_ok != current_position_ok:
                return candidate_position_ok
            if not np.isclose(candidate.position_error_norm, current_best.position_error_norm):
                return candidate.position_error_norm < current_best.position_error_norm
            return candidate.rotation_error_rad < current_best.rotation_error_rad

        for iteration in range(max_iterations + 1):
            q = self._clip_configuration_to_limits(q)
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)

            current_pose = self.data.oMf[frame_id]
            position_error = target_translation - current_pose.translation
            position_error_norm = float(np.linalg.norm(position_error))
            rotation_error_rad = float(np.linalg.norm(
                pin.log3(current_pose.rotation.T @ preferred_rotation_matrix)
            ))

            result = BestEffortIKResult(
                success=position_error_norm < tolerance,
                q=q.copy(),
                error_norm=position_error_norm,
                iterations=iteration,
                position_error_norm=position_error_norm,
                rotation_error_rad=rotation_error_rad,
            )
            if is_better(result, best_result):
                best_result = result

            if iteration == max_iterations:
                break

            jacobian = pin.computeFrameJacobian(
                self.model,
                self.data,
                q,
                frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            position_jacobian = jacobian[:3, :]
            velocity = self._damped_least_squares(position_jacobian, position_error, damping)

            if position_error_norm <= position_priority_tolerance and orientation_weight > 0.0:
                rotation_error_local = pin.log3(current_pose.rotation.T @ preferred_rotation_matrix)
                rotation_error_world = current_pose.rotation @ rotation_error_local
                rotation_jacobian = jacobian[3:, :]
                rotation_velocity = self._damped_least_squares(rotation_jacobian, rotation_error_world, damping)
                jj_t = position_jacobian @ position_jacobian.T
                position_pinv = position_jacobian.T @ np.linalg.solve(
                    jj_t + damping * np.eye(jj_t.shape[0]),
                    np.eye(jj_t.shape[0]),
                )
                nullspace = np.eye(self.nv) - position_pinv @ position_jacobian
                velocity = velocity + orientation_weight * (nullspace @ rotation_velocity)

            next_q = self._clip_configuration_to_limits(
                pin.integrate(self.model, q, step_size * velocity)
            )
            next_q = self.check_joint_angles(
                next_q,
                context=f"Link6 position-preferred IK iteration {iteration} proposed configuration",
            )
            if np.linalg.norm(next_q - q) < 1e-12:
                break
            q = next_q

        if best_result is None:  # pragma: no cover - guarded by max_iterations validation.
            raise RuntimeError("Link6 position-preferred IK did not evaluate any configuration")
        result = BestEffortIKResult(
            success=best_result.position_error_norm < tolerance,
            q=best_result.q,
            error_norm=best_result.position_error_norm,
            iterations=best_result.iterations,
            position_error_norm=best_result.position_error_norm,
            rotation_error_rad=best_result.rotation_error_rad,
        )
        if print_error:
            self._print_best_effort_position_ik_error(result)
            print(f"姿态偏好误差={np.rad2deg(result.rotation_error_rad):.8f} deg")
        return result

    def ikine_position_best_effort(
        self,
        target_position: Iterable[float],
        q0: Iterable[float] | None = None,
        end_effector_link: str | None = None,
        max_iterations: int = 500,
        tolerance: float = 1e-4,
        damping: float = 1e-6,
        step_size: float = 0.4,
        print_error: bool = True,
        link6_angle_rad: float = DEFAULT_LINK6_ANGLE_RAD,
    ) -> BestEffortIKResult:
        """Solve position-only IK and return the closest configuration found.

        ``success`` indicates whether the target position was reached within
        ``tolerance``. Even when it is false, ``q`` is the configuration with
        the smallest position error seen during the search.
        """

        if max_iterations < 0:
            raise ValueError("max_iterations must be greater than or equal to 0")
        if tolerance <= 0.0:
            raise ValueError("tolerance must be greater than 0")
        if damping <= 0.0:
            raise ValueError("damping must be greater than 0")
        if step_size <= 0.0:
            raise ValueError("step_size must be greater than 0")

        target_translation = np.asarray(target_position, dtype=float).reshape(3)
        q = self._as_configuration(q0) if q0 is not None else self.neutral_q.copy()
        q = self._with_link6_angle(
            q,
            link6_angle_rad,
            context="Best-effort position IK link6 angle",
        )
        q = self.check_joint_angles(
            q,
            context="Best-effort position IK initial configuration",
        )
        frame_id = self._get_frame_id(end_effector_link or self.end_effector_link)

        best_result: BestEffortIKResult | None = None

        for iteration in range(max_iterations + 1):
            q = self._clip_configuration_to_limits(q)
            q = self._with_link6_angle(
                q,
                link6_angle_rad,
                context=f"Best-effort position IK iteration {iteration} link6 angle",
            )
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)

            current_translation = self.data.oMf[frame_id].translation
            error = target_translation - current_translation
            position_error_norm = float(np.linalg.norm(error))

            result = BestEffortIKResult(
                success=position_error_norm < tolerance,
                q=q.copy(),
                error_norm=position_error_norm,
                iterations=iteration,
                position_error_norm=position_error_norm,
                rotation_error_rad=0.0,
            )
            if best_result is None or result.error_norm < best_result.error_norm:
                best_result = result

            if result.success:
                if print_error:
                    self._print_best_effort_position_ik_error(result)
                return result

            if iteration == max_iterations:
                break

            jacobian = pin.computeFrameJacobian(
                self.model,
                self.data,
                q,
                frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )[:3, :]
            velocity = self._damped_least_squares(jacobian, error, damping)
            next_q = self._clip_configuration_to_limits(
                pin.integrate(self.model, q, step_size * velocity)
            )
            next_q = self._with_link6_angle(
                next_q,
                link6_angle_rad,
                context=f"Best-effort position IK iteration {iteration} proposed link6 angle",
            )
            if np.linalg.norm(next_q - q) < 1e-12:
                break
            q = next_q

        if best_result is None:  # pragma: no cover - guarded by max_iterations validation.
            raise RuntimeError("Best-effort position IK did not evaluate any configuration")
        result = BestEffortIKResult(
            success=False,
            q=best_result.q,
            error_norm=best_result.error_norm,
            iterations=best_result.iterations,
            position_error_norm=best_result.position_error_norm,
            rotation_error_rad=0.0,
        )
        if print_error:
            self._print_best_effort_position_ik_error(result)
        return result

    def best_effort_ik_solver(
        self,
        *,
        end_effector_link: str | None = None,
        max_iterations: int = 500,
        tolerance: float = 1e-4,
        damping: float = 1e-6,
        step_size: float = 0.4,
        print_error: bool = True,
        link6_angle_rad: float = DEFAULT_LINK6_ANGLE_RAD,
    ) -> "BestEffortIKSolver":
        """Create a reusable best-effort pose IK solver for this robot."""

        return BestEffortIKSolver(
            self,
            end_effector_link=end_effector_link,
            max_iterations=max_iterations,
            tolerance=tolerance,
            damping=damping,
            step_size=step_size,
            print_error=print_error,
            link6_angle_rad=link6_angle_rad,
        )

    def move_joints_rad(
        self,
        joint_angles_rad: Iterable[float],
        *,
        speed: float = DEFAULT_MOVE_SPEED,
        acceleration: float = DEFAULT_MOVE_ACCELERATION,
        wait_response: bool = False,
        response_timeout: float | None = None,
    ) -> dict[str, Any] | None:
        """Move joints using the named-joint arm actuator command."""

        q = self.check_joint_angles(joint_angles_rad, context="Named joint command")
        return self.arm.move_joints_rad(
            q,
            speed=speed,
            acceleration=acceleration,
            wait_response=wait_response,
            response_timeout=response_timeout,
        )

    def move_zero_pose(
        self,
        *,
        speed: float = DEFAULT_MOVE_SPEED,
        acceleration: float = DEFAULT_MOVE_ACCELERATION,
        wait_response: bool = False,
        response_timeout: float | None = None,
    ) -> dict[str, Any] | None:
        """Move all arm joints to the zero configuration."""

        return self.move_joints_rad(
            np.zeros(self.nq),
            speed=speed,
            acceleration=acceleration,
            wait_response=wait_response,
            response_timeout=response_timeout,
        )

    def move_to_pose(
        self,
        target_position: Iterable[float],
        *,
        target_rotation: np.ndarray | None = None,
        q0: Iterable[float] | None = None,
        link6_angle_rad: float = DEFAULT_LINK6_ANGLE_RAD,
        speed: float = DEFAULT_MOVE_SPEED,
        acceleration: float = DEFAULT_MOVE_ACCELERATION,
        wait_response: bool = False,
        response_timeout: float | None = None,
    ) -> IKResult:
        """Solve IK for a target end-effector pose, then move the arm."""

        result = self.ikine(
            target_position=target_position,
            target_rotation=target_rotation,
            q0=q0,
            link6_angle_rad=link6_angle_rad,
        )
        if not result.success:
            raise RuntimeError(f"IK failed with error norm {result.error_norm:.6f}")

        self.check_joint_angles(result.q)
        self.arm.move_joints_rad(
            result.q,
            speed=speed,
            acceleration=acceleration,
            wait_response=wait_response,
            response_timeout=response_timeout,
        )
        return result

    def read_raw_feedback(
        self,
        *,
        response_timeout: float | None = None,
        expected_t: int | None = None,
    ) -> dict[str, Any]:
        return self.arm.read_feedback(
            response_timeout=response_timeout,
            expected_t=expected_t,
        )

    def enable_torque(
        self,
        *,
        wait_response: bool = False,
        response_timeout: float | None = None,
    ) -> dict[str, Any] | None:
        return self.arm.enable_torque(
            wait_response=wait_response,
            response_timeout=response_timeout,
        )

    def disable_torque(
        self,
        *,
        wait_response: bool = False,
        response_timeout: float | None = None,
    ) -> dict[str, Any] | None:
        return self.arm.disable_torque(
            wait_response=wait_response,
            response_timeout=response_timeout,
        )

    def read_arm_feedback(
        self,
        *,
        response_timeout: float | None = None,
        input_unit: Literal["rad", "deg"] = "rad",
        expected_t: int | None = None,
    ) -> ArmFeedback:
        """Read and parse arm joint angle and torque feedback."""

        feedback = self.read_raw_feedback(
            response_timeout=response_timeout,
            expected_t=expected_t,
        )
        q, feedback_unit = self._extract_joint_angles(feedback)
        torque = self._extract_joint_torques(feedback)
        if (feedback_unit or input_unit) == "deg":
            q = np.deg2rad(q)
        q = self.check_joint_angles(q, context="Feedback joint angle")
        return ArmFeedback(raw=feedback, q_rad=q, torque=torque)

    def zero_pose_error(
        self,
        feedback: ArmFeedback,
        *,
        expected_zero_rad: Iterable[float] | None = None,
    ) -> np.ndarray:
        expected = (
            np.zeros(self.nq)
            if expected_zero_rad is None
            else self.check_joint_angles(expected_zero_rad, context="Expected zero configuration")
        )
        return feedback.q_rad - expected

    def open_gripper(
        self,
        *,
        wait_response: bool = False,
        response_timeout: float | None = None,
    ) -> dict[str, Any] | None:
        return self.gripper.open_gripper(
            wait_response=wait_response,
            response_timeout=response_timeout,
        )

    def close_gripper(
        self,
        *,
        wait_response: bool = False,
        response_timeout: float | None = None,
    ) -> dict[str, Any] | None:
        return self.gripper.close_gripper(
            wait_response=wait_response,
            response_timeout=response_timeout,
        )

    def set_gripper_angle(
        self,
        angle_deg: float,
        *,
        wait_response: bool = False,
        response_timeout: float | None = None,
    ) -> dict[str, Any] | None:
        return self.gripper.set_angle(
            angle_deg,
            wait_response=wait_response,
            response_timeout=response_timeout,
        )

    def set_gripper_torque(
        self,
        torque: float,
        *,
        wait_response: bool = False,
        response_timeout: float | None = None,
    ) -> dict[str, Any] | None:
        return self.gripper.set_torque(
            torque,
            wait_response=wait_response,
            response_timeout=response_timeout,
        )

    def hold_gripper_closed(
        self,
        *,
        wait_response: bool = False,
        response_timeout: float | None = None,
    ) -> dict[str, Any] | None:
        return self.gripper.hold_close(
            wait_response=wait_response,
            response_timeout=response_timeout,
        )

    def read_gripper_feedback(
        self,
        *,
        response_timeout: float | None = None,
    ) -> dict[str, Any]:
        return self.gripper.read_feedback(response_timeout=response_timeout)

    def _ikine_position_only(
        self,
        target_translation: np.ndarray,
        q: np.ndarray,
        frame_id: int,
        max_iterations: int,
        tolerance: float,
        damping: float,
        step_size: float,
        link6_angle_rad: float,
    ) -> IKResult:
        last_error_norm = float("inf")

        for iteration in range(1, max_iterations + 1):
            q = self._with_link6_angle(
                q,
                link6_angle_rad,
                context=f"IK iteration {iteration} link6 angle",
            )
            q = self.check_joint_angles(
                q,
                context=f"IK iteration {iteration} configuration",
            )
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)

            current_translation = self.data.oMf[frame_id].translation
            error = target_translation - current_translation
            last_error_norm = float(np.linalg.norm(error))
            if last_error_norm < tolerance:
                return IKResult(True, q, last_error_norm, iteration)

            jacobian = pin.computeFrameJacobian(
                self.model,
                self.data,
                q,
                frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )[:3, :]
            velocity = self._damped_least_squares(jacobian, error, damping)
            q = self.check_joint_angles(
                self._with_link6_angle(
                    pin.integrate(self.model, q, step_size * velocity),
                    link6_angle_rad,
                    context=f"IK iteration {iteration} proposed link6 angle",
                ),
                context=f"IK iteration {iteration} proposed configuration",
            )

        return IKResult(False, q, last_error_norm, max_iterations)

    def _ikine_pose(
        self,
        target_pose: pin.SE3,
        q: np.ndarray,
        frame_id: int,
        max_iterations: int,
        tolerance: float,
        damping: float,
        step_size: float,
        link6_angle_rad: float,
    ) -> IKResult:
        last_error_norm = float("inf")

        for iteration in range(1, max_iterations + 1):
            q = self._with_link6_angle(
                q,
                link6_angle_rad,
                context=f"IK iteration {iteration} link6 angle",
            )
            q = self.check_joint_angles(
                q,
                context=f"IK iteration {iteration} configuration",
            )
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)

            current_pose = self.data.oMf[frame_id]
            frame_error = current_pose.actInv(target_pose)
            error = pin.log(frame_error).vector
            last_error_norm = float(np.linalg.norm(error))
            if last_error_norm < tolerance:
                return IKResult(True, q, last_error_norm, iteration)

            jacobian = pin.computeFrameJacobian(
                self.model,
                self.data,
                q,
                frame_id,
                pin.ReferenceFrame.LOCAL,
            )
            jacobian = -pin.Jlog6(frame_error.inverse()) @ jacobian
            velocity = -self._damped_least_squares(jacobian, error, damping)
            q = self.check_joint_angles(
                self._with_link6_angle(
                    pin.integrate(self.model, q, step_size * velocity),
                    link6_angle_rad,
                    context=f"IK iteration {iteration} proposed link6 angle",
                ),
                context=f"IK iteration {iteration} proposed configuration",
            )

        return IKResult(False, q, last_error_norm, max_iterations)

    @staticmethod
    def _print_best_effort_ik_error(result: BestEffortIKResult) -> None:
        print(
            "尽力逆解"
            f"{'已达到容许误差' if result.success else '未完全收敛，返回最接近构型'}："
            f"总误差={result.error_norm:.8f}，"
            f"位置误差={result.position_error_norm:.8f} m，"
            f"姿态误差={np.rad2deg(result.rotation_error_rad):.8f} deg，"
            f"最优迭代={result.iterations}"
        )

    @staticmethod
    def _print_best_effort_position_ik_error(result: BestEffortIKResult) -> None:
        print(
            "尽力位置逆解"
            f"{'已达到容许误差' if result.success else '未完全收敛，返回最接近构型'}："
            f"位置误差={result.position_error_norm:.8f} m，"
            f"最优迭代={result.iterations}"
        )

    def _get_frame_id(self, frame_name: str) -> int:
        if not self.model.existFrame(frame_name):
            available_frames = [frame.name for frame in self.model.frames]
            raise ValueError(
                f"URDF does not contain frame/link {frame_name!r}. "
                f"Available frames: {available_frames}"
            )
        return self.model.getFrameId(frame_name)

    def _get_actuated_joint_names(self) -> list[str]:
        return [
            self.model.names[index]
            for index, joint in enumerate(self.model.joints)
            if index > 0 and joint.nq > 0
        ]

    def _link6_joint_index(self) -> int:
        try:
            return self.joint_names.index(LINK6_JOINT_NAME)
        except ValueError as exc:
            raise ValueError(
                f"Cannot set link6 angle because joint {LINK6_JOINT_NAME!r} "
                f"is not in actuated joints: {self.joint_names}"
            ) from exc

    def _with_link6_angle(
        self,
        q: Iterable[float],
        link6_angle_rad: float,
        *,
        context: str,
    ) -> np.ndarray:
        q_array = self._as_configuration(q).copy()
        link6_angle = float(link6_angle_rad)
        if not np.isfinite(link6_angle):
            raise ValueError(f"{context} must be finite")

        link6_index = self._link6_joint_index()
        q_array[link6_index] = link6_angle
        return self.check_joint_angles(q_array, context=context)

    @staticmethod
    def _resolve_target_translation(
        *,
        target_position: Iterable[float] | None,
        target_x: float | None,
        target_y: float | None,
        target_z: float | None,
    ) -> np.ndarray:
        split_position = (target_x, target_y, target_z)
        split_position_provided = any(value is not None for value in split_position)
        if target_position is not None and split_position_provided:
            raise ValueError("Provide either target_position or target_x/target_y/target_z, not both")
        if target_position is None:
            if any(value is None for value in split_position):
                raise ValueError("Provide target_position, or provide all of target_x, target_y, and target_z")
            target_translation = np.asarray(split_position, dtype=float).reshape(3)
        else:
            target_translation = np.asarray(target_position, dtype=float).reshape(3)
        if not np.all(np.isfinite(target_translation)):
            raise ValueError("target position must contain only finite values")
        return target_translation

    def _as_configuration(self, q: Iterable[float]) -> np.ndarray:
        q_array = np.asarray(list(q), dtype=float)
        if q_array.shape != (self.nq,):
            raise ValueError(f"Expected {self.nq} joint values, got {q_array.shape[0]}")
        return q_array

    def _clip_configuration_to_limits(self, q: Iterable[float]) -> np.ndarray:
        q_array = self._as_configuration(q)
        return np.clip(
            q_array,
            self.model.lowerPositionLimit,
            self.model.upperPositionLimit,
        )

    def _extract_joint_angles(self, feedback: dict[str, Any]) -> tuple[np.ndarray, str | None]:
        candidates: list[tuple[Any, str | None]] = [
            (feedback.get("joints_rad"), "rad"),
            (feedback.get("angles_rad"), "rad"),
            (feedback.get("joints_deg"), "deg"),
            (feedback.get("angles_deg"), "deg"),
            (feedback.get("joints"), None),
            (feedback.get("q"), None),
        ]

        data = feedback.get("data")
        if isinstance(data, dict):
            candidates.extend(
                [
                    (data.get("joints_rad"), "rad"),
                    (data.get("angles_rad"), "rad"),
                    (data.get("joints_deg"), "deg"),
                    (data.get("angles_deg"), "deg"),
                    (data.get("joints"), None),
                    (data.get("q"), None),
                ]
            )
        elif isinstance(data, list):
            candidates.append((data, None))

        for candidate, unit in candidates:
            if candidate is None:
                continue
            try:
                angles = np.asarray(candidate, dtype=float)
            except (TypeError, ValueError):
                continue
            if angles.shape == (self.nq,):
                return angles, unit

        raise ValueError(
            "Cannot find joint angle feedback. Expected a list field such as "
            "joints_rad, joints_deg, joints, q, data.joints_rad, data.joints, or data.q. "
            f"Raw feedback: {feedback}"
        )

    def _extract_joint_torques(self, feedback: dict[str, Any]) -> np.ndarray:
        candidates: list[Any] = [
            feedback.get("joints_torque"),
            feedback.get("torques"),
            feedback.get("tau"),
        ]

        data = feedback.get("data")
        if isinstance(data, dict):
            candidates.extend(
                [
                    data.get("joints_torque"),
                    data.get("torques"),
                    data.get("tau"),
                ]
            )

        for candidate in candidates:
            if candidate is None:
                continue
            try:
                torques = np.asarray(candidate, dtype=float)
            except (TypeError, ValueError):
                continue
            if torques.shape == (self.nq,):
                return torques

        raise ValueError(
            "Cannot find joint torque feedback. Expected a list field such as "
            "joints_torque, torques, tau, data.joints_torque, data.torques, or data.tau. "
            f"Raw feedback: {feedback}"
        )

    @staticmethod
    def _damped_least_squares(
        jacobian: np.ndarray,
        error: np.ndarray,
        damping: float,
    ) -> np.ndarray:
        jj_t = jacobian @ jacobian.T
        damping_matrix = damping * np.eye(jj_t.shape[0])
        return jacobian.T @ np.linalg.solve(jj_t + damping_matrix, error)

    @staticmethod
    def _build_model_from_assets_urdf() -> pin.Model:
        with tempfile.TemporaryDirectory(prefix="bookarm_urdf_") as temp_dir:
            temp_urdf_path = Path(temp_dir) / DEFAULT_URDF_PATH.name
            shutil.copy2(DEFAULT_URDF_PATH, temp_urdf_path)
            return pin.buildModelFromUrdf(str(temp_urdf_path))


class BestEffortIKSolver:
    """Reusable best-effort pose IK solver.

    The solver always returns a configuration. If the target pose cannot be
    reached within ``tolerance`` and ``max_iterations``, the returned result has
    ``success=False`` and ``q`` contains the closest configuration found.
    """

    def __init__(
        self,
        robot: BookArm,
        *,
        end_effector_link: str | None = None,
        max_iterations: int = 500,
        tolerance: float = 1e-4,
        damping: float = 1e-6,
        step_size: float = 0.4,
        print_error: bool = True,
        link6_angle_rad: float = DEFAULT_LINK6_ANGLE_RAD,
    ) -> None:
        self.robot = robot
        self.end_effector_link = end_effector_link
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.damping = damping
        self.step_size = step_size
        self.print_error = print_error
        self.link6_angle_rad = link6_angle_rad

    def solve(
        self,
        target_position: Iterable[float],
        target_rotation: np.ndarray,
        q0: Iterable[float] | None = None,
        *,
        max_iterations: int | None = None,
        tolerance: float | None = None,
        damping: float | None = None,
        step_size: float | None = None,
        print_error: bool | None = None,
        link6_angle_rad: float | None = None,
    ) -> BestEffortIKResult:
        """Solve IK for a target position and rotation matrix."""

        return self.robot.ikine_best_effort(
            target_position=target_position,
            target_rotation=target_rotation,
            q0=q0,
            end_effector_link=self.end_effector_link,
            max_iterations=self.max_iterations if max_iterations is None else max_iterations,
            tolerance=self.tolerance if tolerance is None else tolerance,
            damping=self.damping if damping is None else damping,
            step_size=self.step_size if step_size is None else step_size,
            print_error=self.print_error if print_error is None else print_error,
            link6_angle_rad=(
                self.link6_angle_rad if link6_angle_rad is None else link6_angle_rad
            ),
        )
