"""Robot joint chain ported from the working legacy Lumen implementation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from core.model import SceneModel


def _translation(vector: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = vector
    return matrix


def _axis_rotation(axis: np.ndarray, degrees: float) -> np.ndarray:
    vector = np.asarray(axis, dtype=np.float64)
    vector /= max(float(np.linalg.norm(vector)), 1e-12)
    x, y, z = vector
    angle = math.radians(float(degrees))
    cosine, sine = math.cos(angle), math.sin(angle)
    one_minus = 1.0 - cosine
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(
        (
            (cosine + x*x*one_minus, x*y*one_minus - z*sine, x*z*one_minus + y*sine),
            (y*x*one_minus + z*sine, cosine + y*y*one_minus, y*z*one_minus - x*sine),
            (z*x*one_minus - y*sine, z*y*one_minus + x*sine, cosine + z*z*one_minus),
        ), dtype=np.float64,
    )
    return matrix


def _smooth_alpha(delta_time: float, response_seconds: float) -> float:
    return 1.0 - math.exp(-max(delta_time, 0.0) / max(response_seconds, 1e-6))


@dataclass(slots=True)
class RobotJoint:
    id: str
    name: str
    parent: str | None
    pivot: np.ndarray
    axis: np.ndarray
    minimum: float
    maximum: float
    control: str
    sensitivity: float
    response_seconds: float
    nodes: set[str]
    current: float = 0.0
    target: float = 0.0

    def add_drag(self, delta_x: float, delta_y: float) -> None:
        movement = delta_x if self.control == "horizontal" else -delta_y
        self.target = float(np.clip(self.target + movement * self.sensitivity, self.minimum, self.maximum))


class RobotController:
    """Five serial joints, smoothing, limits, demo, and the original gripper."""

    def __init__(self, model: SceneModel) -> None:
        self.model = model
        config = model.metadata.get("robotics", {})
        label_nodes: dict[str, set[str]] = {}
        for primitive in model.primitives:
            label_nodes.setdefault(primitive.display_name, set()).add(primitive.node_name)
        self.joints: dict[str, RobotJoint] = {}
        self.order: list[str] = []
        for raw in config.get("joints", []):
            member_nodes: set[str] = set()
            for label in raw.get("members", []):
                member_nodes.update(label_nodes.get(str(label), ()))
            if not member_nodes:
                raise ValueError(f"Robot joint {raw.get('name')} has no matching model meshes")
            axis = np.asarray(raw["axis"], dtype=np.float64)
            axis /= max(float(np.linalg.norm(axis)), 1e-12)
            joint = RobotJoint(
                id=str(raw["id"]),
                name=str(raw["name"]),
                parent=None if raw.get("parent") is None else str(raw["parent"]),
                pivot=np.asarray(raw["pivot"], dtype=np.float64),
                axis=axis,
                minimum=float(raw["min"]),
                maximum=float(raw["max"]),
                control=str(raw["control"]),
                sensitivity=float(raw["sensitivity"]),
                response_seconds=float(raw["response_seconds"]),
                nodes=member_nodes,
            )
            self.joints[joint.id] = joint
            self.order.append(joint.id)

        self.max_speed = float(config.get("joint_max_speed", 90.0))
        self.control_gain = float(config.get("joint_control_gain", 0.55))
        self.demo_segment_seconds = float(config.get("demo_segment_seconds", 3.2))
        self.node_to_joint = {node: joint.id for joint in self.joints.values() for node in joint.nodes}
        self._local_frames: dict[str, np.ndarray] = {}
        for joint_id in self.order:
            joint = self.joints[joint_id]
            rest = _translation(joint.pivot)
            self._local_frames[joint_id] = rest if joint.parent is None else np.linalg.inv(_translation(self.joints[joint.parent].pivot)) @ rest

        self.gripper_spec = dict(config.get("gripper", {}))
        self.gripper_open = 1.0
        self.gripper_target = 1.0
        self.demo_enabled = False
        self.demo_elapsed = 0.0
        self.demo_joint: str | None = None
        self._demo_actions = [*self.order, "gripper"]
        self._apply()

    def joint_for_component(self, component_id: str | None) -> str | None:
        return component_id if component_id in self.joints else None

    def add_drag(self, component_id: str | None, delta_x: float, delta_y: float) -> None:
        if component_id == "gripper":
            self.gripper_target = float(np.clip(self.gripper_target - delta_y * 3.0, 0.0, 1.0))
            return
        joint = self.joints.get(component_id or "")
        if joint is not None:
            joint.add_drag(delta_x * self.control_gain, delta_y * self.control_gain)

    def set_demo(self, enabled: bool) -> None:
        self.demo_enabled = bool(enabled)
        self.demo_elapsed = 0.0
        self.demo_joint = None
        if not enabled:
            self.home()

    def home(self) -> None:
        for joint in self.joints.values():
            joint.target = 0.0
        self.gripper_target = 1.0

    def angle(self, component_id: str | None) -> float | None:
        joint = self.joints.get(component_id or "")
        return None if joint is None else joint.current

    def limits(self, component_id: str | None) -> tuple[float, float] | None:
        joint = self.joints.get(component_id or "")
        return None if joint is None else (joint.minimum, joint.maximum)

    def update(self, delta_time: float) -> None:
        if self.demo_enabled:
            self._advance_demo(delta_time)
        changed = False
        for joint in self.joints.values():
            alpha = _smooth_alpha(delta_time, joint.response_seconds)
            step = float(np.clip((joint.target - joint.current) * alpha, -self.max_speed * delta_time, self.max_speed * delta_time))
            updated = float(np.clip(joint.current + step, joint.minimum, joint.maximum))
            if abs(updated - joint.target) < 1e-4:
                updated = joint.target
            changed = changed or abs(updated - joint.current) > 1e-8
            joint.current = updated
        response = float(self.gripper_spec.get("response_seconds", 0.065))
        old_gripper = self.gripper_open
        self.gripper_open = float(np.clip(self.gripper_open + (self.gripper_target - self.gripper_open) * _smooth_alpha(delta_time, response), 0.0, 1.0))
        if changed:
            self._apply()
        setattr(self.model, "robot_gripper_open", self.gripper_open)
        setattr(self.model, "robot_gripper_spec", self.gripper_spec)
        if abs(old_gripper - self.gripper_open) > 1e-8:
            self.model.matrix_revision += 1

    @staticmethod
    def _ease(first: float, second: float, amount: float) -> float:
        value = float(np.clip(amount, 0.0, 1.0))
        smooth = value * value * (3.0 - 2.0 * value)
        return first + (second - first) * smooth

    def _advance_demo(self, delta_time: float) -> None:
        self.demo_elapsed += max(delta_time, 0.0)
        index = int(self.demo_elapsed / self.demo_segment_seconds) % len(self._demo_actions)
        phase = (self.demo_elapsed % self.demo_segment_seconds) / self.demo_segment_seconds
        action = self._demo_actions[index]
        self.home()
        self.demo_joint = action
        if action == "gripper":
            self.gripper_target = self._ease(1.0, 0.0, phase * 2.0) if phase < 0.5 else self._ease(0.0, 0.5, (phase - 0.5) * 2.0)
            return
        joint = self.joints[action]
        low = joint.minimum * 0.55 if joint.minimum < 0.0 else joint.minimum
        high = joint.maximum * 0.55 if joint.maximum > 0.0 else joint.maximum
        if phase < 0.25:
            joint.target = self._ease(0.0, low, phase * 4.0)
        elif phase < 0.5:
            joint.target = self._ease(low, 0.0, (phase - 0.25) * 4.0)
        elif phase < 0.75:
            joint.target = self._ease(0.0, high, (phase - 0.5) * 4.0)
        else:
            joint.target = self._ease(high, 0.0, (phase - 0.75) * 4.0)

    def _apply(self) -> None:
        world_frames: dict[str, np.ndarray] = {}
        deformations: dict[str, np.ndarray] = {}
        for joint_id in self.order:
            joint = self.joints[joint_id]
            parent_world = np.eye(4, dtype=np.float64) if joint.parent is None else world_frames[joint.parent]
            world = parent_world @ self._local_frames[joint_id] @ _axis_rotation(joint.axis, joint.current)
            world_frames[joint_id] = world
            deformations[joint_id] = world @ _translation(-joint.pivot)
        self.model.set_node_deformations({node: deformations[joint_id] for node, joint_id in self.node_to_joint.items()})


__all__ = ["RobotController", "RobotJoint"]
