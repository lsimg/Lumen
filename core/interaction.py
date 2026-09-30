"""Gesture-to-action state, hover stabilization, and model transforms."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math

import numpy as np

import config
from core.gestures import Gesture, GestureFrame


class InteractionState(str, Enum):
    IDLE = "IDLE"
    MOVE = "MOVE"
    ROTATE = "ROTATE"
    SCALE = "SCALE"
    POINT = "POINT"
    PINCH_PENDING = "PINCH PENDING"
    COMPONENT_SELECTED = "COMPONENT SELECTED"
    STOPPED = "STOPPED"
    UI_INTERACTION = "UI INTERACTION"


class InputMode(str, Enum):
    HAND = "HAND CONTROL"
    MOUSE = "MOUSE CONTROL"


def _rotation_x(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array(
        ((1, 0, 0, 0), (0, cosine, -sine, 0), (0, sine, cosine, 0), (0, 0, 0, 1)),
        dtype=np.float64,
    )


def _rotation_y(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array(
        ((cosine, 0, sine, 0), (0, 1, 0, 0), (-sine, 0, cosine, 0), (0, 0, 0, 1)),
        dtype=np.float64,
    )


def _rotation_z(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array(
        ((cosine, -sine, 0, 0), (sine, cosine, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
        dtype=np.float64,
    )


@dataclass
class ModelTransform:
    position: tuple[float, float, float] = (0.0, -0.05, 0.0)
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: float = 0.88

    def matrix(self) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, 3] = self.position
        scale = np.diag((self.scale, self.scale, self.scale, 1.0))
        return (
            matrix
            @ _rotation_z(self.rotation[2])
            @ _rotation_y(self.rotation[1])
            @ _rotation_x(self.rotation[0])
            @ scale
        )


@dataclass
class ApplicationState:
    transform: ModelTransform
    interaction: InteractionState = InteractionState.IDLE
    previous_interaction: InteractionState = InteractionState.IDLE
    transition_reason: str = "startup"
    input_mode: InputMode = InputMode.HAND
    gesture: Gesture = Gesture.NONE
    pointer_position: tuple[float, float] | None = None
    hovered_component: str | None = None
    selected_component: str | None = None
    pinch_progress: float = 0.0
    debug: bool = False


@dataclass
class HoverStabilizer:
    """Stabilize semantic targets while preserving a short release grace period."""

    stable_target: str | None = None
    _candidate: str | None = None
    _candidate_since: float = 0.0
    _candidate_frames: int = 0
    _stable_pointer: np.ndarray | None = field(default=None, repr=False)

    def reset(self) -> None:
        self.stable_target = None
        self._candidate = None
        self._candidate_since = 0.0
        self._candidate_frames = 0
        self._stable_pointer = None

    def update(
        self,
        candidate: str | None,
        pointer: tuple[float, float] | None,
        timestamp: float,
    ) -> str | None:
        point = None if pointer is None else np.asarray(pointer, dtype=np.float64)
        if candidate == self.stable_target:
            self._candidate = None
            self._candidate_frames = 0
            if point is not None:
                self._stable_pointer = point
            return self.stable_target

        if candidate != self._candidate:
            self._candidate = candidate
            self._candidate_since = float(timestamp)
            self._candidate_frames = 1
            return self.stable_target

        self._candidate_frames += 1
        if candidate is None:
            delay = config.HOVER_RELEASE_SECONDS
        elif self.stable_target is None:
            delay = config.HOVER_ACQUIRE_SECONDS
        else:
            nearby = bool(
                point is not None
                and self._stable_pointer is not None
                and np.linalg.norm(point - self._stable_pointer)
                <= config.HOVER_NEARBY_POINTER_RADIUS
            )
            delay = config.HOVER_SWITCH_SECONDS if nearby else config.HOVER_ACQUIRE_SECONDS

        if (
            self._candidate_frames >= config.HOVER_MINIMUM_FRAMES
            and float(timestamp) - self._candidate_since >= delay
        ):
            self.stable_target = candidate
            self._candidate = None
            self._candidate_frames = 0
            self._stable_pointer = point
        return self.stable_target


class InteractionController:
    """One explicit state machine for hand, mouse, hover, and selection input."""

    def __init__(
        self,
        default_scale: float = 0.88,
        default_rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self._default_scale = float(default_scale)
        self._default_rotation = tuple(map(float, default_rotation))
        transform = ModelTransform(rotation=self._default_rotation, scale=self._default_scale)
        self.state = ApplicationState(transform)
        self.hover = HoverStabilizer()

        self._target_position = np.asarray(transform.position, dtype=np.float64)
        self._target_rotation = np.asarray(transform.rotation, dtype=np.float64)
        self._target_scale = transform.scale
        self._operation_anchor: np.ndarray | float | None = None
        self._operation_start: np.ndarray | float | None = None
        self._active_operation: str | None = None
        self._pinch_target: str | None = None
        self._pinch_since: float | None = None
        self._pinch_frames = 0
        self._pinch_consumed = False
        self._stop_since: float | None = None
        self._stop_frames = 0
        self._resume_since: float | None = None
        self._mouse_down = False

    def _transition(self, state: InteractionState, reason: str) -> None:
        if self.state.interaction == state:
            return
        self.state.previous_interaction = self.state.interaction
        self.state.interaction = state
        self.state.transition_reason = reason
        operation = {
            InteractionState.MOVE: "move",
            InteractionState.ROTATE: "rotate",
            InteractionState.SCALE: "scale",
        }.get(state)
        if operation != self._active_operation:
            self._active_operation = operation
            self._operation_anchor = None
            self._operation_start = None

    def _freeze_targets(self) -> None:
        self._target_position = np.asarray(self.state.transform.position, dtype=np.float64)
        self._target_rotation = np.asarray(self.state.transform.rotation, dtype=np.float64)
        self._target_scale = float(self.state.transform.scale)

    def reset_view(self) -> None:
        if self.state.interaction == InteractionState.STOPPED:
            return
        self._target_position = np.asarray((0.0, -0.05, 0.0), dtype=np.float64)
        self._target_rotation = np.asarray(self._default_rotation, dtype=np.float64)
        self._target_scale = self._default_scale
        # Reset is an explicit command, so apply it immediately.  Otherwise an
        # idle hand update in the same frame would freeze the old transform and
        # cancel the reset targets before smoothing can begin.
        self.state.transform.position = tuple(map(float, self._target_position))
        self.state.transform.rotation = tuple(map(float, self._target_rotation))
        self.state.transform.scale = float(self._target_scale)
        self._operation_anchor = None
        self._operation_start = None
        self._transition(
            InteractionState.COMPONENT_SELECTED
            if self.state.selected_component
            else InteractionState.IDLE,
            "view reset",
        )

    def clear_selection(self) -> None:
        self.state.selected_component = None
        self._reset_pinch()
        # Require release before a held pinch can immediately select again.
        self._pinch_consumed = True
        if self.state.interaction != InteractionState.STOPPED:
            self._transition(InteractionState.IDLE, "selection cleared")

    def stop(self, reason: str = "stop requested") -> None:
        self._freeze_targets()
        self._reset_pinch()
        self.hover.reset()
        self.state.hovered_component = None
        self._resume_since = None
        self._transition(InteractionState.STOPPED, reason)

    def resume(self, reason: str = "resume requested") -> None:
        if self.state.interaction != InteractionState.STOPPED:
            return
        self._resume_since = None
        self._transition(
            InteractionState.COMPONENT_SELECTED
            if self.state.selected_component
            else InteractionState.IDLE,
            reason,
        )

    def toggle_input_mode(self) -> None:
        self.state.input_mode = (
            InputMode.MOUSE if self.state.input_mode == InputMode.HAND else InputMode.HAND
        )
        self.hover.reset()
        self.state.hovered_component = None
        self.state.pointer_position = None
        self._mouse_down = False
        self._reset_pinch()
        self._pinch_consumed = True
        self._stop_since = None
        self._stop_frames = 0
        self._resume_since = None
        self._freeze_targets()
        if self.state.interaction != InteractionState.STOPPED:
            self._transition(
                InteractionState.COMPONENT_SELECTED
                if self.state.selected_component
                else InteractionState.IDLE,
                "input mode changed",
            )

    def _reset_pinch(self) -> None:
        self._pinch_target = None
        self._pinch_since = None
        self._pinch_frames = 0
        self.state.pinch_progress = 0.0

    def _update_stop_gesture(self, frame: GestureFrame, timestamp: float) -> bool:
        if frame.gesture == Gesture.FIST:
            if self._stop_since is None:
                self._stop_since = timestamp
                self._stop_frames = 1
            else:
                self._stop_frames += 1
            if (
                self._stop_frames >= config.STOP_CONFIRM_FRAMES
                and timestamp - self._stop_since >= config.STOP_CONFIRM_SECONDS
            ):
                self.stop("fist stop")
                return True
        else:
            self._stop_since = None
            self._stop_frames = 0
        return False

    def _update_pinch(
        self,
        frame: GestureFrame,
        pointed_component: str | None,
        timestamp: float,
    ) -> bool:
        if not frame.pinch_active:
            self._pinch_consumed = False
            self._reset_pinch()
            return False
        if self._pinch_consumed:
            return True
        if self._pinch_since is None or pointed_component != self._pinch_target:
            self._pinch_target = pointed_component
            self._pinch_since = timestamp
            self._pinch_frames = 1
        else:
            self._pinch_frames += 1
        elapsed = timestamp - self._pinch_since
        self.state.pinch_progress = float(
            np.clip(elapsed / max(config.PINCH_HOLD_SECONDS, 1e-6), 0.0, 1.0)
        )
        self._transition(InteractionState.PINCH_PENDING, "confirming pinch")
        if (
            elapsed >= config.PINCH_HOLD_SECONDS
            and self._pinch_frames >= config.GESTURE_CONFIRM_FRAMES
        ):
            self.state.selected_component = self._pinch_target
            self._pinch_consumed = True
            self.state.pinch_progress = 1.0
            self._transition(
                InteractionState.COMPONENT_SELECTED
                if self._pinch_target is not None
                else InteractionState.IDLE,
                "component selected" if self._pinch_target else "empty pinch cleared selection",
            )
        return True

    def update_hand(
        self,
        frame: GestureFrame,
        pointed_component: str | None,
        *,
        ui_capture: bool,
        timestamp: float,
    ) -> None:
        if self.state.input_mode != InputMode.HAND:
            return
        self.state.gesture = frame.gesture
        self.state.pointer_position = frame.pointer

        if ui_capture:
            self.hover.update(None, frame.pointer, timestamp)
            self.state.hovered_component = None
            self._freeze_targets()
            if frame.pinch_active:
                self._pinch_consumed = True
            else:
                self._pinch_consumed = False
            self._reset_pinch()
            if self.state.interaction != InteractionState.STOPPED:
                self._transition(
                    InteractionState.UI_INTERACTION,
                    "interface captured pointer",
                )
            return

        if self._update_stop_gesture(frame, timestamp):
            return
        if self.state.interaction == InteractionState.STOPPED:
            if frame.gesture == Gesture.OPEN_PALM:
                if self._resume_since is None:
                    self._resume_since = timestamp
                elif timestamp - self._resume_since >= config.RESUME_HOLD_SECONDS:
                    self.resume("open palm resume")
            else:
                self._resume_since = None
            return

        can_point = frame.gesture in (Gesture.POINT, Gesture.PINCH)
        stable_hover = self.hover.update(
            pointed_component if can_point else None,
            frame.pointer,
            timestamp,
        )
        self.state.hovered_component = stable_hover

        pinch_target = stable_hover if pointed_component is not None else None
        if self._update_pinch(frame, pinch_target, timestamp):
            return

        if frame.gesture == Gesture.OPEN_PALM and frame.palm_center is not None:
            self._transition(InteractionState.MOVE, "open palm")
            self._update_move(frame.palm_center)
        elif frame.gesture == Gesture.V_SIGN and frame.pointer is not None:
            self._transition(InteractionState.ROTATE, "v gesture")
            self._update_rotate(frame.pointer)
        elif frame.gesture == Gesture.TWO_HANDS and frame.two_hand_distance is not None:
            self._transition(InteractionState.SCALE, "two hands")
            self._update_scale(frame.two_hand_distance)
        elif frame.gesture == Gesture.POINT:
            self._freeze_targets()
            self._transition(InteractionState.POINT, "pointing")
        else:
            self._freeze_targets()
            self._transition(
                InteractionState.COMPONENT_SELECTED
                if self.state.selected_component
                else InteractionState.IDLE,
                "no active gesture",
            )

    def _update_move(self, center: tuple[float, float]) -> None:
        current = np.asarray(center, dtype=np.float64)
        if self._operation_anchor is None:
            self._operation_anchor = current
            self._operation_start = self._target_position.copy()
            return
        delta = current - np.asarray(self._operation_anchor)
        delta = np.sign(delta) * np.maximum(np.abs(delta) - config.POSITION_DEAD_ZONE, 0.0)
        target = np.asarray(self._operation_start).copy()
        target[0] += delta[0] * config.MODEL_TRANSLATION_SENSITIVITY
        target[1] -= delta[1] * config.MODEL_TRANSLATION_SENSITIVITY
        target[0] = np.clip(target[0], -config.MODEL_POSITION_LIMIT_X, config.MODEL_POSITION_LIMIT_X)
        target[1] = np.clip(target[1], -config.MODEL_POSITION_LIMIT_Y, config.MODEL_POSITION_LIMIT_Y)
        self._target_position = target

    def _update_rotate(self, pointer: tuple[float, float]) -> None:
        current = np.asarray(pointer, dtype=np.float64)
        if self._operation_anchor is None:
            self._operation_anchor = current
            self._operation_start = self._target_rotation.copy()
            return
        delta = current - np.asarray(self._operation_anchor)
        delta = np.sign(delta) * np.maximum(np.abs(delta) - config.ROTATION_DEAD_ZONE, 0.0)
        start = np.asarray(self._operation_start)
        self._target_rotation = start + np.asarray(
            (delta[1], delta[0], 0.0), dtype=np.float64
        ) * config.ROTATION_SENSITIVITY

    def _update_scale(self, distance: float) -> None:
        value = float(distance)
        if self._operation_anchor is None:
            self._operation_anchor = value
            self._operation_start = self._target_scale
            return
        ratio = value / max(float(self._operation_anchor), 1e-5) - 1.0
        ratio = math.copysign(max(abs(ratio) - config.SCALE_DEAD_ZONE, 0.0), ratio)
        self._target_scale = float(
            np.clip(float(self._operation_start) * (1.0 + ratio), config.MIN_SCALE, config.MAX_SCALE)
        )

    def update_mouse(
        self,
        pointer: tuple[float, float] | None,
        left_down: bool,
        scroll_delta: float,
        *,
        ui_capture: bool,
    ) -> None:
        if self.state.input_mode != InputMode.MOUSE:
            return
        self.state.pointer_position = pointer
        self.state.gesture = Gesture.NONE
        if self.state.interaction == InteractionState.STOPPED:
            self._mouse_down = left_down
            return
        if scroll_delta and not ui_capture:
            self._target_scale = float(
                np.clip(
                    self._target_scale
                    * math.exp(np.clip(scroll_delta, -20.0, 20.0) * config.MOUSE_WHEEL_SENSITIVITY),
                    config.MIN_SCALE,
                    config.MAX_SCALE,
                )
            )
        if ui_capture:
            self._mouse_down = left_down
            self._operation_anchor = None
            self._transition(InteractionState.UI_INTERACTION, "interface captured mouse")
            return
        if left_down and pointer is not None:
            # Enter the operation before capturing its anchor.  A transition
            # intentionally clears the previous gesture's anchor.
            self._transition(InteractionState.ROTATE, "mouse drag")
            current = np.asarray(pointer, dtype=np.float64)
            if not self._mouse_down or self._operation_anchor is None:
                self._operation_anchor = current
                self._operation_start = self._target_rotation.copy()
            else:
                delta = current - np.asarray(self._operation_anchor)
                start = np.asarray(self._operation_start)
                self._target_rotation = start + np.asarray(
                    (delta[1], delta[0], 0.0), dtype=np.float64
                ) * config.MOUSE_ROTATION_SENSITIVITY
        else:
            self._operation_anchor = None
            self._operation_start = None
            self._transition(
                InteractionState.COMPONENT_SELECTED
                if self.state.selected_component
                else InteractionState.IDLE,
                "mouse idle",
            )
        self._mouse_down = left_down

    def advance(self, delta_time: float) -> ModelTransform:
        delta = float(np.clip(delta_time, 0.0, 0.05))
        if self.state.interaction == InteractionState.STOPPED:
            return self.state.transform

        def alpha(response: float) -> float:
            return 1.0 - math.exp(-delta / max(response, 1e-6))

        position = np.asarray(self.state.transform.position, dtype=np.float64)
        rotation = np.asarray(self.state.transform.rotation, dtype=np.float64)
        position += (self._target_position - position) * alpha(
            config.MODEL_POSITION_RESPONSE_SECONDS
        )
        rotation += (self._target_rotation - rotation) * alpha(
            config.MODEL_ROTATION_RESPONSE_SECONDS
        )
        scale = self.state.transform.scale + (
            self._target_scale - self.state.transform.scale
        ) * alpha(config.MODEL_SCALE_RESPONSE_SECONDS)
        self.state.transform.position = tuple(map(float, position))
        self.state.transform.rotation = tuple(map(float, rotation))
        self.state.transform.scale = float(np.clip(scale, config.MIN_SCALE, config.MAX_SCALE))
        return self.state.transform


__all__ = [
    "ApplicationState",
    "HoverStabilizer",
    "InputMode",
    "InteractionController",
    "InteractionState",
    "ModelTransform",
]
