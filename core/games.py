"""Two small gesture games built on the same picking and model transforms."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from core.gestures import Gesture, GestureFrame
from core.model import SceneModel


@dataclass(frozen=True, slots=True)
class TargetBox:
    label: str
    kind: str
    rect: tuple[float, float, float, float]
    color_rgb: tuple[int, int, int]


class FruitGames:
    MATCH = "match"
    SIZE = "size"

    def __init__(self, model: SceneModel, mode: str = MATCH) -> None:
        self.model = model
        self.mode = mode
        self.status = ""
        self.status_positive = False
        self.matched: set[str] = set()
        self.object_offsets: dict[str, np.ndarray] = {}
        self.object_scales: dict[str, float] = {}
        self.object_rotations: dict[str, np.ndarray] = {}
        self._base_centers = dict(model._target_centers)
        self._drag_object: str | None = None
        self._drag_anchor: np.ndarray | None = None
        self._drag_start = np.zeros(3, dtype=np.float64)
        self._move_object: str | None = None
        self._move_anchor: np.ndarray | None = None
        self._move_start = np.zeros(3, dtype=np.float64)
        self._rotate_object: str | None = None
        self._rotate_anchor: np.ndarray | None = None
        self._rotate_start = np.zeros(3, dtype=np.float64)
        self._pinch_was_active = False
        self._scale_object: str | None = None
        self._scale_anchor: float | None = None
        self._scale_start = 1.0
        self.size_object = "pear"
        self.target_scale = 1.65
        self.size_tolerance = 0.16
        self.boxes = (
            TargetBox("APPLE", "apple", (0.035, 0.750, 0.255, 0.880), (234, 74, 82)),
            TargetBox("BANANA", "banana", (0.270, 0.750, 0.490, 0.880), (248, 207, 69)),
            TargetBox("ORANGE", "orange", (0.510, 0.750, 0.730, 0.880), (255, 143, 42)),
            TargetBox("PEAR", "pear", (0.745, 0.750, 0.965, 0.880), (112, 205, 112)),
        )

    @staticmethod
    def _kind(object_id: str) -> str:
        if "apple" in object_id:
            return "apple"
        if "banana" in object_id:
            return "banana"
        return object_id

    def set_mode(self, mode: str) -> None:
        self.mode = mode if mode in (self.MATCH, self.SIZE) else self.MATCH
        self.status = ""
        self._drag_object = None
        self._drag_anchor = None
        self._move_object = None
        self._move_anchor = None
        self._rotate_object = None
        self._rotate_anchor = None
        self._scale_object = None
        self._scale_anchor = None

    def reset(self) -> None:
        self.model.clear_object_transforms()
        self.status = ""
        self.matched.clear()
        self.object_offsets.clear()
        self.object_scales.clear()
        self.object_rotations.clear()
        self._drag_object = None
        self._drag_anchor = None
        self._move_object = None
        self._move_anchor = None
        self._rotate_object = None
        self._rotate_anchor = None
        self._scale_object = None
        self._scale_anchor = None

    def _apply(self, object_id: str) -> None:
        offset = self.object_offsets.get(object_id, np.zeros(3, dtype=np.float64))
        scale = self.object_scales.get(object_id, 1.0)
        rotation = np.radians(self.object_rotations.get(object_id, np.zeros(3, dtype=np.float64)))
        center = np.asarray(self._base_centers[object_id], dtype=np.float64)
        translate = np.eye(4, dtype=np.float64)
        translate[:3, 3] = offset
        to_center = np.eye(4, dtype=np.float64)
        to_center[:3, 3] = center
        from_center = np.eye(4, dtype=np.float64)
        from_center[:3, 3] = -center
        scale_matrix = np.diag((scale, scale, scale, 1.0))
        cos_x, sin_x = math.cos(rotation[0]), math.sin(rotation[0])
        cos_y, sin_y = math.cos(rotation[1]), math.sin(rotation[1])
        rotate_x = np.asarray(
            ((1.0, 0.0, 0.0, 0.0), (0.0, cos_x, -sin_x, 0.0), (0.0, sin_x, cos_x, 0.0), (0.0, 0.0, 0.0, 1.0)),
            dtype=np.float64,
        )
        rotate_y = np.asarray(
            ((cos_y, 0.0, sin_y, 0.0), (0.0, 1.0, 0.0, 0.0), (-sin_y, 0.0, cos_y, 0.0), (0.0, 0.0, 0.0, 1.0)),
            dtype=np.float64,
        )
        transform = translate @ to_center @ rotate_y @ rotate_x @ scale_matrix @ from_center
        self.model.set_object_transform(object_id, transform)

    def update(
        self,
        frame: GestureFrame,
        selected_object: str | None,
        root_scale: float,
    ) -> None:
        valid_object = selected_object if selected_object in self._base_centers else None
        self._update_pinch_move(frame, valid_object, root_scale)
        self._update_palm_move(frame, valid_object, root_scale)
        self._update_rotation(frame, valid_object)
        self._update_scale(frame, valid_object)
        if self.mode == self.MATCH:
            self._update_match_drop(frame)
        self._pinch_was_active = frame.pinch_active

    def _update_pinch_move(self, frame: GestureFrame, valid_object: str | None, root_scale: float) -> None:
        if frame.pinch_active and frame.pointer is not None and valid_object:
            pointer = np.asarray(frame.pointer, dtype=np.float64)
            if self._drag_object != valid_object or not self._pinch_was_active:
                self._drag_object = valid_object
                self._drag_anchor = pointer
                self._drag_start = self.object_offsets.get(valid_object, np.zeros(3, dtype=np.float64)).copy()
            if self._drag_anchor is not None:
                delta = pointer - self._drag_anchor
                offset = self._drag_start.copy()
                offset[0] += delta[0] * 2.45 / max(root_scale, 0.25)
                offset[1] -= delta[1] * 1.85 / max(root_scale, 0.25)
                self.object_offsets[valid_object] = offset
                self._apply(valid_object)
            return

        self._drag_anchor = None

    def _update_palm_move(self, frame: GestureFrame, valid_object: str | None, root_scale: float) -> None:
        if frame.gesture == Gesture.OPEN_PALM and frame.palm_center is not None and valid_object:
            center = np.asarray(frame.palm_center, dtype=np.float64)
            if self._move_object != valid_object or self._move_anchor is None:
                self._move_object = valid_object
                self._move_anchor = center
                self._move_start = self.object_offsets.get(valid_object, np.zeros(3, dtype=np.float64)).copy()
            delta = center - self._move_anchor
            offset = self._move_start.copy()
            offset[0] += delta[0] * 2.45 / max(root_scale, 0.25)
            offset[1] -= delta[1] * 1.85 / max(root_scale, 0.25)
            self.object_offsets[valid_object] = offset
            self._apply(valid_object)
            return
        self._move_object = None
        self._move_anchor = None

    def _update_rotation(self, frame: GestureFrame, valid_object: str | None) -> None:
        if frame.gesture == Gesture.V_SIGN and frame.pointer is not None and valid_object:
            pointer = np.asarray(frame.pointer, dtype=np.float64)
            if self._rotate_object != valid_object or self._rotate_anchor is None:
                self._rotate_object = valid_object
                self._rotate_anchor = pointer
                self._rotate_start = self.object_rotations.get(valid_object, np.zeros(3, dtype=np.float64)).copy()
            delta = pointer - self._rotate_anchor
            rotation = self._rotate_start.copy()
            rotation[0] += delta[1] * 230.0
            rotation[1] += delta[0] * 230.0
            self.object_rotations[valid_object] = rotation
            self._apply(valid_object)
            return
        self._rotate_object = None
        self._rotate_anchor = None

    def _update_scale(self, frame: GestureFrame, valid_object: str | None) -> None:
        object_id = valid_object or (self.size_object if self.mode == self.SIZE else None)
        if frame.gesture == Gesture.TWO_HANDS and frame.two_hand_distance is not None and object_id:
            distance = float(frame.two_hand_distance)
            if self._scale_object != object_id or self._scale_anchor is None:
                self._scale_object = object_id
                self._scale_anchor = max(distance, 1e-4)
                self._scale_start = self.object_scales.get(object_id, 1.0)
            ratio = distance / self._scale_anchor
            scale = float(np.clip(self._scale_start * ratio, 0.45, 2.5))
            self.object_scales[object_id] = scale
            self._apply(object_id)
            if self.mode == self.SIZE and object_id == self.size_object:
                relative_error = abs(scale - self.target_scale) / self.target_scale
                if relative_error <= self.size_tolerance:
                    self.status = "CORRECT"
                    self.status_positive = True
                else:
                    self.status = "MAKE IT A LITTLE BIGGER" if scale < self.target_scale else "A LITTLE SMALLER"
                    self.status_positive = False
            return
        self._scale_object = None
        self._scale_anchor = None

    def _update_match_drop(self, frame: GestureFrame) -> None:
        if frame.pinch_active:
            return

        if self._pinch_was_active and self._drag_object and frame.pointer is not None:
            x, y = frame.pointer
            target = next((box for box in self.boxes if box.rect[0] <= x <= box.rect[2] and box.rect[1] <= y <= box.rect[3]), None)
            if target is not None and target.kind == self._kind(self._drag_object):
                self.matched.add(self._drag_object)
                self.status = "CORRECT"
                self.status_positive = True
            else:
                self.status = "TRY THE MATCHING BOX"
                self.status_positive = False
        self._drag_object = None

    @property
    def current_scale(self) -> float:
        return self.object_scales.get(self.size_object, 1.0)

    @property
    def dimension_match(self) -> bool:
        return math.isclose(self.current_scale, self.target_scale, rel_tol=self.size_tolerance)


__all__ = ["FruitGames", "TargetBox"]
