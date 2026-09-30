"""Asynchronous MediaPipe hand tracking with resilient status reporting."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
import threading
import time
from typing import Any

import cv2
import numpy as np

from config import (
    FIST_THUMB_CENTER_RATIO,
    FIST_TIP_FOLD_RATIO,
    HAND_CONNECTIONS,
    HAND_MIN_DETECTION_CONFIDENCE,
    HAND_MIN_PRESENCE_CONFIDENCE,
    HAND_MIN_TRACKING_CONFIDENCE,
    LANDMARK_JUMP_LIMIT,
    MAX_ISOLATED_LANDMARK_OUTLIERS,
    ONE_EURO_BETA,
    ONE_EURO_D_CUTOFF,
    ONE_EURO_MIN_CUTOFF,
    TRACKING_CALLBACK_TIMEOUT_SECONDS,
    TRACKING_FAILURE_LIMIT,
    TRACKING_MAX_FPS,
    TRACKING_MAX_WIDTH,
    TRACK_LOST_RESET_SECONDS,
)

try:
    import mediapipe as mp
except Exception as import_error:  # Missing/incompatible MediaPipe disables tracking only.
    mp = None  # type: ignore[assignment]
    _MEDIAPIPE_IMPORT_ERROR: Exception | None = import_error
else:
    _MEDIAPIPE_IMPORT_ERROR = None


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class TrackedHand:
    """Filtered landmarks and derived, scale-independent hand poses."""

    handedness: str
    confidence: float
    landmarks: tuple[Landmark, ...]
    raw_landmarks: tuple[Landmark, ...] | None = None

    @property
    def score(self) -> float:
        """Compatibility alias for MediaPipe's handedness score."""

        return self.confidence

    @property
    def wrist(self) -> Landmark:
        return self.landmarks[0]

    @property
    def thumb_tip(self) -> Landmark:
        return self.landmarks[4]

    @property
    def index_tip(self) -> Landmark:
        return self.landmarks[8]

    @property
    def raw_index_tip(self) -> Landmark:
        return self.index_tip if self.raw_landmarks is None else self.raw_landmarks[8]

    @property
    def palm_center(self) -> Landmark:
        points = tuple(self.landmarks[index] for index in (0, 5, 9, 13, 17))
        return Landmark(
            x=sum(point.x for point in points) / len(points),
            y=sum(point.y for point in points) / len(points),
            z=sum(point.z for point in points) / len(points),
        )

    @property
    def palm_width(self) -> float:
        first, second = self.landmarks[5], self.landmarks[17]
        return max(math.hypot(first.x - second.x, first.y - second.y), 1e-4)

    @property
    def pinch_ratio(self) -> float:
        distance = math.hypot(
            self.thumb_tip.x - self.index_tip.x,
            self.thumb_tip.y - self.index_tip.y,
        )
        return distance / self.palm_width

    def _finger_extended(self, mcp_index: int, pip_index: int, tip_index: int) -> bool:
        mcp = self.landmarks[mcp_index]
        pip = self.landmarks[pip_index]
        tip = self.landmarks[tip_index]
        wrist_to_tip = math.hypot(tip.x - self.wrist.x, tip.y - self.wrist.y)
        wrist_to_pip = math.hypot(pip.x - self.wrist.x, pip.y - self.wrist.y)
        base_vector = np.asarray((mcp.x - pip.x, mcp.y - pip.y), dtype=np.float64)
        tip_vector = np.asarray((tip.x - pip.x, tip.y - pip.y), dtype=np.float64)
        denominator = max(
            float(np.linalg.norm(base_vector) * np.linalg.norm(tip_vector)), 1e-8
        )
        straightness = float(np.dot(base_vector, tip_vector) / denominator)
        return bool(wrist_to_tip > wrist_to_pip * 1.07 and straightness < -0.48)

    @property
    def index_extended(self) -> bool:
        return self._finger_extended(5, 6, 8)

    @property
    def middle_extended(self) -> bool:
        return self._finger_extended(9, 10, 12)

    @property
    def ring_extended(self) -> bool:
        return self._finger_extended(13, 14, 16)

    @property
    def pinky_extended(self) -> bool:
        return self._finger_extended(17, 18, 20)

    @property
    def point(self) -> bool:
        return bool(
            self.index_extended
            and not self.middle_extended
            and not self.ring_extended
            and not self.pinky_extended
        )

    @property
    def v_sign(self) -> bool:
        return bool(
            self.index_extended
            and self.middle_extended
            and not self.ring_extended
            and not self.pinky_extended
        )

    @property
    def rock_sign(self) -> bool:
        return bool(
            self.index_extended
            and not self.middle_extended
            and not self.ring_extended
            and self.pinky_extended
        )

    @property
    def open_palm(self) -> bool:
        return bool(
            self.index_extended
            and self.middle_extended
            and self.ring_extended
            and self.pinky_extended
        )

    @property
    def fist(self) -> bool:
        # Thumb-to-center compactness prevents a side-on point or pinch from
        # becoming a false emergency-stop gesture.
        center = self.palm_center
        thumb_tucked = math.hypot(
            self.thumb_tip.x - center.x, self.thumb_tip.y - center.y
        ) < self.palm_width * FIST_THUMB_CENTER_RATIO
        folded = all(
            math.hypot(
                self.landmarks[tip].x - self.wrist.x,
                self.landmarks[tip].y - self.wrist.y,
            )
            <= math.hypot(
                self.landmarks[pip].x - self.wrist.x,
                self.landmarks[pip].y - self.wrist.y,
            )
            * FIST_TIP_FOLD_RATIO
            for pip, tip in ((6, 8), (10, 12), (14, 16), (18, 20))
        )
        return bool(thumb_tucked and folded)


@dataclass(frozen=True)
class TrackingSnapshot:
    hands: tuple[TrackedHand, ...] = ()
    sequence: int = 0
    timestamp_ms: int = -1
    latency_ms: float = 0.0
    received_at: float = 0.0
    processing_ms: float = 0.0
    captured_at: float = 0.0
    status: str = "STARTING"

    @property
    def average_confidence(self) -> float:
        if not self.hands:
            return 0.0
        return sum(hand.confidence for hand in self.hands) / len(self.hands)


class _OneEuroFilter:
    """Vector-valued One Euro filter for low-jitter, responsive landmarks."""

    def __init__(self, min_cutoff: float, beta: float, derivative_cutoff: float) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.derivative_cutoff = float(derivative_cutoff)
        self._previous_raw: np.ndarray | None = None
        self._previous_filtered: np.ndarray | None = None
        self._previous_derivative: np.ndarray | None = None
        self._previous_time: float | None = None

    @staticmethod
    def _alpha(cutoff: np.ndarray | float, delta_time: float) -> np.ndarray | float:
        safe_cutoff = np.maximum(cutoff, 1e-6)
        tau = 1.0 / (2.0 * math.pi * safe_cutoff)
        return 1.0 / (1.0 + tau / max(delta_time, 1e-6))

    def filter(self, value: np.ndarray, timestamp: float) -> np.ndarray:
        raw = np.asarray(value, dtype=np.float64)
        if self._previous_raw is None or self._previous_time is None:
            self._previous_raw = raw.copy()
            self._previous_filtered = raw.copy()
            self._previous_derivative = np.zeros_like(raw)
            self._previous_time = float(timestamp)
            return raw.copy()

        delta_time = max(float(timestamp) - self._previous_time, 1e-4)
        derivative = (raw - self._previous_raw) / delta_time
        derivative_alpha = float(self._alpha(self.derivative_cutoff, delta_time))
        filtered_derivative = (
            derivative_alpha * derivative
            + (1.0 - derivative_alpha) * self._previous_derivative
        )
        cutoff = self.min_cutoff + self.beta * np.abs(filtered_derivative)
        value_alpha = self._alpha(cutoff, delta_time)
        filtered = value_alpha * raw + (1.0 - value_alpha) * self._previous_filtered

        self._previous_raw = raw.copy()
        self._previous_filtered = filtered.copy()
        self._previous_derivative = filtered_derivative.copy()
        self._previous_time = float(timestamp)
        return filtered


def _stabilize_landmark_outliers(
    coordinates: np.ndarray,
    previous: np.ndarray | None,
) -> tuple[np.ndarray, bool]:
    """Hold isolated bad points while accepting coherent whole-hand movement."""

    current = np.asarray(coordinates, dtype=np.float64)
    if previous is None or previous.shape != current.shape:
        return current, False
    displacement = np.linalg.norm(current[:, :2] - previous[:, :2], axis=1)
    outliers = displacement > LANDMARK_JUMP_LIMIT
    outlier_count = int(np.count_nonzero(outliers))
    if 0 < outlier_count <= MAX_ISOLATED_LANDMARK_OUTLIERS:
        stable = current.copy()
        stable[outliers] = previous[outliers]
        return stable, False
    coherent_jump = float(np.median(displacement)) > LANDMARK_JUMP_LIMIT
    return current, coherent_jump


class HandTracker:
    """MediaPipe LIVE_STREAM tracker that never blocks the render loop."""

    def __init__(self, model_path: str | Path, max_hands: int = 2) -> None:
        self._lock = threading.Lock()
        self._snapshot = TrackingSnapshot()
        self._filters: dict[str, _OneEuroFilter] = {}
        self._last_raw: dict[str, np.ndarray] = {}
        self._last_seen: dict[str, float] = {}
        self._submitted_at: dict[int, float] = {}
        self._capture_times: dict[int, float] = {}
        self._clock_origin_ns = time.monotonic_ns()
        self._last_submit_timestamp = -1
        self._last_submit_time = -float("inf")
        self._failure_count = 0
        self._error: str | None = None
        self._enabled = False
        self._closed = False
        self._landmarker: Any | None = None

        path = Path(model_path)
        if not path.is_file():
            self._disable("MODEL_MISSING", f"Hand tracking model not found: {path.name}")
            return
        if mp is None:
            detail = "MediaPipe is unavailable"
            if _MEDIAPIPE_IMPORT_ERROR is not None:
                detail = f"{detail}: {_MEDIAPIPE_IMPORT_ERROR}"
            self._disable("DISABLED", detail)
            return

        try:
            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=str(path),
                    delegate=mp.tasks.BaseOptions.Delegate.CPU,
                ),
                running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
                num_hands=max(1, int(max_hands)),
                min_hand_detection_confidence=HAND_MIN_DETECTION_CONFIDENCE,
                min_hand_presence_confidence=HAND_MIN_PRESENCE_CONFIDENCE,
                min_tracking_confidence=HAND_MIN_TRACKING_CONFIDENCE,
                result_callback=self._on_result,
            )
            self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        except Exception as error:
            self._disable("DISABLED", f"Could not initialize hand tracking: {error}")
            return

        with self._lock:
            self._enabled = True
            self._snapshot = replace(self._snapshot, status="READY")

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled and not self._closed

    @property
    def status(self) -> str:
        with self._lock:
            return self._snapshot.status

    @property
    def error(self) -> str | None:
        with self._lock:
            return self._error

    def _disable(self, status: str, error: str) -> None:
        with self._lock:
            self._enabled = False
            self._error = error
            self._snapshot = replace(self._snapshot, hands=(), status=status)

    def _filter_for(self, identity: str) -> _OneEuroFilter:
        tracker_filter = self._filters.get(identity)
        if tracker_filter is None:
            tracker_filter = _OneEuroFilter(
                ONE_EURO_MIN_CUTOFF,
                ONE_EURO_BETA,
                ONE_EURO_D_CUTOFF,
            )
            self._filters[identity] = tracker_filter
        return tracker_filter

    def _on_result(self, result: Any, _output_image: Any, timestamp_ms: int) -> None:
        processing_started = time.perf_counter()
        try:
            timestamp = timestamp_ms / 1000.0
            raw_hands: list[tuple[str, str, float, np.ndarray]] = []
            handedness_counts: dict[str, int] = {}
            result_hands = getattr(result, "hand_landmarks", ()) or ()
            result_handedness = getattr(result, "handedness", ()) or ()
            for index, hand_landmarks in enumerate(result_hands):
                categories = result_handedness[index] if index < len(result_handedness) else ()
                category = categories[0] if categories else None
                handedness = str(
                    getattr(category, "category_name", "") or f"Hand_{index + 1}"
                )
                occurrence = handedness_counts.get(handedness, 0) + 1
                handedness_counts[handedness] = occurrence
                identity = f"{handedness}:{occurrence}"
                confidence = float(getattr(category, "score", 0.0) or 0.0)
                coordinates = np.asarray(
                    [(point.x, point.y, point.z) for point in hand_landmarks],
                    dtype=np.float64,
                )
                if coordinates.shape == (21, 3) and np.isfinite(coordinates).all():
                    raw_hands.append((identity, handedness, confidence, coordinates))

            with self._lock:
                if self._closed:
                    return
                seen = {identity for identity, _, _, _ in raw_hands}
                for identity in tuple(self._filters):
                    last_seen = self._last_seen.get(identity, timestamp)
                    if identity not in seen and timestamp - last_seen > TRACK_LOST_RESET_SECONDS:
                        self._filters.pop(identity, None)
                        self._last_raw.pop(identity, None)
                        self._last_seen.pop(identity, None)

                hands: list[TrackedHand] = []
                for identity, handedness, confidence, coordinates in raw_hands:
                    raw_coordinates = coordinates.copy()
                    previous = self._last_raw.get(identity)
                    recent = (
                        previous is not None
                        and timestamp - self._last_seen.get(identity, timestamp)
                        < TRACK_LOST_RESET_SECONDS
                    )
                    coherent_jump = False
                    if recent:
                        coordinates, coherent_jump = _stabilize_landmark_outliers(
                            coordinates, previous
                        )
                    self._last_raw[identity] = raw_coordinates
                    if coherent_jump:
                        self._filters.pop(identity, None)
                    filtered = self._filter_for(identity).filter(coordinates, timestamp)
                    self._last_seen[identity] = timestamp
                    landmarks = tuple(
                        Landmark(*map(float, point)) for point in filtered
                    )
                    raw_landmarks = tuple(
                        Landmark(*map(float, point)) for point in raw_coordinates
                    )
                    hands.append(
                        TrackedHand(
                            handedness=handedness,
                            confidence=confidence,
                            landmarks=landmarks,
                            raw_landmarks=raw_landmarks,
                        )
                    )

                captured_at = self._capture_times.pop(timestamp_ms, 0.0)
                submitted_at = self._submitted_at.pop(timestamp_ms, None)
                now = time.perf_counter()
                latency_ms = (
                    0.0 if submitted_at is None else (now - submitted_at) * 1000.0
                )
                # Discard bookkeeping for callbacks MediaPipe skipped internally.
                self._submitted_at = {
                    key: value for key, value in self._submitted_at.items() if key > timestamp_ms
                }
                self._capture_times = {
                    key: value for key, value in self._capture_times.items() if key > timestamp_ms
                }
                self._failure_count = 0
                self._error = None
                self._snapshot = TrackingSnapshot(
                    hands=tuple(hands),
                    sequence=self._snapshot.sequence + 1,
                    timestamp_ms=int(timestamp_ms),
                    latency_ms=latency_ms,
                    received_at=now,
                    processing_ms=(now - processing_started) * 1000.0,
                    captured_at=captured_at,
                    status="TRACKING" if hands else "NO_HANDS",
                )
        except Exception as error:
            with self._lock:
                self._submitted_at.pop(timestamp_ms, None)
                self._capture_times.pop(timestamp_ms, None)
                self._failure_count += 1
                self._error = f"Hand tracking callback failed: {error}"
                repeated = self._failure_count >= max(int(TRACKING_FAILURE_LIMIT), 1)
                self._enabled = True
                self._snapshot = replace(
                    self._snapshot,
                    hands=(),
                    status="DEGRADED" if repeated else "RETRYING",
                    received_at=time.perf_counter(),
                )

    def _expire_lost_callback(self, now: float) -> bool:
        """Release a stuck in-flight slot and surface the loss as status."""

        if not self._submitted_at:
            return False
        oldest = min(self._submitted_at.values())
        if now - oldest <= max(float(TRACKING_CALLBACK_TIMEOUT_SECONDS), 0.1):
            return False
        self._submitted_at.clear()
        self._capture_times.clear()
        self._failure_count += 1
        self._error = "MediaPipe did not return a tracking callback in time."
        self._enabled = True
        self._snapshot = replace(
            self._snapshot,
            hands=(),
            status="CALLBACK_TIMEOUT",
            received_at=now,
        )
        return True

    def submit(self, bgr_frame: np.ndarray, captured_at: float = 0.0) -> bool:
        if self._closed or self._landmarker is None:
            return False
        if not isinstance(bgr_frame, np.ndarray) or bgr_frame.ndim != 3:
            return False
        now = time.perf_counter()
        if now - self._last_submit_time < 0.98 / max(int(TRACKING_MAX_FPS), 1):
            return False
        with self._lock:
            self._expire_lost_callback(now)
            if not self._enabled or self._submitted_at:
                return False
            self._last_submit_time = now

        height, width = bgr_frame.shape[:2]
        if width > TRACKING_MAX_WIDTH:
            target_height = max(1, round(height * TRACKING_MAX_WIDTH / width))
            working = cv2.resize(
                bgr_frame,
                (TRACKING_MAX_WIDTH, target_height),
                interpolation=cv2.INTER_AREA,
            )
        else:
            working = bgr_frame

        try:
            rgb = cv2.cvtColor(working, cv2.COLOR_BGR2RGB)
            image = mp.Image(  # type: ignore[union-attr]
                image_format=mp.ImageFormat.SRGB,  # type: ignore[union-attr]
                data=np.ascontiguousarray(rgb),
            )
            timestamp_ms = (
                time.monotonic_ns() - self._clock_origin_ns
            ) // 1_000_000
            timestamp_ms = max(int(timestamp_ms), self._last_submit_timestamp + 1)
            self._last_submit_timestamp = timestamp_ms
            with self._lock:
                if not self._enabled or self._closed:
                    return False
                self._submitted_at[timestamp_ms] = now
                self._capture_times[timestamp_ms] = float(captured_at)
            self._landmarker.detect_async(image, timestamp_ms)
            return True
        except Exception as error:
            with self._lock:
                if "timestamp_ms" in locals():
                    self._submitted_at.pop(timestamp_ms, None)
                    self._capture_times.pop(timestamp_ms, None)
                self._failure_count += 1
                self._error = f"Could not submit a tracking frame: {error}"
                repeated = self._failure_count >= max(int(TRACKING_FAILURE_LIMIT), 1)
                self._enabled = True
                self._snapshot = replace(
                    self._snapshot,
                    hands=(),
                    status="DEGRADED" if repeated else "RETRYING",
                    received_at=now,
                )
            return False

    def latest(self) -> TrackingSnapshot:
        with self._lock:
            return self._snapshot

    def draw_debug(
        self,
        frame: np.ndarray,
        hands: tuple[TrackedHand, ...] | None = None,
    ) -> None:
        """Draw compact debug landmarks onto a BGR frame in place."""

        visible_hands = self.latest().hands if hands is None else hands
        height, width = frame.shape[:2]
        for hand in visible_hands:
            pixels = [
                (
                    int(np.clip(point.x, 0.0, 1.0) * (width - 1)),
                    int(np.clip(point.y, 0.0, 1.0) * (height - 1)),
                )
                for point in hand.landmarks
            ]
            for start, end in HAND_CONNECTIONS:
                cv2.line(
                    frame, pixels[start], pixels[end], (255, 190, 40), 2, cv2.LINE_AA
                )
            for landmark_index, pixel in enumerate(pixels):
                is_pointer = landmark_index == 8
                cv2.circle(
                    frame,
                    pixel,
                    6 if is_pointer else 3,
                    (40, 255, 255) if is_pointer else (255, 100, 30),
                    -1,
                    cv2.LINE_AA,
                )
            wrist_x, wrist_y = pixels[0]
            cv2.putText(
                frame,
                f"{hand.handedness} {hand.confidence:.2f}",
                (wrist_x + 8, wrist_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (235, 245, 245),
                1,
                cv2.LINE_AA,
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._enabled = False
            self._submitted_at.clear()
            self._capture_times.clear()
            self._snapshot = replace(self._snapshot, hands=(), status="CLOSED")
            landmarker = self._landmarker
            self._landmarker = None
        if landmarker is not None:
            try:
                landmarker.close()
            except Exception:
                pass

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = ["HandTracker", "Landmark", "TrackedHand", "TrackingSnapshot"]
