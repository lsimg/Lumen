"""Small, explicit gesture recognizer for Lumen's study controls."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np

from config import (
    CONTROL_MIN_CONFIDENCE,
    GESTURE_STABILITY_FRAMES,
    GESTURE_STABILITY_SECONDS,
    LEADING_HAND,
    PINCH_OFF_RATIO,
    PINCH_ON_RATIO,
)
from core.tracking import TrackedHand, TrackingSnapshot


class Gesture(str, Enum):
    NONE = "NONE"
    POINT = "POINT"
    PINCH = "PINCH"
    OPEN_PALM = "OPEN_PALM"
    V_SIGN = "V_SIGN"
    FIST = "FIST"
    TWO_HANDS = "TWO_HANDS"
    ROCK_SIGN = "ROCK_SIGN"


@dataclass(frozen=True)
class GestureFrame:
    """One stabilized interpretation of the latest tracking result."""

    gesture: Gesture
    pointer: tuple[float, float] | None
    palm_center: tuple[float, float] | None
    two_hand_distance: float | None
    pinch_active: bool
    hands: tuple[TrackedHand, ...]
    timestamp: float
    confidence: float

    @property
    def hand_count(self) -> int:
        return len(self.hands)


class GestureRecognizer:
    """Classify filtered landmarks with pinch hysteresis and pose debounce."""

    def __init__(self) -> None:
        self._pinch_active = False
        self._stable_gesture = Gesture.NONE
        self._candidate_gesture: Gesture | None = None
        self._candidate_since = 0.0
        self._candidate_frames = 0

    @property
    def pinch_active(self) -> bool:
        return self._pinch_active

    @property
    def stable_gesture(self) -> Gesture:
        return self._stable_gesture

    def reset(self) -> None:
        self._pinch_active = False
        self._stable_gesture = Gesture.NONE
        self._candidate_gesture = None
        self._candidate_since = 0.0
        self._candidate_frames = 0

    @staticmethod
    def _is_valid(hand: TrackedHand) -> bool:
        if hand.confidence < CONTROL_MIN_CONFIDENCE or len(hand.landmarks) != 21:
            return False
        return bool(
            np.isfinite(
                [(landmark.x, landmark.y, landmark.z) for landmark in hand.landmarks]
            ).all()
        )

    @staticmethod
    def _pointer(hand: TrackedHand) -> tuple[float, float]:
        return (
            float(np.clip(hand.index_tip.x, 0.0, 1.0)),
            float(np.clip(hand.index_tip.y, 0.0, 1.0)),
        )

    @staticmethod
    def _palm_center(hand: TrackedHand) -> tuple[float, float]:
        return hand.palm_center.x, hand.palm_center.y

    @staticmethod
    def _leading_hand(hands: tuple[TrackedHand, ...]) -> TrackedHand:
        preferred = str(LEADING_HAND).casefold()
        return next(
            (
                hand
                for hand in hands
                if hand.handedness.casefold() == preferred
            ),
            hands[0],
        )

    def _stabilize(self, raw_gesture: Gesture, timestamp: float) -> Gesture:
        if raw_gesture == self._stable_gesture:
            self._candidate_gesture = None
            self._candidate_frames = 0
            return self._stable_gesture
        if raw_gesture != self._candidate_gesture:
            self._candidate_gesture = raw_gesture
            self._candidate_since = timestamp
            self._candidate_frames = 1
            return self._stable_gesture

        self._candidate_frames += 1
        stable_for = timestamp - self._candidate_since
        if (
            self._candidate_frames >= max(int(GESTURE_STABILITY_FRAMES), 1)
            and stable_for >= max(float(GESTURE_STABILITY_SECONDS), 0.0)
        ):
            self._stable_gesture = raw_gesture
            self._candidate_gesture = None
            self._candidate_frames = 0
        return self._stable_gesture

    def _single_hand_gesture(self, hand: TrackedHand) -> Gesture:
        threshold = PINCH_OFF_RATIO if self._pinch_active else PINCH_ON_RATIO
        self._pinch_active = hand.pinch_ratio < threshold

        # STOP must win over a small thumb-index distance in a closed fist.
        if hand.fist:
            self._pinch_active = False
            return Gesture.FIST
        if hand.rock_sign:
            self._pinch_active = False
            return Gesture.ROCK_SIGN
        if self._pinch_active:
            return Gesture.PINCH
        if hand.open_palm:
            return Gesture.OPEN_PALM
        if hand.v_sign:
            return Gesture.V_SIGN
        if hand.point:
            return Gesture.POINT
        return Gesture.NONE

    def update(self, snapshot: TrackingSnapshot, timestamp: float) -> GestureFrame:
        now = float(timestamp)
        if not math.isfinite(now):
            now = 0.0
        hands = tuple(hand for hand in snapshot.hands if self._is_valid(hand))

        if not hands:
            self.reset()
            return GestureFrame(
                gesture=Gesture.NONE,
                pointer=None,
                palm_center=None,
                two_hand_distance=None,
                pinch_active=False,
                hands=(),
                timestamp=now,
                confidence=0.0,
            )

        leading = self._leading_hand(hands)
        pointer = self._pointer(leading)
        palm_center = self._palm_center(leading)
        confidence = sum(hand.confidence for hand in hands) / len(hands)

        # STOP is an emergency gesture and must win even when a second hand is
        # also visible to the tracker.
        if any(hand.fist for hand in hands):
            self._pinch_active = False
            gesture = self._stabilize(Gesture.FIST, now)
            return GestureFrame(
                gesture=gesture,
                pointer=pointer,
                palm_center=palm_center,
                two_hand_distance=None,
                pinch_active=False,
                hands=hands,
                timestamp=now,
                confidence=float(confidence),
            )

        if len(hands) >= 2:
            self._pinch_active = False
            first, second = hands[:2]
            two_hand_distance = math.hypot(
                first.index_tip.x - second.index_tip.x,
                first.index_tip.y - second.index_tip.y,
            )
            gesture = self._stabilize(Gesture.TWO_HANDS, now)
            return GestureFrame(
                gesture=gesture,
                pointer=pointer,
                palm_center=palm_center,
                two_hand_distance=float(two_hand_distance),
                pinch_active=False,
                hands=hands,
                timestamp=now,
                confidence=float(confidence),
            )

        raw_gesture = self._single_hand_gesture(leading)
        gesture = self._stabilize(raw_gesture, now)
        return GestureFrame(
            gesture=gesture,
            pointer=pointer,
            palm_center=palm_center,
            two_hand_distance=None,
            pinch_active=self._pinch_active,
            hands=hands,
            timestamp=now,
            confidence=float(confidence),
        )


__all__ = ["Gesture", "GestureFrame", "GestureRecognizer"]
