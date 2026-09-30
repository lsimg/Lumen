"""Resilient, low-latency webcam capture for Lumen.

The capture thread always has a valid frame to return.  If a camera is absent,
busy, or briefly stops producing frames, the application keeps running while
this module reconnects in the background.
"""

from __future__ import annotations

from dataclasses import dataclass
import sys
import threading
import time

import cv2
import numpy as np

from config import (
    CAMERA_AUTO_EXPOSURE,
    CAMERA_AUTOFOCUS,
    CAMERA_FAILURE_TOLERANCE,
    CAMERA_MANUAL_EXPOSURE,
    CAMERA_MIRROR,
    CAMERA_PREFER_MJPG,
    CAMERA_RETRY_SECONDS,
)


@dataclass(frozen=True)
class CameraFrame:
    """Latest BGR camera image and its capture timing."""

    image: np.ndarray
    sequence: int
    captured_at: float
    capture_ms: float = 0.0


@dataclass(frozen=True)
class CameraInfo:
    """Properties reported by the active capture backend."""

    backend: str
    width: int
    height: int
    requested_fps: int
    reported_fps: float
    format_name: str


def _fourcc_name(value: float) -> str:
    integer = max(int(value), 0)
    text = "".join(chr((integer >> (8 * index)) & 0xFF) for index in range(4))
    return text if text and all(32 <= ord(character) < 127 for character in text) else "NATIVE"


class CameraCapture:
    """Continuously publish the newest webcam frame without blocking rendering."""

    def __init__(self, index: int, width: int, height: int, fps: int) -> None:
        self.index = int(index)
        self.requested_width = max(int(width), 1)
        self.requested_height = max(int(height), 1)
        self.requested_fps = max(int(fps), 1)

        placeholder = np.zeros(
            (self.requested_height, self.requested_width, 3), dtype=np.uint8
        )
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._capture: cv2.VideoCapture | None = None
        self._latest = CameraFrame(placeholder, 0, time.perf_counter(), 0.0)
        self._info = CameraInfo(
            backend="UNAVAILABLE",
            width=self.requested_width,
            height=self.requested_height,
            requested_fps=self.requested_fps,
            reported_fps=0.0,
            format_name="NATIVE",
        )
        self._status = "CONNECTING"
        self._error: str | None = None
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="lumen-camera",
            daemon=True,
        )
        self._thread.start()

    @property
    def info(self) -> CameraInfo:
        with self._lock:
            return self._info

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    @property
    def error(self) -> str | None:
        with self._lock:
            return self._error

    def _set_state(self, status: str, error: str | None = None) -> None:
        with self._lock:
            self._status = status
            self._error = error

    @staticmethod
    def _backend_name(capture: cv2.VideoCapture) -> str:
        try:
            return str(capture.getBackendName())
        except (cv2.error, AttributeError):
            return "AUTO"

    @staticmethod
    def _configure_capture(
        capture: cv2.VideoCapture,
        width: int,
        height: int,
        fps: int,
        *,
        backend: int,
        use_mjpg: bool,
    ) -> None:
        if use_mjpg:
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if hasattr(cv2, "CAP_PROP_AUTOFOCUS"):
            capture.set(cv2.CAP_PROP_AUTOFOCUS, 1.0 if CAMERA_AUTOFOCUS else 0.0)
        if hasattr(cv2, "CAP_PROP_AUTO_WB"):
            capture.set(cv2.CAP_PROP_AUTO_WB, 1.0)
        if hasattr(cv2, "CAP_PROP_AUTO_EXPOSURE"):
            automatic = CAMERA_AUTO_EXPOSURE and CAMERA_MANUAL_EXPOSURE is None
            if sys.platform == "win32" and backend == cv2.CAP_DSHOW:
                capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if automatic else 0.25)
            else:
                capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0 if automatic else 0.0)
        if CAMERA_MANUAL_EXPOSURE is not None:
            capture.set(cv2.CAP_PROP_EXPOSURE, float(CAMERA_MANUAL_EXPOSURE))

    def _capture_attempts(self) -> tuple[tuple[int, bool], ...]:
        backends = (
            (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY)
            if sys.platform == "win32"
            else (cv2.CAP_ANY,)
        )
        attempts: list[tuple[int, bool]] = []
        for backend in backends:
            if CAMERA_PREFER_MJPG and backend != cv2.CAP_ANY:
                attempts.append((backend, True))
            attempts.append((backend, False))
        return tuple(dict.fromkeys(attempts))

    def _open_camera(
        self,
    ) -> tuple[cv2.VideoCapture | None, np.ndarray | None, float]:
        for backend, use_mjpg in self._capture_attempts():
            if self._stop.is_set():
                break
            capture: cv2.VideoCapture | None = None
            try:
                capture = cv2.VideoCapture(self.index, backend)
                if not capture.isOpened():
                    capture.release()
                    continue
                self._configure_capture(
                    capture,
                    self.requested_width,
                    self.requested_height,
                    self.requested_fps,
                    backend=backend,
                    use_mjpg=use_mjpg,
                )
                image: np.ndarray | None = None
                read_ms = 0.0
                # A few startup reads let exposure settle without blocking app startup,
                # because this method only runs on the capture thread.
                for _ in range(4):
                    started = time.perf_counter()
                    ok, candidate = capture.read()
                    read_ms = (time.perf_counter() - started) * 1000.0
                    if ok and candidate is not None:
                        image = candidate
                if image is None:
                    capture.release()
                    continue
                return capture, image, read_ms
            except (cv2.error, OSError, RuntimeError):
                if capture is not None:
                    capture.release()
        return None, None, 0.0

    def _publish(self, image: np.ndarray, capture_ms: float) -> None:
        frame = cv2.flip(image, 1) if CAMERA_MIRROR else image
        if frame.ndim != 3 or frame.shape[2] != 3:
            return
        captured_at = time.perf_counter()
        with self._lock:
            self._latest = CameraFrame(
                np.ascontiguousarray(frame),
                self._latest.sequence + 1,
                captured_at,
                float(capture_ms),
            )

    def _capture_loop(self) -> None:
        capture: cv2.VideoCapture | None = None
        failed_reads = 0
        try:
            while not self._stop.is_set():
                if capture is None:
                    self._set_state("CONNECTING")
                    capture, first_image, read_ms = self._open_camera()
                    if capture is None or first_image is None:
                        self._set_state(
                            "UNAVAILABLE",
                            f"Camera {self.index} is unavailable; Lumen will keep retrying.",
                        )
                        self._stop.wait(max(float(CAMERA_RETRY_SECONDS), 0.1))
                        continue

                    with self._lock:
                        self._capture = capture
                        self._info = CameraInfo(
                            backend=self._backend_name(capture),
                            width=int(
                                capture.get(cv2.CAP_PROP_FRAME_WIDTH)
                                or first_image.shape[1]
                            ),
                            height=int(
                                capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
                                or first_image.shape[0]
                            ),
                            requested_fps=self.requested_fps,
                            reported_fps=float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
                            format_name=_fourcc_name(capture.get(cv2.CAP_PROP_FOURCC)),
                        )
                    self._publish(first_image, read_ms)
                    self._set_state("READY")
                    failed_reads = 0

                started = time.perf_counter()
                try:
                    ok, image = capture.read()
                except (cv2.error, OSError, RuntimeError):
                    ok, image = False, None
                capture_ms = (time.perf_counter() - started) * 1000.0
                if ok and image is not None:
                    failed_reads = 0
                    self._publish(image, capture_ms)
                    if self.status != "READY":
                        self._set_state("READY")
                    continue

                failed_reads += 1
                self._set_state(
                    "RECONNECTING",
                    f"Camera {self.index} temporarily stopped returning frames.",
                )
                if failed_reads < max(int(CAMERA_FAILURE_TOLERANCE), 1):
                    self._stop.wait(0.01)
                    continue

                capture.release()
                capture = None
                failed_reads = 0
                with self._lock:
                    self._capture = None
                self._stop.wait(max(float(CAMERA_RETRY_SECONDS), 0.1))
        except Exception as error:  # A backend must never terminate the application.
            self._set_state("UNAVAILABLE", f"Camera error: {error}")
        finally:
            if capture is not None:
                capture.release()
            with self._lock:
                self._capture = None
                if self._status != "CLOSED":
                    self._status = "CLOSED" if self._stop.is_set() else "UNAVAILABLE"

    def latest(self) -> CameraFrame:
        with self._lock:
            return self._latest

    def close(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        self._set_state("CLOSED")
        with self._lock:
            capture = self._capture
        # Releasing from the caller can unblock a backend stuck in read().
        if capture is not None:
            capture.release()
        self._thread.join(timeout=1.5)

    def __enter__(self) -> "CameraCapture":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = ["CameraCapture", "CameraFrame", "CameraInfo"]
