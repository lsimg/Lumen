"""Non-blocking microphone capture, in-memory WAV encoding, and PCM playback."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import io
import queue
import threading
import time
import wave

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None  # type: ignore[assignment]

try:
    import soundfile as sf
except ImportError:
    sf = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class InputDeviceInfo:
    index: int
    name: str
    max_input_channels: int
    default_sample_rate: int


def pcm_rms(pcm16_mono: bytes) -> float:
    samples = np.frombuffer(pcm16_mono, dtype="<i2")
    if not len(samples):
        return 0.0
    normalized = samples.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(normalized * normalized)))


def resample_pcm16_mono(pcm16_mono: bytes, source_rate: int, target_rate: int) -> bytes:
    """Convert mono PCM16 without pulling a heavyweight DSP dependency."""

    source_rate, target_rate = int(source_rate), int(target_rate)
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("Audio sample rates must be positive")
    if source_rate == target_rate or not pcm16_mono:
        return bytes(pcm16_mono)
    source = np.frombuffer(pcm16_mono, dtype="<i2")
    if len(source) < 2:
        return bytes(source.astype("<i2").tobytes())
    target_frames = max(1, int(round(len(source) * target_rate / source_rate)))
    source_positions = np.arange(len(source), dtype=np.float64)
    target_positions = np.linspace(0.0, len(source) - 1, target_frames)
    converted = np.interp(target_positions, source_positions, source.astype(np.float64))
    return np.clip(np.rint(converted), -32768, 32767).astype("<i2").tobytes()


def pcm_to_wav(pcm16_mono: bytes, sample_rate: int) -> bytes:
    """Return a mono, 16-bit PCM WAV held entirely in memory."""

    target = io.BytesIO()
    if sf is not None:
        samples = np.frombuffer(pcm16_mono, dtype="<i2")
        sf.write(target, samples, int(sample_rate), format="WAV", subtype="PCM_16")
        return target.getvalue()
    # Lumen can still launch if an installation was interrupted. The normal
    # requirements install soundfile and use the branch above.
    with wave.open(target, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(int(sample_rate))
        output.writeframes(pcm16_mono)
    return target.getvalue()


def input_devices() -> tuple[InputDeviceInfo, ...]:
    if sd is None:
        return ()
    result: list[InputDeviceInfo] = []
    for index, raw in enumerate(sd.query_devices()):
        channels = int(raw.get("max_input_channels", 0))
        if channels <= 0:
            continue
        result.append(
            InputDeviceInfo(
                index=index,
                name=str(raw.get("name", f"Input device {index}")),
                max_input_channels=channels,
                default_sample_rate=int(round(float(raw.get("default_samplerate", 0) or 0))),
            )
        )
    return tuple(result)


def _default_input_index() -> int | None:
    if sd is None:
        return None
    default = sd.default.device
    try:
        value = default[0]
    except (IndexError, TypeError):
        value = default
    try:
        index = int(value)
    except (TypeError, ValueError):
        return None
    return index if index >= 0 else None


def _device_candidates(target_sample_rate: int) -> tuple[tuple[InputDeviceInfo, int], ...]:
    """Return working input configurations, preferring Windows' default mic."""

    if sd is None:
        return ()
    devices = input_devices()
    by_index = {device.index: device for device in devices}
    default_index = _default_input_index()
    ordered_indexes = list(by_index)
    if default_index in by_index:
        ordered_indexes.remove(default_index)
        ordered_indexes.insert(0, default_index)

    compatible: list[tuple[InputDeviceInfo, int]] = []
    for index in ordered_indexes:
        device = by_index[index]
        rates = [int(target_sample_rate)]
        if device.default_sample_rate > 0 and device.default_sample_rate not in rates:
            rates.append(device.default_sample_rate)
        for rate in rates:
            try:
                sd.check_input_settings(
                    device=device.index,
                    channels=1,
                    dtype="int16",
                    samplerate=rate,
                )
            except Exception:
                continue
            compatible.append((device, rate))
    return tuple(compatible)


class MicrophoneInput:
    """PortAudio callback that only resamples and copies into a bounded queue."""

    def __init__(self, sample_rate: int = 16_000, block_seconds: float = 0.08) -> None:
        self.sample_rate = int(sample_rate)
        self.block_seconds = max(0.02, float(block_seconds))
        self.block_frames = max(160, int(self.sample_rate * self.block_seconds))
        self.device_index: int | None = None
        self.device_name = ""
        self.device_sample_rate = self.sample_rate
        self._blocks: queue.Queue[bytes] = queue.Queue(maxsize=96)
        self._stream = None
        self._lock = threading.Lock()
        self.error: str | None = None

    @property
    def available(self) -> bool:
        return sd is not None

    def _callback(self, indata: object, _frames: int, _time_info: object, status: object) -> None:
        if status:
            self.error = f"Microphone stream warning: {status}"
        block = bytes(indata)
        if self.device_sample_rate != self.sample_rate:
            block = resample_pcm16_mono(block, self.device_sample_rate, self.sample_rate)
        try:
            self._blocks.put_nowait(block)
        except queue.Full:
            try:
                self._blocks.get_nowait()
            except queue.Empty:
                pass
            try:
                self._blocks.put_nowait(block)
            except queue.Full:
                pass

    def start(self) -> None:
        if sd is None:
            raise RuntimeError("sounddevice is not installed")
        with self._lock:
            if self._stream is not None:
                return
            candidates = _device_candidates(self.sample_rate)
            if not candidates:
                raise RuntimeError("No working microphone/input device was found")
            errors: list[str] = []
            for device, device_rate in candidates:
                stream = None
                try:
                    block_frames = max(160, int(device_rate * self.block_seconds))
                    stream = sd.RawInputStream(
                        device=device.index,
                        samplerate=device_rate,
                        blocksize=block_frames,
                        channels=1,
                        dtype="int16",
                        callback=self._callback,
                    )
                    self.device_index = device.index
                    self.device_name = device.name
                    self.device_sample_rate = device_rate
                    self.block_frames = block_frames
                    stream.start()
                    # Some Windows host APIs report a successful open but leave
                    # the stream inactive and never invoke the callback. Do not
                    # select such a pseudo-input as Jeremy's microphone.
                    deadline = time.monotonic() + 0.75
                    first_block: bytes | None = None
                    while time.monotonic() < deadline:
                        try:
                            first_block = self._blocks.get(timeout=0.05)
                            break
                        except queue.Empty:
                            if not stream.active:
                                break
                    if first_block is None:
                        raise RuntimeError("stream opened but produced no audio blocks")
                    self._blocks.put_nowait(first_block)
                    self._stream = stream
                    self.error = None
                    return
                except Exception as error:
                    errors.append(f"{device.index} ({device.name}): {error}")
                    if stream is not None:
                        try:
                            if stream.active:
                                stream.stop()
                        except Exception:
                            pass
                        try:
                            stream.close()
                        except Exception:
                            pass
                    self.flush()
            raise RuntimeError("Microphone unavailable: " + " | ".join(errors))

    def read(self, timeout: float = 0.1) -> bytes | None:
        try:
            return self._blocks.get(timeout=max(float(timeout), 0.0))
        except queue.Empty:
            return None

    def flush(self) -> None:
        while True:
            try:
                self._blocks.get_nowait()
            except queue.Empty:
                return

    def close(self) -> None:
        with self._lock:
            stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
            except Exception as error:
                self.error = f"Could not stop microphone stream: {error}"
            try:
                stream.close()
            except Exception as error:
                self.error = f"Could not close microphone stream: {error}"
        self.flush()


class AudioRecorder:
    """Fixed-duration recorder used by the standalone microphone diagnostic."""

    def __init__(self, microphone: MicrophoneInput | None = None, sample_rate: int = 16_000) -> None:
        self.microphone = microphone or MicrophoneInput(sample_rate)
        self.sample_rate = self.microphone.sample_rate

    def record(self, duration_seconds: float) -> bytes:
        self.microphone.start()
        self.microphone.flush()
        blocks: list[bytes] = []
        duration = max(0.1, float(duration_seconds))
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            block = self.microphone.read(timeout=min(0.1, max(0.0, deadline - time.monotonic())))
            if block:
                blocks.append(block)
        target_frames = int(duration * self.sample_rate)
        return b"".join(blocks)[: target_frames * 2]

    def close(self) -> None:
        self.microphone.close()


class AudioPlayer:
    def play_pcm(self, pcm16_mono: bytes, sample_rate: int) -> None:
        if sd is None:
            raise RuntimeError("sounddevice is not installed")
        if not pcm16_mono:
            raise RuntimeError("TTS returned empty audio")
        with sd.RawOutputStream(
            samplerate=int(sample_rate), channels=1, dtype="int16"
        ) as output:
            output.write(bytes(pcm16_mono))


class UtteranceRecorder:
    """Tiny RMS-based endpoint detector used only after the local wake word."""

    def __init__(
        self,
        sample_rate: int,
        *,
        rms_threshold: float,
        silence_seconds: float,
        max_seconds: float,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.rms_threshold = float(rms_threshold)
        self.silence_seconds = float(silence_seconds)
        self.max_seconds = float(max_seconds)
        self._blocks: list[bytes] = []
        self._frames = 0
        self._silent_frames = 0
        self.speech_started = False
        self.last_end_reason = ""

    def start(self, pre_roll: tuple[bytes, ...] = (), *, speech_started: bool = False) -> None:
        self._blocks = list(pre_roll)
        self._frames = sum(len(block) // 2 for block in pre_roll)
        self._silent_frames = 0
        self.speech_started = bool(speech_started)

    def add(self, block: bytes) -> bytes | None:
        if not block:
            return None
        frames = len(block) // 2
        self._blocks.append(block)
        self._frames += frames
        if pcm_rms(block) >= self.rms_threshold:
            self.speech_started = True
            self._silent_frames = 0
        elif self.speech_started:
            self._silent_frames += frames

        total_seconds = self._frames / max(self.sample_rate, 1)
        silence = self._silent_frames / max(self.sample_rate, 1)
        if self.speech_started and (
            silence >= self.silence_seconds or total_seconds >= self.max_seconds
        ):
            result = b"".join(self._blocks)
            reason = "silence" if silence >= self.silence_seconds else "max"
            self.start()
            self.last_end_reason = reason
            return result
        return None


def recent_audio_buffer(max_blocks: int = 16) -> deque[bytes]:
    return deque(maxlen=max(1, int(max_blocks)))


__all__ = [
    "AudioPlayer",
    "AudioRecorder",
    "InputDeviceInfo",
    "MicrophoneInput",
    "UtteranceRecorder",
    "input_devices",
    "pcm_rms",
    "pcm_to_wav",
    "recent_audio_buffer",
    "resample_pcm16_mono",
]
