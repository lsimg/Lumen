"""Optional local faster-whisper STT backend; never downloads a model."""

from __future__ import annotations

from pathlib import Path
import subprocess
import time

import numpy as np

from .stt_vosk import STTResult, merge_segments


class FasterWhisperSTT:
    def __init__(self, model_path: str | Path, sample_rate: int = 16_000,
                 requested_device: str = "auto") -> None:
        self.model_path = Path(model_path)
        self.sample_rate = sample_rate
        self.requested_device = requested_device
        self.device = "cpu"
        self.compute_type = "int8"
        self.load_seconds = 0.0
        self.finalization_seconds = 0.0
        self._model = None

    def initialize(self) -> None:
        if self._model is not None:
            return
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"Local faster-whisper model not found: {self.model_path}")
        try:
            import ctranslate2
            from faster_whisper import WhisperModel
        except ImportError as error:
            raise RuntimeError("Install faster-whisper and ctranslate2 for this optional backend") from error

        cuda = self.requested_device in {"auto", "cuda"} and ctranslate2.get_cuda_device_count() > 0
        if cuda:
            try:
                output = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                    text=True, timeout=2, stderr=subprocess.DEVNULL,
                )
                cuda = int(output.splitlines()[0].strip()) >= 3072
            except Exception:
                cuda = False
        devices = ["cuda", "cpu"] if cuda else ["cpu"]
        started = time.monotonic()
        for device in devices:
            try:
                supported = ctranslate2.get_supported_compute_types(device)
                precision = ("float16" if device == "cuda" and "float16" in supported
                             else "int8_float16" if device == "cuda" and "int8_float16" in supported
                             else "int8")
                self._model = WhisperModel(str(self.model_path), device=device, compute_type=precision)
                self.device, self.compute_type = device, precision
                break
            except Exception:
                if device == "cpu":
                    raise
        self.load_seconds = time.monotonic() - started

    def transcribe(self, pcm16_mono: bytes) -> STTResult:
        self.initialize()
        if not pcm16_mono:
            return STTResult("", "English", 0.0)
        started = time.monotonic()
        audio = np.frombuffer(pcm16_mono, dtype="<i2").astype(np.float32) / 32768.0
        segments, _ = self._model.transcribe(audio, language="en", vad_filter=False, beam_size=3)
        parts = list(segments)
        self.finalization_seconds = time.monotonic() - started
        return STTResult(merge_segments([part.text for part in parts]), "English",
                         sum(part.avg_logprob for part in parts) / len(parts) if parts else 0.0)

    def shutdown(self) -> None:
        self._model = None


__all__ = ["FasterWhisperSTT"]
