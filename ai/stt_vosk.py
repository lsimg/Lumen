"""Local English-only Vosk speech recognition for a completed utterance."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from statistics import fmean
import time
from typing import Any

try:
    from vosk import KaldiRecognizer, Model, SetLogLevel
except ImportError:
    KaldiRecognizer = Model = None  # type: ignore[assignment]
    SetLogLevel = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class STTResult:
    text: str
    language: str
    confidence: float


def merge_segments(segments: list[str]) -> str:
    """Join finalized Vosk segments, removing only exact repeated word overlap."""
    merged: list[str] = []
    for segment in segments:
        words = str(segment or "").split()
        if not words:
            continue
        limit = min(len(merged), len(words))
        overlap = next(
            (size for size in range(limit, 0, -1)
             if [word.casefold() for word in merged[-size:]] ==
                [word.casefold() for word in words[:size]]), 0
        )
        merged.extend(words[overlap:])
    return " ".join(merged).strip()


class VoskSTT:
    """Load one large English model once and finalize each captured utterance."""

    def __init__(self, en_model_path: str | Path, sample_rate: int = 16_000) -> None:
        self.en_model_path = Path(en_model_path)
        self.sample_rate = int(sample_rate)
        self._model: Any | None = None
        self.device = "cpu"  # The installed Vosk Python recognizer has no CUDA backend.
        self.load_seconds = 0.0
        self.finalization_seconds = 0.0

    def initialize(self) -> None:
        if self._model is not None:
            return
        if Model is None or KaldiRecognizer is None:
            raise RuntimeError("vosk is not installed")
        if not (self.en_model_path / "am" / "final.mdl").is_file():
            raise FileNotFoundError(
                f"JEREMY / ENGLISH SPEECH MODEL NOT FOUND: {self.en_model_path}"
            )
        if SetLogLevel is not None:
            SetLogLevel(-1)
        started = time.monotonic()
        self._model = Model(str(self.en_model_path))
        self.load_seconds = time.monotonic() - started

    def transcribe(self, pcm16_mono: bytes) -> STTResult:
        if self._model is None:
            self.initialize()
        if not pcm16_mono:
            return STTResult("", "English", 0.0)
        recognizer = KaldiRecognizer(self._model, float(self.sample_rate))
        recognizer.SetWords(True)
        started = time.monotonic()
        payloads: list[dict[str, Any]] = []
        for offset in range(0, len(pcm16_mono), 8_000):
            if recognizer.AcceptWaveform(pcm16_mono[offset : offset + 8_000]):
                payloads.append(json.loads(recognizer.Result() or "{}"))
        payloads.append(json.loads(recognizer.FinalResult() or "{}"))
        self.finalization_seconds = time.monotonic() - started
        text = merge_segments([str(payload.get("text", "")) for payload in payloads])
        words = [word for payload in payloads for word in (payload.get("result") or [])]
        confidences = [float(word.get("conf", 0.0)) for word in words if isinstance(word, dict)]
        confidence = fmean(confidences) if confidences else 0.0
        return STTResult(text=text, language="English", confidence=confidence)

    def shutdown(self) -> None:
        self._model = None


__all__ = ["STTResult", "VoskSTT", "merge_segments"]
