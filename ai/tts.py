"""Local PyTorch TTS abstraction used by Jeremy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from .settings import JeremySettings


@dataclass(frozen=True, slots=True)
class SpeechAudio:
    pcm16_mono: bytes
    sample_rate: int


class JeremyTTS:
    """Load configured local speech models once; currently implements Silero."""

    def __init__(self, settings: JeremySettings) -> None:
        self.settings = settings
        self.engine = settings.tts_engine
        self.device = settings.tts_device
        self._models: dict[str, Any] = {}
        self._speaker = settings.silero_en_speaker
        self.load_seconds = 0.0
        self.synthesis_seconds = 0.0
        self.gpu_vram_bytes = 0
        self.benchmark_seconds: dict[str, float] = {}

    @staticmethod
    def _load_package(path: Path) -> Any:
        if not path.is_file():
            raise FileNotFoundError(f"Silero TTS model not found: {path}")
        return torch.package.PackageImporter(str(path)).load_pickle("tts_models", "model")

    def initialize(self) -> None:
        if self._models:
            return
        if self.engine != "silero":
            raise RuntimeError(
                f"Unsupported TTS_ENGINE={self.engine!r}; this Python 3.12 build uses silero"
            )
        if torch is None:
            raise RuntimeError("PyTorch is not installed")
        started = time.monotonic()
        requested = self.device
        candidates: list[str] = []
        if requested in {"auto", "cuda"} and torch.cuda.is_available():
            try:
                free_bytes, _ = torch.cuda.mem_get_info()
                if free_bytes >= 2 * 1024**3:
                    candidates.append("cuda")
            except Exception:
                pass
        if requested in {"auto", "mps"} and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            candidates.append("mps")
        candidates.append("cpu")

        model = None
        for device in candidates:
            try:
                before = torch.cuda.memory_allocated() if device == "cuda" else 0
                model = self._load_package(Path(self.settings.silero_en_model_path))
                if hasattr(model, "to"):
                    model.to(device)
                with torch.inference_mode():
                    model.apply_tts(text="Ready.", speaker=self._speaker,
                                    sample_rate=self.settings.tts_sample_rate)
                self.device = device
                self.gpu_vram_bytes = max(0, torch.cuda.memory_allocated() - before) if device == "cuda" else 0
                break
            except Exception:
                model = None
                if device == "cuda":
                    torch.cuda.empty_cache()
                if device == "cpu":
                    raise
        assert model is not None
        speakers = tuple(str(value) for value in getattr(model, "speakers", ()) or ())
        if self._speaker not in speakers:
            raise RuntimeError(
                f"Silero English speaker {self._speaker!r} is unavailable; "
                f"available: {', '.join(speakers) or '<not reported>'}"
            )
        self._models["en"] = model
        if self.settings.benchmark:
            self._benchmark()
        self.load_seconds = time.monotonic() - started

    def _benchmark(self) -> None:
        def run(candidate: Any, device: str) -> float:
            started = time.monotonic()
            with torch.inference_mode():
                candidate.apply_tts(text="The CPU processes instructions.", speaker=self._speaker,
                                    sample_rate=self.settings.tts_sample_rate)
            if device == "cuda":
                torch.cuda.synchronize()
            return time.monotonic() - started

        model = self._models["en"]
        run(model, self.device)
        self.benchmark_seconds[self.device] = run(model, self.device)
        if self.device == "cpu":
            return
        cpu_model = self._load_package(Path(self.settings.silero_en_model_path))
        cpu_model.to("cpu")
        run(cpu_model, "cpu")  # warm-up before comparison
        self.benchmark_seconds["cpu"] = run(cpu_model, "cpu")
        if self.settings.tts_device == "auto" and self.benchmark_seconds["cpu"] < self.benchmark_seconds[self.device] * .9:
            self._models["en"] = cpu_model
            self.device = "cpu"
            self.gpu_vram_bytes = 0
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def synthesize(self, text: str, language: str = "English") -> SpeechAudio:
        if not self._models:
            self.initialize()
        model = self._models["en"]
        started = time.monotonic()
        with torch.inference_mode():
            audio = model.apply_tts(
                text=str(text),
                speaker=self._speaker,
                sample_rate=self.settings.tts_sample_rate,
            )
        samples = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        pcm = np.clip(np.rint(samples * 32767.0), -32768, 32767).astype("<i2")
        self.synthesis_seconds = time.monotonic() - started
        return SpeechAudio(pcm.tobytes(), self.settings.tts_sample_rate)

    def shutdown(self) -> None:
        self._models.clear()
        if torch is not None and self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = ["JeremyTTS", "SpeechAudio"]
