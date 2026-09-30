"""Environment-backed configuration for Jeremy's hybrid voice pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # Lumen itself must still start after an interrupted install.
    load_dotenv = None  # type: ignore[assignment]


def _float_env(name: str, fallback: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, fallback))
    except (TypeError, ValueError):
        return fallback
    return max(minimum, min(maximum, value))


def _int_env(name: str, fallback: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, fallback))
    except (TypeError, ValueError):
        return fallback
    return max(minimum, min(maximum, value))


def _project_path(project_dir: Path, name: str, fallback: str) -> Path:
    value = os.getenv(name, "").strip() or fallback
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_dir / path).resolve()


def _discover_english_vosk_model(project_dir: Path, configured: Path) -> Path:
    """Find the largest valid English model inside/next to the Lumen project."""
    candidates: set[Path] = set()
    if (configured / "am" / "final.mdl").is_file():
        candidates.add(configured.resolve())
    for base in (project_dir, *project_dir.parents[:3]):
        if not base.is_dir():
            continue
        for pattern in (
            "vosk-model*",
            "models/vosk*",
            "models/speech/vosk/*",
            "models/voice_models/*",
            "models/speech_models/*",
            "assets/ai_models/vosk/*",
        ):
            for path in base.glob(pattern):
                if (
                    path.is_dir()
                    and "en" in path.name.casefold()
                    and (path / "am" / "final.mdl").is_file()
                ):
                    candidates.add(path.resolve())
    if not candidates:
        return configured

    def model_size(path: Path) -> int:
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())

    # Explicitly insist on a large model. Never silently fall back to a compact one.
    large = [(model_size(path), path) for path in candidates]
    large = [(size, path) for size, path in large if size >= 500_000_000]
    if not large:
        return configured
    return max(large, key=lambda item: item[0])[1]


@dataclass(frozen=True, slots=True)
class JeremySettings:
    api_key: str
    llm_model: str
    fallback_models: tuple[str, ...] = ("openai/gpt-4.1-mini", "mistralai/mistral-small-3.2-24b-instruct")
    project_dir: Path = Path(".")
    vosk_en_model_path: Path = Path("../vosk-model-en-us-0.22")
    stt_backend: str = "vosk"
    stt_device: str = "auto"
    whisper_model_path: Path = Path("assets/ai_models/faster-whisper")
    tts_engine: str = "silero"
    silero_en_model_path: Path = Path("assets/ai_models/silero/v3_en.pt")
    silero_en_speaker: str = "en_1"
    tts_device: str = "auto"
    tts_sample_rate: int = 24_000
    microphone_sample_rate: int = 16_000
    listening_timeout: float = 7.0
    utterance_silence_seconds: float = 1.35
    utterance_max_seconds: float = 25.0
    speech_rms_threshold: float = 0.004
    wake_threshold: float = 1e-20
    request_timeout: float = 9.0
    connect_timeout: float = 3.0
    total_timeout: float = 28.0
    request_retries: int = 1
    max_response_tokens: int = 180
    debug: bool = False
    benchmark: bool = False

    @classmethod
    def load(cls, project_dir: str | Path) -> "JeremySettings":
        project = Path(project_dir).resolve()
        env_path = project / ".env"
        if load_dotenv is not None:
            load_dotenv(dotenv_path=env_path, override=False)

        en_path = _project_path(
            project,
            "VOSK_EN_MODEL_PATH",
            "../vosk-model-en-us-0.22",
        )
        return cls(
            api_key=os.getenv("OPENROUTER_API_KEY", "").strip(),
            llm_model=os.getenv("OPENROUTER_LLM_MODEL", "google/gemini-3.1-flash-lite").strip()
            or "google/gemini-3.1-flash-lite",
            fallback_models=tuple(dict.fromkeys(
                value.strip() for value in os.getenv(
                    "OPENROUTER_FALLBACK_MODELS", "openai/gpt-4.1-mini,mistralai/mistral-small-3.2-24b-instruct"
                ).split(",") if value.strip()
            )),
            project_dir=project,
            vosk_en_model_path=_discover_english_vosk_model(project, en_path),
            stt_backend=os.getenv("JEREMY_STT_BACKEND", "vosk").strip().casefold() or "vosk",
            stt_device=os.getenv("JEREMY_STT_DEVICE", "auto").strip().casefold() or "auto",
            whisper_model_path=_project_path(project, "JEREMY_WHISPER_MODEL_PATH", "assets/ai_models/faster-whisper"),
            tts_engine=os.getenv("TTS_ENGINE", "silero").strip().casefold() or "silero",
            silero_en_model_path=_project_path(
                project, "SILERO_EN_MODEL_PATH", "assets/ai_models/silero/v3_en.pt"
            ),
            silero_en_speaker=os.getenv("SILERO_EN_SPEAKER", "en_1").strip() or "en_1",
            # TTS chooses an available backend while retaining a CPU fallback.
            tts_device=os.getenv("JEREMY_TTS_DEVICE", "auto").strip().casefold() or "auto",
            tts_sample_rate=_int_env("JEREMY_TTS_SAMPLE_RATE", 24_000, 8_000, 48_000),
            microphone_sample_rate=_int_env("JEREMY_MIC_SAMPLE_RATE", 16_000, 8_000, 48_000),
            listening_timeout=_float_env("JEREMY_LISTENING_TIMEOUT", 7.0, 3.0, 30.0),
            utterance_silence_seconds=_float_env("JEREMY_SILENCE_SECONDS", 1.35, 0.35, 3.0),
            utterance_max_seconds=_float_env("JEREMY_MAX_UTTERANCE_SECONDS", 25.0, 3.0, 45.0),
            speech_rms_threshold=_float_env("JEREMY_SPEECH_RMS", 0.004, 0.001, 0.25),
            wake_threshold=_float_env("JEREMY_WAKE_THRESHOLD", 1e-20, 1e-40, 1e-5),
            request_timeout=_float_env("OPENROUTER_TIMEOUT_SECONDS", 9.0, 4.0, 30.0),
            connect_timeout=_float_env("OPENROUTER_CONNECT_TIMEOUT_SECONDS", 3.0, 1.0, 15.0),
            total_timeout=_float_env("OPENROUTER_TOTAL_TIMEOUT_SECONDS", 28.0, 10.0, 90.0),
            request_retries=_int_env("OPENROUTER_RETRIES", 1, 0, 3),
            max_response_tokens=_int_env("JEREMY_MAX_RESPONSE_TOKENS", 180, 64, 512),
            debug=os.getenv("JEREMY_DEBUG", "").strip().casefold() in {"1", "true", "yes"},
            benchmark=os.getenv("JEREMY_DEBUG_BENCHMARK", "").strip().casefold() in {"1", "true", "yes"},
        )

    @property
    def configuration_error(self) -> str | None:
        missing: list[str] = []
        if not self.api_key:
            missing.append("OPENROUTER_API_KEY")
        if not self.llm_model:
            missing.append("OPENROUTER_LLM_MODEL")
        return None if not missing else "Set " + ", ".join(missing) + " in .env"


__all__ = ["JeremySettings"]
