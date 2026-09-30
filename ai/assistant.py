"""Jeremy's explicit state machine and background hybrid voice pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
import threading
import time
from typing import Any, Protocol

from .audio import AudioPlayer, MicrophoneInput, UtteranceRecorder, pcm_to_wav, recent_audio_buffer
from .context import LumenAIContext, LumenContextBuilder
from .openrouter_client import OpenRouterClient, OpenRouterError
from .settings import JeremySettings
from .stt_vosk import STTResult, VoskSTT
from .stt_whisper import FasterWhisperSTT
from .tts import JeremyTTS, SpeechAudio
from .wake_word import WakeWordDetector


LOGGER = logging.getLogger("lumen.jeremy")
if not LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(_handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False


def _console_safe(value: object) -> str:
    text = str(value)
    handler = LOGGER.handlers[0] if LOGGER.handlers else None
    encoding = getattr(getattr(handler, "stream", None), "encoding", None) or "utf-8"
    return text.encode(encoding, errors="backslashreplace").decode(encoding, errors="replace")


class JeremyState(str, Enum):
    SLEEPING = "SLEEPING"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    SPEAKING = "SPEAKING"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class JeremyStatus:
    state: JeremyState
    label: str = ""
    detail: str = ""
    enabled: bool = True


class _LegacyJeremyClient(Protocol):
    """Compatibility seam retained for isolated state-machine tests."""

    def transcribe_audio(self, wav_bytes: bytes) -> str: ...
    def generate_response(self, question: str, context: LumenAIContext) -> str: ...
    def generate_speech(self, text: str) -> bytes: ...
    def close(self) -> None: ...


class JeremyAssistant:
    """Own one worker; the render thread only publishes context and reads status."""

    def __init__(
        self,
        settings: JeremySettings,
        *,
        wake_detector: WakeWordDetector | None = None,
        microphone: MicrophoneInput | None = None,
        player: AudioPlayer | None = None,
        stt: Any | None = None,
        llm: Any | None = None,
        tts: Any | None = None,
        client: _LegacyJeremyClient | None = None,
    ) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = JeremyState.SLEEPING
        self._context = LumenAIContext()
        self._activation_deadline = 0.0
        self._status_label = ""
        self._status_detail = ""
        self._status_expires = 0.0
        self._disabled_reason = settings.configuration_error
        self._initializing = False

        self._wake = wake_detector or WakeWordDetector(
            settings.microphone_sample_rate, settings.wake_threshold
        )
        self._microphone = microphone or MicrophoneInput(settings.microphone_sample_rate)
        self._player = player or AudioPlayer()
        self._legacy_client = client
        if client is not None:
            self._stt = self._llm = self._tts = client
        else:
            self._stt = stt or (
                FasterWhisperSTT(settings.whisper_model_path, settings.microphone_sample_rate,
                                 settings.stt_device)
                if settings.stt_backend == "faster-whisper" else
                VoskSTT(settings.vosk_en_model_path, settings.microphone_sample_rate)
            )
            self._llm = llm
            self._tts = tts or JeremyTTS(settings)
            if self._disabled_reason is None and self._llm is None:
                try:
                    self._llm = OpenRouterClient(settings)
                except Exception as error:
                    self._disabled_reason = str(error)

        # Pre-roll is local and bounded. It preserves the spoken name for Vosk;
        # no audio reaches OpenRouter under any state.
        self._recent = recent_audio_buffer(max_blocks=10)
        self._session_audio: list[bytes] = []
        self._session_frames = 0
        self._recorder = UtteranceRecorder(
            settings.microphone_sample_rate,
            rms_threshold=settings.speech_rms_threshold,
            silence_seconds=settings.utterance_silence_seconds,
            max_seconds=settings.utterance_max_seconds,
        )

        if self._disabled_reason is None and not self._wake.available:
            self._disabled_reason = self._wake.error or "Wake-word detector unavailable"
        if self._disabled_reason is None and not self._microphone.available:
            self._disabled_reason = "sounddevice is not installed"

    @property
    def state(self) -> JeremyState:
        with self._lock:
            return self._state

    def update_context(self, context: LumenAIContext) -> None:
        with self._lock:
            self._context = context

    def status(self) -> JeremyStatus:
        now = time.monotonic()
        with self._lock:
            if self._disabled_reason:
                configuration = self.settings.configuration_error is not None
                reason = self._disabled_reason
                if configuration:
                    label = "AI NOT CONFIGURED"
                    detail = "Set OPENROUTER_API_KEY"
                elif "microphone" in reason.casefold() or "input device" in reason.casefold():
                    label = "MICROPHONE UNAVAILABLE"
                    detail = "Microphone unavailable"
                elif "english speech model not found" in reason.casefold():
                    label = "ENGLISH MODEL MISSING"
                    detail = "English speech model missing"
                else:
                    label = "UNAVAILABLE"
                    detail = "Speech system unavailable"
                return JeremyStatus(
                    self._state,
                    label,
                    detail,
                    False,
                )
            if self._initializing:
                return JeremyStatus(self._state, "INITIALIZING", "Loading local speech models", True)
            if self._status_label and now < self._status_expires:
                return JeremyStatus(self._state, self._status_label, self._status_detail, True)
            label = {
                JeremyState.SLEEPING: "",
                JeremyState.LISTENING: "LISTENING",
                JeremyState.PROCESSING: "THINKING",
                JeremyState.SPEAKING: "SPEAKING",
                JeremyState.ERROR: "ERROR",
            }[self._state]
            return JeremyStatus(self._state, label, "", True)

    def start(self) -> bool:
        LOGGER.info(
            "[Jeremy] OpenRouter: %s",
            "configured" if self.settings.api_key else "API key missing",
        )
        if self._disabled_reason is not None:
            LOGGER.error("[Jeremy][ERROR] Disabled: %s", _console_safe(self._disabled_reason))
            return False
        if self._thread is not None:
            return True
        LOGGER.info("[Jeremy] Initializing")
        with self._lock:
            self._initializing = True
        self._thread = threading.Thread(
            target=self._worker_loop, name="lumen-jeremy", daemon=True
        )
        self._thread.start()
        return True

    def _initialize_providers(self) -> None:
        try:
            self._microphone.start()
        except Exception as error:
            raise RuntimeError(f"Microphone unavailable: {error}") from error

        device_index = getattr(self._microphone, "device_index", None)
        device_name = getattr(self._microphone, "device_name", "")
        input_rate = int(
            getattr(self._microphone, "device_sample_rate", self.settings.microphone_sample_rate)
        )
        LOGGER.info("[Jeremy] Microphone initialized")
        LOGGER.info(
            "[Jeremy] Input device: %s",
            _console_safe(
                f"{device_index} - {device_name}"
                if device_index is not None and device_name
                else device_name or "configured test input"
            ),
        )
        LOGGER.info("[Jeremy] Input channels: 1")
        LOGGER.info("[Jeremy] Native sample rate: %d Hz", input_rate)
        if input_rate != self.settings.microphone_sample_rate:
            LOGGER.info(
                "[Jeremy] Speech sample rate: %d Hz mono PCM16 (resampled)",
                self.settings.microphone_sample_rate,
            )
        if self._stop.is_set():
            return
        initialize = getattr(self._stt, "initialize", None)
        if callable(initialize):
            initialize()
        if isinstance(self._stt, VoskSTT):
            LOGGER.info("[Jeremy] STT model: %s", _console_safe(self._stt.en_model_path.name))
            LOGGER.info("[Jeremy] STT language: English")
        LOGGER.info("[Jeremy] STT backend=%s device=%s", self.settings.stt_backend,
                    getattr(self._stt, "device", "cpu"))
        if self.settings.stt_backend == "vosk" and self.settings.stt_device == "cuda":
            LOGGER.warning("[Jeremy] Vosk Python STT is CPU-only; CUDA request ignored")
        if self.settings.debug:
            LOGGER.info("[Jeremy][DEBUG] STT load=%.2fs", getattr(self._stt, "load_seconds", 0.0))

        if self._stop.is_set():
            return
        initialize = getattr(self._tts, "initialize", None)
        if callable(initialize):
            initialize()
        if isinstance(self._tts, JeremyTTS):
            LOGGER.info("[Jeremy] TTS engine: %s", self._tts.engine.upper())
            LOGGER.info("[Jeremy] TTS device: %s", self._tts.device.upper())
            LOGGER.info("[Jeremy] TTS voice: English male %s", self.settings.silero_en_speaker)
        if self.settings.debug:
            LOGGER.info("[Jeremy][DEBUG] TTS load=%.2fs VRAM=%d MiB",
                        getattr(self._tts, "load_seconds", 0.0),
                        getattr(self._tts, "gpu_vram_bytes", 0) // 1024**2)
            if self.settings.benchmark:
                LOGGER.info("[Jeremy][DEBUG] TTS benchmark: %s",
                            getattr(self._tts, "benchmark_seconds", {}))
            self._log_hardware()
        validate = getattr(self._llm, "validate_models", None)
        if callable(validate) and not self._stop.is_set():
            validate()
        self._microphone.flush()
        with self._lock:
            self._initializing = False
        LOGGER.info("[Jeremy] Sleeping")

    def _log_hardware(self) -> None:
        try:
            import torch
            import subprocess
            gpu = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
                text=True, timeout=2, stderr=subprocess.DEVNULL,
            ).splitlines()[0]
            LOGGER.info("[Jeremy][DEBUG] GPU: %s", gpu)
            LOGGER.info("[Jeremy][DEBUG] PyTorch CUDA=%s build=%s MPS=%s STT=%s/%s TTS=%s LLM=OpenRouter/remote",
                        torch.cuda.is_available(), torch.version.cuda,
                        bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
                        self.settings.stt_backend, getattr(self._stt, "device", "cpu"),
                        getattr(self._tts, "device", "cpu"))
        except Exception as error:
            LOGGER.info("[Jeremy][DEBUG] Hardware detail unavailable: %s", error.__class__.__name__)

    def _set_state(self, state: JeremyState) -> None:
        with self._lock:
            changed = self._state != state
            self._state = state
        if changed:
            LOGGER.info("[Jeremy] %s", state.value.title())

    def _wake_up(self, now: float) -> None:
        LOGGER.info("[Jeremy] Wake word detected")
        LOGGER.info("[Jeremy] Recording started")
        self._activation_deadline = now + self.settings.listening_timeout
        self._recorder.start(tuple(self._recent), speech_started=True)
        self._session_audio.clear()
        self._session_frames = 0
        self._recent.clear()
        self._set_state(JeremyState.LISTENING)

    def _sleep(self, *, preserve_status: bool = False) -> None:
        self._recorder.start()
        self._session_audio.clear()
        self._session_frames = 0
        self._activation_deadline = 0.0
        self._recent.clear()
        self._wake.reset()
        self._microphone.flush()
        if not preserve_status:
            with self._lock:
                self._status_label = ""
                self._status_detail = ""
                self._status_expires = 0.0
        self._set_state(JeremyState.SLEEPING)

    def _fail(self, label: str, detail: str, *, disable: bool = False) -> None:
        self._set_state(JeremyState.ERROR)
        with self._lock:
            self._status_label = label
            self._status_detail = detail
            self._status_expires = time.monotonic() + 5.0
        self._stop.wait(0.08)
        self._sleep(preserve_status=True)
        if disable:
            with self._lock:
                self._disabled_reason = detail

    def _active_context(self, question: str = "", recognized_language: str = "English") -> LumenAIContext:
        with self._lock:
            context = self._context
        return context.with_language("English")

    def _transcribe(self, pcm16_mono: bytes) -> STTResult:
        transcribe = getattr(self._stt, "transcribe", None)
        if callable(transcribe):
            result = transcribe(pcm16_mono)
            if isinstance(result, STTResult):
                return result
            return STTResult(str(result or "").strip(), "unknown", 0.0)
        legacy = getattr(self._stt, "transcribe_audio", None)
        if not callable(legacy):
            raise RuntimeError("Local STT provider is unavailable")
        text = legacy(pcm_to_wav(pcm16_mono, self.settings.microphone_sample_rate))
        return STTResult(str(text or "").strip(), "English", 0.0)

    def _synthesize(self, text: str, language: str) -> SpeechAudio:
        synthesize = getattr(self._tts, "synthesize", None)
        if callable(synthesize):
            result = synthesize(text, language)
            if isinstance(result, SpeechAudio):
                return result
            if isinstance(result, bytes):
                return SpeechAudio(result, self.settings.tts_sample_rate)
        legacy = getattr(self._tts, "generate_speech", None)
        if not callable(legacy):
            raise RuntimeError("Local TTS provider is unavailable")
        return SpeechAudio(bytes(legacy(text)), self.settings.tts_sample_rate)

    def _process_utterance(self, pcm16_mono: bytes) -> None:
        try:
            LOGGER.info("[Jeremy] Recording finished")
            LOGGER.info("[Jeremy] Transcribing locally with %s", self.settings.stt_backend)
            result = self._transcribe(pcm16_mono)
            LOGGER.info("[Jeremy] STT language: English")
            LOGGER.info("[Jeremy] Recognized: %s", _console_safe(result.text or "<empty>"))
            if self.settings.debug:
                LOGGER.info("[Jeremy][DEBUG] STT finalization=%.0f ms",
                            getattr(self._stt, "finalization_seconds", 0.0) * 1000)
        except Exception as error:
            LOGGER.error("[Jeremy][ERROR] Local STT failed: %s", _console_safe(error))
            self._fail("STT ERROR", str(error))
            return

        # This is the application-level privacy gate. No wake means this method
        # can never be entered, and a wake-only transcript never reaches the LLM.
        question = " ".join(WakeWordDetector.strip_wake_word(result.text).split()).strip()
        if not question:
            if time.monotonic() >= self._activation_deadline:
                self._sleep()
            else:
                self._recorder.start()
                self._set_state(JeremyState.LISTENING)
            return
        if not any(character.isalpha() for character in question) or question.casefold() in {
            "uh", "um", "hmm", "the", "a", "and"
        }:
            self._sleep()
            return

        self._set_state(JeremyState.PROCESSING)
        context = self._active_context(question, result.language)
        try:
            LOGGER.info("[Jeremy] Building Lumen context")
            LOGGER.info("[Jeremy] Sending request to OpenRouter")
            stream = getattr(self._llm, "stream_response", None)
            if callable(stream):
                answer = stream(question, context, self._on_stream_text, self._stop.is_set)
            else:
                answer = self._llm.generate_response(question, context)
            with self._lock:
                self._status_label = self._status_detail = ""
            LOGGER.info("[Jeremy] Response received")
        except InterruptedError:
            return
        except OpenRouterError as error:
            LOGGER.error("[Jeremy] AI temporarily unavailable: model=%s HTTP=%s type=%s",
                         error.model, error.status_code, error.exception_type)
            self._fail("OFFLINE", "AI temporarily unavailable")
            return
        except Exception as error:
            LOGGER.error("[Jeremy][ERROR] LLM failed: %s", _console_safe(error))
            self._fail("OFFLINE", "AI temporarily unavailable")
            return

        if self._stop.is_set():
            return

        try:
            LOGGER.info("[Jeremy] Generating local speech")
            speech = self._synthesize(answer, "English")
            if self.settings.debug:
                LOGGER.info("[Jeremy][DEBUG] TTS synthesis=%.0f ms",
                            getattr(self._tts, "synthesis_seconds", 0.0) * 1000)
            self._set_state(JeremyState.SPEAKING)
            self._player.play_pcm(speech.pcm16_mono, speech.sample_rate)
        except Exception as error:
            LOGGER.error("[Jeremy][ERROR] Local TTS/playback failed: %s", _console_safe(error))
            # The generated answer remains briefly visible through the existing HUD.
            self._fail("VOICE ERROR", "Speech playback unavailable")
            return

        # Do not let output lingering in the room reactivate Jeremy.
        self._stop.wait(0.25)
        self._sleep()

    def _on_stream_text(self, text: str) -> None:
        with self._lock:
            self._status_label = "THINKING"
            self._status_detail = text[-34:].strip()
            self._status_expires = time.monotonic() + 4.0

    def _worker_loop(self) -> None:
        try:
            self._initialize_providers()
        except Exception as error:
            LOGGER.error("[Jeremy][ERROR] Initialization failed: %s", _console_safe(error))
            with self._lock:
                self._initializing = False
            self._fail("UNAVAILABLE", str(error), disable=True)
            self._microphone.close()
            return

        if self._stop.is_set():
            self._microphone.close()
            return

        while not self._stop.is_set():
            block = self._microphone.read(timeout=0.10)
            now = time.monotonic()
            state = self.state
            if state == JeremyState.SLEEPING:
                if block is None:
                    continue
                self._recent.append(block)
                if self._wake.process_audio(block):
                    self._wake_up(now)
                continue

            # PROCESSING, SPEAKING and ERROR deliberately ignore microphone data.
            if state != JeremyState.LISTENING:
                continue
            if now >= self._activation_deadline and not self._recorder.speech_started:
                self._sleep()
                continue
            if block is None:
                continue
            utterance = self._recorder.add(block)
            if utterance is not None:
                self._session_audio.append(utterance)
                self._session_frames += len(utterance) // 2
                if (self._recorder.last_end_reason == "max" and
                    self._session_frames / self.settings.microphone_sample_rate < 60):
                    self._recorder.start(speech_started=True)
                    continue
                full_utterance = b"".join(self._session_audio)
                self._session_audio.clear()
                self._session_frames = 0
                self._process_utterance(full_utterance)

    def close(self) -> None:
        self._stop.set()
        self._microphone.close()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=3.0)
        self._wake.close()
        # A timed-out HTTP request may still own these objects on the daemon
        # worker. Let process shutdown reclaim them instead of racing its use.
        if thread is not None and thread.is_alive():
            return
        closed: set[int] = set()
        for provider in (self._stt, self._llm, self._tts):
            if provider is None or id(provider) in closed:
                continue
            closed.add(id(provider))
            closer = getattr(provider, "shutdown", None) or getattr(provider, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    pass


__all__ = ["JeremyAssistant", "JeremyState", "JeremyStatus"]
