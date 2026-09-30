from __future__ import annotations

from dataclasses import replace
import io
from pathlib import Path
import queue
import threading
import time
import unittest
import wave

import numpy as np

from ai.assistant import JeremyAssistant, JeremyState
from ai.audio import pcm_to_wav, resample_pcm16_mono
from ai.context import ComponentContext, LumenAIContext, LumenContextBuilder
from ai.openrouter_client import OpenRouterError
from ai.settings import JeremySettings
from ai.stt_vosk import VoskSTT
from ai.wake_word import WakeWordDetector
from core.gestures import Gesture
from core.interaction import ApplicationState, InteractionState, ModelTransform
from core.model import ComponentMetadata, SceneModel


def _pcm(amplitude: int, frames: int = 1600) -> bytes:
    return np.full(frames, amplitude, dtype="<i2").tobytes()


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


class _FakeWake:
    available = True
    error = None

    def __init__(self, detections: list[bool]) -> None:
        self.detections = queue.Queue()
        for value in detections:
            self.detections.put(value)

    def process_audio(self, _block: bytes) -> bool:
        try:
            return self.detections.get_nowait()
        except queue.Empty:
            return False

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass


class _FakeMicrophone:
    available = True

    def __init__(self) -> None:
        self.blocks: queue.Queue[bytes | None] = queue.Queue()
        self.error = None

    def start(self) -> None:
        pass

    def push(self, block: bytes) -> None:
        self.blocks.put(block)

    def read(self, timeout: float = 0.1) -> bytes | None:
        try:
            return self.blocks.get(timeout=timeout)
        except queue.Empty:
            return None

    def flush(self) -> None:
        while True:
            try:
                self.blocks.get_nowait()
            except queue.Empty:
                return

    def close(self) -> None:
        self.blocks.put(None)


class _FakePlayer:
    def __init__(self) -> None:
        self.played = 0

    def play_pcm(self, _audio: bytes, _sample_rate: int) -> None:
        self.played += 1


class _BlockingPlayer(_FakePlayer):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def play_pcm(self, audio: bytes, sample_rate: int) -> None:
        super().play_pcm(audio, sample_rate)
        self.started.set()
        self.release.wait(timeout=2.0)


class _FakeClient:
    def __init__(self, transcripts: list[str], *, fail_stage: str = "") -> None:
        self.transcripts = queue.Queue()
        for transcript in transcripts:
            self.transcripts.put(transcript)
        self.fail_stage = fail_stage
        self.stt_calls = 0
        self.llm_calls = 0
        self.tts_calls = 0
        self.questions: list[str] = []
        self.contexts: list[LumenAIContext] = []

    def transcribe_audio(self, _wav: bytes) -> str:
        self.stt_calls += 1
        if self.fail_stage == "STT":
            raise OpenRouterError("STT", "offline")
        return self.transcripts.get_nowait()

    def generate_response(self, question: str, context: LumenAIContext) -> str:
        self.llm_calls += 1
        self.questions.append(question)
        self.contexts.append(context)
        if self.fail_stage == "LLM":
            raise OpenRouterError("LLM", "offline")
        return "Short answer"

    def generate_speech(self, _text: str) -> bytes:
        self.tts_calls += 1
        if self.fail_stage == "TTS":
            raise OpenRouterError("TTS", "offline")
        return b"\0\0" * 100

    def close(self) -> None:
        pass


class _BlockingClient(_FakeClient):
    def __init__(self, transcripts: list[str]) -> None:
        super().__init__(transcripts)
        self.started = threading.Event()
        self.release = threading.Event()

    def generate_response(self, question: str, context: LumenAIContext) -> str:
        self.started.set()
        self.release.wait(timeout=2.0)
        return super().generate_response(question, context)


def _settings() -> JeremySettings:
    return JeremySettings(
        api_key="test-key",
        llm_model="test-llm",
        listening_timeout=0.22,
        utterance_silence_seconds=0.05,
        utterance_max_seconds=1.0,
        speech_rms_threshold=0.01,
    )


class JeremyTests(unittest.TestCase):
    def _assistant(
        self,
        transcripts: list[str],
        detections: list[bool],
        *,
        fail_stage: str = "",
    ) -> tuple[JeremyAssistant, _FakeMicrophone, _FakeClient, _FakePlayer]:
        microphone = _FakeMicrophone()
        client = _FakeClient(transcripts, fail_stage=fail_stage)
        player = _FakePlayer()
        assistant = JeremyAssistant(
            _settings(),
            wake_detector=_FakeWake(detections),
            microphone=microphone,
            player=player,
            client=client,
        )
        assistant.start()
        _wait_until(lambda: assistant.status().label != "INITIALIZING")
        return assistant, microphone, client, player

    def test_wake_word_text_matching_is_case_and_punctuation_insensitive(self) -> None:
        for value in ("Jeremy", "JEREMY!", "Jeremy, what is this?", "(jeremy)"):
            self.assertTrue(WakeWordDetector.contains_wake_word(value), value)
        self.assertFalse(WakeWordDetector.contains_wake_word("How do I zoom?"))
        self.assertEqual(
            WakeWordDetector.extract_invocation("Jeremy, how do I rotate the model?"),
            (True, "how do I rotate the model"),
        )

    def test_microphone_audio_is_resampled_and_encoded_as_mono_pcm16_wav(self) -> None:
        source = np.arange(4410, dtype="<i2").tobytes()
        converted = resample_pcm16_mono(source, 44_100, 16_000)
        self.assertEqual(len(converted), 1600 * 2)
        wav_bytes = pcm_to_wav(converted, 16_000)
        with wave.open(io.BytesIO(wav_bytes), "rb") as recorded:
            self.assertEqual(recorded.getnchannels(), 1)
            self.assertEqual(recorded.getsampwidth(), 2)
            self.assertEqual(recorded.getframerate(), 16_000)
            self.assertEqual(recorded.getnframes(), 1600)

    def test_vosk_reports_the_exact_missing_local_model_path(self) -> None:
        stt = VoskSTT("missing-english-model")
        with self.assertRaises(FileNotFoundError) as caught:
            stt.initialize()
        self.assertIn("ENGLISH SPEECH MODEL NOT FOUND", str(caught.exception))
        self.assertIn("missing-english-model", str(caught.exception))

    def test_context_builder_exposes_real_lumen_state_and_controls(self) -> None:
        component = ComponentMetadata(
            "cpu", "CPU SOCKET", "Holds the CPU", "Connects the processor", fields={"TYPE": "Socket"}
        )
        model = SceneModel(
            components={"cpu": component},
            model_name="TEST BOARD",
            category_name="COMPUTER HARDWARE",
            loaded=True,
        )
        state = ApplicationState(
            ModelTransform(position=(0.2, -0.1, 0.0), rotation=(10.0, 20.0, 0.0), scale=1.25),
            interaction=InteractionState.ROTATE,
        )
        context = LumenContextBuilder().build(
            model=model,
            model_path="test.glb",
            category="computer",
            application_state=state,
            selected_id="cpu",
            hovered_id=None,
            active_gesture=Gesture.V_SIGN,
            tracking_status="TRACKING",
            camera_status="READY",
        )
        self.assertEqual(context.current_model, "TEST BOARD")
        self.assertEqual(context.selected_object.name, "CPU SOCKET")
        self.assertEqual(context.current_mode, "ROTATE")
        self.assertEqual(context.active_gesture, "V_SIGN")
        self.assertEqual(context.current_transform.zoom_scale, 1.25)
        self.assertIn("V gesture -> rotate the model", context.available_controls)

    def test_a_ambient_question_never_reaches_online_pipeline(self) -> None:
        assistant, microphone, client, player = self._assistant([], [False, False])
        try:
            microphone.push(_pcm(9000))
            microphone.push(_pcm(0))
            time.sleep(0.18)
            self.assertEqual(client.stt_calls, 0)
            self.assertEqual(client.llm_calls, 0)
            self.assertEqual(client.tts_calls, 0)
            self.assertEqual(player.played, 0)
            self.assertEqual(assistant.state, JeremyState.SLEEPING)
        finally:
            assistant.close()

    def test_b_and_c_called_question_uses_llm_and_tts_in_same_language(self) -> None:
        for transcript in ("Jeremy, how do I rotate the model?", "Jeremy, how do I zoom?"):
            assistant, microphone, client, player = self._assistant([transcript], [True])
            try:
                microphone.push(_pcm(9000))
                microphone.push(_pcm(0))
                self.assertTrue(_wait_until(lambda: player.played == 1))
                self.assertEqual(client.llm_calls, 1)
                self.assertEqual(client.tts_calls, 1)
                self.assertEqual(client.contexts[0].user_language, "English")
                self.assertTrue(
                    _wait_until(lambda: assistant.state == JeremyState.SLEEPING)
                )
            finally:
                assistant.close()

    def test_d_wake_only_waits_for_follow_up_and_uses_current_selection(self) -> None:
        assistant, microphone, client, player = self._assistant(
            ["Jeremy.", "What is this?"], [True]
        )
        selected = ComponentContext("biceps", "Biceps brachii", "Arm muscle", "Flexes the elbow", {})
        assistant.update_context(LumenAIContext(selected_object=selected))
        try:
            microphone.push(_pcm(9000))
            microphone.push(_pcm(0))
            self.assertTrue(_wait_until(lambda: client.stt_calls == 1))
            self.assertEqual(client.llm_calls, 0)
            self.assertEqual(assistant.state, JeremyState.LISTENING)
            microphone.push(_pcm(9000))
            microphone.push(_pcm(0))
            self.assertTrue(_wait_until(lambda: player.played == 1))
            self.assertEqual(client.contexts[0].selected_object.name, "Biceps brachii")
        finally:
            assistant.close()

    def test_e_wake_only_times_out_silently(self) -> None:
        assistant, microphone, client, player = self._assistant(["Jeremy."], [True])
        try:
            microphone.push(_pcm(9000))
            microphone.push(_pcm(0))
            self.assertTrue(_wait_until(lambda: client.stt_calls == 1))
            self.assertTrue(_wait_until(lambda: assistant.state == JeremyState.SLEEPING))
            self.assertEqual(client.llm_calls, 0)
            self.assertEqual(client.tts_calls, 0)
            self.assertEqual(player.played, 0)
        finally:
            assistant.close()

    def test_f_follow_up_without_new_wake_remains_silent(self) -> None:
        assistant, microphone, client, player = self._assistant(
            ["Jeremy, how do I zoom?"], [True, False, False]
        )
        try:
            microphone.push(_pcm(9000))
            microphone.push(_pcm(0))
            self.assertTrue(_wait_until(lambda: player.played == 1))
            self.assertTrue(_wait_until(lambda: assistant.state == JeremyState.SLEEPING))
            microphone.push(_pcm(9000))
            microphone.push(_pcm(0))
            time.sleep(0.18)
            self.assertEqual(client.llm_calls, 1)
            self.assertEqual(client.tts_calls, 1)
        finally:
            assistant.close()

    def test_g_new_wake_word_starts_a_new_one_shot_session(self) -> None:
        assistant, microphone, client, player = self._assistant(
            ["Jeremy, first question", "Jeremy, second question"], [True, True]
        )
        try:
            microphone.push(_pcm(9000))
            microphone.push(_pcm(0))
            self.assertTrue(_wait_until(lambda: player.played == 1))
            self.assertTrue(_wait_until(lambda: assistant.state == JeremyState.SLEEPING))
            microphone.push(_pcm(9000))
            microphone.push(_pcm(0))
            self.assertTrue(_wait_until(lambda: player.played == 2))
            self.assertEqual(client.llm_calls, 2)
            self.assertEqual(client.tts_calls, 2)
            self.assertTrue(_wait_until(lambda: assistant.state == JeremyState.SLEEPING))
        finally:
            assistant.close()

    def test_h_network_failure_returns_to_sleep_without_crashing(self) -> None:
        assistant, microphone, client, _player = self._assistant(
            ["Jeremy, help me"], [True], fail_stage="LLM"
        )
        try:
            microphone.push(_pcm(9000))
            microphone.push(_pcm(0))
            self.assertTrue(
                _wait_until(
                    lambda: assistant.status().label == "OFFLINE"
                    and assistant.state == JeremyState.SLEEPING
                )
            )
            self.assertEqual(assistant.state, JeremyState.SLEEPING)
            self.assertEqual(client.llm_calls, 1)
        finally:
            assistant.close()

    def test_i_missing_key_disables_only_jeremy(self) -> None:
        settings = replace(_settings(), api_key="")
        assistant = JeremyAssistant(
            settings,
            wake_detector=_FakeWake([]),
            microphone=_FakeMicrophone(),
            player=_FakePlayer(),
            client=_FakeClient([]),
        )
        try:
            self.assertFalse(assistant.start())
            status = assistant.status()
            self.assertFalse(status.enabled)
            self.assertEqual(status.label, "AI NOT CONFIGURED")
            self.assertIn("OPENROUTER_API_KEY", status.detail)
        finally:
            assistant.close()

    def test_j_slow_online_generation_does_not_block_state_updates(self) -> None:
        microphone = _FakeMicrophone()
        client = _BlockingClient(["Jeremy, explain this"])
        assistant = JeremyAssistant(
            _settings(),
            wake_detector=_FakeWake([True]),
            microphone=microphone,
            player=_FakePlayer(),
            client=client,
        )
        assistant.start()
        self.assertTrue(_wait_until(lambda: assistant.status().label != "INITIALIZING"))
        try:
            microphone.push(_pcm(9000))
            microphone.push(_pcm(0))
            self.assertTrue(client.started.wait(timeout=1.0))
            started = time.perf_counter()
            for _ in range(1000):
                assistant.update_context(LumenAIContext(current_model="Responsive"))
                assistant.status()
            elapsed = time.perf_counter() - started
            self.assertLess(elapsed, 0.15)
            self.assertEqual(assistant.state, JeremyState.PROCESSING)
            client.release.set()
            self.assertTrue(_wait_until(lambda: assistant.state == JeremyState.SLEEPING))
        finally:
            client.release.set()
            assistant.close()

    def test_missing_vosk_model_disables_only_jeremy_with_exact_path(self) -> None:
        client = _FakeClient([])
        assistant = JeremyAssistant(
            replace(
                _settings(),
                vosk_en_model_path=Path("missing/en/vosk"),
            ),
            wake_detector=_FakeWake([]),
            microphone=_FakeMicrophone(),
            player=_FakePlayer(),
            stt=VoskSTT("missing/en/vosk"),
            llm=client,
            tts=client,
        )
        try:
            self.assertTrue(assistant.start())
            self.assertTrue(_wait_until(lambda: not assistant.status().enabled))
            self.assertEqual(assistant.state, JeremyState.SLEEPING)
            self.assertEqual(assistant.status().label, "ENGLISH MODEL MISSING")
            self.assertEqual(assistant.status().detail, "English speech model missing")
        finally:
            assistant.close()

    def test_speaker_output_cannot_reactivate_jeremy(self) -> None:
        microphone = _FakeMicrophone()
        client = _FakeClient(["Jeremy, explain this"])
        player = _BlockingPlayer()
        assistant = JeremyAssistant(
            _settings(),
            wake_detector=_FakeWake([True, True]),
            microphone=microphone,
            player=player,
            client=client,
        )
        assistant.start()
        self.assertTrue(_wait_until(lambda: assistant.status().label != "INITIALIZING"))
        try:
            microphone.push(_pcm(9000))
            microphone.push(_pcm(0))
            self.assertTrue(player.started.wait(timeout=1.0))
            self.assertEqual(assistant.state, JeremyState.SPEAKING)
            # Simulate Jeremy's own voice reaching the mic while playback owns
            # the worker. These queued blocks must be discarded after playback.
            microphone.push(_pcm(9000))
            microphone.push(_pcm(0))
            player.release.set()
            self.assertTrue(_wait_until(lambda: assistant.state == JeremyState.SLEEPING))
            time.sleep(0.12)
            self.assertEqual(client.stt_calls, 1)
            self.assertEqual(client.llm_calls, 1)
            self.assertEqual(client.tts_calls, 1)
        finally:
            player.release.set()
            assistant.close()


if __name__ == "__main__":
    unittest.main()
