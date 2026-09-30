"""Regression coverage for finalized speech and streaming OpenRouter failover."""

from __future__ import annotations

from dataclasses import replace
import json
import threading
import time
import unittest
from unittest.mock import patch

import httpx

from ai.audio import UtteranceRecorder
from ai.assistant import JeremyAssistant, JeremyState
from ai.context import LumenAIContext
from ai.openrouter_client import OpenRouterClient, OpenRouterError
from ai.settings import JeremySettings
from ai.stt_vosk import VoskSTT, merge_segments
from tests.test_jeremy import _FakeMicrophone, _FakePlayer, _FakeWake, _pcm, _wait_until


def _sse(*parts: str, model: str = "google/gemini-3.1-flash-lite") -> httpx.Response:
    lines = ["data: " + json.dumps({"model": model, "choices": [{"delta": {"content": part}}]})
             for part in parts]
    return httpx.Response(200, text="\n\n".join([*lines, "data: [DONE]"]) + "\n\n")


def _settings(**changes: object) -> JeremySettings:
    return replace(JeremySettings(api_key="unit-test-key", llm_model="google/gemini-3.1-flash-lite",
                                  fallback_models=("openai/gpt-4.1-mini", "mistralai/mistral-small-3.2-24b-instruct"),
                                  request_retries=0, request_timeout=1.0,
                                  connect_timeout=1.0, total_timeout=4.0), **changes)


def _client(handler, **settings):
    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(base_url=OpenRouterClient.BASE_URL, transport=transport)
    return OpenRouterClient(_settings(**settings), client=http_client)


class SpeechTests(unittest.TestCase):
    def test_short_and_long_speech_and_duplicate_fragment_removal(self) -> None:
        self.assertEqual(merge_segments(["what is this"]), "what is this")
        self.assertEqual(merge_segments(["explain how the processor works",
                                         "works and why the cache is important"]),
                         "explain how the processor works and why the cache is important")
        self.assertEqual(merge_segments(["explain the CPU", "explain the CPU"]), "explain the CPU")
        self.assertEqual(merge_segments(["", "  "]), "")

    def test_natural_pause_does_not_end_utterance(self) -> None:
        recorder = UtteranceRecorder(16000, rms_threshold=.004,
                                     silence_seconds=1.35, max_seconds=25)
        loud = (1000).to_bytes(2, "little", signed=True) * 1600
        quiet = bytes(3200)
        recorder.start(speech_started=True)
        self.assertIsNone(recorder.add(loud))
        for _ in range(7):  # 700 ms, a natural pause
            self.assertIsNone(recorder.add(quiet))
        self.assertIsNone(recorder.add(loud))
        result = None
        for _ in range(14):
            result = recorder.add(quiet) or result
        self.assertIsNotNone(result)
        self.assertEqual(result.count(loud), 2)

    def test_vosk_collects_multiple_final_segments(self) -> None:
        class Recognizer:
            count = 0
            def __init__(self, *_): self.count = 0
            def SetWords(self, _): pass
            def AcceptWaveform(self, _):
                self.count += 1
                return self.count in (1, 2)
            def Result(self):
                return json.dumps({"text": "explain how the CPU works" if self.count == 1
                                   else "works and why its cache matters"})
            def FinalResult(self): return json.dumps({"text": "cache matters"})
        with patch("ai.stt_vosk.Model", lambda *_: object()), \
             patch("ai.stt_vosk.KaldiRecognizer", Recognizer), \
             patch("ai.stt_vosk.Path.is_file", return_value=True):
            stt = VoskSTT("fake-english-model")
            self.assertEqual(stt.transcribe(bytes(24_000)).text,
                             "explain how the CPU works and why its cache matters")
            self.assertEqual(stt.transcribe(b"").text, "")


class OpenRouterTests(unittest.TestCase):
    def test_streaming_response_assembly_and_history(self) -> None:
        seen = []
        requests = []
        def handler(request):
            requests.append(json.loads(request.content))
            return _sse("The processor ", "handles instructions.")
        client = _client(handler)
        try:
            answer = client.stream_response("What does the CPU do?", LumenAIContext(), seen.append)
            self.assertEqual(answer, "The processor handles instructions.")
            self.assertEqual(seen[-1], answer)
            self.assertTrue(requests[0]["stream"])
            self.assertEqual(requests[0]["provider"]["sort"], "latency")
            self.assertEqual(requests[0]["models"], ["openai/gpt-4.1-mini", "mistralai/mistral-small-3.2-24b-instruct"])
            client.generate_response("Explain it more simply", LumenAIContext())
            self.assertEqual(len(requests[1]["messages"]), 4)
        finally:
            client.close()

    def test_primary_failure_fallback_success(self) -> None:
        attempted = []
        def handler(request):
            model = json.loads(request.content)["model"]
            attempted.append(model)
            return httpx.Response(503, json={"error": "provider busy"}) if len(attempted) == 1 else _sse("A useful answer.", model=model)
        client = _client(handler)
        try:
            self.assertEqual(client.generate_response("What is this?", LumenAIContext()), "A useful answer.")
            self.assertEqual(attempted[:2], ["google/gemini-3.1-flash-lite", "openai/gpt-4.1-mini"])
        finally:
            client.close()

    def test_http_429_retries_and_http_5xx_falls_back(self) -> None:
        for status in (429, 500, 503):
            calls = []
            def handler(request):
                calls.append(json.loads(request.content)["model"])
                return httpx.Response(status, json={"error": "temporary"}) if len(calls) == 1 else _sse("Ready.")
            client = _client(handler, request_retries=1)
            try:
                self.assertEqual(client.generate_response("Hello", LumenAIContext()), "Ready.")
                self.assertEqual(len(calls), 2)
            finally:
                client.close()

    def test_timeout_and_api_unavailable_are_bounded(self) -> None:
        for failure in (httpx.ReadTimeout("timed out"), httpx.ConnectError("offline")):
            client = _client(lambda _request: (_ for _ in ()).throw(failure))
            try:
                with self.assertRaises(OpenRouterError):
                    client.generate_response("What is this?", LumenAIContext())
            finally:
                client.close()

    def test_malformed_and_empty_response_use_fallback(self) -> None:
        for first in (httpx.Response(200, text="data: broken\n\ndata: [DONE]\n\n"),
                      httpx.Response(200, text="data: [DONE]\n\n")):
            count = 0
            def handler(_request):
                nonlocal count
                count += 1
                return first if count == 1 else _sse("Recovered.")
            client = _client(handler)
            try:
                self.assertEqual(client.generate_response("Explain this", LumenAIContext()), "Recovered.")
            finally:
                client.close()

    def test_invalid_request_fails_immediately_with_safe_diagnostic(self) -> None:
        key = "unit-test-key"
        client = _client(lambda _request: httpx.Response(400, text=f"invalid {key}"))
        try:
            with self.assertRaises(OpenRouterError) as caught:
                client.generate_response("Hello", LumenAIContext())
            self.assertEqual(caught.exception.status_code, 400)
            self.assertNotIn(key, caught.exception.response_text)
        finally:
            client.close()

    def test_next_request_works_after_previous_failure(self) -> None:
        calls = 0
        def handler(_request):
            nonlocal calls
            calls += 1
            return httpx.Response(503, text="offline") if calls <= 3 else _sse("I am back.")
        client = _client(handler)
        try:
            with self.assertRaises(OpenRouterError):
                client.generate_response("First", LumenAIContext())
            self.assertEqual(client.generate_response("Second", LumenAIContext()), "I am back.")
        finally:
            client.close()

    def test_stream_cancellation_returns_promptly(self) -> None:
        stop = False
        client = _client(lambda _request: _sse("First token", "second token"))
        def on_text(_):
            nonlocal stop
            stop = bool(_)
        try:
            with self.assertRaises(InterruptedError):
                client.stream_response("Hello", LumenAIContext(), on_text, lambda: stop)
        finally:
            client.close()

    def test_model_validation_keeps_only_live_ids(self) -> None:
        def handler(request):
            self.assertEqual(request.url.path, "/api/v1/models")
            return httpx.Response(200, json={"data": [{"id": "openai/gpt-4.1-mini"}]})
        client = _client(handler)
        try:
            client.validate_models()
            self.assertEqual(client.models, ("openai/gpt-4.1-mini",))
        finally:
            client.close()


class AssistantStreamingTests(unittest.TestCase):
    def test_streamed_text_reaches_existing_status_and_shutdown_does_not_hang(self) -> None:
        class Client:
            def __init__(self):
                self.started = threading.Event()
                self.closed = False
            def transcribe_audio(self, _): return "Jeremy, explain this component"
            def stream_response(self, _question, _context, on_text, cancelled):
                on_text("The processor handles instructions.")
                self.started.set()
                while not cancelled():
                    time.sleep(.01)
                raise InterruptedError()
            def generate_speech(self, _): raise AssertionError("cancelled stream must not speak")
            def close(self): self.closed = True
        client = Client()
        mic = _FakeMicrophone()
        assistant = JeremyAssistant(_settings(utterance_silence_seconds=0.05), client=client, microphone=mic,
                                    wake_detector=_FakeWake([True]), player=_FakePlayer())
        assistant.start()
        self.assertTrue(_wait_until(lambda: assistant.status().label != "INITIALIZING"))
        mic.push(_pcm(9000))
        mic.push(_pcm(0))
        self.assertTrue(client.started.wait(timeout=1))
        self.assertEqual(assistant.state, JeremyState.PROCESSING)
        self.assertIn("processor", assistant.status().detail)
        started = time.monotonic()
        assistant.close()
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertTrue(client.closed)


if __name__ == "__main__":
    unittest.main()
