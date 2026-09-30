"""Streaming OpenRouter text generation with bounded retries and model failover."""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Callable

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

from .context import LumenAIContext
from .prompts import JEREMY_SYSTEM_PROMPT
from .settings import JeremySettings


LOGGER = logging.getLogger("lumen.jeremy")


class OpenRouterError(RuntimeError):
    def __init__(self, stage: str, message: str, status_code: int | None = None,
                 *, model: str = "", response_text: str = "", exception_type: str = "",
                 elapsed: float = 0.0, retry: int = 0) -> None:
        super().__init__(message)
        self.stage = stage
        self.status_code = status_code
        self.model = model
        self.response_text = response_text
        self.exception_type = exception_type
        self.elapsed = elapsed
        self.retry = retry


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
    return ""


class OpenRouterClient:
    """Online reasoning only; audio remains entirely local."""

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, settings: JeremySettings, client: Any | None = None) -> None:
        if httpx is None:
            raise RuntimeError("httpx is not installed")
        self.settings = settings
        self._models = tuple(dict.fromkeys((settings.llm_model, *settings.fallback_models)))
        self._history: list[dict[str, str]] = []
        self._history_lock = threading.Lock()
        self._client = client or httpx.Client(
            base_url=self.BASE_URL,
            timeout=httpx.Timeout(connect=settings.connect_timeout,
                                  read=settings.request_timeout,
                                  write=settings.connect_timeout,
                                  pool=settings.connect_timeout),
            headers={"Authorization": f"Bearer {settings.api_key}",
                     "X-OpenRouter-Title": "Lumen / Jeremy",
                     "X-OpenRouter-Metadata": "enabled"},
        )
        self.last_model = ""
        self.last_provider = ""
        self.last_status: int | None = None
        self.first_token_seconds: float | None = None
        self.total_seconds = 0.0

    @property
    def models(self) -> tuple[str, ...]:
        return self._models

    def validate_models(self) -> None:
        """Check live IDs on the worker; temporary catalog failure is nonfatal."""
        try:
            response = self._client.get("/models", timeout=self.settings.connect_timeout + 2)
            response.raise_for_status()
            available = {item.get("id") for item in response.json().get("data", [])}
            valid = tuple(model for model in self._models if model in available)
            for model in self._models:
                if model not in available:
                    LOGGER.warning("[Jeremy] OpenRouter model unavailable: %s", model)
            if valid:
                self._models = valid
            else:
                LOGGER.warning("[Jeremy] No configured OpenRouter model found in live catalog")
        except Exception as error:
            LOGGER.warning("[Jeremy] OpenRouter model validation unavailable: %s",
                           error.__class__.__name__)

    def _request_messages(self, question: str, context: LumenAIContext) -> list[dict[str, str]]:
        user = json.dumps({"lumen_state": context.to_dict(), "user_request": question},
                          ensure_ascii=False, separators=(",", ":"))
        with self._history_lock:
            history = list(self._history)
        return [{"role": "system", "content": JEREMY_SYSTEM_PROMPT},
                *history, {"role": "user", "content": user}]

    def _error(self, error: Exception, model: str, started: float, retry: int) -> OpenRouterError:
        status = (error.response.status_code if isinstance(error, httpx.HTTPStatusError)
                  else self.last_status)
        body = ""
        if isinstance(error, httpx.HTTPStatusError):
            body = " ".join(error.response.text.split())[:500]
        else:
            body = " ".join(str(error).split())[:500]
        if self.settings.api_key:
            body = body.replace(self.settings.api_key, "[redacted]")
        result = OpenRouterError("LLM", "OpenRouter request failed", status,
                                 model=model, response_text=body,
                                 exception_type=error.__class__.__name__,
                                 elapsed=time.monotonic() - started, retry=retry)
        LOGGER.warning(
            "[Jeremy] OpenRouter error: model=%s HTTP=%s type=%s elapsed=%.2fs "
            "connect_timeout=%.1fs read_timeout=%.1fs retry=%d body=%s",
            model, status or "-", result.exception_type, result.elapsed,
            self.settings.connect_timeout, self.settings.request_timeout,
            retry, body or "-",
        )
        return result

    @staticmethod
    def _retryable(error: OpenRouterError) -> bool:
        return error.status_code in {429, 500, 502, 503, 504} or error.exception_type in {
            "ConnectError", "ConnectTimeout", "ReadTimeout", "ReadError", "RemoteProtocolError"
        }

    def _stream_once(self, payload: dict[str, Any], on_text: Callable[[str], None],
                     cancelled: Callable[[], bool], retry: int, deadline: float) -> str:
        requested = str(payload["model"])
        started = time.monotonic()
        chunks: list[str] = []
        self.first_token_seconds = None
        self.last_status = None
        self.last_model = ""
        self.last_provider = ""
        try:
            with self._client.stream("POST", "/chat/completions", json=payload) as response:
                self.last_status = response.status_code
                if response.status_code >= 400:
                    response.read()
                    response.raise_for_status()
                for line in response.iter_lines():
                    if cancelled():
                        raise InterruptedError("Jeremy stopped")
                    if time.monotonic() > deadline:
                        raise TimeoutError("OpenRouter total stream timeout")
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                    except ValueError:
                        continue
                    if event.get("error"):
                        raise ValueError(str(event["error"])[:500])
                    self.last_model = str(event.get("model") or self.last_model or requested)
                    metadata = event.get("openrouter_metadata") or {}
                    self.last_provider = str(event.get("provider") or metadata.get("provider")
                                             or self.last_provider or "")
                    choices = event.get("choices") or []
                    delta = (choices[0].get("delta") or {}) if choices else {}
                    token = _message_text(delta.get("content"))
                    if token:
                        if self.first_token_seconds is None:
                            self.first_token_seconds = time.monotonic() - started
                        chunks.append(token)
                        on_text("".join(chunks))
        except InterruptedError:
            raise
        except Exception as error:
            # A completed sentence already streamed to the user is more useful
            # than discarding it after a late transport interruption.
            partial = "".join(chunks).strip()
            failure = self._error(error, requested, started, retry)
            if self._retryable(failure) and len(partial) >= 25:
                boundary = max(partial.rfind("."), partial.rfind("?"), partial.rfind("!"))
                if boundary >= 15:
                    self.total_seconds = time.monotonic() - started
                    return partial[:boundary + 1]
            raise failure from error
        answer = "".join(chunks).strip()
        self.total_seconds = time.monotonic() - started
        if not answer:
            raise self._error(ValueError("Empty streamed answer"), requested, started, retry)
        return answer

    def stream_response(self, question: str, context: LumenAIContext,
                        on_text: Callable[[str], None],
                        cancelled: Callable[[], bool] = lambda: False) -> str:
        messages = self._request_messages(question, context)
        base: dict[str, Any] = {
            "messages": messages, "temperature": 0.25,
            "max_tokens": self.settings.max_response_tokens, "stream": True,
            "provider": {"allow_fallbacks": True, "sort": "latency"},
        }
        last_error: OpenRouterError | None = None
        models = self._models
        deadline = time.monotonic() + self.settings.total_timeout
        for index, model in enumerate(models):
            attempts = self.settings.request_retries + 1 if index == 0 else 1
            for retry in range(attempts):
                if cancelled():
                    raise InterruptedError("Jeremy stopped")
                if time.monotonic() >= deadline:
                    raise last_error or OpenRouterError("LLM", "OpenRouter total timeout",
                                                       model=model, exception_type="TimeoutError")
                on_text("")
                payload = dict(base, model=model)
                if index == 0 and len(models) > 1:
                    payload["models"] = list(models[1:])
                try:
                    answer = self._stream_once(payload, on_text, cancelled, retry, deadline)
                    with self._history_lock:
                        self._history.extend((messages[-1], {"role": "assistant", "content": answer}))
                        self._history = self._history[-4:]
                    if self.settings.debug:
                        LOGGER.info("[Jeremy][DEBUG] LLM requested=%s used=%s provider=%s HTTP=%s first_token=%s total=%.2fs",
                                    model, self.last_model or model, self.last_provider or "unknown",
                                    self.last_status, self.first_token_seconds, self.total_seconds)
                    return answer
                except OpenRouterError as error:
                    last_error = error
                    if error.status_code in {400, 401, 403, 422}:
                        raise
                    if not self._retryable(error) or retry + 1 >= attempts:
                        break
                    if cancelled():
                        raise InterruptedError("Jeremy stopped")
                    time.sleep(min(0.3 * (2 ** retry), 0.7))
        raise last_error or OpenRouterError("LLM", "OpenRouter unavailable")

    def generate_response(self, question: str, context: LumenAIContext) -> str:
        return self.stream_response(question, context, lambda _text: None)

    def close(self) -> None:
        self._client.close()


__all__ = ["OpenRouterClient", "OpenRouterError"]
