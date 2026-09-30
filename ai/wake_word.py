"""Offline wake-word detection and invocation parsing for Jeremy."""

from __future__ import annotations

import os
import re
import threading

try:
    from pocketsphinx import Decoder
except ImportError:
    Decoder = None  # type: ignore[assignment]


_WAKE_PATTERN = re.compile(r"(?<!\w)(jeremy)(?!\w)", re.IGNORECASE)
_TRIM_PUNCTUATION = " \t\r\n,.:;!?—–-\"'()[]{}"


class WakeWordDetector:
    """Small replaceable keyword spotter.

    PocketSphinx runs entirely on-device and uses a constrained one-word search.
    The English keyphrase is processed locally; only the final Vosk utterance
    transcript is used for the user request.
    """

    def __init__(self, sample_rate: int = 16_000, threshold: float = 1e-24) -> None:
        self.sample_rate = int(sample_rate)
        self.threshold = float(threshold)
        self._lock = threading.Lock()
        self._decoder = None
        self._error: str | None = None
        if Decoder is None:
            self._error = "PocketSphinx is not installed"
            return
        try:
            self._decoder = Decoder(
                keyphrase="jeremy",
                kws_threshold=self.threshold,
                samprate=self.sample_rate,
                logfn=os.devnull,
            )
            self._decoder.start_utt()
        except Exception as error:
            self._decoder = None
            self._error = f"Wake-word detector is unavailable: {error}"

    @property
    def available(self) -> bool:
        return self._decoder is not None

    @property
    def error(self) -> str | None:
        return self._error

    @staticmethod
    def contains_wake_word(text: str) -> bool:
        return bool(_WAKE_PATTERN.search(str(text or "")))

    @staticmethod
    def extract_invocation(text: str) -> tuple[bool, str]:
        value = str(text or "")
        match = _WAKE_PATTERN.search(value)
        if match is None:
            return False, ""
        remainder = value[: match.start()] + " " + value[match.end() :]
        remainder = " ".join(remainder.split()).strip(_TRIM_PUNCTUATION)
        return True, remainder

    @staticmethod
    def strip_wake_word(text: str) -> str:
        found, remainder = WakeWordDetector.extract_invocation(text)
        return remainder if found else str(text or "").strip()

    def reset(self) -> None:
        decoder = self._decoder
        if decoder is None:
            return
        with self._lock:
            try:
                decoder.end_utt()
            except Exception:
                pass
            decoder.start_utt()

    def process_audio(self, pcm16_mono: bytes) -> bool:
        """Return True only when the local decoder hears the configured name."""

        decoder = self._decoder
        if decoder is None or not pcm16_mono:
            return False
        with self._lock:
            try:
                decoder.process_raw(bytes(pcm16_mono), False, False)
                hypothesis = decoder.hyp()
                detected = bool(
                    hypothesis is not None
                    and self.contains_wake_word(getattr(hypothesis, "hypstr", ""))
                )
                if detected:
                    decoder.end_utt()
                    decoder.start_utt()
                return detected
            except Exception as error:
                self._error = f"Wake-word detection failed: {error}"
                return False

    def close(self) -> None:
        decoder, self._decoder = self._decoder, None
        if decoder is not None:
            try:
                decoder.end_utt()
            except Exception:
                pass


__all__ = ["WakeWordDetector"]
