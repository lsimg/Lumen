"""Exercise Jeremy's microphone, local Vosk STT, and local Silero TTS."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ai.audio import AudioPlayer, AudioRecorder, MicrophoneInput, input_devices, pcm_rms
from ai.settings import JeremySettings
from ai.stt_vosk import VoskSTT
from ai.tts import JeremyTTS


def _safe(value: object) -> str:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return str(value).encode(encoding, errors="backslashreplace").decode(encoding, errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=5.0, help="microphone recording duration")
    parser.add_argument("--no-play", action="store_true", help="synthesize but do not play samples")
    args = parser.parse_args()

    settings = JeremySettings.load(PROJECT_DIR)
    print("[Jeremy] Available input devices:")
    for device in input_devices():
        print(
            f"  {device.index}: {_safe(device.name)} | inputs={device.max_input_channels} "
            f"| default_rate={device.default_sample_rate} Hz"
        )

    microphone = MicrophoneInput(settings.microphone_sample_rate)
    recorder = AudioRecorder(microphone)
    stt = VoskSTT(settings.vosk_en_model_path, settings.microphone_sample_rate)
    tts = JeremyTTS(settings)
    player = AudioPlayer()
    try:
        microphone.start()
        print("[Jeremy] Microphone initialized")
        print(f"[Jeremy] Input device: {microphone.device_index} - {_safe(microphone.device_name)}")
        print("[Jeremy] Input channels: 1")
        print(f"[Jeremy] Native sample rate: {microphone.device_sample_rate} Hz")
        print(f"[Jeremy] Recording started ({args.seconds:.1f} seconds). Speak now...")
        audio = recorder.record(args.seconds)
        print("[Jeremy] Recording finished")
        print(f"[Jeremy] RMS/volume level: {pcm_rms(audio):.6f}")

        print("[Jeremy] Loading local Vosk models")
        stt.initialize()
        print(f"[Jeremy] STT model: {settings.vosk_en_model_path.name}")
        print("[Jeremy] STT language: English")
        result = stt.transcribe(audio)
        print(f"[Jeremy] STT language: {result.language}")
        print(f"[Jeremy] STT confidence: {result.confidence:.3f}")
        print(f"[Jeremy] Transcript: {_safe(result.text or '<empty>')}")

        print("[Jeremy] Loading local Silero TTS")
        tts.initialize()
        print(f"[Jeremy] TTS engine: {tts.engine.upper()}")
        print(f"[Jeremy] TTS device: {tts.device.upper()}")
        speech = tts.synthesize("Jeremy's English voice system is working.")
        print(
            f"[Jeremy] English male TTS generated: "
            f"{len(speech.pcm16_mono) // 2} samples at {speech.sample_rate} Hz"
        )
        if not args.no_play:
            print("[Jeremy] Playing English sample")
            player.play_pcm(speech.pcm16_mono, speech.sample_rate)
        print("[Jeremy] Local audio diagnostic passed")
        return 0
    except Exception as error:
        print(f"[Jeremy][ERROR] {_safe(error)}", file=sys.stderr)
        return 1
    finally:
        recorder.close()
        stt.shutdown()
        tts.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
