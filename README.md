# Lumen

Lumen is one local, gesture-controlled educational 3D platform with four paths:
Robotics, Computer Components, Human Anatomy, and Interactive Development for
Children. It keeps the original camera HUD and gesture workflow while preserving
each model's authored materials, textures, colors, and separate parts. A subtle
Lumen highlight is added only to the hovered or selected part.

Jeremy is an optional voice layer built into the same application. He uses the
current model, selected/hovered component, interaction mode, gesture, tracking,
camera, transform, error, robotics, and game state when answering. Jeremy does
not replace or modify the renderer, models, gesture controls, or model UI.

## Requirements

- Python 3.12
- A webcam
- A graphics driver with OpenGL 3.3 support
- Windows, Linux, or macOS with access to the camera
- A microphone; internet is needed only for Jeremy's generated answer
- An OpenRouter API key and a configured text LLM
- Local large English Vosk and PyTorch/Silero English speech models

## Run

On Windows, double-click `run.bat`. The launcher creates the local Python
environment and installs the required packages automatically on its first run.
Later launches start Lumen immediately.

From any Command Prompt window, the same launcher can be started with:

```bat
"C:\Users\gulsy\OneDrive\Desktop\Golography\Lumen\run.bat"
```

From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

On Linux or macOS, activate the environment with
`source .venv/bin/activate` instead.

## Jeremy voice assistant

Copy `.env.example` to `.env` in this directory, then set:

```dotenv
OPENROUTER_API_KEY=your-key-here
OPENROUTER_LLM_MODEL=google/gemini-3.1-flash-lite
OPENROUTER_FALLBACK_MODELS=openai/gpt-4.1-mini,mistralai/mistral-small-3.2-24b-instruct
OPENROUTER_CONNECT_TIMEOUT_SECONDS=3
OPENROUTER_TIMEOUT_SECONDS=9
OPENROUTER_TOTAL_TIMEOUT_SECONDS=28

VOSK_EN_MODEL_PATH=../vosk-model-en-us-0.22
JEREMY_STT_BACKEND=vosk
JEREMY_STT_DEVICE=auto

TTS_ENGINE=silero
SILERO_EN_MODEL_PATH=assets/ai_models/silero/v3_en.pt
SILERO_EN_SPEAKER=en_1
JEREMY_TTS_DEVICE=auto
JEREMY_TTS_SAMPLE_RATE=24000
JEREMY_SILENCE_SECONDS=1.35
JEREMY_MAX_UTTERANCE_SECONDS=25
```

The key is loaded with `python-dotenv`; `.env` is ignored by Git and must never
be committed. Jeremy streams the answer through OpenRouter and uses the
configured backup models when the primary fails. The free router is the last
backup and may select different models over time. Audio remains local.

Jeremy uses the large English model `vosk-model-en-us-0.22` located in the
workspace next to the `Lumen` project. The path is relative to this project and
auto-discovery selects the largest valid local English Vosk model; small models
are never used as fallback. The local Silero English model is stored in
`assets/ai_models/silero/v3_en.pt` and uses the male `en_1` speaker.

Jeremy starts in `SLEEPING`. PocketSphinx performs a one-keyphrase search on
the microphone locally. No ambient audio is sent to OpenRouter in this state.
Only a local match for “Jeremy” opens a short recording session. The completed
utterance is transcribed locally by the English Vosk model. All finalized Vosk
segments in the same utterance are joined before the text and current Lumen
state are sent to OpenRouter. A short pause does not end the question.
The answer is synthesized locally by PyTorch/Silero in English and played. A wake-only
utterance waits silently for the question; after about seven seconds it returns
to sleep. Every answer is one-shot, so the name must be spoken again for
another question.

All recording, Vosk work, network work, synthesis, and playback run on Jeremy's background worker.
The OpenGL loop, camera thread, MediaPipe callbacks, gesture state machine, and
UI continue independently. Temporary request audio is held in memory and is
not written to disk.

If the OpenRouter key or English model is missing, the rest of Lumen starts
normally and the existing HUD reports the specific Jeremy connection/model
status. Microphone or network errors also remain isolated to Jeremy.

`JEREMY_DEBUG=1` enables timing and hardware logs without adding technical
details to the HUD. `JEREMY_DEBUG_BENCHMARK=1` compares CPU and accelerator TTS
latency when an accelerator is available. Vosk's installed Python backend uses
CPU. For optional GPU-capable STT, install `faster-whisper` and `ctranslate2`,
place an already downloaded model in `assets/ai_models/faster-whisper`, then set
`JEREMY_STT_BACKEND=faster-whisper`. No model is downloaded automatically.
`JEREMY_STT_DEVICE` and `JEREMY_TTS_DEVICE` can independently be `auto`, `cuda`,
`mps`, or `cpu`; unsupported or unavailable accelerators fall back to CPU.

### Jeremy microphone diagnostic

Close Lumen first so it releases the microphone, then run this from the
`Lumen` directory and speak an English sentence for five seconds:

```bat
.venv\Scripts\python.exe -m ai.microphone_test
```

The diagnostic lists every Windows input device, prefers the default input
device, records mono PCM16 audio, converts unsupported device rates to 16 kHz,
runs the large local English Vosk model, prints the final transcript, then synthesizes
and plays one local English male voice sample. It never needs the API key or
uploads audio. To synthesize without speaker playback, run:

```bat
.venv\Scripts\python.exe -m ai.microphone_test --no-play
```

The motherboard remains in `assets/`. The existing robot, anatomy, fruit, and
reference video files remain in `../Models_Examples` and are registered in
`config.MODEL_CATALOG`; they are not duplicated or replaced. Missing model,
metadata, tracking, texture, or camera input is reported without repeatedly
flooding the console. Mouse mode remains available when hand tracking is not.

## Controls

| Input | Action |
| --- | --- |
| Point | Hover a component and show its name and short purpose |
| Pinch + hold | Select a component and show its longer description |
| Pinch empty space | Clear the selection |
| Open palm | Move the model |
| V gesture | Rotate the model |
| Two hands | Scale the model |
| Rock sign (robotics) | Hold to start/stop the joint demonstration |
| Pinch and drag (robotics) | Move the selected joint or gripper |
| Pinch and drag (Match the Fruit) | Move one logical fruit into its matching box |
| Two hands (Match the Size) | Scale the pear toward the tolerant target size |
| Fist | Stop interaction |
| Palm after stop | Resume interaction |
| Left mouse drag | Rotate in mouse mode |
| Mouse wheel | Scale in mouse mode |

Keyboard shortcuts:

- `H` — tutorial
- `D` — debug information
- `R` — reset the view
- `X` — clear the selected component
- `Space` — stop interaction
- `Enter` — resume interaction
- `B` — switch between hand and mouse control
- `Esc` or `Q` — exit

The MODELS panel now contains the four education paths and the two interactive
lessons. MODELS and HELP capture the pointer, so their pinch actions never pass
through to the 3D model.

## Project structure

```text
Lumen/
├── main.py
├── run.bat
├── config.py
├── requirements.txt
├── README.md
├── core/
│   ├── camera.py
│   ├── tracking.py
│   ├── gestures.py
│   ├── renderer.py
│   ├── model.py
│   ├── interaction.py
│   ├── robotics.py
│   ├── games.py
│   └── ui.py
├── ai/
│   ├── assistant.py
│   ├── audio.py
│   ├── context.py
│   ├── openrouter_client.py
│   ├── prompts.py
│   ├── settings.py
│   ├── stt_vosk.py
│   ├── tts.py
│   └── wake_word.py
├── tools/
│   └── test_jeremy_audio.py
├── assets/
│   ├── motherboard.glb
│   ├── motherboard.json
│   ├── robot.json
│   ├── anatomy.json
│   ├── fruits.json
│   └── hand_landmarker.task
├── tests/
│   └── test_platform.py
└── shaders/
    ├── screen.vert
    ├── camera.frag
    ├── model.vert
    ├── model.frag
    └── overlay.frag
```

Jeremy adds `httpx`, `python-dotenv`, `sounddevice`, `soundfile`,
`pocketsphinx`, `vosk`, `torch`, `torchaudio`, and `omegaconf`.
PocketSphinx is used only for the local English wake name, Vosk handles local English STT,
OpenRouter handles text reasoning, and Silero handles local TTS. Wake accuracy
depends on the microphone, room noise, and accent; `JEREMY_WAKE_THRESHOLD` and
`JEREMY_SPEECH_RMS` can be tuned in `.env` without changing code.

## Model metadata

The JSON metadata files keep educational content out of the renderer. Each
component has a stable ID, name, short purpose, longer description, importance
flag, and exact GLB selectors. Selectors may match authored node, mesh,
ancestor, or material names. Geometry without verified metadata remains visible
and blocks raycasts, but it is not given an invented label.

`core/model.py` supports GLB, GLTF, STL, OBJ, PLY, and 3MF through the shared
scene loader. It keeps hierarchy transforms, individual primitives, PBR base
colors/textures, triangle or native line topology, local/world bounds, and
raycast caches. Debug mode (`D`) shows the exact node, mesh, material, and target
under the pointer so generic names can be mapped manually without guessing.

Run validation with:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

## Motherboard model attribution

`motherboard.glb` is **MotherBoard + Components** by
[Daniel Cardona](https://sketchfab.com/DanielCardonaArt), obtained from
[Sketchfab](https://sketchfab.com/3d-models/motherboard-components-3bc94057328243d4b341a55f59160f8a)
and used under the
[Creative Commons Attribution 4.0 license](https://creativecommons.org/licenses/by/4.0/).
