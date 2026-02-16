# Quick Start Guide - Talking Characters

Get up and running in 5 minutes! 🚀

## Prerequisites

- ComfyUI running (`python main.py`)
- Python 3.10+
- 8GB+ VRAM (GPU) or 16GB+ RAM

## Installation (2 minutes)

```bash
# Navigate to project
cd D:\AI\ComfyUI\talking-characters

# Install dependencies
pip install -r requirements.txt

# Setup TTS (choose one)

# Option 1: Piper (Polish - RECOMMENDED)
pip install piper-tts
piper --download --voice pl_PL-dark-medium

# Option 2: Bark (Natural)
pip install bark

# Option 3: Google TTS
pip install gtts
```

## Testing (1 minute)

```bash
# Verify everything works
python -m scripts.test_system

# Expected output:
# ✅ PASS - Python Version
# ✅ PASS - FFmpeg
# ✅ PASS - Piper TTS
# ✅ PASS - ComfyUI API
# ... etc
```

## Web UI (2 minutes)

```bash
# Start web server
python -m scripts.web_ui

# Open browser
# http://localhost:5000

# Done! Generate videos from browser
```

## Command Line

```bash
# Generate single video
python -m scripts.video_generator

# Or custom script
python scripts/example_usage.py
```

## Try Pre-Scripted Scenes

```python
from scripts.dialogue_scenes import get_jessica_scenes, get_gigi_scenes
from scripts.video_generator import TalkingCharacterGenerator

# Get Jessica scenes
jessica_scenes = get_jessica_scenes()
scene = jessica_scenes[0]  # Get first scene

# Generate
gen = TalkingCharacterGenerator(
    character="jessica",
    dialogue_text=scene.dialogue,
    tts_engine="piper",
    output_name=scene.id
)
gen.generate()
```

## Directory Structure

```
Generated files are saved to:
├── talking-characters/audio/    ← Audio files (.wav)
├── talking-characters/video/    ← Video files (.mp4)
└── talking-characters/models/   ← Downloaded models
```

## Troubleshooting

### FFmpeg not found
```bash
# Windows
choco install ffmpeg

# or download: https://ffmpeg.org/download.html
```

### Piper not working
```bash
pip install --upgrade piper-tts
piper --download --voice pl_PL-dark-medium
```

### ComfyUI connection error
```bash
# Make sure ComfyUI is running in another terminal:
cd ..
python main.py
```

### Out of memory
- Reduce workflow steps (40 → 20)
- Use Piper instead of Bark
- Reduce image resolution

## Next Steps

1. ✅ Run `test_system.py` to verify setup
2. ✅ Open web UI: `python -m scripts.web_ui`
3. ✅ Generate first video
4. ✅ Check output in `talking-characters/video/`
5. ✅ Try pre-scripted scenes from `dialogue_scenes.py`

## Features Available

| Feature | Status | Time |
|---------|--------|------|
| TTS (Piper) | ✅ | 5-10s |
| TTS (Bark) | ✅ | 30-60s |
| Flux Image | ✅ | 30-60s |
| Lip-sync | ✅ | 60-120s |
| Web UI | ✅ | instant |

## Support

- **README.md** - Full documentation
- **SETUP.md** - Detailed setup guide
- **scripts/test_system.py** - System diagnostics
- **scripts/example_usage.py** - Code examples

---

**Questions?** Check README.md for detailed documentation.

**Ready?** Start the web UI: `python -m scripts.web_ui`
