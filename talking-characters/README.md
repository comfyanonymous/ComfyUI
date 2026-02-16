# Talking Characters - Flux Video Generation System

Generate talking character videos locally using Flux AI image generation, text-to-speech, and Wav2Lip lip-sync synchronization.

## 🎯 Features

- ✅ **Local Processing** - No external APIs required (except optional gTTS)
- ✅ **Multiple TTS Engines** - Piper (Polish), Bark, Google TTS
- ✅ **Lip-Sync** - Wav2Lip integration for realistic mouth movements
- ✅ **Character Support** - Jessica, Gigi, extensible for more
- ✅ **Polish Language** - Full Polish text-to-speech support
- ✅ **Batch Processing** - Generate multiple videos sequentially

## 📋 System Architecture

```
Text Input
    ↓
[TTS Engine] → Speech Audio
    ↓
[Flux Workflow] → Character Image
    ↓
[Image to Video] → Base Video
    ↓
[Wav2Lip] → Lip-synced Video
    ↓
[Audio Merge] → Final Video
```

## 🚀 Quick Start

### 1. Installation

```bash
cd talking-characters

# Install dependencies
pip install -r requirements.txt

# Optional: Install FFmpeg
# Windows: choco install ffmpeg
# macOS: brew install ffmpeg
# Linux: sudo apt install ffmpeg
```

### 2. Setup TTS (Choose One)

**Option A: Piper (Recommended for Polish)**
```bash
pip install piper-tts
piper --download --voice pl_PL-dark-medium
```

**Option B: Bark (Natural sounding)**
```bash
pip install bark
```

**Option C: Google TTS (Simple, requires internet)**
```bash
pip install gtts
```

### 3. Setup Wav2Lip (Optional, for lip-sync)

```bash
# Wav2Lip will auto-download on first use
# Or manually:
git clone https://github.com/Rudrabha/Wav2Lip.git
cd Wav2Lip
wget "https://www.adrianbulat.com/downloads/deep_learning/state_dict" -O checkpoints/wav2lip.pth
```

### 4. Ensure ComfyUI is Running

```bash
# Terminal 1: Start ComfyUI
cd ..
python main.py
```

### 5. Generate Video

```bash
# Terminal 2: Generate talking character
cd talking-characters
python -m scripts.video_generator
```

## 📝 Configuration

Edit `config.py` to customize:

- **TTS Engine**: Change `TTS_ENGINE` to "piper", "bark", or "gtts"
- **Language**: Set `TTS_LANGUAGE` for your locale
- **Characters**: Add new characters to `CHARACTERS` dict
- **Video Settings**: FPS, codec, bitrate

## 🎬 Usage Examples

### Single Video Generation

```python
from scripts.video_generator import TalkingCharacterGenerator

# Create Jessica saying something
generator = TalkingCharacterGenerator(
    character="jessica",
    dialogue_text="Cześć, jak się masz?",
    tts_engine="piper",
    output_name="jessica_hello"
)

generator.generate()
```

### Batch Generation

```python
from scripts.video_generator import TalkingCharacterGenerator

scenes = [
    ("jessica", "Cześć, czekam na ciebie"),
    ("jessica", "Chcę się z tobą bawić"),
    ("gigi", "Cześć, jestem gotowa"),
    ("gigi", "Tak, tak, jeszcze!"),
]

for character, text in scenes:
    generator = TalkingCharacterGenerator(
        character=character,
        dialogue_text=text,
        tts_engine="piper"
    )
    generator.generate()
```

### Custom Workflow

```python
from scripts.video_generator import TalkingCharacterGenerator
from config import CHARACTERS

# Add custom character to config first
CHARACTERS["custom"] = {
    "name": "Custom",
    "workflow": "custom_workflow.json",
    "character_strength": 0.7
}

# Then use it
generator = TalkingCharacterGenerator(
    character="custom",
    dialogue_text="Custom dialogue..."
)
```

## 📂 Directory Structure

```
talking-characters/
├── scripts/
│   ├── tts_engine.py       # Text-to-speech engines
│   ├── lip_sync.py         # Wav2Lip synchronization
│   └── video_generator.py  # Main pipeline
├── audio/                  # Generated audio files
├── video/                  # Generated videos
├── models/                 # AI model checkpoints
├── workflows/              # ComfyUI workflows
├── docs/                   # Documentation
├── config.py              # Configuration
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## 🔧 Troubleshooting

### Piper not found
```bash
pip install piper-tts
piper --download --voice pl_PL-dark-medium
```

### FFmpeg not found
- **Windows**: `choco install ffmpeg` or download from ffmpeg.org
- **Linux**: `sudo apt install ffmpeg`
- **macOS**: `brew install ffmpeg`

### Wav2Lip model download fails
```bash
# Manual download
wget "https://www.adrianbulat.com/downloads/deep_learning/state_dict" \
  -O models/wav2lip.pth
```

### ComfyUI API connection error
- Ensure ComfyUI is running: `python main.py`
- Check API is on localhost:8188
- Verify workflows exist in `user/default/workflows/`

### Out of memory (GPU)
- Reduce video resolution in workflow
- Use Piper instead of Bark (lighter)
- Reduce CFG/steps in Flux workflow

## 📊 Performance

Approximate times (GPU with 8GB VRAM):

| Step | Time | Notes |
|------|------|-------|
| Flux image | 30-60s | Depends on steps/CFG |
| TTS (Piper) | 5-10s | Fast, no GPU needed |
| TTS (Bark) | 30-60s | GPU accelerated |
| Image→Video | 2-5s | FFmpeg, fast |
| Wav2Lip | 60-120s | GPU heavy, most time |
| Audio merge | 2-5s | FFmpeg, fast |
| **Total** | **2-5 min** | Per video |

## 🎨 Customization

### Add New Character

1. Create workflow in ComfyUI
2. Save to `user/default/workflows/`
3. Add to `config.py`:

```python
CHARACTERS["new_char"] = {
    "name": "New Character",
    "workflow": "TEST_NewChar_v1.json",
    "character_strength": 0.7,
    "prompt_template": "New character {scene_description}..."
}
```

### Add Dialogue Scenes

```python
DIALOGUE_SCENES["new_scene"] = {
    "character": "new_char",
    "text": "Nowy dialog",
    "duration": 3
}
```

## 🚀 Advanced Features

### Custom TTS Voice (Bark)

```python
from scripts.tts_engine import TTSEngine

tts = TTSEngine("bark")
tts.synthesize(
    "Custom text",
    "output.wav",
    speaker="v2/en_speaker_9"  # Different voice
)
```

### Frame Interpolation (Smooth Motion)

```bash
# Add between image→video and wav2lip for smoother results
ffmpeg -i input.mp4 -vf "fps=60" -c:v libx264 smooth.mp4
```

### Video Enhancement

```bash
# Upscale video quality
ffmpeg -i input.mp4 -vf "scale=1920:1080" output.mp4

# Add effects
ffmpeg -i input.mp4 -vf "eq=brightness=0.1" output.mp4
```

## 📝 License & Attribution

- **Flux**: Developed by Black Forest Labs
- **Wav2Lip**: https://github.com/Rudrabha/Wav2Lip
- **Piper TTS**: https://github.com/rhasspy/piper
- **Bark**: https://github.com/suno-ai/bark

## 🤝 Contributing

To add features:

1. Create feature branch
2. Add code to appropriate module
3. Update config.py if needed
4. Test thoroughly
5. Document changes

## 📞 Support

For issues:
- Check config.py settings
- Verify all dependencies installed
- Check ComfyUI is running
- Review logs in DEBUG mode

---

**Created**: February 2026
**Status**: Active Development
**Branch**: feature/talking-characters
