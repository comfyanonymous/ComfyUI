# Setup Guide - Talking Characters

Complete step-by-step guide to set up the talking characters system.

## Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- 8GB+ VRAM (GPU) or 16GB+ RAM (CPU)
- FFmpeg
- ComfyUI running

## 1. Python Environment

```bash
# Create virtual environment (recommended)
cd talking-characters
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

## 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Install FFmpeg

### Windows
```bash
# Using Chocolatey
choco install ffmpeg

# Or download from: https://ffmpeg.org/download.html
```

### macOS
```bash
brew install ffmpeg
```

### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install ffmpeg
```

## 4. Install Text-to-Speech Engine

### Option A: Piper (Recommended for Polish)

```bash
pip install piper-tts

# Download Polish voice model
piper --download --voice pl_PL-dark-medium

# Test
echo "Cześć świecie" | piper --model pl_PL-dark-medium --output_file test.wav
```

### Option B: Bark

```bash
pip install bark

# Test (will auto-download models on first run)
python -c "
from bark import generate_audio, save_wav
from scipy.io import wavfile

audio_array = generate_audio('Hello world')
save_wav(audio_array, 'test.wav')
"
```

### Option C: Google TTS

```bash
pip install gtts

# Test
python -c "
from gtts import gTTS
tts = gTTS('Hello world', lang='en')
tts.save('test.wav')
"
```

## 5. Install Wav2Lip (Optional, for lip-sync)

```bash
# Auto-setup on first run, or manual:
cd ..
git clone https://github.com/Rudrabha/Wav2Lip.git
cd Wav2Lip

# Download model checkpoint
mkdir -p checkpoints
wget "https://www.adrianbulat.com/downloads/deep_learning/state_dict" \
  -O checkpoints/wav2lip.pth

# Install Wav2Lip dependencies
pip install -r requirements.txt

cd ../talking-characters
```

## 6. Configure Settings

Edit `config.py`:

```python
# TTS Engine (default: piper)
TTS_ENGINE = "piper"

# Language (default: Polish)
TTS_LANGUAGE = "pl_PL"

# Piper model
TTS_PIPER_MODEL = "dark-medium"

# Video settings
VIDEO_FPS = 24
VIDEO_CODEC = "libx264"

# Debug mode
DEBUG = True
```

## 7. Verify Installation

```bash
# Test TTS
python -c "
from scripts.tts_engine import test_tts_engines
test_tts_engines()
"

# Test Wav2Lip setup
python -c "
from scripts.lip_sync import setup_wav2lip
setup_wav2lip()
"

# Test API connection
python -c "
import requests
try:
    response = requests.get('http://localhost:8188/system_stats')
    print('✅ ComfyUI API connected')
except:
    print('❌ ComfyUI not running on localhost:8188')
"
```

## 8. Test Generation

```bash
# Ensure ComfyUI is running in another terminal:
# python main.py

# Then run:
python -m scripts.video_generator
```

## 🎉 Setup Complete!

You're ready to generate talking character videos. See `README.md` for usage examples.

## 📦 GPU Acceleration

For NVIDIA GPUs:

```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For Bark
export CUDA_VISIBLE_DEVICES=0
```

For AMD GPUs (experimental):

```bash
# Install ROCm PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

## 💾 Storage Requirements

- Flux checkpoint: ~13GB
- Wav2Lip checkpoint: ~350MB
- Audio files: ~1-2MB per video
- Video files: ~10-50MB per video (depends on resolution/duration)

**Total**: ~13.5GB minimum

## 🐛 Common Issues & Solutions

### "piper: command not found"
```bash
# Reinstall with user flag
pip install --user piper-tts
# Or add to PATH: ~/.local/bin
```

### "No module named 'bark'"
```bash
pip install bark
# First run will be slow (downloading ~1.5GB models)
```

### "CUDA out of memory"
```python
# In config.py, reduce:
# - Steps in workflow (40 → 20)
# - CFG scale (3.5 → 2.5)
# Or switch to CPU (slower)
```

### "Wav2Lip checkpoint not found"
```bash
cd talking-characters/models
wget "https://www.adrianbulat.com/downloads/deep_learning/state_dict" \
  -O wav2lip.pth
```

### ComfyUI connection error
```bash
# Terminal 1: Start ComfyUI
cd ..
python main.py

# Terminal 2: Use talking-characters
cd talking-characters
python -m scripts.video_generator
```

## 📚 Next Steps

1. Read `README.md` for usage guide
2. Check `config.py` for customization options
3. Explore `scripts/` directory for advanced features
4. Create your first talking character video!

---

**Last Updated**: February 2026
