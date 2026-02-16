"""
Configuration for talking-characters system
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
WORKFLOWS_DIR = PROJECT_ROOT / "workflows"
AUDIO_DIR = PROJECT_ROOT / "audio"
VIDEO_DIR = PROJECT_ROOT / "video"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

# ComfyUI paths
COMFYUI_ROOT = PROJECT_ROOT.parent
COMFYUI_OUTPUT = COMFYUI_ROOT / "output"
COMFYUI_INPUT = COMFYUI_ROOT / "input"
COMFYUI_WORKFLOWS = COMFYUI_ROOT / "user" / "default" / "workflows"

# API Configuration
COMFYUI_API_URL = "http://localhost:8188"
COMFYUI_API_TIMEOUT = 300

# TTS Configuration
TTS_ENGINE = "piper"  # Options: piper, bark, gtts
TTS_LANGUAGE = "pl_PL"  # Polish
TTS_PIPER_MODEL = "pl_PL-dark-medium"
TTS_BARK_SPEAKER = "v2/en_speaker_6"

# Video Configuration
VIDEO_FPS = 24
VIDEO_CODEC = "libx264"
VIDEO_BITRATE = "5000k"

# Wav2Lip Configuration
WAV2LIP_CHECKPOINT = MODELS_DIR / "wav2lip.pth"
WAV2LIP_REPO = "https://github.com/Rudrabha/Wav2Lip.git"

# Character Configurations
CHARACTERS = {
    "jessica": {
        "name": "Jessica",
        "workflow": "TEST_Jessica_Penetration_Missionary_v1.json",
        "character_strength": 0.55,
        "prompt_template": "Jessica {scene_description}, intimate moment, sensual, 4k, cinematic"
    },
    "gigi": {
        "name": "Gigi",
        "workflow": "TEST_Gigi_DP_DoublePenetration_v1.json",
        "character_strength": 0.85,
        "prompt_template": "Gigi {scene_description}, sensual, intimate, 4k, cinematic"
    }
}

# Dialogue Scenes
DIALOGUE_SCENES = {
    "jessica_seduction": {
        "character": "jessica",
        "text": "Cześć kochanie, czekam na ciebie. Chcę się z tobą bawić.",
        "duration": 3
    },
    "jessica_pleasure": {
        "character": "jessica",
        "text": "Mmmm, to jest takie dobre. Nie przestawaj, proszę...",
        "duration": 2.5
    },
    "jessica_satisfaction": {
        "character": "jessica",
        "text": "Oooh! Jeszcze... jeszcze bardziej... tak dobrze!",
        "duration": 3
    },
    "gigi_seduction": {
        "character": "gigi",
        "text": "Cześć, jestem Gigi. Jestem gotowa dla ciebie.",
        "duration": 2.5
    },
    "gigi_pleasure": {
        "character": "gigi",
        "text": "Tak, tak! Dokładnie tam! Mmmm...",
        "duration": 2.5
    }
}

# Debug/Logging
DEBUG = True
VERBOSE = True
KEEP_INTERMEDIATE = True  # Zachowaj pośrednie pliki do debugowania
