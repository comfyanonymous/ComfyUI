"""
Talking Characters - Local AI Video Generation System
Combines Flux, TTS, and Wav2Lip for talking character videos
"""

from .tts_engine import TTSEngine
from .lip_sync import Wav2LipSynchronizer
from .video_generator import TalkingCharacterGenerator

__version__ = "0.1.0"
__author__ = "Claude Code"
__all__ = [
    "TTSEngine",
    "Wav2LipSynchronizer",
    "TalkingCharacterGenerator"
]
