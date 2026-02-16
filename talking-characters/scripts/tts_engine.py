"""
Text-to-Speech engine with multiple backends
Supports: Piper (local), Bark (local), gTTS (online)
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TTSEngine:
    """Base TTS Engine class"""

    def __init__(self, engine_type: str = "piper"):
        self.engine_type = engine_type
        self.is_ready = False
        self._setup()

    def _setup(self):
        """Setup specific TTS engine"""
        if self.engine_type == "piper":
            self._setup_piper()
        elif self.engine_type == "bark":
            self._setup_bark()
        elif self.engine_type == "gtts":
            self._setup_gtts()
        else:
            raise ValueError(f"Unknown TTS engine: {self.engine_type}")

    def _setup_piper(self):
        """Setup Piper TTS (recommended for Polish)"""
        try:
            import piper
            logger.info("✅ Piper TTS loaded")
            self.is_ready = True
        except ImportError:
            logger.error("❌ Piper not installed. Install with: pip install piper-tts")
            self.is_ready = False

    def _setup_bark(self):
        """Setup Bark TTS"""
        try:
            import bark
            logger.info("✅ Bark TTS loaded")
            self.is_ready = True
        except ImportError:
            logger.error("❌ Bark not installed. Install with: pip install bark")
            self.is_ready = False

    def _setup_gtts(self):
        """Setup Google TTS"""
        try:
            from gtts import gTTS
            logger.info("✅ gTTS loaded")
            self.is_ready = True
        except ImportError:
            logger.error("❌ gTTS not installed. Install with: pip install gtts")
            self.is_ready = False

    def synthesize(self, text: str, output_path: str, **kwargs) -> bool:
        """
        Synthesize speech from text

        Args:
            text: Text to synthesize
            output_path: Where to save audio
            **kwargs: Engine-specific options

        Returns:
            True if successful
        """
        if not self.is_ready:
            logger.error(f"❌ {self.engine_type} TTS not ready")
            return False

        if self.engine_type == "piper":
            return self._synthesize_piper(text, output_path, **kwargs)
        elif self.engine_type == "bark":
            return self._synthesize_bark(text, output_path, **kwargs)
        elif self.engine_type == "gtts":
            return self._synthesize_gtts(text, output_path, **kwargs)

    def _synthesize_piper(self, text: str, output_path: str,
                         language: str = "pl_PL", model: str = "dark-medium") -> bool:
        """Synthesize with Piper"""
        try:
            voice = f"{language}-{model}"

            cmd = [
                "piper",
                "--model", voice,
                "--output_file", output_path
            ]

            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            process.communicate(input=text)

            if process.returncode == 0:
                logger.info(f"✅ Audio generated: {output_path}")
                return True
            else:
                logger.error(f"❌ Piper error: {process.stderr}")
                return False

        except Exception as e:
            logger.error(f"❌ Piper error: {e}")
            return False

    def _synthesize_bark(self, text: str, output_path: str,
                        speaker: str = "v2/en_speaker_6") -> bool:
        """Synthesize with Bark"""
        try:
            import soundfile as sf
            from bark import generate_audio, preload_models, SAMPLE_RATE

            logger.info("Loading Bark models...")
            preload_models()

            logger.info("Generating audio...")
            audio_array = generate_audio(text, history_prompt=speaker)

            sf.write(output_path, audio_array, SAMPLE_RATE)

            logger.info(f"✅ Audio generated: {output_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Bark error: {e}")
            return False

    def _synthesize_gtts(self, text: str, output_path: str,
                        language: str = "pl", slow: bool = False) -> bool:
        """Synthesize with Google TTS (requires internet)"""
        try:
            from gtts import gTTS

            logger.info("Generating audio with Google TTS...")
            tts = gTTS(text=text, lang=language, slow=slow)
            tts.save(output_path)

            logger.info(f"✅ Audio generated: {output_path}")
            return True

        except Exception as e:
            logger.error(f"❌ gTTS error: {e}")
            return False


def test_tts_engines():
    """Test all available TTS engines"""
    engines = ["piper", "bark", "gtts"]
    test_text = "Cześć, to jest test."

    for engine in engines:
        logger.info(f"\nTesting {engine}...")
        try:
            tts = TTSEngine(engine)
            if tts.is_ready:
                output = f"/tmp/{engine}_test.wav"
                if tts.synthesize(test_text, output):
                    logger.info(f"  ✅ {engine} works!")
                else:
                    logger.info(f"  ❌ {engine} failed")
            else:
                logger.info(f"  ⚠️  {engine} not installed")
        except Exception as e:
            logger.error(f"  ❌ {engine} error: {e}")


if __name__ == "__main__":
    test_tts_engines()
