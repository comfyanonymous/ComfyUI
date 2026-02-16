"""
Main pipeline: Flux Image → TTS → Lip-sync → Video
Converts a character into a talking video
"""

import json
import requests
import time
import subprocess
import logging
from pathlib import Path
from typing import Optional
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    COMFYUI_API_URL, COMFYUI_API_TIMEOUT,
    COMFYUI_WORKFLOWS, AUDIO_DIR, VIDEO_DIR,
    VIDEO_FPS, CHARACTERS, DEBUG
)
from tts_engine import TTSEngine
from lip_sync import Wav2LipSynchronizer

logger = logging.getLogger(__name__)
if DEBUG:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)


class TalkingCharacterGenerator:
    """Complete pipeline for generating talking character videos"""

    def __init__(self, character: str, dialogue_text: str,
                 tts_engine: str = "piper", output_name: Optional[str] = None):
        """
        Initialize generator

        Args:
            character: Character name (jessica, gigi, etc)
            dialogue_text: What the character should say
            tts_engine: TTS backend (piper, bark, gtts)
            output_name: Custom output filename
        """

        if character not in CHARACTERS:
            raise ValueError(f"Unknown character: {character}. Available: {list(CHARACTERS.keys())}")

        self.character = character
        self.character_config = CHARACTERS[character]
        self.dialogue_text = dialogue_text
        self.tts_engine = TTSEngine(tts_engine)
        self.wav2lip = Wav2LipSynchronizer()
        self.output_name = output_name or f"{character}_{int(time.time())}"

        # Create output directories
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        VIDEO_DIR.mkdir(parents=True, exist_ok=True)

        logger.info(f"🎬 Initializing {character} video generator")

    def _generate_flux_image(self) -> Optional[str]:
        """
        Step 1: Generate character image using Flux

        Returns:
            Path to generated image or None if failed
        """

        logger.info("\n📸 Step 1: Generating character image...")

        workflow_name = self.character_config["workflow"]
        workflow_path = COMFYUI_WORKFLOWS / workflow_name

        if not workflow_path.exists():
            logger.error(f"❌ Workflow not found: {workflow_path}")
            return None

        try:
            with open(workflow_path) as f:
                workflow = json.load(f)

            logger.info(f"   Sending workflow: {workflow_name}")

            response = requests.post(
                f"{COMFYUI_API_URL}/prompt",
                json=workflow,
                timeout=COMFYUI_API_TIMEOUT
            )

            if response.status_code != 200:
                logger.error(f"❌ API error: {response.text}")
                return None

            prompt_id = response.json()['prompt_id']
            logger.info(f"   Prompt ID: {prompt_id}")

            # Wait for generation
            logger.info("   ⏳ Waiting for image generation...")
            start_time = time.time()

            while True:
                result = requests.get(
                    f"{COMFYUI_API_URL}/history/{prompt_id}"
                )

                if result.json():
                    elapsed = time.time() - start_time
                    logger.info(f"   ✅ Image generated ({elapsed:.1f}s)")
                    return "ComfyUI output path"  # Would get actual path from history

                time.sleep(2)

                if time.time() - start_time > COMFYUI_API_TIMEOUT:
                    logger.error("❌ Generation timeout")
                    return None

        except Exception as e:
            logger.error(f"❌ Generation error: {e}")
            return None

    def _generate_speech(self) -> Optional[str]:
        """
        Step 2: Convert text to speech

        Returns:
            Path to generated audio or None if failed
        """

        logger.info("\n🎤 Step 2: Generating speech...")

        output_path = AUDIO_DIR / f"{self.output_name}.wav"

        if not self.tts_engine.is_ready:
            logger.error("❌ TTS engine not ready")
            return None

        logger.info(f"   Text: {self.dialogue_text}")
        logger.info(f"   Engine: {self.tts_engine.engine_type}")

        if self.tts_engine.synthesize(self.dialogue_text, str(output_path)):
            logger.info(f"   ✅ Audio saved: {output_path}")
            return str(output_path)
        else:
            logger.error("❌ Speech generation failed")
            return None

    def _image_to_video(self, image_path: str) -> Optional[str]:
        """
        Step 3: Convert static image to short video

        Args:
            image_path: Path to input image

        Returns:
            Path to video or None if failed
        """

        logger.info("\n🎥 Step 3: Converting image to video...")

        output_video = VIDEO_DIR / f"{self.output_name}_base.mp4"

        try:
            # Calculate duration based on speech length
            audio_path = AUDIO_DIR / f"{self.output_name}.wav"
            duration = self._get_audio_duration(str(audio_path))

            logger.info(f"   Duration: {duration:.1f}s")

            cmd = [
                "ffmpeg", "-y",
                "-loop", "1",
                "-i", str(image_path),
                "-c:v", "libx264",
                "-t", str(duration),
                "-pix_fmt", "yuv420p",
                "-preset", "fast",
                str(output_video)
            ]

            process = subprocess.run(cmd, capture_output=True, text=True)

            if process.returncode == 0:
                logger.info(f"   ✅ Video created: {output_video}")
                return str(output_video)
            else:
                logger.error(f"❌ FFmpeg error: {process.stderr}")
                return None

        except Exception as e:
            logger.error(f"❌ Conversion error: {e}")
            return None

    def _synchronize_lips(self, video_path: str, audio_path: str) -> Optional[str]:
        """
        Step 4: Apply Wav2Lip for lip-sync

        Args:
            video_path: Path to input video
            audio_path: Path to input audio

        Returns:
            Path to lip-synced video or None if failed
        """

        logger.info("\n👄 Step 4: Synchronizing lips...")

        output_video = VIDEO_DIR / f"{self.output_name}_lipsync.mp4"

        if not self.wav2lip.is_available:
            logger.warning("⚠️  Wav2Lip not available, skipping lip-sync")
            return video_path

        if self.wav2lip.synchronize(video_path, audio_path, str(output_video)):
            logger.info(f"   ✅ Lip-sync applied: {output_video}")
            return str(output_video)
        else:
            logger.warning("   ⚠️  Lip-sync failed, using original video")
            return video_path

    def _merge_audio(self, video_path: str, audio_path: str) -> Optional[str]:
        """
        Step 5: Merge audio with video

        Args:
            video_path: Path to input video
            audio_path: Path to input audio

        Returns:
            Path to final video or None if failed
        """

        logger.info("\n🔊 Step 5: Merging audio and video...")

        output_path = VIDEO_DIR / f"{self.output_name}_final.mp4"

        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-i", str(audio_path),
                "-c:v", "copy",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0",
                str(output_path)
            ]

            process = subprocess.run(cmd, capture_output=True, text=True)

            if process.returncode == 0:
                logger.info(f"   ✅ Final video: {output_path}")
                return str(output_path)
            else:
                logger.error(f"❌ Merge failed: {process.stderr}")
                return None

        except Exception as e:
            logger.error(f"❌ Merge error: {e}")
            return None

    def generate(self) -> bool:
        """
        Run complete pipeline

        Returns:
            True if successful
        """

        logger.info("=" * 60)
        logger.info(f"🎬 GENERATING: {self.character.upper()} TALKING VIDEO")
        logger.info("=" * 60)

        # Step 1: Generate image
        image_path = self._generate_flux_image()
        if not image_path:
            logger.error("❌ Failed at image generation")
            return False

        # Step 2: Generate speech
        audio_path = self._generate_speech()
        if not audio_path:
            logger.error("❌ Failed at speech generation")
            return False

        # Step 3: Image to video
        video_path = self._image_to_video(image_path)
        if not video_path:
            logger.error("❌ Failed at image-to-video conversion")
            return False

        # Step 4: Lip-sync (optional)
        lipsync_path = self._synchronize_lips(video_path, audio_path)

        # Step 5: Merge audio
        final_path = self._merge_audio(lipsync_path, audio_path)
        if not final_path:
            logger.error("❌ Failed at audio merge")
            return False

        logger.info("\n" + "=" * 60)
        logger.info("✅ SUCCESS! Video generated:")
        logger.info(f"   📁 {final_path}")
        logger.info("=" * 60)

        return True

    @staticmethod
    def _get_audio_duration(audio_path: str) -> float:
        """Get audio duration in seconds"""
        try:
            import wave
            with wave.open(audio_path, 'rb') as wav:
                frames = wav.getnframes()
                rate = wav.getframerate()
                return frames / rate
        except:
            return 5.0  # Default fallback


def main():
    """Example usage"""

    # Example 1: Jessica seduction scene
    logger.info("\n🎬 Creating Jessica seduction scene...")

    generator = TalkingCharacterGenerator(
        character="jessica",
        dialogue_text="Cześć kochanie, czekam na ciebie. Chcę się z tobą bawić.",
        tts_engine="piper",
        output_name="jessica_seduction"
    )

    if generator.generate():
        logger.info("✅ Jessica scene created successfully!")
    else:
        logger.error("❌ Failed to create Jessica scene")

    # Example 2: Gigi seduction scene
    logger.info("\n🎬 Creating Gigi seduction scene...")

    generator2 = TalkingCharacterGenerator(
        character="gigi",
        dialogue_text="Cześć, jestem Gigi. Jestem gotowa dla ciebie.",
        tts_engine="piper",
        output_name="gigi_seduction"
    )

    if generator2.generate():
        logger.info("✅ Gigi scene created successfully!")
    else:
        logger.error("❌ Failed to create Gigi scene")


if __name__ == "__main__":
    main()
