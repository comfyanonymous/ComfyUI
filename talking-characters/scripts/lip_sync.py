"""
Lip-sync synchronization using Wav2Lip
Converts static image + audio into video with synchronized lips
"""

import subprocess
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class Wav2LipSynchronizer:
    """Handles lip-sync using Wav2Lip model"""

    def __init__(self, checkpoint_path: Optional[str] = None,
                 wav2lip_repo: Optional[str] = None):
        """
        Initialize Wav2Lip synchronizer

        Args:
            checkpoint_path: Path to wav2lip.pth model
            wav2lip_repo: Path to Wav2Lip repository
        """
        self.checkpoint_path = checkpoint_path or "models/wav2lip.pth"
        self.wav2lip_repo = wav2lip_repo or "Wav2Lip"
        self.is_available = self._check_availability()

    def _check_availability(self) -> bool:
        """Check if Wav2Lip is available"""
        try:
            # Check if repository exists
            repo_path = Path(self.wav2lip_repo)
            if not repo_path.exists():
                logger.warning(f"⚠️  Wav2Lip repo not found at {self.wav2lip_repo}")
                return False

            # Check if checkpoint exists
            ckpt_path = Path(self.checkpoint_path)
            if not ckpt_path.exists():
                logger.warning(f"⚠️  Checkpoint not found at {self.checkpoint_path}")
                return False

            logger.info("✅ Wav2Lip is available")
            return True

        except Exception as e:
            logger.error(f"❌ Wav2Lip check failed: {e}")
            return False

    def synchronize(self, video_path: str, audio_path: str,
                   output_path: str, pads: list = None) -> bool:
        """
        Synchronize lips in video with audio

        Args:
            video_path: Path to input video/image
            audio_path: Path to input audio
            output_path: Where to save output
            pads: Padding [top, bottom, left, right] - default auto-detect

        Returns:
            True if successful
        """

        if not self.is_available:
            logger.error("❌ Wav2Lip not available")
            return False

        try:
            logger.info(f"🔄 Synchronizing lips...")
            logger.info(f"  Video: {video_path}")
            logger.info(f"  Audio: {audio_path}")

            # Construct command
            cmd = [
                "python",
                f"{self.wav2lip_repo}/inference.py",
                "--checkpoint_path", str(self.checkpoint_path),
                "--face", str(video_path),
                "--audio", str(audio_path),
                "--outfile", str(output_path),
                "--device", "cuda",  # Use GPU if available
            ]

            # Add optional padding
            if pads:
                cmd.extend(["--pads", *map(str, pads)])

            # Run Wav2Lip
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600
            )

            if process.returncode == 0:
                logger.info(f"✅ Lip-sync complete: {output_path}")
                return True
            else:
                logger.error(f"❌ Wav2Lip failed: {process.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("❌ Wav2Lip timeout")
            return False
        except Exception as e:
            logger.error(f"❌ Wav2Lip error: {e}")
            return False

    def download_model(self, output_path: Optional[str] = None):
        """
        Download Wav2Lip model checkpoint

        Args:
            output_path: Where to save checkpoint
        """
        output_path = output_path or self.checkpoint_path

        logger.info("📥 Downloading Wav2Lip checkpoint...")

        url = "https://www.adrianbulat.com/downloads/deep_learning/state_dict"

        try:
            import urllib.request
            urllib.request.urlretrieve(url, output_path)
            logger.info(f"✅ Checkpoint downloaded: {output_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            return False

    def setup_repo(self):
        """Clone Wav2Lip repository if not present"""
        if Path(self.wav2lip_repo).exists():
            logger.info(f"✅ Wav2Lip repo already exists")
            return True

        logger.info("📥 Cloning Wav2Lip repository...")

        try:
            cmd = [
                "git",
                "clone",
                "https://github.com/Rudrabha/Wav2Lip.git",
                self.wav2lip_repo
            ]

            process = subprocess.run(cmd, capture_output=True, text=True)

            if process.returncode == 0:
                logger.info(f"✅ Wav2Lip cloned to {self.wav2lip_repo}")
                return True
            else:
                logger.error(f"❌ Clone failed: {process.stderr}")
                return False

        except Exception as e:
            logger.error(f"❌ Setup error: {e}")
            return False


def setup_wav2lip():
    """One-time setup for Wav2Lip"""
    logger.info("🔧 Setting up Wav2Lip...")

    sync = Wav2LipSynchronizer()

    # Setup repo
    if not Path(sync.wav2lip_repo).exists():
        if not sync.setup_repo():
            return False

    # Download model
    if not Path(sync.checkpoint_path).exists():
        if not sync.download_model():
            return False

    logger.info("✅ Wav2Lip setup complete")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    setup_wav2lip()
