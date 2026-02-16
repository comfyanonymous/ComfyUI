"""
Comprehensive testing system for talking-characters
Tests all components: TTS, Wav2Lip, ComfyUI API, FFmpeg
"""

import subprocess
import sys
import json
from pathlib import Path
import time
import logging
from typing import Dict, List, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tts_engine import TTSEngine
from lip_sync import Wav2LipSynchronizer
from config import COMFYUI_API_URL, COMFYUI_WORKFLOWS
import requests


class TestResult:
    """Represents a test result"""
    def __init__(self, name: str, passed: bool, message: str = "", time_taken: float = 0):
        self.name = name
        self.passed = passed
        self.message = message
        self.time_taken = time_taken

    def __str__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{status} - {self.name} ({self.time_taken:.2f}s) - {self.message}"


class SystemTester:
    """Complete system testing suite"""

    def __init__(self):
        self.results: List[TestResult] = []
        self.failed_tests = 0
        self.passed_tests = 0

    def test_python_version(self) -> TestResult:
        """Test Python 3.10+"""
        logger.info("\n🔍 Testing Python version...")
        start = time.time()

        try:
            version = sys.version_info
            if version.major >= 3 and version.minor >= 10:
                result = TestResult(
                    "Python Version",
                    True,
                    f"Python {version.major}.{version.minor}.{version.micro}",
                    time.time() - start
                )
            else:
                result = TestResult(
                    "Python Version",
                    False,
                    f"Need 3.10+, found {version.major}.{version.minor}",
                    time.time() - start
                )
        except Exception as e:
            result = TestResult("Python Version", False, str(e), time.time() - start)

        self.results.append(result)
        return result

    def test_ffmpeg(self) -> TestResult:
        """Test FFmpeg installation"""
        logger.info("🔍 Testing FFmpeg...")
        start = time.time()

        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                test_result = TestResult(
                    "FFmpeg",
                    True,
                    version_line,
                    time.time() - start
                )
            else:
                test_result = TestResult(
                    "FFmpeg",
                    False,
                    "FFmpeg not working",
                    time.time() - start
                )

        except FileNotFoundError:
            test_result = TestResult(
                "FFmpeg",
                False,
                "FFmpeg not installed",
                time.time() - start
            )
        except Exception as e:
            test_result = TestResult("FFmpeg", False, str(e), time.time() - start)

        self.results.append(test_result)
        return test_result

    def test_piper_tts(self) -> TestResult:
        """Test Piper TTS"""
        logger.info("🔍 Testing Piper TTS...")
        start = time.time()

        try:
            tts = TTSEngine("piper")

            if not tts.is_ready:
                test_result = TestResult(
                    "Piper TTS",
                    False,
                    "Not installed",
                    time.time() - start
                )
            else:
                # Try to generate audio
                test_file = Path("/tmp/piper_test.wav")
                if tts.synthesize("Test", str(test_file)):
                    if test_file.exists():
                        test_result = TestResult(
                            "Piper TTS",
                            True,
                            "Working",
                            time.time() - start
                        )
                    else:
                        test_result = TestResult(
                            "Piper TTS",
                            False,
                            "No output file",
                            time.time() - start
                        )
                else:
                    test_result = TestResult(
                        "Piper TTS",
                        False,
                        "Synthesis failed",
                        time.time() - start
                    )

        except Exception as e:
            test_result = TestResult("Piper TTS", False, str(e), time.time() - start)

        self.results.append(test_result)
        return test_result

    def test_bark_tts(self) -> TestResult:
        """Test Bark TTS"""
        logger.info("🔍 Testing Bark TTS...")
        start = time.time()

        try:
            tts = TTSEngine("bark")

            if not tts.is_ready:
                test_result = TestResult(
                    "Bark TTS",
                    False,
                    "Not installed",
                    time.time() - start
                )
            else:
                test_result = TestResult(
                    "Bark TTS",
                    True,
                    "Installed",
                    time.time() - start
                )

        except Exception as e:
            test_result = TestResult("Bark TTS", False, str(e), time.time() - start)

        self.results.append(test_result)
        return test_result

    def test_gtts(self) -> TestResult:
        """Test Google TTS"""
        logger.info("🔍 Testing Google TTS...")
        start = time.time()

        try:
            tts = TTSEngine("gtts")

            if not tts.is_ready:
                test_result = TestResult(
                    "Google TTS",
                    False,
                    "Not installed",
                    time.time() - start
                )
            else:
                test_result = TestResult(
                    "Google TTS",
                    True,
                    "Installed",
                    time.time() - start
                )

        except Exception as e:
            test_result = TestResult("Google TTS", False, str(e), time.time() - start)

        self.results.append(test_result)
        return test_result

    def test_wav2lip(self) -> TestResult:
        """Test Wav2Lip availability"""
        logger.info("🔍 Testing Wav2Lip...")
        start = time.time()

        try:
            wav2lip = Wav2LipSynchronizer()

            if wav2lip.is_available:
                test_result = TestResult(
                    "Wav2Lip",
                    True,
                    "Available",
                    time.time() - start
                )
            else:
                test_result = TestResult(
                    "Wav2Lip",
                    False,
                    "Not available (optional)",
                    time.time() - start
                )

        except Exception as e:
            test_result = TestResult("Wav2Lip", False, str(e), time.time() - start)

        self.results.append(test_result)
        return test_result

    def test_comfyui_api(self) -> TestResult:
        """Test ComfyUI API connection"""
        logger.info("🔍 Testing ComfyUI API...")
        start = time.time()

        try:
            response = requests.get(
                f"{COMFYUI_API_URL}/system_stats",
                timeout=5
            )

            if response.status_code == 200:
                test_result = TestResult(
                    "ComfyUI API",
                    True,
                    f"Connected to {COMFYUI_API_URL}",
                    time.time() - start
                )
            else:
                test_result = TestResult(
                    "ComfyUI API",
                    False,
                    f"Status {response.status_code}",
                    time.time() - start
                )

        except requests.ConnectionError:
            test_result = TestResult(
                "ComfyUI API",
                False,
                f"Cannot connect to {COMFYUI_API_URL}",
                time.time() - start
            )
        except Exception as e:
            test_result = TestResult("ComfyUI API", False, str(e), time.time() - start)

        self.results.append(test_result)
        return test_result

    def test_workflows(self) -> TestResult:
        """Test required workflow files exist"""
        logger.info("🔍 Testing workflow files...")
        start = time.time()

        try:
            workflows = list(COMFYUI_WORKFLOWS.glob("TEST_*.json"))

            if len(workflows) > 0:
                test_result = TestResult(
                    "Workflow Files",
                    True,
                    f"Found {len(workflows)} workflows",
                    time.time() - start
                )
            else:
                test_result = TestResult(
                    "Workflow Files",
                    False,
                    f"No workflows in {COMFYUI_WORKFLOWS}",
                    time.time() - start
                )

        except Exception as e:
            test_result = TestResult("Workflow Files", False, str(e), time.time() - start)

        self.results.append(test_result)
        return test_result

    def test_torch_gpu(self) -> TestResult:
        """Test PyTorch and GPU availability"""
        logger.info("🔍 Testing PyTorch/GPU...")
        start = time.time()

        try:
            import torch

            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                test_result = TestResult(
                    "PyTorch GPU",
                    True,
                    f"GPU: {device_name}",
                    time.time() - start
                )
            else:
                test_result = TestResult(
                    "PyTorch GPU",
                    True,
                    "CPU mode (slower)",
                    time.time() - start
                )

        except ImportError:
            test_result = TestResult(
                "PyTorch GPU",
                False,
                "PyTorch not installed",
                time.time() - start
            )
        except Exception as e:
            test_result = TestResult("PyTorch GPU", False, str(e), time.time() - start)

        self.results.append(test_result)
        return test_result

    def test_disk_space(self) -> TestResult:
        """Test available disk space"""
        logger.info("🔍 Testing disk space...")
        start = time.time()

        try:
            import shutil
            total, used, free = shutil.disk_usage("/")
            free_gb = free / (1024**3)

            if free_gb > 10:  # At least 10GB
                test_result = TestResult(
                    "Disk Space",
                    True,
                    f"{free_gb:.1f}GB available",
                    time.time() - start
                )
            else:
                test_result = TestResult(
                    "Disk Space",
                    False,
                    f"Only {free_gb:.1f}GB, need 10GB+",
                    time.time() - start
                )

        except Exception as e:
            test_result = TestResult("Disk Space", False, str(e), time.time() - start)

        self.results.append(test_result)
        return test_result

    def run_all_tests(self) -> Tuple[int, int]:
        """Run all tests and return (passed, failed)"""
        logger.info("=" * 70)
        logger.info("TALKING CHARACTERS - SYSTEM TEST SUITE")
        logger.info("=" * 70)

        # Run tests
        self.test_python_version()
        self.test_ffmpeg()
        self.test_piper_tts()
        self.test_bark_tts()
        self.test_gtts()
        self.test_wav2lip()
        self.test_comfyui_api()
        self.test_workflows()
        self.test_torch_gpu()
        self.test_disk_space()

        # Print results
        logger.info("\n" + "=" * 70)
        logger.info("TEST RESULTS")
        logger.info("=" * 70)

        for result in self.results:
            logger.info(str(result))
            if result.passed:
                self.passed_tests += 1
            else:
                self.failed_tests += 1

        # Summary
        logger.info("\n" + "=" * 70)
        total = self.passed_tests + self.failed_tests
        percentage = (self.passed_tests / total * 100) if total > 0 else 0

        logger.info(f"SUMMARY: {self.passed_tests}/{total} passed ({percentage:.0f}%)")

        if self.failed_tests == 0:
            logger.info("✅ ALL SYSTEMS GO! Ready to generate videos.")
        else:
            logger.warning(f"⚠️  {self.failed_tests} test(s) failed. Check above for details.")

        logger.info("=" * 70)

        return self.passed_tests, self.failed_tests

    def print_recommendations(self):
        """Print recommendations based on test results"""
        logger.info("\n📋 RECOMMENDATIONS:")

        for result in self.results:
            if not result.passed:
                if "FFmpeg" in result.name:
                    logger.info("  → Install FFmpeg: choco install ffmpeg (Windows)")
                elif "Piper" in result.name:
                    logger.info("  → Install Piper: pip install piper-tts")
                elif "ComfyUI" in result.name:
                    logger.info("  → Start ComfyUI: python main.py")
                elif "Workflow" in result.name:
                    logger.info("  → Copy workflows to: D:\\AI\\ComfyUI\\user\\default\\workflows\\")


def main():
    """Run test suite"""
    tester = SystemTester()
    passed, failed = tester.run_all_tests()

    if failed > 0:
        tester.print_recommendations()

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
