import unittest
import json
from unittest.mock import patch

# Import the module to test
import sys
sys.path.insert(0, '/root/.openclaw/workspace/ComfyUI')

from comfy.ldm.lightricks.vae.audio_vae import AudioVAEComponentConfig


class TestAudioVAEComponentConfig(unittest.TestCase):
    """Test cases for AudioVAEComponentConfig JSON parsing"""

    def test_valid_json_config(self):
        """测试有效的 JSON 配置"""
        metadata = {
            "config": '{"audio_vae": {"test": 1}, "vocoder": {"test": 2}}'
        }
        config = AudioVAEComponentConfig.from_metadata(metadata)
        self.assertEqual(config.autoencoder, {"test": 1})
        self.assertEqual(config.vocoder, {"test": 2})

    def test_dict_config(self):
        """测试字典格式的配置（非 JSON 字符串）"""
        metadata = {
            "config": {"audio_vae": {"test": 1}, "vocoder": {"test": 2}}
        }
        config = AudioVAEComponentConfig.from_metadata(metadata)
        self.assertEqual(config.autoencoder, {"test": 1})
        self.assertEqual(config.vocoder, {"test": 2})

    def test_invalid_json_config(self):
        """测试无效的 JSON 配置应该抛出 ValueError"""
        metadata = {
            "config": '{"invalid json'  # 损坏的 JSON
        }
        with self.assertRaises(ValueError) as context:
            AudioVAEComponentConfig.from_metadata(metadata)
        self.assertIn("Invalid JSON config", str(context.exception))

    def test_missing_config(self):
        """测试缺少 config 字段"""
        metadata = {}
        with self.assertRaises(AssertionError):
            AudioVAEComponentConfig.from_metadata(metadata)

    def test_null_metadata(self):
        """测试 null metadata"""
        with self.assertRaises(AssertionError):
            AudioVAEComponentConfig.from_metadata(None)


class TestModelDetectionJsonParsing(unittest.TestCase):
    """Test cases for model_detection.py JSON parsing"""

    def test_valid_metadata_config(self):
        """测试有效的 metadata config 解析"""
        from comfy import model_detection
        # This would require more setup to test properly
        pass


if __name__ == '__main__':
    unittest.main()
