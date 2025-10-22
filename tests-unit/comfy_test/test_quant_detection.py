"""
Integration tests for quantization detection.
Tests Phase 2: Detection & Integration
"""

import unittest
import torch
import sys
import os

# Add comfy to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from comfy import model_detection


class TestNormalizeLayerName(unittest.TestCase):
    """Test the normalize_layer_name helper function"""
    
    def test_strip_prefix_and_suffix(self):
        """Test stripping prefix and suffix"""
        known_prefixes = ["model.diffusion_model."]
        result = model_detection.normalize_layer_name(
            "model.diffusion_model.layer1.weight",
            known_prefixes
        )
        self.assertEqual(result, "layer1")
    
    def test_strip_multiple_prefixes(self):
        """Test with multiple known prefixes"""
        known_prefixes = ["model.diffusion_model.", "model.model.", "net."]
        
        result1 = model_detection.normalize_layer_name(
            "model.diffusion_model.block.attn.weight",
            known_prefixes
        )
        self.assertEqual(result1, "block.attn")
        
        result2 = model_detection.normalize_layer_name(
            "model.model.encoder.layer.weight",
            known_prefixes
        )
        self.assertEqual(result2, "encoder.layer")
        
        result3 = model_detection.normalize_layer_name(
            "net.transformer.blocks.0.weight",
            known_prefixes
        )
        self.assertEqual(result3, "transformer.blocks.0")
    
    def test_strip_scale_weight_suffix(self):
        """Test stripping scale_weight suffix"""
        known_prefixes = ["model.diffusion_model."]
        result = model_detection.normalize_layer_name(
            "model.diffusion_model.layer1.scale_weight",
            known_prefixes
        )
        self.assertEqual(result, "layer1")
    
    def test_strip_bias_suffix(self):
        """Test stripping bias suffix"""
        known_prefixes = ["model.diffusion_model."]
        result = model_detection.normalize_layer_name(
            "model.diffusion_model.layer1.bias",
            known_prefixes
        )
        self.assertEqual(result, "layer1")
    
    def test_no_prefix_match(self):
        """Test with no prefix match"""
        known_prefixes = ["model.diffusion_model."]
        result = model_detection.normalize_layer_name(
            "other.model.layer1.weight",
            known_prefixes
        )
        # Should strip suffix but not prefix
        self.assertEqual(result, "other.model.layer1")


class TestDetectLayerQuantization(unittest.TestCase):
    """Test the detect_layer_quantization function"""
    
    def test_no_quantization(self):
        """Test with no quantization markers"""
        state_dict = {
            "model.diffusion_model.layer1.weight": torch.randn(10, 20),
            "model.diffusion_model.layer2.weight": torch.randn(20, 30),
        }
        result = model_detection.detect_layer_quantization(state_dict, "model.diffusion_model.")
        self.assertIsNone(result)
    
    def test_legacy_scaled_fp8(self):
        """Test that legacy scaled_fp8 marker returns None"""
        # Create FP8 tensor by converting from float32
        fp8_weight = torch.randn(10, 20, dtype=torch.float32).to(torch.float8_e4m3fn)
        state_dict = {
            "model.diffusion_model.scaled_fp8": torch.tensor([], dtype=torch.float8_e4m3fn),
            "model.diffusion_model.layer1.weight": fp8_weight,
            "model.diffusion_model.layer1.scale_weight": torch.tensor(1.0),
        }
        result = model_detection.detect_layer_quantization(state_dict, "model.diffusion_model.")
        # Should return None to trigger legacy path
        self.assertIsNone(result)
    
    def test_metadata_format(self):
        """Test with new metadata format"""
        metadata = {
            "format_version": "1.0",
            "layers": {
                "layer1": {
                    "format": "fp8_e4m3fn_scaled",
                    "params": {"use_fp8_matmul": True}
                },
                "layer2": {
                    "format": "fp8_e5m2_scaled",
                    "params": {"use_fp8_matmul": True}
                }
            }
        }
        state_dict = {
            "model.diffusion_model._quantization_metadata": metadata,
            "model.diffusion_model.layer1.weight": torch.randn(10, 20),
        }
        result = model_detection.detect_layer_quantization(state_dict, "model.diffusion_model.")
        
        self.assertIsNotNone(result)
        self.assertIn("layer1", result)
        self.assertIn("layer2", result)
        self.assertEqual(result["layer1"]["format"], "fp8_e4m3fn_scaled")
        self.assertEqual(result["layer2"]["format"], "fp8_e5m2_scaled")
        # Metadata should be popped from state_dict
        self.assertNotIn("model.diffusion_model._quantization_metadata", state_dict)
    
    def test_mixed_precision_detection(self):
        """Test detection of mixed precision via scale patterns"""
        # Create FP8 tensors by converting from float32
        fp8_weight1 = torch.randn(10, 20, dtype=torch.float32).to(torch.float8_e4m3fn)
        fp8_weight3 = torch.randn(30, 40, dtype=torch.float32).to(torch.float8_e4m3fn)
        state_dict = {
            # Layer 1: FP8 (has scale_weight)
            "model.diffusion_model.layer1.weight": fp8_weight1,
            "model.diffusion_model.layer1.scale_weight": torch.tensor(1.0),
            # Layer 2: Standard (no scale_weight)
            "model.diffusion_model.layer2.weight": torch.randn(20, 30, dtype=torch.bfloat16),
            # Layer 3: FP8 (has scale_weight)
            "model.diffusion_model.layer3.weight": fp8_weight3,
            "model.diffusion_model.layer3.scale_weight": torch.tensor(1.0),
        }
        result = model_detection.detect_layer_quantization(state_dict, "model.diffusion_model.")
        
        self.assertIsNotNone(result)
        self.assertIn("layer1", result)
        self.assertIn("layer3", result)
        self.assertNotIn("layer2", result)  # Layer 2 not quantized
        self.assertEqual(result["layer1"]["format"], "fp8_e4m3fn_scaled")
        self.assertEqual(result["layer3"]["format"], "fp8_e4m3fn_scaled")
    
    def test_all_layers_quantized(self):
        """Test that uniform quantization (all layers) returns None"""
        # Create FP8 tensors by converting from float32
        fp8_weight1 = torch.randn(10, 20, dtype=torch.float32).to(torch.float8_e4m3fn)
        fp8_weight2 = torch.randn(20, 30, dtype=torch.float32).to(torch.float8_e4m3fn)
        state_dict = {
            # All layers have scale_weight
            "model.diffusion_model.layer1.weight": fp8_weight1,
            "model.diffusion_model.layer1.scale_weight": torch.tensor(1.0),
            "model.diffusion_model.layer2.weight": fp8_weight2,
            "model.diffusion_model.layer2.scale_weight": torch.tensor(1.0),
        }
        result = model_detection.detect_layer_quantization(state_dict, "model.diffusion_model.")
        
        # If all layers are quantized, it's not mixed precision
        # Should return None to use legacy scaled_fp8_ops path
        self.assertIsNone(result)
    
    def test_fp8_e5m2_detection(self):
        """Test detection of FP8 E5M2 format"""
        # Create FP8 E5M2 tensor by converting from float32
        fp8_weight = torch.randn(10, 20, dtype=torch.float32).to(torch.float8_e5m2)
        state_dict = {
            "model.diffusion_model.layer1.weight": fp8_weight,
            "model.diffusion_model.layer1.scale_weight": torch.tensor(1.0),
            "model.diffusion_model.layer2.weight": torch.randn(20, 30, dtype=torch.bfloat16),
        }
        result = model_detection.detect_layer_quantization(state_dict, "model.diffusion_model.")
        
        self.assertIsNotNone(result)
        self.assertIn("layer1", result)
        self.assertEqual(result["layer1"]["format"], "fp8_e5m2_scaled")
    
    def test_invalid_metadata(self):
        """Test with invalid metadata format"""
        state_dict = {
            "model.diffusion_model._quantization_metadata": "invalid_string",
            "model.diffusion_model.layer1.weight": torch.randn(10, 20),
        }
        result = model_detection.detect_layer_quantization(state_dict, "model.diffusion_model.")
        # Should return None on invalid metadata
        self.assertIsNone(result)
    
    def test_different_prefix(self):
        """Test with different model prefix (audio model)"""
        # Create FP8 tensor by converting from float32
        fp8_weight = torch.randn(10, 20, dtype=torch.float32).to(torch.float8_e4m3fn)
        state_dict = {
            "model.model.layer1.weight": fp8_weight,
            "model.model.layer1.scale_weight": torch.tensor(1.0),
            "model.model.layer2.weight": torch.randn(20, 30, dtype=torch.bfloat16),
        }
        result = model_detection.detect_layer_quantization(state_dict, "model.model.")
        
        self.assertIsNotNone(result)
        self.assertIn("layer1", result)


class TestPickOperationsIntegration(unittest.TestCase):
    """Test pick_operations with model_config parameter"""
    
    def test_backward_compatibility(self):
        """Test that pick_operations works without model_config (legacy)"""
        from comfy import ops
        
        # Should work without model_config parameter
        result = ops.pick_operations(None, None)
        self.assertIsNotNone(result)
        self.assertEqual(result, ops.disable_weight_init)
    
    def test_with_model_config_no_quant(self):
        """Test with model_config but no quantization"""
        from comfy import ops, supported_models_base
        
        model_config = supported_models_base.BASE({})
        model_config.layer_quant_config = None
        
        result = ops.pick_operations(None, None, model_config=model_config)
        self.assertIsNotNone(result)
        # Should use standard path
        self.assertEqual(result, ops.disable_weight_init)
    
    def test_legacy_scaled_fp8(self):
        """Test that legacy scaled_fp8 still works"""
        from comfy import ops, supported_models_base
        
        model_config = supported_models_base.BASE({})
        model_config.scaled_fp8 = torch.float8_e4m3fn
        
        result = ops.pick_operations(
            None, None, 
            scaled_fp8=torch.float8_e4m3fn,
            model_config=model_config
        )
        self.assertIsNotNone(result)
        # Should return scaled_fp8_ops (the returned class is the inner class)
        # Check that it's not the standard disable_weight_init
        self.assertNotEqual(result, ops.disable_weight_init)
        # Verify it has Linear class
        self.assertTrue(hasattr(result, 'Linear'))


if __name__ == "__main__":
    unittest.main()

