"""
End-to-end tests for mixed precision quantization.
Tests Phase 3: Mixed Precision Operations
"""

import unittest
import torch
import sys
import os

# Add comfy to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from comfy import ops


class SimpleModel(torch.nn.Module):
    """Simple model for testing mixed precision"""
    def __init__(self, operations=ops.disable_weight_init):
        super().__init__()
        self.layer1 = operations.Linear(10, 20, device="cpu", dtype=torch.bfloat16)
        self.layer2 = operations.Linear(20, 30, device="cpu", dtype=torch.bfloat16)
        self.layer3 = operations.Linear(30, 40, device="cpu", dtype=torch.bfloat16)
    
    def forward(self, x):
        x = self.layer1(x)
        x = torch.nn.functional.relu(x)
        x = self.layer2(x)
        x = torch.nn.functional.relu(x)
        x = self.layer3(x)
        return x


class TestMixedPrecisionOps(unittest.TestCase):
    """Test MixedPrecisionOps end-to-end"""
    
    def test_all_layers_standard(self):
        """Test that model with no quantization works normally"""
        # Configure no quantization
        ops.MixedPrecisionOps._layer_quant_config = {}
        
        # Create model
        model = SimpleModel(operations=ops.MixedPrecisionOps)
        
        # Initialize weights manually
        model.layer1.weight = torch.nn.Parameter(torch.randn(20, 10, dtype=torch.bfloat16))
        model.layer1.bias = torch.nn.Parameter(torch.randn(20, dtype=torch.bfloat16))
        model.layer2.weight = torch.nn.Parameter(torch.randn(30, 20, dtype=torch.bfloat16))
        model.layer2.bias = torch.nn.Parameter(torch.randn(30, dtype=torch.bfloat16))
        model.layer3.weight = torch.nn.Parameter(torch.randn(40, 30, dtype=torch.bfloat16))
        model.layer3.bias = torch.nn.Parameter(torch.randn(40, dtype=torch.bfloat16))
        
        # Initialize weight_function and bias_function
        for layer in [model.layer1, model.layer2, model.layer3]:
            layer.weight_function = []
            layer.bias_function = []
        
        # Forward pass
        input_tensor = torch.randn(5, 10, dtype=torch.bfloat16)
        output = model(input_tensor)
        
        self.assertEqual(output.shape, (5, 40))
        self.assertEqual(output.dtype, torch.bfloat16)
    
    def test_mixed_precision_load(self):
        """Test loading a mixed precision model from state dict"""
        # Configure mixed precision: layer1 is FP8, layer2 and layer3 are standard
        layer_quant_config = {
            "layer1": {
                "format": "fp8_e4m3fn_scaled",
                "params": {"use_fp8_matmul": False}  # Disable for CPU testing
            },
            "layer3": {
                "format": "fp8_e5m2_scaled",
                "params": {"use_fp8_matmul": False}
            }
        }
        ops.MixedPrecisionOps._layer_quant_config = layer_quant_config
        
        # Create state dict with mixed precision
        fp8_weight1 = torch.randn(20, 10, dtype=torch.float32).to(torch.float8_e4m3fn)
        fp8_weight3 = torch.randn(40, 30, dtype=torch.float32).to(torch.float8_e5m2)
        
        state_dict = {
            # Layer 1: FP8 E4M3FN
            "layer1.weight": fp8_weight1,
            "layer1.bias": torch.randn(20, dtype=torch.bfloat16),
            "layer1.scale_weight": torch.tensor(2.0, dtype=torch.float32),
            
            # Layer 2: Standard BF16
            "layer2.weight": torch.randn(30, 20, dtype=torch.bfloat16),
            "layer2.bias": torch.randn(30, dtype=torch.bfloat16),
            
            # Layer 3: FP8 E5M2
            "layer3.weight": fp8_weight3,
            "layer3.bias": torch.randn(40, dtype=torch.bfloat16),
            "layer3.scale_weight": torch.tensor(1.5, dtype=torch.float32),
        }
        
        # Create model and load state dict
        model = SimpleModel(operations=ops.MixedPrecisionOps)
        model.load_state_dict(state_dict)
        
        # Verify handlers are set up correctly
        self.assertIsNotNone(model.layer1.quant_handler)
        self.assertIsNone(model.layer2.quant_handler)  # No quantization
        self.assertIsNotNone(model.layer3.quant_handler)
        
        # Verify scales were loaded
        self.assertEqual(model.layer1.scale_weight.item(), 2.0)
        self.assertEqual(model.layer3.scale_weight.item(), 1.5)
        
        # Forward pass
        input_tensor = torch.randn(5, 10, dtype=torch.bfloat16)
        output = model(input_tensor)
        
        self.assertEqual(output.shape, (5, 40))
    
    def test_state_dict_round_trip(self):
        """Test saving and loading state dict preserves quantization"""
        # Configure mixed precision
        layer_quant_config = {
            "layer1": {
                "format": "fp8_e4m3fn_scaled",
                "params": {"use_fp8_matmul": False}
            }
        }
        ops.MixedPrecisionOps._layer_quant_config = layer_quant_config
        
        # Create and load model
        fp8_weight = torch.randn(20, 10, dtype=torch.float32).to(torch.float8_e4m3fn)
        state_dict1 = {
            "layer1.weight": fp8_weight,
            "layer1.bias": torch.randn(20, dtype=torch.bfloat16),
            "layer1.scale_weight": torch.tensor(3.0, dtype=torch.float32),
            "layer2.weight": torch.randn(30, 20, dtype=torch.bfloat16),
            "layer2.bias": torch.randn(30, dtype=torch.bfloat16),
            "layer3.weight": torch.randn(40, 30, dtype=torch.bfloat16),
            "layer3.bias": torch.randn(40, dtype=torch.bfloat16),
        }
        
        model1 = SimpleModel(operations=ops.MixedPrecisionOps)
        model1.load_state_dict(state_dict1)
        
        # Save state dict
        state_dict2 = model1.state_dict()
        
        # Verify scale_weight is saved
        self.assertIn("layer1.scale_weight", state_dict2)
        self.assertEqual(state_dict2["layer1.scale_weight"].item(), 3.0)
        
        # Load into new model
        model2 = SimpleModel(operations=ops.MixedPrecisionOps)
        model2.load_state_dict(state_dict2)
        
        # Verify handler is set up
        self.assertIsNotNone(model2.layer1.quant_handler)
        self.assertEqual(model2.layer1.scale_weight.item(), 3.0)
        
        # Verify forward passes match
        input_tensor = torch.randn(5, 10, dtype=torch.bfloat16)
        output1 = model1(input_tensor)
        output2 = model2(input_tensor)
        
        torch.testing.assert_close(output1, output2, rtol=1e-3, atol=1e-3)
    
    def test_weight_function_compatibility(self):
        """Test that weight_function (LoRA) works with quantized layers"""
        # Configure FP8 quantization
        layer_quant_config = {
            "layer1": {
                "format": "fp8_e4m3fn_scaled",
                "params": {"use_fp8_matmul": False}
            }
        }
        ops.MixedPrecisionOps._layer_quant_config = layer_quant_config
        
        # Create and load model
        fp8_weight = torch.randn(20, 10, dtype=torch.float32).to(torch.float8_e4m3fn)
        state_dict = {
            "layer1.weight": fp8_weight,
            "layer1.bias": torch.randn(20, dtype=torch.bfloat16),
            "layer1.scale_weight": torch.tensor(2.0, dtype=torch.float32),
            "layer2.weight": torch.randn(30, 20, dtype=torch.bfloat16),
            "layer2.bias": torch.randn(30, dtype=torch.bfloat16),
            "layer3.weight": torch.randn(40, 30, dtype=torch.bfloat16),
            "layer3.bias": torch.randn(40, dtype=torch.bfloat16),
        }
        
        model = SimpleModel(operations=ops.MixedPrecisionOps)
        model.load_state_dict(state_dict)
        
        # Add a weight function (simulating LoRA)
        # LoRA delta must match weight shape (20, 10)
        def apply_lora(weight):
            # Generate LoRA delta matching weight shape
            lora_delta = torch.randn_like(weight) * 0.01
            return weight + lora_delta
        
        model.layer1.weight_function.append(apply_lora)
        
        # Forward pass should work with LoRA
        input_tensor = torch.randn(5, 10, dtype=torch.bfloat16)
        output = model(input_tensor)
        
        self.assertEqual(output.shape, (5, 40))
    
    def test_error_handling_unknown_format(self):
        """Test that unknown formats fall back gracefully"""
        # Configure with unknown format
        layer_quant_config = {
            "layer1": {
                "format": "unknown_format_xyz",
                "params": {}
            }
        }
        ops.MixedPrecisionOps._layer_quant_config = layer_quant_config
        
        # Create state dict
        state_dict = {
            "layer1.weight": torch.randn(20, 10, dtype=torch.bfloat16),
            "layer1.bias": torch.randn(20, dtype=torch.bfloat16),
            "layer2.weight": torch.randn(30, 20, dtype=torch.bfloat16),
            "layer2.bias": torch.randn(30, dtype=torch.bfloat16),
            "layer3.weight": torch.randn(40, 30, dtype=torch.bfloat16),
            "layer3.bias": torch.randn(40, dtype=torch.bfloat16),
        }
        
        # Load should not crash, just log warning
        model = SimpleModel(operations=ops.MixedPrecisionOps)
        model.load_state_dict(state_dict)
        
        # Handler should be None (fallback to standard)
        self.assertIsNone(model.layer1.quant_handler)
        
        # Forward pass should still work
        input_tensor = torch.randn(5, 10, dtype=torch.bfloat16)
        output = model(input_tensor)
        self.assertEqual(output.shape, (5, 40))


class TestPickOperationsWithMixedPrecision(unittest.TestCase):
    """Test pick_operations with mixed precision config"""
    
    def test_pick_operations_with_layer_quant_config(self):
        """Test that pick_operations returns MixedPrecisionOps when config present"""
        from comfy import supported_models_base
        
        # Create model config with layer_quant_config
        model_config = supported_models_base.BASE({})
        model_config.layer_quant_config = {
            "layer1": {"format": "fp8_e4m3fn_scaled", "params": {}}
        }
        
        result = ops.pick_operations(None, None, model_config=model_config)
        
        self.assertEqual(result, ops.MixedPrecisionOps)
        self.assertEqual(ops.MixedPrecisionOps._layer_quant_config, model_config.layer_quant_config)
    
    def test_pick_operations_without_layer_quant_config(self):
        """Test that pick_operations falls back to standard when no config"""
        from comfy import supported_models_base
        
        model_config = supported_models_base.BASE({})
        model_config.layer_quant_config = None
        
        result = ops.pick_operations(None, None, model_config=model_config)
        
        self.assertEqual(result, ops.disable_weight_init)


if __name__ == "__main__":
    unittest.main()

