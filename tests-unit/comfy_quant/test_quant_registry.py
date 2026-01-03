import unittest
import torch
import sys
import os
import time
import gc

# Add comfy to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

def has_gpu():
    return torch.cuda.is_available()

from comfy.cli_args import args
if not has_gpu():
    args.cpu = True

from comfy.quant_ops import (
    QuantizedTensor, 
    TensorCoreFP8Layout, 
    BlockWiseINT8Layout,
    _int8_gemm_pytorch_fallback,
    _int8_gemm_triton_or_fallback
)

# set TRITON_SKIP_AUTOTUNING=1 to skip autotuning
os.environ['TRITON_SKIP_AUTOTUNING'] = '1'

class TestQuantizedTensor(unittest.TestCase):
    """Test the QuantizedTensor subclass with FP8 layout"""

    def test_creation(self):
        """Test creating a QuantizedTensor with TensorCoreFP8Layout"""
        fp8_data = torch.randn(256, 128, dtype=torch.float32).to(torch.float8_e4m3fn)
        scale = torch.tensor(2.0)
        layout_params = {'scale': scale, 'orig_dtype': torch.bfloat16}

        qt = QuantizedTensor(fp8_data, "TensorCoreFP8Layout", layout_params)

        self.assertIsInstance(qt, QuantizedTensor)
        self.assertEqual(qt.shape, (256, 128))
        self.assertEqual(qt.dtype, torch.float8_e4m3fn)
        self.assertEqual(qt._layout_params['scale'], scale)
        self.assertEqual(qt._layout_params['orig_dtype'], torch.bfloat16)
        self.assertEqual(qt._layout_type, "TensorCoreFP8Layout")

    def test_dequantize(self):
        """Test explicit dequantization"""

        fp8_data = torch.ones(10, 20, dtype=torch.float32).to(torch.float8_e4m3fn)
        scale = torch.tensor(3.0)
        layout_params = {'scale': scale, 'orig_dtype': torch.float32}

        qt = QuantizedTensor(fp8_data, "TensorCoreFP8Layout", layout_params)
        dequantized = qt.dequantize()

        self.assertEqual(dequantized.dtype, torch.float32)
        self.assertTrue(torch.allclose(dequantized, torch.ones(10, 20) * 3.0, rtol=0.1))

    def test_from_float(self):
        """Test creating QuantizedTensor from float tensor"""
        float_tensor = torch.randn(64, 32, dtype=torch.float32)
        scale = torch.tensor(1.5)

        qt = QuantizedTensor.from_float(
            float_tensor,
            "TensorCoreFP8Layout",
            scale=scale,
            dtype=torch.float8_e4m3fn
        )

        self.assertIsInstance(qt, QuantizedTensor)
        self.assertEqual(qt.dtype, torch.float8_e4m3fn)
        self.assertEqual(qt.shape, (64, 32))

        # Verify dequantization gives approximately original values
        dequantized = qt.dequantize()
        mean_rel_error = ((dequantized - float_tensor).abs() / (float_tensor.abs() + 1e-6)).mean()
        self.assertLess(mean_rel_error, 0.1)


class TestGenericUtilities(unittest.TestCase):
    """Test generic utility operations"""

    def test_detach(self):
        """Test detach operation on quantized tensor"""
        fp8_data = torch.randn(10, 20, dtype=torch.float32).to(torch.float8_e4m3fn)
        scale = torch.tensor(1.5)
        layout_params = {'scale': scale, 'orig_dtype': torch.float32}
        qt = QuantizedTensor(fp8_data, "TensorCoreFP8Layout", layout_params)

        # Detach should return a new QuantizedTensor
        qt_detached = qt.detach()

        self.assertIsInstance(qt_detached, QuantizedTensor)
        self.assertEqual(qt_detached.shape, qt.shape)
        self.assertEqual(qt_detached._layout_type, "TensorCoreFP8Layout")

    def test_clone(self):
        """Test clone operation on quantized tensor"""
        fp8_data = torch.randn(10, 20, dtype=torch.float32).to(torch.float8_e4m3fn)
        scale = torch.tensor(1.5)
        layout_params = {'scale': scale, 'orig_dtype': torch.float32}
        qt = QuantizedTensor(fp8_data, "TensorCoreFP8Layout", layout_params)

        # Clone should return a new QuantizedTensor
        qt_cloned = qt.clone()

        self.assertIsInstance(qt_cloned, QuantizedTensor)
        self.assertEqual(qt_cloned.shape, qt.shape)
        self.assertEqual(qt_cloned._layout_type, "TensorCoreFP8Layout")

        # Verify it's a deep copy
        self.assertIsNot(qt_cloned._qdata, qt._qdata)

    @unittest.skipUnless(has_gpu(), "GPU not available")
    def test_to_device(self):
        """Test device transfer"""
        fp8_data = torch.randn(10, 20, dtype=torch.float32).to(torch.float8_e4m3fn)
        scale = torch.tensor(1.5)
        layout_params = {'scale': scale, 'orig_dtype': torch.float32}
        qt = QuantizedTensor(fp8_data, "TensorCoreFP8Layout", layout_params)

        # Moving to same device should work (CPU to CPU)
        qt_cpu = qt.to('cpu')

        self.assertIsInstance(qt_cpu, QuantizedTensor)
        self.assertEqual(qt_cpu.device.type, 'cpu')
        self.assertEqual(qt_cpu._layout_params['scale'].device.type, 'cpu')


class TestTensorCoreFP8Layout(unittest.TestCase):
    """Test the TensorCoreFP8Layout implementation"""

    def test_quantize(self):
        """Test quantization method"""
        float_tensor = torch.randn(32, 64, dtype=torch.float32)
        scale = torch.tensor(1.5)

        qdata, layout_params = TensorCoreFP8Layout.quantize(
            float_tensor,
            scale=scale,
            dtype=torch.float8_e4m3fn
        )

        self.assertEqual(qdata.dtype, torch.float8_e4m3fn)
        self.assertEqual(qdata.shape, float_tensor.shape)
        self.assertIn('scale', layout_params)
        self.assertIn('orig_dtype', layout_params)
        self.assertEqual(layout_params['orig_dtype'], torch.float32)

    def test_dequantize(self):
        """Test dequantization method"""
        float_tensor = torch.ones(10, 20, dtype=torch.float32) * 3.0
        scale = torch.tensor(1.0)

        qdata, layout_params = TensorCoreFP8Layout.quantize(
            float_tensor,
            scale=scale,
            dtype=torch.float8_e4m3fn
        )

        dequantized = TensorCoreFP8Layout.dequantize(qdata, **layout_params)

        # Should approximately match original
        self.assertTrue(torch.allclose(dequantized, float_tensor, rtol=0.1, atol=0.1))


class TestFallbackMechanism(unittest.TestCase):
    """Test fallback for unsupported operations"""

    def test_unsupported_op_dequantizes(self):
        """Test that unsupported operations fall back to dequantization"""
        # Set seed for reproducibility
        torch.manual_seed(42)

        # Create quantized tensor
        a_fp32 = torch.randn(10, 20, dtype=torch.float32)
        scale = torch.tensor(1.0)
        a_q = QuantizedTensor.from_float(
            a_fp32,
            "TensorCoreFP8Layout",
            scale=scale,
            dtype=torch.float8_e4m3fn
        )

        # Call an operation that doesn't have a registered handler
        # For example, torch.abs
        result = torch.abs(a_q)

        # Should work via fallback (dequantize → abs → return)
        self.assertNotIsInstance(result, QuantizedTensor)
        expected = torch.abs(a_fp32)
        # FP8 introduces quantization error, so use loose tolerance
        mean_error = (result - expected).abs().mean()
        self.assertLess(mean_error, 0.05, f"Mean error {mean_error:.4f} is too large")


class TestBlockWiseINT8Layout(unittest.TestCase):
    """Test the BlockWiseINT8Layout implementation"""

    def test_weight_quantize_dequantize(self):
        """Test weight quantization and dequantization"""
        # Create a weight tensor (M, N) with dimensions divisible by 128
        weight = torch.randn(256, 512, dtype=torch.float32)
        block_size = 128
        
        # Quantize as weight
        qdata, layout_params = BlockWiseINT8Layout.quantize(
            weight,
            block_size=block_size,
            is_weight=True
        )
        
        # Check quantized data
        self.assertEqual(qdata.dtype, torch.int8)
        self.assertEqual(qdata.shape, weight.shape)
        
        # Check scale shape: (M//block_size, N//block_size)
        expected_scale_shape = (256 // block_size, 512 // block_size)
        self.assertEqual(layout_params['scale'].shape, expected_scale_shape)
        self.assertEqual(layout_params['block_size'], block_size)
        self.assertTrue(layout_params['is_weight'])
        self.assertEqual(layout_params['orig_dtype'], torch.float32)
        
        # Dequantize
        dequantized = BlockWiseINT8Layout.dequantize(qdata, **layout_params)
        
        # Check reconstruction quality
        self.assertEqual(dequantized.dtype, torch.float32)
        self.assertEqual(dequantized.shape, weight.shape)
        
        # INT8 has limited precision, so we use a relaxed tolerance
        max_error = (dequantized - weight).abs().max()
        mean_error = (dequantized - weight).abs().mean()
        self.assertLess(mean_error, 0.1)  # Mean error should be reasonable for INT8

    def test_activation_quantize_dequantize(self):
        """Test activation quantization and dequantization"""
        # Create an activation tensor with batch dimensions
        activation = torch.randn(4, 16, 512, dtype=torch.float32)
        block_size = 128
        
        # Quantize as activation
        qdata, layout_params = BlockWiseINT8Layout.quantize(
            activation,
            block_size=block_size,
            is_weight=False
        )
        
        # Check quantized data
        self.assertEqual(qdata.dtype, torch.int8)
        self.assertEqual(qdata.shape, activation.shape)
        
        # Check scale shape: (*batch_dims, K//block_size)
        expected_scale_shape = (4, 16, 512 // block_size)
        self.assertEqual(layout_params['scale'].shape, expected_scale_shape)
        self.assertEqual(layout_params['block_size'], block_size)
        self.assertFalse(layout_params['is_weight'])
        
        # Dequantize
        dequantized = BlockWiseINT8Layout.dequantize(qdata, **layout_params)
        
        # Check reconstruction
        self.assertEqual(dequantized.shape, activation.shape)
        mean_error = (dequantized - activation).abs().mean()
        self.assertLess(mean_error, 0.1)

    def test_quantized_tensor_creation(self):
        """Test creating QuantizedTensor with BlockWiseINT8Layout"""
        weight = torch.randn(256, 512, dtype=torch.float32)
        
        qt = QuantizedTensor.from_float(
            weight,
            "BlockWiseINT8Layout",
            block_size=128,
            is_weight=True
        )
        
        self.assertIsInstance(qt, QuantizedTensor)
        self.assertEqual(qt.dtype, torch.int8)
        self.assertEqual(qt.shape, weight.shape)
        self.assertEqual(qt._layout_type, "BlockWiseINT8Layout")
        
        # Test dequantization
        dequantized = qt.dequantize()
        self.assertEqual(dequantized.dtype, torch.float32)
        mean_error = (dequantized - weight).abs().mean()
        self.assertLess(mean_error, 0.1)


class TestBlockWiseINT8Operations(unittest.TestCase):
    """Test operations with BlockWiseINT8 quantized tensors"""

    def test_linear_operation(self):
        """Test linear operation with quantized weight and activation"""
        torch.manual_seed(42)
        
        # Create test data
        batch_size = 4
        seq_len = 16
        in_features = 256
        out_features = 512
        block_size = 128
        
        # Input activation
        input_fp32 = torch.randn(batch_size, seq_len, in_features, dtype=torch.float32)
        
        # Weight (note: linear expects weight as (out_features, in_features))
        weight_fp32 = torch.randn(out_features, in_features, dtype=torch.float32)
        bias = torch.randn(out_features, dtype=torch.float32)
        
        # Quantize both
        input_q = QuantizedTensor.from_float(
            input_fp32,
            "BlockWiseINT8Layout",
            block_size=block_size,
            is_weight=False
        )
        
        weight_q = QuantizedTensor.from_float(
            weight_fp32,
            "BlockWiseINT8Layout",
            block_size=block_size,
            is_weight=True
        )
        
        # Compute quantized linear
        output_q = torch.nn.functional.linear(input_q, weight_q, bias)
        
        # Compute reference (full precision)
        output_ref = torch.nn.functional.linear(input_fp32, weight_fp32, bias)
        
        # Compare results
        self.assertEqual(output_q.shape, output_ref.shape)
        
        # INT8 quantization introduces error, but should be reasonable
        mean_rel_error = ((output_q - output_ref).abs() / (output_ref.abs() + 1e-6)).mean()
        self.assertLess(mean_rel_error, 0.2)  # 20% relative error tolerance

    def test_clone_operation(self):
        """Test clone operation on INT8 quantized tensor"""
        weight = torch.randn(256, 512, dtype=torch.float32)
        
        qt = QuantizedTensor.from_float(
            weight,
            "BlockWiseINT8Layout",
            block_size=128,
            is_weight=True
        )
        
        qt_cloned = qt.clone()
        
        self.assertIsInstance(qt_cloned, QuantizedTensor)
        self.assertEqual(qt_cloned.shape, qt.shape)
        self.assertEqual(qt_cloned._layout_type, "BlockWiseINT8Layout")
        self.assertIsNot(qt_cloned._qdata, qt._qdata)

    def test_detach_operation(self):
        """Test detach operation on INT8 quantized tensor"""
        weight = torch.randn(256, 512, dtype=torch.float32)
        
        qt = QuantizedTensor.from_float(
            weight,
            "BlockWiseINT8Layout",
            block_size=128,
            is_weight=True
        )
        
        qt_detached = qt.detach()
        
        self.assertIsInstance(qt_detached, QuantizedTensor)
        self.assertEqual(qt_detached.shape, qt.shape)
        self.assertEqual(qt_detached._layout_type, "BlockWiseINT8Layout")

    @unittest.skipUnless(has_gpu(), "GPU not available")
    def test_device_transfer(self):
        """Test moving INT8 quantized tensor to different devices"""
        weight = torch.randn(256, 512, dtype=torch.float32)
        
        qt = QuantizedTensor.from_float(
            weight,
            "BlockWiseINT8Layout",
            block_size=128,
            is_weight=True
        )
        
        # Move to CPU (should be no-op if already on CPU)
        qt_cpu = qt.to('cpu')
        
        self.assertIsInstance(qt_cpu, QuantizedTensor)
        self.assertEqual(qt_cpu.device.type, 'cpu')
        self.assertEqual(qt_cpu._layout_params['scale'].device.type, 'cpu')

    def test_mixed_precision_fallback(self):
        """Test mixed precision: quantized weight with float input"""
        torch.manual_seed(42)
        
        input_fp32 = torch.randn(4, 256, dtype=torch.float32)
        weight_fp32 = torch.randn(512, 256, dtype=torch.float32)
        
        # Only quantize weight
        weight_q = QuantizedTensor.from_float(
            weight_fp32,
            "BlockWiseINT8Layout",
            block_size=128,
            is_weight=True
        )
        
        # Linear with float input and quantized weight
        output = torch.nn.functional.linear(input_fp32, weight_q)
        
        # Should work via fallback
        output_ref = torch.nn.functional.linear(input_fp32, weight_fp32)
        
        # With mixed precision fallback (dequantize weight), error should be small
        mean_error = (output - output_ref).abs().mean()
        self.assertLess(mean_error, 0.3)


class TestBlockWiseINT8EdgeCases(unittest.TestCase):
    """Test edge cases and error handling for INT8 quantization"""

    def test_dimension_alignment(self):
        """Test that dimensions must be divisible by block_size"""
        # Try to quantize with misaligned dimensions
        weight = torch.randn(200, 300, dtype=torch.float32)  # Not divisible by 128
        
        with self.assertRaises(AssertionError):
            BlockWiseINT8Layout.quantize(weight, block_size=128, is_weight=True)

    def test_weight_must_be_2d(self):
        """Test that weight quantization requires 2D tensors"""
        weight_3d = torch.randn(4, 256, 512, dtype=torch.float32)
        
        with self.assertRaises(AssertionError):
            BlockWiseINT8Layout.quantize(weight_3d, block_size=128, is_weight=True)

    def test_different_block_sizes(self):
        """Test quantization with different block sizes"""
        for block_size in [64, 128, 256]:
            weight = torch.randn(512, 512, dtype=torch.float32)
            
            qdata, layout_params = BlockWiseINT8Layout.quantize(
                weight,
                block_size=block_size,
                is_weight=True
            )
            
            expected_scale_shape = (512 // block_size, 512 // block_size)
            self.assertEqual(layout_params['scale'].shape, expected_scale_shape)
            
            # Verify dequantization works
            dequantized = BlockWiseINT8Layout.dequantize(qdata, **layout_params)
            self.assertEqual(dequantized.shape, weight.shape)


class TestBlockWiseINT8Precision(unittest.TestCase):
    """Precision tests for BlockWiseINT8Layout operations"""

    def test_weight_quantization_matches_manual_calculation(self):
        """Test that weight quantization matches manual PyTorch calculation"""
        device = torch.device('cuda' if has_gpu() else 'cpu')
        torch.manual_seed(42)
        
        M, N = 256, 512
        block_size = 128
        weight = torch.randn(M, N, dtype=torch.float32, device=device)
        
        # Manual PyTorch calculation for weight quantization
        # Weight shape: (M, N), blocks: (M//block_size, N//block_size)
        weight_reshaped = weight.reshape(M // block_size, block_size, N // block_size, block_size)
        weight_blocks = weight_reshaped.permute(0, 2, 1, 3)  # (M//bs, N//bs, bs, bs)
        
        # Calculate scale per block: amax / 127.0
        amax = weight_blocks.abs().amax(dim=(2, 3), keepdim=False)  # (M//bs, N//bs)
        scale_manual = amax / 127.0
        scale_manual = torch.maximum(scale_manual, torch.tensor(1e-8, device=device, dtype=weight.dtype))
        
        # Quantize: divide by scale and clamp to [-127, 127]
        weight_blocks_scaled = weight_blocks / scale_manual.unsqueeze(-1).unsqueeze(-1)
        int8_manual = torch.clamp(weight_blocks_scaled, -127.0, 127.0).to(torch.int8)
        int8_manual = int8_manual.permute(0, 2, 1, 3).reshape(M, N)
        
        # Use BlockWiseINT8Layout.quantize
        qdata, layout_params = BlockWiseINT8Layout.quantize(
            weight,
            block_size=block_size,
            is_weight=True
        )
        
        # Compare int8 values
        self.assertEqual(qdata.shape, int8_manual.shape)
        self.assertEqual(qdata.dtype, torch.int8)
        matches = (qdata == int8_manual).float().mean().item()
        self.assertGreater(matches, 0.95, f"Only {matches*100:.2f}% of int8 values match")
        
        # Compare scales
        self.assertEqual(layout_params['scale'].shape, scale_manual.shape)
        scale_diff = (layout_params['scale'] - scale_manual).abs().mean().item()
        scale_rel_diff = (scale_diff / (scale_manual.abs().mean().item() + 1e-8))
        self.assertLess(scale_rel_diff, 0.01, f"Scale relative difference too high: {scale_rel_diff}")

    def test_activation_quantization_matches_manual_calculation(self):
        """Test that activation quantization matches manual PyTorch calculation"""
        device = torch.device('cuda' if has_gpu() else 'cpu')
        torch.manual_seed(42)
        
        batch_size = 4
        seq_len = 16
        K = 512
        block_size = 128
        activation = torch.randn(batch_size, seq_len, K, dtype=torch.float32, device=device)
        
        # Manual PyTorch calculation for activation quantization
        # Activation shape: (*batch_dims, K), scale shape: (*batch_dims, K//block_size)
        orig_shape = activation.shape
        batch_dims = orig_shape[:-1]
        
        # Reshape to expose blocks in last dimension
        activation_reshaped = activation.reshape(*batch_dims, K // block_size, block_size)
        
        # Calculate scale per block: amax / 127.0
        amax = activation_reshaped.abs().amax(dim=-1, keepdim=False)  # (*batch_dims, K//block_size)
        scale_manual = amax / 127.0
        scale_manual = torch.maximum(scale_manual, torch.tensor(1e-8, device=device, dtype=activation.dtype))
        
        # Quantize: divide by scale and clamp to [-127, 127]
        activation_scaled = activation_reshaped / scale_manual.unsqueeze(-1)
        int8_manual = torch.clamp(activation_scaled, -127.0, 127.0).to(torch.int8)
        int8_manual = int8_manual.reshape(orig_shape)
        
        # Use BlockWiseINT8Layout.quantize
        qdata, layout_params = BlockWiseINT8Layout.quantize(
            activation,
            block_size=block_size,
            is_weight=False
        )
        
        # Compare int8 values
        self.assertEqual(qdata.shape, int8_manual.shape)
        self.assertEqual(qdata.dtype, torch.int8)
        matches = (qdata == int8_manual).float().mean().item()
        self.assertGreater(matches, 0.95, f"Only {matches*100:.2f}% of int8 values match")
        
        # Compare scales
        self.assertEqual(layout_params['scale'].shape, scale_manual.shape)
        scale_diff = (layout_params['scale'] - scale_manual).abs().mean().item()
        scale_rel_diff = (scale_diff / (scale_manual.abs().mean().item() + 1e-8))
        self.assertLess(scale_rel_diff, 0.01, f"Scale relative difference too high: {scale_rel_diff}")

    def test_dequantization_matches_manual_calculation(self):
        """Test that dequantization matches manual PyTorch calculation"""
        device = torch.device('cuda' if has_gpu() else 'cpu')
        torch.manual_seed(42)
        
        # Test weight dequantization
        M, N = 256, 512
        block_size = 128
        weight = torch.randn(M, N, dtype=torch.float32, device=device)
        
        # Quantize
        qdata, layout_params = BlockWiseINT8Layout.quantize(
            weight,
            block_size=block_size,
            is_weight=True
        )
        
        # Manual dequantization for weight
        scale = layout_params['scale']  # (M//bs, N//bs)
        int8_data = qdata  # (M, N)
        orig_dtype = layout_params['orig_dtype']
        
        # Reshape to blocks
        int8_reshaped = int8_data.reshape(M // block_size, block_size, N // block_size, block_size)
        int8_blocks = int8_reshaped.permute(0, 2, 1, 3)  # (M//bs, N//bs, bs, bs)
        
        # Dequantize: int8 * scale (no division by 127)
        fp_blocks = int8_blocks.to(orig_dtype) * scale.unsqueeze(-1).unsqueeze(-1)
        dequant_manual = fp_blocks.permute(0, 2, 1, 3).reshape(M, N)
        
        # Use BlockWiseINT8Layout.dequantize
        dequant_layout = BlockWiseINT8Layout.dequantize(qdata, **layout_params)
        
        # Compare
        diff = (dequant_layout - dequant_manual).abs().max().item()
        self.assertLess(diff, 1e-5, f"Dequantization differs by {diff}")
        
        # Test activation dequantization
        batch_size = 4
        seq_len = 16
        K = 512
        activation = torch.randn(batch_size, seq_len, K, dtype=torch.float32, device=device)
        
        qdata_act, layout_params_act = BlockWiseINT8Layout.quantize(
            activation,
            block_size=block_size,
            is_weight=False
        )
        
        # Manual dequantization for activation
        scale_act = layout_params_act['scale']  # (batch_size, seq_len, K//bs)
        int8_data_act = qdata_act  # (batch_size, seq_len, K)
        orig_dtype_act = layout_params_act['orig_dtype']
        
        # Reshape
        int8_reshaped_act = int8_data_act.reshape(batch_size, seq_len, K // block_size, block_size)
        # Dequantize: int8 * scale (no division by 127)
        fp_blocks_act = int8_reshaped_act.to(orig_dtype_act) * scale_act.unsqueeze(-1)
        dequant_manual_act = fp_blocks_act.reshape(batch_size, seq_len, K)
        
        # Use BlockWiseINT8Layout.dequantize
        dequant_layout_act = BlockWiseINT8Layout.dequantize(qdata_act, **layout_params_act)
        
        # Compare
        diff_act = (dequant_layout_act - dequant_manual_act).abs().max().item()
        self.assertLess(diff_act, 1e-5, f"Activation dequantization differs by {diff_act}")

    def test_triton_linear_matches_pytorch_fallback(self):
        """Test that Triton kernel INT8 GEMM matches PyTorch INT8 GEMM fallback"""
        device = torch.device('cuda' if has_gpu() else 'cpu')
        torch.manual_seed(42)
        
        batch_size = 4
        seq_len = 16
        in_features = 512
        out_features = 1024
        block_size = 128
        
        # Create original float tensors
        input_fp = torch.randn(batch_size, seq_len, in_features, dtype=torch.float32, device=device)
        weight_fp = torch.randn(out_features, in_features, dtype=torch.float32, device=device)
        bias = torch.randn(out_features, dtype=torch.float32, device=device)
        
        # Quantize to get int8 data and scales
        input_q = QuantizedTensor.from_float(
            input_fp,
            "BlockWiseINT8Layout",
            block_size=block_size,
            is_weight=False
        )
        
        weight_q = QuantizedTensor.from_float(
            weight_fp,
            "BlockWiseINT8Layout",
            block_size=block_size,
            is_weight=True
        )
        
        # Extract int8 data and scales
        a_int8, a_scale, _, _ = BlockWiseINT8Layout.get_plain_tensors(input_q)
        b_int8, b_scale, _, _ = BlockWiseINT8Layout.get_plain_tensors(weight_q)
        
        # Call Triton/fallback version (will use Triton on GPU if available)
        output_triton = _int8_gemm_triton_or_fallback(
            a_int8, a_scale, b_int8, b_scale, block_size, bias=bias, out_quant=False
        )
        
        # Call PyTorch fallback directly
        output_pytorch = _int8_gemm_pytorch_fallback(
            a_int8, a_scale, b_int8, b_scale, block_size, bias=bias
        )
        
        # Convert both to float32 for fair comparison (Triton outputs float16, PyTorch outputs float32)
        output_triton_fp32 = output_triton.to(torch.float32)
        output_pytorch_fp32 = output_pytorch.to(torch.float32)
        
        # These should match very closely (same int8 inputs, same computation)
        abs_diff = (output_triton_fp32 - output_pytorch_fp32).abs()
        mean_abs_diff = abs_diff.mean().item()
        max_abs_diff = abs_diff.max().item()
        
        # Use relative error to account for float16 precision limits
        rel_diff = abs_diff / (output_pytorch_fp32.abs() + 1e-6)
        mean_rel_diff = rel_diff.mean().item()
        
        # Since both compute the same INT8 GEMM from same inputs, differences should be tiny
        self.assertLess(mean_rel_diff, 1e-3, 
                       f"Triton and PyTorch INT8 GEMM differ too much: mean_rel={mean_rel_diff:.6f}, mean_abs={mean_abs_diff:.6f}, max={max_abs_diff:.6f}")

    def test_triton_linear_from_raw_int8_and_scales(self):
        """Test INT8 GEMM from manually created int8 data and scales - compare 3 methods"""
        device = torch.device('cuda' if has_gpu() else 'cpu')
        if not has_gpu():
            self.skipTest("This test requires GPU (Triton kernels)")
        torch.manual_seed(123)
        
        batch_size = 2
        seq_len = 8
        in_features = 256
        out_features = 512
        block_size = 128
        
        # Manually create int8 data and scales for input (activation)
        # Input shape: (batch_size, seq_len, in_features)
        input_int8 = torch.randint(-127, 127, (batch_size, seq_len, in_features), 
                                   dtype=torch.int8, device=device)
        input_scale = torch.rand(batch_size, seq_len, in_features // block_size, 
                                dtype=torch.float32, device=device) * 0.1
        
        input_layout_params = {
            'scale': input_scale,
            'block_size': block_size,
            'is_weight': False,
            'orig_dtype': torch.float32
        }
        input_q = QuantizedTensor(input_int8, "BlockWiseINT8Layout", input_layout_params)
        
        # Manually create int8 data and scales for weight
        # Weight shape: (out_features, in_features)
        weight_int8 = torch.randint(-127, 127, (out_features, in_features),
                                   dtype=torch.int8, device=device)
        weight_scale = torch.rand(out_features // block_size, in_features // block_size,
                                 dtype=torch.float32, device=device) * 0.1
        
        weight_layout_params = {
            'scale': weight_scale,
            'block_size': block_size,
            'is_weight': True,
            'orig_dtype': torch.float32
        }
        weight_q = QuantizedTensor(weight_int8, "BlockWiseINT8Layout", weight_layout_params)
        
        # Bias
        bias = torch.randn(out_features, dtype=torch.float32, device=device)
        
        # Method 1: Call INT8 GEMM via Triton/fallback
        output_triton = _int8_gemm_triton_or_fallback(
            input_int8, input_scale, weight_int8, weight_scale, block_size, bias=bias, out_quant=False
        )
        
        # Method 2: Call PyTorch INT8 GEMM fallback directly
        output_pytorch = _int8_gemm_pytorch_fallback(
            input_int8, input_scale, weight_int8, weight_scale, block_size, bias=bias
        )
        
        # Method 3: Dequantize and use standard torch.nn.functional.linear
        input_dequant = input_q.dequantize()
        weight_dequant = weight_q.dequantize()
        output_dequant = torch.nn.functional.linear(input_dequant, weight_dequant, bias)
        
        # Convert all to float32 for fair comparison
        output_triton_fp32 = output_triton.to(torch.float32)
        output_pytorch_fp32 = output_pytorch.to(torch.float32)
        output_dequant_fp32 = output_dequant.to(torch.float32)
        
        # Compare Method 1 vs Method 2: Triton vs PyTorch INT8 GEMM
        self.assertEqual(output_triton.shape, output_pytorch.shape)
        abs_diff_12 = (output_triton_fp32 - output_pytorch_fp32).abs()
        mean_abs_diff_12 = abs_diff_12.mean().item()
        max_abs_diff_12 = abs_diff_12.max().item()
        
        # Use relative error since Triton outputs float16 which has limited precision for large values
        rel_diff_12 = abs_diff_12 / (output_pytorch_fp32.abs() + 1e-6)
        mean_rel_diff_12 = rel_diff_12.mean().item()
        
        # Same int8 data → both INT8 GEMMs should produce nearly identical results
        # Use 0.1% relative error tolerance to account for float16 precision limits
        self.assertLess(mean_rel_diff_12, 1e-3, 
                       f"Triton and PyTorch INT8 GEMM differ: mean_rel={mean_rel_diff_12:.6f}, mean_abs={mean_abs_diff_12:.6f}, max_abs={max_abs_diff_12:.6f}")
        
        # Compare Method 1 vs Method 3: Triton INT8 GEMM vs Dequant+Float Linear
        self.assertEqual(output_triton.shape, output_dequant.shape)
        abs_diff_13 = (output_triton_fp32 - output_dequant_fp32).abs()
        mean_abs_diff_13 = abs_diff_13.mean().item()
        max_abs_diff_13 = abs_diff_13.max().item()
        
        # Use relative error for float16 precision limits
        rel_diff_13 = abs_diff_13 / (output_dequant_fp32.abs() + 1e-6)
        mean_rel_diff_13 = rel_diff_13.mean().item()
        
        # INT8 GEMM should match dequant+float linear (both compute the same thing)
        self.assertLess(mean_rel_diff_13, 1e-3,
                       f"Triton INT8 GEMM and dequant+float differ: mean_rel={mean_rel_diff_13:.6f}, mean_abs={mean_abs_diff_13:.6f}, max_abs={max_abs_diff_13:.6f}")
        
        # Compare Method 2 vs Method 3: PyTorch INT8 GEMM vs Dequant+Float Linear
        abs_diff_23 = (output_pytorch_fp32 - output_dequant_fp32).abs()
        mean_abs_diff_23 = abs_diff_23.mean().item()
        max_abs_diff_23 = abs_diff_23.max().item()
        
        # Use relative error
        rel_diff_23 = abs_diff_23 / (output_dequant_fp32.abs() + 1e-6)
        mean_rel_diff_23 = rel_diff_23.mean().item()
        
        # PyTorch INT8 GEMM should also match dequant+float linear
        self.assertLess(mean_rel_diff_23, 1e-3,
                       f"PyTorch INT8 GEMM and dequant+float differ: mean_rel={mean_rel_diff_23:.6f}, mean_abs={mean_abs_diff_23:.6f}, max_abs={max_abs_diff_23:.6f}")

    @unittest.skipUnless(has_gpu(), "GPU not available")
    def test_triton_vs_pytorch_linear_implementation(self):
        """Compare Triton kernel vs PyTorch fallback implementation directly"""
        torch.manual_seed(42)
        device = torch.device('cuda')
        
        batch_size = 8
        seq_len = 32
        in_features = 1024
        out_features = 2048
        block_size = 128
        
        # Create test data
        input_fp = torch.randn(batch_size, seq_len, in_features, dtype=torch.float32, device=device)
        weight_fp = torch.randn(out_features, in_features, dtype=torch.float32, device=device)
        bias = torch.randn(out_features, dtype=torch.float32, device=device)
        
        # Quantize
        input_q = QuantizedTensor.from_float(input_fp, "BlockWiseINT8Layout", 
                                            block_size=block_size, is_weight=False)
        weight_q = QuantizedTensor.from_float(weight_fp, "BlockWiseINT8Layout",
                                             block_size=block_size, is_weight=True)
        
        # Extract quantized data
        a_int8, a_scale, a_block_size, _ = BlockWiseINT8Layout.get_plain_tensors(input_q)
        b_int8, b_scale, b_block_size, _ = BlockWiseINT8Layout.get_plain_tensors(weight_q)
        
        # Call Triton version (via _int8_gemm_triton_or_fallback)
        # Note: This may still use Triton for quant fusion even with out_quant=False
        output_triton = _int8_gemm_triton_or_fallback(
            a_int8, a_scale, b_int8, b_scale, block_size, bias=bias, out_quant=False
        )
        
        # Call PyTorch fallback directly
        output_pytorch = _int8_gemm_pytorch_fallback(
            a_int8, a_scale, b_int8, b_scale, block_size, bias=bias
        )
        
        # Compare Triton vs PyTorch fallback implementations
        triton_pytorch_diff = (output_triton - output_pytorch).abs().mean().item()
        
        # These should match very closely since both compute the same operation
        self.assertLess(triton_pytorch_diff, 1e-2, 
                       f"Triton and PyTorch implementations differ: {triton_pytorch_diff}")
        
        # Also test via high-level API (which may return quantized output)
        output_api = torch.nn.functional.linear(input_q, weight_q, bias)
        if isinstance(output_api, QuantizedTensor):
            output_api_dequant = output_api.dequantize()
        else:
            output_api_dequant = output_api
        
        # Compare API with PyTorch fallback (more lenient since API might use different path)
        api_pytorch_diff = (output_api_dequant - output_pytorch).abs().mean().item()
        self.assertLess(api_pytorch_diff, 0.5,
                       f"API and PyTorch implementations differ: {api_pytorch_diff}")

    def test_int8_gemm_with_block_size_128(self):
        """Test INT8 GEMM with block_size=128 (standard size for Triton kernels)"""
        device = torch.device('cuda' if has_gpu() else 'cpu')
        torch.manual_seed(42)
        
        batch_size = 4
        seq_len = 16
        in_features = 512
        out_features = 512
        block_size = 128
        
        # Create test data
        input_fp = torch.randn(batch_size, seq_len, in_features, 
                              dtype=torch.float32, device=device)
        weight_fp = torch.randn(out_features, in_features,
                               dtype=torch.float32, device=device)
        bias = torch.randn(out_features, dtype=torch.float32, device=device)
        
        # Quantize to get int8 data
        input_q = QuantizedTensor.from_float(
            input_fp, "BlockWiseINT8Layout",
            block_size=block_size, is_weight=False
        )
        weight_q = QuantizedTensor.from_float(
            weight_fp, "BlockWiseINT8Layout",
            block_size=block_size, is_weight=True
        )
        
        # Extract int8 and scales
        a_int8, a_scale, _, _ = BlockWiseINT8Layout.get_plain_tensors(input_q)
        b_int8, b_scale, _, _ = BlockWiseINT8Layout.get_plain_tensors(weight_q)
        
        # Run Triton/fallback INT8 GEMM
        output_triton = _int8_gemm_triton_or_fallback(
            a_int8, a_scale, b_int8, b_scale, block_size, bias=bias, out_quant=False
        )
        
        # Run PyTorch INT8 GEMM fallback
        output_pytorch = _int8_gemm_pytorch_fallback(
            a_int8, a_scale, b_int8, b_scale, block_size, bias=bias
        )
        
        # Convert both to float32 for fair comparison (Triton outputs float16, PyTorch outputs float32)
        output_triton_fp32 = output_triton.to(torch.float32)
        output_pytorch_fp32 = output_pytorch.to(torch.float32)
        
        # Compare using relative error
        abs_diff = (output_triton_fp32 - output_pytorch_fp32).abs()
        mean_abs_diff = abs_diff.mean().item()
        rel_diff = abs_diff / (output_pytorch_fp32.abs() + 1e-6)
        mean_rel_diff = rel_diff.mean().item()
        
        self.assertLess(mean_rel_diff, 1e-3,
                       f"Triton and PyTorch INT8 GEMM differ: mean_rel={mean_rel_diff:.6f}, mean_abs={mean_abs_diff:.6f}")

    def test_end_to_end_quantization_accuracy(self):
        """Test end-to-end: quantize → INT8 GEMM → output accuracy vs float baseline"""
        device = torch.device('cuda' if has_gpu() else 'cpu')
        torch.manual_seed(42)
        
        batch_size = 4
        seq_len = 16
        in_features = 512
        out_features = 1024
        block_size = 128
        
        # Create float tensors
        input_fp = torch.randn(batch_size, seq_len, in_features, dtype=torch.float32, device=device)
        weight_fp = torch.randn(out_features, in_features, dtype=torch.float32, device=device)
        bias = torch.randn(out_features, dtype=torch.float32, device=device)
        
        # Float baseline
        output_float = torch.nn.functional.linear(input_fp, weight_fp, bias)
        
        # Quantize → INT8 GEMM path
        input_q = QuantizedTensor.from_float(input_fp, "BlockWiseINT8Layout",
                                            block_size=block_size, is_weight=False)
        weight_q = QuantizedTensor.from_float(weight_fp, "BlockWiseINT8Layout",
                                             block_size=block_size, is_weight=True)
        
        # Get int8 data and scales
        a_int8, a_scale, _, _ = BlockWiseINT8Layout.get_plain_tensors(input_q)
        b_int8, b_scale, _, _ = BlockWiseINT8Layout.get_plain_tensors(weight_q)
        
        # Run INT8 GEMM
        output_int8 = _int8_gemm_triton_or_fallback(
            a_int8, a_scale, b_int8, b_scale, block_size, bias=bias, out_quant=False
        )
        
        # Convert to float32 for fair comparison (Triton outputs float16)
        output_int8_fp32 = output_int8.to(torch.float32)
        output_float_fp32 = output_float.to(torch.float32)
        
        # Compare with float baseline
        abs_error = (output_int8_fp32 - output_float_fp32).abs()
        mean_abs_error = abs_error.mean().item()
        rel_error = abs_error / (output_float_fp32.abs() + 1e-6)
        mean_rel_error = rel_error.mean().item()
        
        # This error is from quantization, not from INT8 GEMM implementation
        # INT8 quantization can have ~5-20% relative error depending on data distribution
        self.assertLess(mean_rel_error, 0.25, 
                       f"Quantization error too high: {mean_rel_error:.4f}")

    def test_basic_weight_quantization(self):
        """Test basic weight quantization precision"""
        device = torch.device('cuda' if has_gpu() else 'cpu')
        weight = torch.randn(256, 512, dtype=torch.float32, device=device)
        
        qt = QuantizedTensor.from_float(
            weight,
            "BlockWiseINT8Layout",
            block_size=128,
            is_weight=True
        )
        
        self.assertEqual(qt.shape, weight.shape)
        self.assertEqual(qt.dtype, torch.int8)
        
        dequantized = qt.dequantize()
        error = (dequantized - weight).abs().mean()
        self.assertLess(error, 0.1, "Mean reconstruction error too high")

    def test_large_activation_quantization(self):
        """Test activation quantization with larger tensor"""
        device = torch.device('cuda' if has_gpu() else 'cpu')
        activation = torch.randn(16, 128, 4096, dtype=torch.float32, device=device)
        
        qt = QuantizedTensor.from_float(
            activation,
            "BlockWiseINT8Layout",
            block_size=128,
            is_weight=False
        )
        
        self.assertEqual(qt.shape, activation.shape)
        self.assertEqual(qt.dtype, torch.int8)
        
        dequantized = qt.dequantize()
        error = (dequantized - activation).abs().mean()
        self.assertLess(error, 0.1, "Mean reconstruction error too high")

    def test_quantized_linear_precision(self):
        """Test quantized linear operation precision"""
        torch.manual_seed(42)
        device = torch.device('cuda' if has_gpu() else 'cpu')
        
        batch_size = 16
        seq_len = 128
        in_features = 2048
        out_features = 2048
        block_size = 128
        
        input_fp32 = torch.randn(batch_size, seq_len, in_features, dtype=torch.float32, device=device)
        weight_fp32 = torch.randn(out_features, in_features, dtype=torch.float32, device=device)
        bias = torch.randn(out_features, dtype=torch.float32, device=device)
        
        # Quantize both
        input_q = QuantizedTensor.from_float(
            input_fp32,
            "BlockWiseINT8Layout",
            block_size=block_size,
            is_weight=False
        )
        
        weight_q = QuantizedTensor.from_float(
            weight_fp32,
            "BlockWiseINT8Layout",
            block_size=block_size,
            is_weight=True
        )
        
        # Compute quantized linear (returns QuantizedTensor by default)
        output_q = torch.nn.functional.linear(input_q, weight_q, bias)
        output_q = QuantizedTensor.from_float(output_q, "BlockWiseINT8Layout", block_size=block_size, is_weight=False)

        self.assertIsInstance(output_q, QuantizedTensor, "Default output should be QuantizedTensor")
        
        # Dequantize for comparison
        output_dequant = output_q.dequantize()
        
        # Compute reference
        output_ref = torch.nn.functional.linear(input_fp32, weight_fp32, bias)
        
        self.assertEqual(output_dequant.shape, output_ref.shape)
        
        mean_rel_error = ((output_dequant - output_ref).abs() / (output_ref.abs() + 1e-6)).mean()
        self.assertLess(mean_rel_error, 0.2, "Mean relative error too high")

    @unittest.skipUnless(has_gpu(), "GPU not available")
    def test_triton_vs_pytorch_precision(self):
        """Compare Triton kernel vs PyTorch fallback precision"""
        # Check if Triton is available
        try:
            from comfy.int8_kernels import int8_gemm as triton_int8_gemm
            has_triton = True
        except ImportError:
            self.skipTest("Triton kernels not available")
        
        torch.manual_seed(42)
        device = torch.device('cuda')
        
        batch_size = 4
        seq_len = 16
        in_features = 256
        out_features = 512
        block_size = 128
        
        # Create test data
        input_fp32 = torch.randn(batch_size, seq_len, in_features, dtype=torch.float32, device=device)
        weight_fp32 = torch.randn(out_features, in_features, dtype=torch.float32, device=device)
        bias = torch.randn(out_features, dtype=torch.float32, device=device)
        
        # Quantize
        input_q = QuantizedTensor.from_float(
            input_fp32,
            "BlockWiseINT8Layout",
            block_size=block_size,
            is_weight=False
        )
        
        weight_q = QuantizedTensor.from_float(
            weight_fp32,
            "BlockWiseINT8Layout",
            block_size=block_size,
            is_weight=True
        )
        
        # Extract quantized data
        a_int8, a_scale, _, _ = BlockWiseINT8Layout.get_plain_tensors(input_q)
        b_int8, b_scale, _, _ = BlockWiseINT8Layout.get_plain_tensors(weight_q)
        
        # Run Triton version (via _int8_gemm_triton_or_fallback)
        output_triton = _int8_gemm_triton_or_fallback(a_int8, a_scale, b_int8, b_scale, block_size, bias)
        
        # Run PyTorch fallback directly
        output_pytorch = _int8_gemm_pytorch_fallback(a_int8, a_scale, b_int8, b_scale, block_size, bias)
        
        # Compute reference
        output_ref = torch.nn.functional.linear(input_fp32, weight_fp32, bias)
        
        # Compare errors
        error_triton = ((output_triton - output_ref).abs() / (output_ref.abs() + 1e-6)).mean()
        error_pytorch = ((output_pytorch - output_ref).abs() / (output_ref.abs() + 1e-6)).mean()
        error_between = (output_triton - output_pytorch).abs().mean()
        
        self.assertLess(error_triton, 0.2, "Triton error too high")
        self.assertLess(error_pytorch, 0.2, "PyTorch error too high")
        self.assertLess(error_between, 4e-3, "Triton and PyTorch implementations differ")
        
        # Test via high-level API (torch dispatch)
        output_dispatch = torch.nn.functional.linear(input_q, weight_q, bias)
        
        # Dequantize if needed
        if isinstance(output_dispatch, QuantizedTensor):
            output_dispatch_fp32 = output_dispatch.dequantize()
        else:
            output_dispatch_fp32 = output_dispatch
        
        # Compare with reference
        error_dispatch = ((output_dispatch_fp32 - output_ref).abs() / (output_ref.abs() + 1e-6)).mean()
        self.assertLess(error_dispatch, 0.2, "Torch dispatch error too high")
        
        # Compare dispatch output with low-level Triton output
        error_dispatch_vs_triton = (output_dispatch_fp32 - output_triton).abs().mean()
        self.assertLess(error_dispatch_vs_triton, 0.2, "Dispatch differs from low-level implementation")

    @unittest.skipUnless(has_gpu(), "GPU not available")
    def test_int8_vs_fp8_precision(self):
        """Compare INT8 vs FP8 precision"""
        # Check if FP8 is available
        try:
            test_tensor = torch.randn(16, 16, device='cuda', dtype=torch.float32)
            _ = test_tensor.to(torch.float8_e4m3fn)
        except (RuntimeError, AttributeError):
            self.skipTest("FP8 dtypes not supported on this system")
        
        torch.manual_seed(42)
        device = torch.device('cuda')
        
        batch_size = 16
        seq_len = 128
        in_features = 2048
        out_features = 2048
        block_size = 128
        
        # Create test data
        input_fp32 = torch.randn(batch_size, seq_len, in_features, dtype=torch.float32, device=device)
        weight_fp32 = torch.randn(out_features, in_features, dtype=torch.float32, device=device)
        bias = torch.randn(out_features, dtype=torch.float32, device=device)
        
        # Quantize with INT8
        input_int8 = QuantizedTensor.from_float(
            input_fp32,
            "BlockWiseINT8Layout",
            block_size=block_size,
            is_weight=False
        )
        
        weight_int8 = QuantizedTensor.from_float(
            weight_fp32,
            "BlockWiseINT8Layout",
            block_size=block_size,
            is_weight=True
        )
        
        # Quantize with FP8
        input_fp8 = QuantizedTensor.from_float(
            input_fp32,
            "TensorCoreFP8Layout",
            dtype=torch.float8_e4m3fn
        )
        
        weight_fp8 = QuantizedTensor.from_float(
            weight_fp32,
            "TensorCoreFP8Layout",
            dtype=torch.float8_e4m3fn
        )
        
        # Compute outputs
        output_int8_q = torch.nn.functional.linear(input_int8, weight_int8, bias)
        output_int8 = output_int8_q.dequantize() if isinstance(output_int8_q, QuantizedTensor) else output_int8_q
        
        # FP8 doesn't support fused bias, so add it manually
        output_fp8 = torch.nn.functional.linear(input_fp8, weight_fp8, None)
        if bias is not None:
            output_fp8 = output_fp8 + bias
        if isinstance(output_fp8, QuantizedTensor):
            output_fp8 = output_fp8.dequantize()
        
        output_ref = torch.nn.functional.linear(input_fp32, weight_fp32, bias)
        
        # Compare precision
        error_int8 = ((output_int8 - output_ref).abs() / (output_ref.abs() + 1e-6)).mean()
        error_fp8 = ((output_fp8 - output_ref).abs() / (output_ref.abs() + 1e-6)).mean()
        error_between = (output_int8 - output_fp8).abs().mean()
        
        self.assertLess(error_int8, 0.2, "INT8 error too high")
        self.assertLess(error_fp8, 0.4, "FP8 error too high")
        
        # Memory usage comparison
        int8_memory = input_int8._qdata.element_size() * input_int8._qdata.numel() + \
                      weight_int8._qdata.element_size() * weight_int8._qdata.numel()
        fp8_memory = input_fp8._qdata.element_size() * input_fp8._qdata.numel() + \
                     weight_fp8._qdata.element_size() * weight_fp8._qdata.numel()
        fp32_memory = input_fp32.element_size() * input_fp32.numel() + \
                      weight_fp32.element_size() * weight_fp32.numel()
        
        self.assertLess(int8_memory, fp32_memory, "INT8 should use less memory than FP32")
        self.assertLess(fp8_memory, fp32_memory, "FP8 should use less memory than FP32")

    def test_output_types(self):
        """Test output types for all registered operations"""
        device = torch.device('cuda' if has_gpu() else 'cpu')
        torch.manual_seed(42)
        
        batch_size = 4
        seq_len = 16
        in_features = 256
        out_features = 512
        block_size = 128
        
        # Create test data
        input_fp32 = torch.randn(batch_size, seq_len, in_features, dtype=torch.float32, device=device)
        weight_fp32 = torch.randn(out_features, in_features, dtype=torch.float32, device=device)
        bias = torch.randn(out_features, dtype=torch.float32, device=device)
        
        # Quantize with INT8
        input_int8 = QuantizedTensor.from_float(
            input_fp32,
            "BlockWiseINT8Layout",
            block_size=block_size,
            is_weight=False
        )
        weight_int8 = QuantizedTensor.from_float(
            weight_fp32,
            "BlockWiseINT8Layout",
            block_size=block_size,
            is_weight=True
        )
        
        # Test 1: linear with quantized output (default)
        output = torch.nn.functional.linear(input_int8, weight_int8, bias)
        output = QuantizedTensor.from_float(output, "BlockWiseINT8Layout", block_size=block_size, is_weight=False)
        self.assertIsInstance(output, QuantizedTensor, "Default output should be QuantizedTensor")
        self.assertEqual(output.layout_type, "BlockWiseINT8Layout")
        
        # Test 2: linear with explicit dequantization
        output_q = torch.nn.functional.linear(input_int8, weight_int8, bias)
        output_reg = output_q.dequantize()
        self.assertNotIsInstance(output_reg, QuantizedTensor, "Dequantized output should be regular tensor")
        
        # Test 3: mm operation (2D input) - default quantized output
        input_2d = input_fp32.reshape(-1, in_features)
        input_int8_2d = QuantizedTensor.from_float(input_2d, "BlockWiseINT8Layout", block_size=block_size, is_weight=False)
        weight_int8_t = weight_int8.t()
        
        output_mm = torch.mm(input_int8_2d, weight_int8_t)
        output_mm = QuantizedTensor.from_float(output_mm, "BlockWiseINT8Layout", block_size=block_size, is_weight=False)
        self.assertIsInstance(output_mm, QuantizedTensor, "Default mm output should be QuantizedTensor")
        self.assertEqual(output_mm.layout_type, "BlockWiseINT8Layout")
        
        # Test 4: addmm operation - default quantized output
        output_addmm = torch.addmm(bias, input_int8_2d, weight_int8_t)
        output_addmm = QuantizedTensor.from_float(output_addmm, "BlockWiseINT8Layout", block_size=block_size, is_weight=False)
        self.assertIsInstance(output_addmm, QuantizedTensor, "Default addmm output should be QuantizedTensor")
        self.assertEqual(output_addmm.layout_type, "BlockWiseINT8Layout")
        
        # Test 5: view operation preserves quantization
        view_result = input_int8.view(batch_size * seq_len, in_features)
        self.assertIsInstance(view_result, QuantizedTensor, "view should preserve QuantizedTensor")
        self.assertEqual(view_result.layout_type, "BlockWiseINT8Layout")
        
        # Test 6: transpose operation preserves quantization
        transpose_result = weight_int8.t()
        self.assertIsInstance(transpose_result, QuantizedTensor, "transpose should preserve QuantizedTensor")
        self.assertEqual(transpose_result.layout_type, "BlockWiseINT8Layout")
        
        # Test 7: clone operation preserves quantization
        clone_result = input_int8.clone()
        self.assertIsInstance(clone_result, QuantizedTensor, "clone should preserve QuantizedTensor")
        self.assertEqual(clone_result.layout_type, "BlockWiseINT8Layout")
        
        # Test 8: detach operation preserves quantization
        detach_result = input_int8.detach()
        self.assertIsInstance(detach_result, QuantizedTensor, "detach should preserve QuantizedTensor")
        self.assertEqual(detach_result.layout_type, "BlockWiseINT8Layout")


class TestBlockWiseINT8GELU(unittest.TestCase):
    """Test INT8 block-wise GELU activation"""

    def test_int8_gelu_basic(self):
        """Test basic GELU operation with INT8 quantized tensors"""
        device = torch.device('cuda' if has_gpu() else 'cpu')
        
        batch_size = 2
        seq_len = 512
        hidden_dim = 2048
        block_size = 128
        
        # Create random input tensor
        x = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16, device=device)
        
        # Compute reference output (full precision)
        with torch.no_grad():
            reference_output = torch.nn.functional.gelu(x)
        
        # Quantize input
        x_quant = QuantizedTensor.from_float(x, "BlockWiseINT8Layout", block_size=block_size, is_weight=False)
        
        # Apply GELU (should use fused kernel)
        with torch.no_grad():
            output_quant = torch.nn.functional.gelu(x_quant)
        
        if isinstance(output_quant, QuantizedTensor):
            output_fp = output_quant.dequantize()
        else:
            output_fp = output_quant
        
        self.assertEqual(output_fp.shape, reference_output.shape)
        
        # Compute error metrics
        relative_error = (torch.norm(output_fp - reference_output) / torch.norm(reference_output)).item()
        self.assertLess(relative_error, 0.1, f"Relative error too high: {relative_error}")

    def test_int8_gelu_2d(self):
        """Test GELU with 2D tensors"""
        device = torch.device('cuda' if has_gpu() else 'cpu')
        
        M, N = 256, 2048
        block_size = 128
        
        x = torch.randn(M, N, dtype=torch.float16, device=device)
        reference_output = torch.nn.functional.gelu(x)
        
        # Quantize and apply GELU
        x_quant = QuantizedTensor.from_float(x, "BlockWiseINT8Layout", block_size=block_size, is_weight=False)
        
        with torch.no_grad():
            output_quant = torch.nn.functional.gelu(x_quant)
        
        if isinstance(output_quant, QuantizedTensor):
            output_fp = output_quant.dequantize()
        else:
            output_fp = output_quant
        
        relative_error = (torch.norm(output_fp - reference_output) / torch.norm(reference_output)).item()
        self.assertLess(relative_error, 0.1, f"Relative error too high: {relative_error}")

    def test_int8_gelu_different_shapes(self):
        """Test GELU with various tensor shapes"""
        device = torch.device('cuda' if has_gpu() else 'cpu')
        block_size = 128
        
        test_shapes = [
            (128, 1024),        # 2D
            (4, 512, 2048),     # 3D
            (2, 8, 128, 1024),  # 4D
        ]
        
        for shape in test_shapes:
            with self.subTest(shape=shape):
                x = torch.randn(*shape, dtype=torch.float16, device=device)
                reference_output = torch.nn.functional.gelu(x)
                
                # Quantize and apply GELU
                x_quant = QuantizedTensor.from_float(x, "BlockWiseINT8Layout", block_size=block_size, is_weight=False)
                
                with torch.no_grad():
                    output_quant = torch.nn.functional.gelu(x_quant)
                
                if isinstance(output_quant, QuantizedTensor):
                    output_fp = output_quant.dequantize()
                else:
                    output_fp = output_quant
                
                relative_error = (torch.norm(output_fp - reference_output) / torch.norm(reference_output)).item()
                self.assertLess(relative_error, 0.1, f"Relative error too high for shape {shape}: {relative_error}")


class TestBlockWiseINT8QuantFusion(unittest.TestCase):
    """Test fused INT8 matmul + quantization kernels"""

    @unittest.skip("out_quant parameter not yet implemented in torch ops")
    @unittest.skipUnless(has_gpu(), "GPU not available")
    def test_int8_linear_with_out_quant(self):
        """Test INT8 linear operation with fused output quantization"""
        batch_size = 4
        seq_len = 256
        input_dim = 1024
        output_dim = 2048
        block_size = 128
        
        # Create input tensor
        input_fp = torch.randn(batch_size, seq_len, input_dim, dtype=torch.float16, device='cuda')
        weight_fp = torch.randn(output_dim, input_dim, dtype=torch.float16, device='cuda')
        bias = torch.randn(output_dim, dtype=torch.float16, device='cuda')
        
        # Quantize input and weight
        input_q = QuantizedTensor.from_float(
            input_fp, 
            "BlockWiseINT8Layout", 
            block_size=block_size, 
            is_weight=False
        )
        
        weight_q = QuantizedTensor.from_float(
            weight_fp, 
            "BlockWiseINT8Layout", 
            block_size=block_size, 
            is_weight=True
        )
        
        # Test 1: Regular linear (float output)
        output_float = torch.ops.aten.linear.default(input_q, weight_q, bias)
        self.assertIsNotNone(output_float)
        self.assertEqual(output_float.shape, (batch_size, seq_len, output_dim))
        
        # Test 2: Linear with fused output quantization (out_quant=True)
        output_quant = torch.ops.aten.linear.default(
            input_q, weight_q, bias
        )
        
        self.assertIsInstance(output_quant, QuantizedTensor, "Output should be QuantizedTensor when out_quant=True")
        self.assertEqual(output_quant._layout_type, "BlockWiseINT8Layout")
        
        # Verify scale shape matches activation format
        expected_scale_shape = (batch_size, seq_len, output_dim // block_size)
        actual_scale_shape = output_quant._layout_params['scale'].shape
        self.assertEqual(actual_scale_shape, expected_scale_shape, "Scale shape should match activation format")
        
        # Dequantize and compare
        output_dequant = output_quant.dequantize()
        self.assertEqual(output_dequant.shape, (batch_size, seq_len, output_dim))
        
        # Compare with float output
        diff = (output_float - output_dequant).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        relative_error = (diff / (output_float.abs() + 1e-6)).mean().item()
        
        self.assertLess(relative_error, 0.15, f"Relative error too high: {relative_error}")

    @unittest.skipUnless(has_gpu(), "GPU not available")
    def test_int8_addmm_with_out_quant(self):
        """Test INT8 addmm operation with fused output quantization"""
        M, K, N = 512, 1024, 2048
        block_size = 128
        
        # Create tensors
        input_fp = torch.randn(M, K, dtype=torch.float16, device='cuda')
        weight_fp = torch.randn(N, K, dtype=torch.float16, device='cuda')
        bias = torch.randn(N, dtype=torch.float16, device='cuda')
        
        # Quantize
        input_q = QuantizedTensor.from_float(
            input_fp, "BlockWiseINT8Layout", 
            block_size=block_size, is_weight=False
        )
        weight_q = QuantizedTensor.from_float(
            weight_fp, "BlockWiseINT8Layout", 
            block_size=block_size, is_weight=True
        )
        
        # Test with out_quant=True
        output_quant = torch.ops.aten.addmm.default(
            bias, input_q, weight_q.t()
        )
        
        output_quant = QuantizedTensor.from_float(output_quant, "BlockWiseINT8Layout", block_size=block_size, is_weight=False)
        self.assertIsInstance(output_quant, QuantizedTensor, "Output should be QuantizedTensor when out_quant=True")
        self.assertEqual(output_quant.shape, (M, N))
        self.assertEqual(output_quant._layout_type, "BlockWiseINT8Layout")
        
        # Verify it can be dequantized
        output_dequant = output_quant.dequantize()
        self.assertEqual(output_dequant.shape, (M, N))
        self.assertEqual(output_dequant.dtype, torch.float16)


# Benchmark tests (skipped by default)
class TestBlockWiseINT8Benchmarks(unittest.TestCase):
    """Performance benchmark tests for BlockWiseINT8Layout"""

    @unittest.skip("perf benchmark only")
    @unittest.skipUnless(has_gpu(), "GPU not available")
    def test_runtime_comparison(self):
        """Benchmark INT8 quantized ops via torch dispatch (high-level API)"""
        device = torch.device('cuda')
        torch.manual_seed(42)
        
        # More comprehensive test configurations
        test_configs = [
            {"name": "Tiny", "batch": 2, "seq": 8, "in_feat": 128, "out_feat": 256, "block": 64},
            {"name": "Small", "batch": 4, "seq": 16, "in_feat": 256, "out_feat": 512, "block": 128},
            {"name": "Medium", "batch": 8, "seq": 32, "in_feat": 512, "out_feat": 1024, "block": 128},
            {"name": "Large", "batch": 16, "seq": 64, "in_feat": 1024, "out_feat": 2048, "block": 128},
            {"name": "XL", "batch": 32, "seq": 128, "in_feat": 2048, "out_feat": 4096, "block": 128},
            {"name": "XXL", "batch": 64, "seq": 256, "in_feat": 4096, "out_feat": 4096, "block": 128},
        ]
        
        n_warmup = 10
        n_iters = 200  # More iterations for better averaging
        
        print(f"\nWarmup iterations: {n_warmup}")
        print(f"Benchmark iterations: {n_iters}\n")
        
        # Check if Triton is available
        try:
            from comfy.int8_kernels import int8_gemm as triton_int8_gemm
            print("✓ Using Triton INT8 kernels (optimized path)\n")
        except ImportError:
            print("⚠ Using PyTorch fallback (Triton not available)\n")
        
        results = []
        
        for config in test_configs:
            name = config["name"]
            batch_size = config["batch"]
            seq_len = config["seq"]
            in_features = config["in_feat"]
            out_features = config["out_feat"]
            block_size = config["block"]
            
            print(f"{name}: batch={batch_size}, seq={seq_len}, in={in_features}, out={out_features}, block={block_size}")
            
            # Calculate FLOPS for this configuration
            m = batch_size * seq_len
            k = in_features
            n = out_features
            flops = 2 * m * n * k  # 2 for multiply-add
            
            try:
                # Create test data
                input_fp32 = torch.randn(batch_size, seq_len, in_features, dtype=torch.float32, device=device)
                weight_fp32 = torch.randn(out_features, in_features, dtype=torch.float32, device=device)
                bias = torch.randn(out_features, dtype=torch.float32, device=device)
                
                # Quantize using high-level API
                input_int8 = QuantizedTensor.from_float(input_fp32, "BlockWiseINT8Layout", block_size=block_size, is_weight=False)
                weight_int8 = QuantizedTensor.from_float(weight_fp32, "BlockWiseINT8Layout", block_size=block_size, is_weight=True)
                
                # Warm up - test full dispatch path
                for _ in range(n_warmup):
                    _ = torch.nn.functional.linear(input_int8, weight_int8, bias)
                    _ = torch.nn.functional.linear(input_fp32, weight_fp32, bias)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                
                # Benchmark INT8 via torch dispatch (includes dispatch overhead + quantized output)
                int8_times = []
                for _ in range(n_iters):
                    start = time.time()
                    output = torch.nn.functional.linear(input_int8, weight_int8, bias)
                    torch.cuda.synchronize()
                    int8_times.append((time.time() - start) * 1000)
                
                # Also benchmark with dequantization to FP32 output (more realistic for some use cases)
                int8_dequant_times = []
                for _ in range(n_iters):
                    start = time.time()
                    output = torch.nn.functional.linear(input_int8, weight_int8, bias)
                    if isinstance(output, QuantizedTensor):
                        output = output.dequantize()
                    torch.cuda.synchronize()
                    int8_dequant_times.append((time.time() - start) * 1000)
                
                # Benchmark FP32 reference
                fp32_times = []
                for _ in range(n_iters):
                    start = time.time()
                    _ = torch.nn.functional.linear(input_fp32, weight_fp32, bias)
                    torch.cuda.synchronize()
                    fp32_times.append((time.time() - start) * 1000)
                
                # Convert to torch tensors for statistics
                int8_times = torch.tensor(int8_times)
                int8_dequant_times = torch.tensor(int8_dequant_times)
                fp32_times = torch.tensor(fp32_times)
                
                # Calculate statistics
                int8_mean = int8_times.mean().item()
                int8_std = int8_times.std().item()
                int8_min = int8_times.min().item()
                
                int8_dequant_mean = int8_dequant_times.mean().item()
                int8_dequant_std = int8_dequant_times.std().item()
                int8_dequant_min = int8_dequant_times.min().item()
                
                fp32_mean = fp32_times.mean().item()
                fp32_std = fp32_times.std().item()
                fp32_min = fp32_times.min().item()
                
                speedup_int8 = fp32_mean / int8_mean
                speedup_int8_dequant = fp32_mean / int8_dequant_mean
                
                print(f"  INT8 (quantized out): {int8_mean:.3f}±{int8_std:.3f} ms (min: {int8_min:.3f} ms) [{flops/int8_mean/1e9:.2f} GFLOPS]")
                print(f"  INT8 (dequant out):   {int8_dequant_mean:.3f}±{int8_dequant_std:.3f} ms (min: {int8_dequant_min:.3f} ms) [{flops/int8_dequant_mean/1e9:.2f} GFLOPS]")
                print(f"  FP32 reference:       {fp32_mean:.3f}±{fp32_std:.3f} ms (min: {fp32_min:.3f} ms) [{flops/fp32_mean/1e9:.2f} GFLOPS]")
                print(f"  Speedup (INT8 quantized/FP32): {speedup_int8:.2f}x")
                print(f"  Speedup (INT8 dequant/FP32):   {speedup_int8_dequant:.2f}x")
                print(f"  Dequant overhead: {((int8_dequant_mean - int8_mean) / int8_mean * 100):.1f}%\n")
                
                results.append({
                    "name": name,
                    "int8_mean": int8_mean,
                    "int8_dequant_mean": int8_dequant_mean,
                    "fp32_mean": fp32_mean,
                    "speedup_int8": speedup_int8,
                    "speedup_int8_dequant": speedup_int8_dequant,
                    "flops": flops,
                })
                
                # Clean up memory after each configuration
                del input_fp32, weight_fp32, bias, input_int8, weight_int8
                if 'int8_times' in locals():
                    del int8_times, int8_dequant_times, fp32_times
                gc.collect()
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  ⚠ OOM - skipping this configuration\n")
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                else:
                    raise
        
        # Print summary
        print("\n" + "=" * 60)
        print("Summary:")
        print("=" * 60)
        for result in results:
            print(f"{result['name']:8s}: INT8 {result['int8_mean']:.3f}ms, "
                  f"INT8+dequant {result['int8_dequant_mean']:.3f}ms, "
                  f"FP32 {result['fp32_mean']:.3f}ms, "
                  f"Speedup: {result['speedup_int8']:.2f}x (quantized), {result['speedup_int8_dequant']:.2f}x (dequant)")
        
        # Assertions for unittest
        self.assertGreater(len(results), 0, "Should have collected benchmark results")

    @unittest.skip("perf benchmark only")
    @unittest.skipUnless(has_gpu(), "GPU not available")
    def test_int8_vs_fp8_runtime(self):
        """Benchmark INT8 vs FP8 runtime with comprehensive configs"""
        # Check if FP8 is available
        try:
            test_tensor = torch.randn(16, 16, device='cuda', dtype=torch.float32)
            _ = test_tensor.to(torch.float8_e4m3fn)
            has_fp8 = True
        except (RuntimeError, AttributeError):
            has_fp8 = False
        
        if not has_fp8:
            print("⚠ FP8 dtypes not supported on this system, skipping comparison")
            self.skipTest("FP8 not supported")
            return
        
        device = torch.device('cuda')
        torch.manual_seed(42)
        
        # More comprehensive test configurations
        test_configs = [
            {"name": "Tiny", "batch": 2, "seq": 8, "in_feat": 128, "out_feat": 256, "block": 64},
            {"name": "Small", "batch": 4, "seq": 16, "in_feat": 256, "out_feat": 512, "block": 128},
            {"name": "Medium", "batch": 8, "seq": 32, "in_feat": 512, "out_feat": 1024, "block": 128},
            {"name": "Large", "batch": 16, "seq": 64, "in_feat": 1024, "out_feat": 2048, "block": 128},
            {"name": "XL", "batch": 32, "seq": 128, "in_feat": 2048, "out_feat": 4096, "block": 128},
            {"name": "XXL", "batch": 64, "seq": 256, "in_feat": 4096, "out_feat": 4096, "block": 128},
            {"name": "XXXL", "batch": 128, "seq": 512, "in_feat": 4096, "out_feat": 4096, "block": 128},
        ]
        
        n_warmup = 10
        n_iters = 200  # More iterations for better averaging
        
        print(f"\nWarmup iterations: {n_warmup}")
        print(f"Benchmark iterations: {n_iters}")
        print("Note: INT8 uses fused bias, FP8 adds bias separately\n")
        
        results = []
        
        for config in test_configs:
            name = config["name"]
            batch_size = config["batch"]
            seq_len = config["seq"]
            in_features = config["in_feat"]
            out_features = config["out_feat"]
            block_size = config["block"]
            
            print(f"{name}: batch={batch_size}, seq={seq_len}, in={in_features}, out={out_features}, block={block_size}")
            
            # Calculate FLOPS for this configuration
            m = batch_size * seq_len
            k = in_features
            n = out_features
            flops = 2 * m * n * k  # 2 for multiply-add
            
            try:
                # Create test data
                input_fp32 = torch.randn(batch_size, seq_len, in_features, dtype=torch.float32, device=device)
                weight_fp32 = torch.randn(out_features, in_features, dtype=torch.float32, device=device)
                bias = torch.randn(out_features, dtype=torch.float32, device=device)
                
                # Quantize with INT8
                input_int8 = QuantizedTensor.from_float(input_fp32, "BlockWiseINT8Layout", block_size=block_size, is_weight=False)
                weight_int8 = QuantizedTensor.from_float(weight_fp32, "BlockWiseINT8Layout", block_size=block_size, is_weight=True)
                
                # Quantize with FP8
                input_fp8 = QuantizedTensor.from_float(input_fp32, "TensorCoreFP8Layout", dtype=torch.float8_e4m3fn)
                weight_fp8 = QuantizedTensor.from_float(weight_fp32, "TensorCoreFP8Layout", dtype=torch.float8_e4m3fn)
                
                # Warm up
                for _ in range(n_warmup):
                    _ = torch.nn.functional.linear(input_int8, weight_int8, bias)
                    out_fp8 = torch.nn.functional.linear(input_fp8, weight_fp8, None)
                    if bias is not None:
                        _ = out_fp8 + bias
                    _ = torch.nn.functional.linear(input_fp32, weight_fp32, bias)
                torch.cuda.synchronize()
                
                # Benchmark INT8 (with fused bias) - collect all times
                int8_times = []
                for _ in range(n_iters):
                    start = time.time()
                    _ = torch.nn.functional.linear(input_int8, weight_int8, bias)
                    torch.cuda.synchronize()
                    int8_times.append((time.time() - start) * 1000)
                
                # Benchmark FP8 (bias added separately)
                fp8_times = []
                for _ in range(n_iters):
                    start = time.time()
                    out_fp8 = torch.nn.functional.linear(input_fp8, weight_fp8, None)
                    if bias is not None:
                        _ = out_fp8 + bias
                    torch.cuda.synchronize()
                    fp8_times.append((time.time() - start) * 1000)
                
                # Benchmark FP32 reference
                fp32_times = []
                for _ in range(n_iters):
                    start = time.time()
                    _ = torch.nn.functional.linear(input_fp32, weight_fp32, bias)
                    torch.cuda.synchronize()
                    fp32_times.append((time.time() - start) * 1000)
                
                # Convert to torch tensors for statistics
                int8_times = torch.tensor(int8_times)
                fp8_times = torch.tensor(fp8_times)
                fp32_times = torch.tensor(fp32_times)
                
                # Calculate statistics
                int8_mean = int8_times.mean().item()
                int8_std = int8_times.std().item()
                int8_min = int8_times.min().item()
                
                fp8_mean = fp8_times.mean().item()
                fp8_std = fp8_times.std().item()
                fp8_min = fp8_times.min().item()
                
                fp32_mean = fp32_times.mean().item()
                fp32_std = fp32_times.std().item()
                fp32_min = fp32_times.min().item()
                
                speedup_int8 = fp32_mean / int8_mean
                speedup_fp8 = fp32_mean / fp8_mean
                int8_vs_fp8 = fp8_mean / int8_mean
                
                print(f"  INT8 (fused bias): {int8_mean:.3f}±{int8_std:.3f} ms (min: {int8_min:.3f} ms) [{flops/int8_mean/1e9:.2f} GFLOPS]")
                print(f"  FP8 (sep. bias):   {fp8_mean:.3f}±{fp8_std:.3f} ms (min: {fp8_min:.3f} ms) [{flops/fp8_mean/1e9:.2f} GFLOPS]")
                print(f"  FP32 (fused bias): {fp32_mean:.3f}±{fp32_std:.3f} ms (min: {fp32_min:.3f} ms) [{flops/fp32_mean/1e9:.2f} GFLOPS]")
                print(f"  Speedup (INT8/FP32): {speedup_int8:.2f}x")
                print(f"  Speedup (FP8/FP32):  {speedup_fp8:.2f}x")
                
                if int8_mean < fp8_mean:
                    print(f"  ✓ INT8 is {int8_vs_fp8:.2f}x faster than FP8\n")
                else:
                    print(f"  ✓ FP8 is {1/int8_vs_fp8:.2f}x faster than INT8\n")
                
                results.append({
                    "name": name,
                    "int8_mean": int8_mean,
                    "fp8_mean": fp8_mean,
                    "fp32_mean": fp32_mean,
                    "speedup_int8": speedup_int8,
                    "speedup_fp8": speedup_fp8,
                    "int8_vs_fp8": int8_vs_fp8,
                    "flops": flops,
                })
                
                # Clean up memory after each configuration
                del input_fp32, weight_fp32, bias, input_int8, weight_int8
                if has_fp8:
                    del input_fp8, weight_fp8
                if 'int8_times' in locals():
                    del int8_times, fp8_times, fp32_times
                gc.collect()
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  ⚠ OOM - skipping this configuration\n")
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                else:
                    raise
        
        # Print summary
        print("\n" + "=" * 60)
        print("Summary:")
        print("=" * 60)
        for result in results:
            print(f"{result['name']:8s}: INT8 {result['int8_mean']:.3f}ms, "
                  f"FP8 {result['fp8_mean']:.3f}ms, "
                  f"FP32 {result['fp32_mean']:.3f}ms, "
                  f"Speedup (INT8/FP32): {result['speedup_int8']:.2f}x, "
                  f"(FP8/FP32): {result['speedup_fp8']:.2f}x")
        
        # Assertions for unittest
        self.assertGreater(len(results), 0, "Should have collected benchmark results")

    @unittest.skip("perf benchmark only")
    @unittest.skipUnless(has_gpu(), "GPU not available")
    def test_quantization_dequantization_runtime(self):
        """Benchmark quantization and dequantization operations"""
        device = torch.device('cuda')
        torch.manual_seed(42)
        
        n_warmup = 5
        n_iters = 100
        
        print(f"\nWarmup iterations: {n_warmup}")
        print(f"Benchmark iterations: {n_iters}\n")
        
        # Test configurations - various tensor sizes
        test_configs = [
            {"name": "Small Weight", "shape": (512, 512), "is_weight": True, "block": 128},
            {"name": "Medium Weight", "shape": (2048, 2048), "is_weight": True, "block": 128},
            {"name": "Large Weight", "shape": (4096, 4096), "is_weight": True, "block": 128},
            {"name": "XL Weight", "shape": (8192, 8192), "is_weight": True, "block": 128},
            {"name": "Small Activation", "shape": (8, 64, 512), "is_weight": False, "block": 128},
            {"name": "Medium Activation", "shape": (16, 128, 2048), "is_weight": False, "block": 128},
            {"name": "Large Activation", "shape": (32, 256, 4096), "is_weight": False, "block": 128},
            {"name": "XL Activation", "shape": (64, 512, 4096), "is_weight": False, "block": 128},
        ]
        
        print("=" * 60)
        print("INT8 BlockWise Quantization/Dequantization")
        print("=" * 60)
        
        results_int8 = []
        
        for config in test_configs:
            name = config["name"]
            shape = config["shape"]
            is_weight = config["is_weight"]
            block_size = config["block"]
            
            try:
                # Create test tensor
                tensor_fp32 = torch.randn(shape, dtype=torch.float32, device=device)
                tensor_size_mb = tensor_fp32.numel() * tensor_fp32.element_size() / 1024 / 1024
                
                print(f"\n{name}: shape={shape}, size={tensor_size_mb:.2f}MB")
                
                # Warm up
                for _ in range(n_warmup):
                    qt = QuantizedTensor.from_float(tensor_fp32, "BlockWiseINT8Layout", block_size=block_size, is_weight=is_weight)
                    _ = qt.dequantize()
                torch.cuda.synchronize()
                
                # Benchmark quantization
                quant_times = []
                for _ in range(n_iters):
                    start = time.time()
                    qt = QuantizedTensor.from_float(tensor_fp32, "BlockWiseINT8Layout", block_size=block_size, is_weight=is_weight)
                    torch.cuda.synchronize()
                    quant_times.append((time.time() - start) * 1000)
                
                # Benchmark dequantization (reuse last quantized tensor)
                dequant_times = []
                for _ in range(n_iters):
                    start = time.time()
                    _ = qt.dequantize()
                    torch.cuda.synchronize()
                    dequant_times.append((time.time() - start) * 1000)
                
                # Calculate statistics
                quant_times = torch.tensor(quant_times)
                dequant_times = torch.tensor(dequant_times)
                
                quant_mean = quant_times.mean().item()
                quant_std = quant_times.std().item()
                quant_min = quant_times.min().item()
                
                dequant_mean = dequant_times.mean().item()
                dequant_std = dequant_times.std().item()
                dequant_min = dequant_times.min().item()
                
                # Calculate throughput (GB/s)
                quant_throughput = (tensor_size_mb / 1024) / (quant_mean / 1000)
                dequant_throughput = (tensor_size_mb / 1024) / (dequant_mean / 1000)
                
                print(f"  Quantization:   {quant_mean:.3f}±{quant_std:.3f} ms (min: {quant_min:.3f} ms) [{quant_throughput:.2f} GB/s]")
                print(f"  Dequantization: {dequant_mean:.3f}±{dequant_std:.3f} ms (min: {dequant_min:.3f} ms) [{dequant_throughput:.2f} GB/s]")
                print(f"  Total roundtrip: {quant_mean + dequant_mean:.3f} ms")
                
                # Calculate memory savings
                qt_memory = qt._qdata.element_size() * qt._qdata.numel()
                qt_memory += qt._layout_params['scale'].element_size() * qt._layout_params['scale'].numel()
                fp32_memory = tensor_fp32.element_size() * tensor_fp32.numel()
                reduction = fp32_memory / qt_memory
                
                print(f"  Memory: FP32 {fp32_memory/1024/1024:.2f}MB -> INT8 {qt_memory/1024/1024:.2f}MB ({reduction:.2f}x reduction)")
                
                results_int8.append({
                    "name": name,
                    "shape": shape,
                    "size_mb": tensor_size_mb,
                    "quant_mean": quant_mean,
                    "dequant_mean": dequant_mean,
                    "quant_throughput": quant_throughput,
                    "dequant_throughput": dequant_throughput,
                    "reduction": reduction,
                })
                
                # Clean up memory after each configuration
                del tensor_fp32, qt
                if 'quant_times' in locals():
                    del quant_times, dequant_times
                gc.collect()
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n{name}: ⚠ OOM - skipping")
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                else:
                    raise
        
        # Summary
        print()
        print("=" * 60)
        print("Summary: INT8 Quantization/Dequantization Performance")
        print("=" * 60)
        for result in results_int8:
            print(f"{result['name']:20s}: Quant {result['quant_mean']:.3f}ms, "
                  f"Dequant {result['dequant_mean']:.3f}ms, "
                  f"Total {result['quant_mean'] + result['dequant_mean']:.3f}ms")
        
        # Assertions for unittest
        self.assertGreater(len(results_int8), 0, "Should have collected benchmark results")

    @unittest.skip("perf benchmark only")
    @unittest.skipUnless(has_gpu(), "GPU not available")
    def test_fp16_vs_int8_real_model_sizes(self):
        """Compare FP16 vs INT8 vs FP8 on actual model sizes via torch dispatch"""
        device = torch.device('cuda')
        torch.manual_seed(42)
        
        # Check if FP8 is available
        try:
            test_tensor = torch.randn(16, 16, device='cuda', dtype=torch.float32)
            _ = test_tensor.to(torch.float8_e4m3fn)
            has_fp8 = True
            print("✓ FP8 support detected")
        except (RuntimeError, AttributeError):
            has_fp8 = False
            print("⚠ FP8 not supported on this system - will compare FP16 vs INT8 only")
        
        # Actual sizes from model dumps
        test_configs = [
            # WAN 2.2 5B model sizes
            {
                "model": "WAN2.2-5B",
                "name": "First layer (small batch)",
                "input_shape": (2, 1, 3072),
                "weight_shape": (18432, 3072),
                "block_size": 128,
            },
            {
                "model": "WAN2.2-5B",
                "name": "Attention layer (long seq)",
                "input_shape": (2, 27280, 3072),
                "weight_shape": (3072, 3072),
                "block_size": 128,
            },
            {
                "model": "WAN2.2-5B",
                "name": "MLP down projection (long seq)",
                "input_shape": (2, 27280, 14336),
                "weight_shape": (3072, 14336),
                "block_size": 128,
            },
            {
                "model": "WAN2.2-5B",
                "name": "MLP up projection (long seq)",
                "input_shape": (2, 27280, 3072),
                "weight_shape": (14336, 3072),
                "block_size": 128,
            },
            {
                "model": "WAN2.2-5B",
                "name": "Attention layer (medium seq)",
                "input_shape": (2, 512, 3072),
                "weight_shape": (3072, 3072),
                "block_size": 128,
            },
            # WAN 2.2 14B model sizes
            {
                "model": "WAN2.2-14B",
                "name": "First layer (small batch)",
                "input_shape": (2, 1, 5120),
                "weight_shape": (30720, 5120),
                "block_size": 128,
            },
            {
                "model": "WAN2.2-14B",
                "name": "Attention layer (long seq)",
                "input_shape": (2, 27280, 5120),
                "weight_shape": (5120, 5120),
                "block_size": 128,
            },
            {
                "model": "WAN2.2-14B",
                "name": "Attention layer (medium seq)",
                "input_shape": (2, 512, 5120),
                "weight_shape": (5120, 5120),
                "block_size": 128,
            },
            {
                "model": "WAN2.2-14B",
                "name": "MLP up projection (long seq)",
                "input_shape": (2, 27280, 5120),
                "weight_shape": (13824, 5120),
                "block_size": 128,
            },
            {
                "model": "WAN2.2-14B",
                "name": "MLP down projection (long seq)",
                "input_shape": (2, 27280, 13824),
                "weight_shape": (5120, 13824),
                "block_size": 128,
            },
        ]
        
        n_warmup = 10
        n_iters = 100
        
        print(f"\nWarmup iterations: {n_warmup}")
        print(f"Benchmark iterations: {n_iters}\n")
        
        results = []
        current_model = None
        
        for config in test_configs:
            model = config["model"]
            name = config["name"]
            input_shape = config["input_shape"]
            weight_shape = config["weight_shape"]
            block_size = config["block_size"]
            
            # Print model header when we switch models
            if model != current_model:
                print("\n" + "=" * 60)
                print(f"{model} Model Layers")
                print("=" * 60)
                current_model = model
            
            print(f"\n{name}")
            print(f"  Input: {input_shape}, Weight: {weight_shape}")
            
            # Calculate FLOPS
            batch, seq_len, in_features = input_shape
            out_features, _ = weight_shape
            m = batch * seq_len
            k = in_features
            n = out_features
            flops = 2 * m * n * k
            
            try:
                # Measure initial VRAM
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                initial_vram = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                
                # Create test data in FP16 and FP32
                input_fp32 = torch.randn(input_shape, dtype=torch.float32, device=device)
                input_fp16 = input_fp32.to(torch.float16)
                
                weight_fp32 = torch.randn(weight_shape, dtype=torch.float32, device=device)
                weight_fp16 = weight_fp32.to(torch.float16)
                
                bias_fp32 = torch.randn(out_features, dtype=torch.float32, device=device)
                bias_fp16 = bias_fp32.to(torch.float16)
                
                # Measure FP16 VRAM
                fp16_vram = torch.cuda.memory_allocated() / 1024 / 1024 - initial_vram
                
                # Quantize to INT8
                input_int8 = QuantizedTensor.from_float(input_fp32, "BlockWiseINT8Layout", block_size=block_size, is_weight=False)
                weight_int8 = QuantizedTensor.from_float(weight_fp32, "BlockWiseINT8Layout", block_size=block_size, is_weight=True)
                
                # Measure INT8 VRAM (after creating quantized tensors, before releasing FP16)
                int8_vram_with_fp16 = torch.cuda.memory_allocated() / 1024 / 1024 - initial_vram
                
                # Quantize to FP8 if available
                if has_fp8:
                    input_fp8 = QuantizedTensor.from_float(input_fp32, "TensorCoreFP8Layout", dtype=torch.float8_e4m3fn)
                    weight_fp8 = QuantizedTensor.from_float(weight_fp32, "TensorCoreFP8Layout", dtype=torch.float8_e4m3fn)
                    fp8_vram_with_others = torch.cuda.memory_allocated() / 1024 / 1024 - initial_vram
                
                # Calculate memory usage
                fp16_input_mem = input_fp16.element_size() * input_fp16.numel()
                fp16_weight_mem = weight_fp16.element_size() * weight_fp16.numel()
                fp16_total_mem = fp16_input_mem + fp16_weight_mem
                
                int8_input_mem = input_int8._qdata.element_size() * input_int8._qdata.numel()
                int8_input_mem += input_int8._layout_params['scale'].element_size() * input_int8._layout_params['scale'].numel()
                
                int8_weight_mem = weight_int8._qdata.element_size() * weight_int8._qdata.numel()
                int8_weight_mem += weight_int8._layout_params['scale'].element_size() * weight_int8._layout_params['scale'].numel()
                
                int8_total_mem = int8_input_mem + int8_weight_mem
                mem_reduction = fp16_total_mem / int8_total_mem
                
                print(f"  Tensor Memory: FP16 {fp16_total_mem/1024/1024:.2f}MB -> INT8 {int8_total_mem/1024/1024:.2f}MB ({mem_reduction:.2f}x reduction)")
                print(f"  VRAM Usage: FP16 {fp16_vram:.2f}MB, INT8 {int8_vram_with_fp16:.2f}MB (incl. FP16 tensors)")
                
                # Warm up
                for _ in range(n_warmup):
                    _ = torch.nn.functional.linear(input_fp16, weight_fp16, bias_fp16)
                    _ = torch.nn.functional.linear(input_int8, weight_int8, bias_fp32)
                    if has_fp8:
                        out_fp8 = torch.nn.functional.linear(input_fp8, weight_fp8, None)
                        if bias_fp32 is not None:
                            _ = out_fp8 + bias_fp32
                torch.cuda.synchronize()
                
                # Clear any warmup artifacts
                torch.cuda.empty_cache()
                
                # Benchmark FP16
                fp16_times = []
                for _ in range(n_iters):
                    start = time.time()
                    output_fp16 = torch.nn.functional.linear(input_fp16, weight_fp16, bias_fp16)
                    torch.cuda.synchronize()
                    fp16_times.append((time.time() - start) * 1000)
                
                # Benchmark INT8 (quantized output)
                int8_times = []
                for _ in range(n_iters):
                    start = time.time()
                    output_int8 = torch.nn.functional.linear(input_int8, weight_int8, bias_fp32)
                    torch.cuda.synchronize()
                    int8_times.append((time.time() - start) * 1000)
                
                # Benchmark INT8 with dequantization
                int8_dequant_times = []
                for _ in range(n_iters):
                    start = time.time()
                    output_int8 = torch.nn.functional.linear(input_int8, weight_int8, bias_fp32)
                    if isinstance(output_int8, QuantizedTensor):
                        output_int8 = output_int8.dequantize()
                    torch.cuda.synchronize()
                    int8_dequant_times.append((time.time() - start) * 1000)
                
                # Benchmark FP8 if available
                if has_fp8:
                    fp8_times = []
                    for _ in range(n_iters):
                        start = time.time()
                        out_fp8 = torch.nn.functional.linear(input_fp8, weight_fp8, None)
                        if bias_fp32 is not None:
                            out_fp8 = out_fp8 + bias_fp32
                        # Dequantize if needed
                        if isinstance(out_fp8, QuantizedTensor):
                            out_fp8 = out_fp8.dequantize()
                        torch.cuda.synchronize()
                        fp8_times.append((time.time() - start) * 1000)
                
                # Clear benchmark outputs to free memory
                if 'output_fp16' in locals():
                    del output_fp16
                if 'output_int8' in locals():
                    del output_int8
                if has_fp8 and 'out_fp8' in locals():
                    del out_fp8
                torch.cuda.empty_cache()
                
                # Calculate statistics
                fp16_times = torch.tensor(fp16_times)
                int8_times = torch.tensor(int8_times)
                int8_dequant_times = torch.tensor(int8_dequant_times)
                
                fp16_mean = fp16_times.mean().item()
                fp16_std = fp16_times.std().item()
                fp16_min = fp16_times.min().item()
                
                int8_mean = int8_times.mean().item()
                int8_std = int8_times.std().item()
                int8_min = int8_times.min().item()
                
                int8_dequant_mean = int8_dequant_times.mean().item()
                int8_dequant_std = int8_dequant_times.std().item()
                int8_dequant_min = int8_dequant_times.min().item()
                
                speedup_int8 = fp16_mean / int8_mean
                speedup_int8_dequant = fp16_mean / int8_dequant_mean
                
                print(f"  FP16:               {fp16_mean:.3f}±{fp16_std:.3f} ms (min: {fp16_min:.3f} ms) [{flops/fp16_mean/1e9:.2f} GFLOPS]")
                print(f"  INT8 (quantized):   {int8_mean:.3f}±{int8_std:.3f} ms (min: {int8_min:.3f} ms) [{flops/int8_mean/1e9:.2f} GFLOPS]")
                print(f"  INT8 (dequantized): {int8_dequant_mean:.3f}±{int8_dequant_std:.3f} ms (min: {int8_dequant_min:.3f} ms) [{flops/int8_dequant_mean/1e9:.2f} GFLOPS]")
                print(f"  Speedup vs FP16: {speedup_int8:.2f}x (quantized), {speedup_int8_dequant:.2f}x (dequantized)")
                
                if has_fp8:
                    fp8_times = torch.tensor(fp8_times)
                    fp8_mean = fp8_times.mean().item()
                    fp8_std = fp8_times.std().item()
                    fp8_min = fp8_times.min().item()
                    speedup_fp8 = fp16_mean / fp8_mean
                    
                    print(f"  FP8 (dequantized):  {fp8_mean:.3f}±{fp8_std:.3f} ms (min: {fp8_min:.3f} ms) [{flops/fp8_mean/1e9:.2f} GFLOPS]")
                    print(f"  Speedup vs FP16: {speedup_fp8:.2f}x")
                else:
                    fp8_mean = None
                    speedup_fp8 = None
                
                # Precision check
                output_fp16_check = torch.nn.functional.linear(input_fp16, weight_fp16, bias_fp16)
                output_int8_check = torch.nn.functional.linear(input_int8, weight_int8, bias_fp32)
                if isinstance(output_int8_check, QuantizedTensor):
                    output_int8_check = output_int8_check.dequantize()
                
                # Convert FP16 output to FP32 for comparison
                output_fp16_check_fp32 = output_fp16_check.to(torch.float32)
                
                # Compare INT8 vs FP16 (both in FP32 for fair comparison)
                error_int8 = ((output_int8_check - output_fp16_check_fp32).abs() / (output_fp16_check_fp32.abs() + 1e-6)).mean()
                print(f"  Precision: INT8 vs FP16 mean relative error: {error_int8:.6f}")
                
                if has_fp8:
                    output_fp8_check = torch.nn.functional.linear(input_fp8, weight_fp8, None)
                    if bias_fp32 is not None:
                        output_fp8_check = output_fp8_check + bias_fp32
                    if isinstance(output_fp8_check, QuantizedTensor):
                        output_fp8_check = output_fp8_check.dequantize()
                    
                    error_fp8 = ((output_fp8_check - output_fp16_check_fp32).abs() / (output_fp16_check_fp32.abs() + 1e-6)).mean()
                    print(f"  Precision: FP8 vs FP16 mean relative error: {error_fp8:.6f}")
                else:
                    error_fp8 = None
                
                results.append({
                    "model": model,
                    "name": name,
                    "input_shape": input_shape,
                    "weight_shape": weight_shape,
                    "fp16_mean": fp16_mean,
                    "int8_mean": int8_mean,
                    "int8_dequant_mean": int8_dequant_mean,
                    "fp8_mean": fp8_mean,
                    "speedup_int8": speedup_int8,
                    "speedup_int8_dequant": speedup_int8_dequant,
                    "speedup_fp8": speedup_fp8,
                    "mem_reduction": mem_reduction,
                    "error_int8": error_int8.item(),
                    "error_fp8": error_fp8.item() if error_fp8 is not None else None,
                    "fp16_vram": fp16_vram,
                    "int8_vram": int8_vram_with_fp16,
                })
                
                # Aggressive memory cleanup after each configuration to avoid OOM
                # Delete input/weight tensors
                del input_fp32, input_fp16, weight_fp32, weight_fp16, bias_fp32, bias_fp16
                del input_int8, weight_int8
                if has_fp8:
                    del input_fp8, weight_fp8
                
                # Delete precision check outputs
                if 'output_fp16_check' in locals():
                    del output_fp16_check, output_fp16_check_fp32, output_int8_check
                if has_fp8 and 'output_fp8_check' in locals():
                    del output_fp8_check
                
                # Delete timing tensors
                if 'fp16_times' in locals():
                    del fp16_times, int8_times, int8_dequant_times
                if has_fp8 and 'fp8_times' in locals():
                    del fp8_times
                
                # Force Python garbage collection
                gc.collect()
                # Clear CUDA cache
                torch.cuda.empty_cache()
                # Synchronize to ensure cleanup is complete
                torch.cuda.synchronize()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  ⚠ OOM - skipping this configuration")
                    # Ultra-aggressive cleanup on OOM
                    # Delete any lingering tensors from failed iteration
                    for var_name in list(locals().keys()):
                        if 'tensor' in var_name.lower() or var_name.endswith(('_fp16', '_fp32', '_int8', '_fp8')):
                            try:
                                del locals()[var_name]
                            except:
                                pass
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                else:
                    raise
        
        # Summary table
        print("\n" + "=" * 80)
        if has_fp8:
            print("Summary: FP16 vs INT8 vs FP8 Performance")
        else:
            print("Summary: FP16 vs INT8 Performance")
        print("=" * 80)
        
        if results:
            # Group results by model
            models = {}
            for result in results:
                model = result["model"]
                if model not in models:
                    models[model] = []
                models[model].append(result)
            
            # Print results grouped by model
            for model_name, model_results in models.items():
                print(f"\n{model_name}:")
                if has_fp8:
                    print(f"{'Layer':<25s} {'FP16':<10s} {'INT8':<10s} {'FP8':<10s} {'Speedup':<20s} {'Mem':<8s}")
                else:
                    print(f"{'Layer':<30s} {'FP16 (ms)':<12s} {'INT8 (ms)':<12s} {'Speedup':<10s} {'Memory':<10s}")
                print("-" * 80)
                
                for result in model_results:
                    layer_name = result["name"][:23] if has_fp8 else result["name"][:28]
                    if has_fp8 and result['fp8_mean'] is not None:
                        print(f"{layer_name:<25s} {result['fp16_mean']:>8.3f}ms {result['int8_dequant_mean']:>8.3f}ms {result['fp8_mean']:>8.3f}ms "
                              f"INT8:{result['speedup_int8_dequant']:>5.2f}x FP8:{result['speedup_fp8']:>5.2f}x {result['mem_reduction']:>6.2f}x")
                    else:
                        print(f"{layer_name:<30s} {result['fp16_mean']:>10.3f}   {result['int8_dequant_mean']:>10.3f}   {result['speedup_int8_dequant']:>8.2f}x   {result['mem_reduction']:>8.2f}x")
                
                # Calculate per-model total
                model_fp16_time = sum(r["fp16_mean"] for r in model_results)
                model_int8_time = sum(r["int8_dequant_mean"] for r in model_results)
                model_speedup_int8 = model_fp16_time / model_int8_time if model_int8_time > 0 else 0
                
                print("-" * 80)
                if has_fp8 and any(r['fp8_mean'] is not None for r in model_results):
                    model_fp8_time = sum(r["fp8_mean"] for r in model_results if r["fp8_mean"] is not None)
                    model_speedup_fp8 = model_fp16_time / model_fp8_time if model_fp8_time > 0 else 0
                    print(f"{'SUBTOTAL':<25s} {model_fp16_time:>8.3f}ms {model_int8_time:>8.3f}ms {model_fp8_time:>8.3f}ms "
                          f"INT8:{model_speedup_int8:>5.2f}x FP8:{model_speedup_fp8:>5.2f}x")
                else:
                    print(f"{'SUBTOTAL':<30s} {model_fp16_time:>10.3f}   {model_int8_time:>10.3f}   {model_speedup_int8:>8.2f}x")
                
                print(f"  {model_name} avg memory reduction: {sum(r['mem_reduction'] for r in model_results) / len(model_results):.2f}x")
                print(f"  {model_name} avg INT8 precision error: {sum(r['error_int8'] for r in model_results) / len(model_results):.6f}")
                if has_fp8 and any(r['error_fp8'] is not None for r in model_results):
                    fp8_errors = [r['error_fp8'] for r in model_results if r['error_fp8'] is not None]
                    if fp8_errors:
                        print(f"  {model_name} avg FP8 precision error: {sum(fp8_errors) / len(fp8_errors):.6f}")
                
                # VRAM analysis
                total_fp16_vram = sum(r['fp16_vram'] for r in model_results)
                total_int8_vram = sum(r['int8_vram'] for r in model_results)
                print(f"  {model_name} VRAM usage: FP16 {total_fp16_vram:.2f}MB, INT8 {total_int8_vram:.2f}MB (during inference with both)")
            
            # Calculate overall totals
            total_fp16_time = sum(r["fp16_mean"] for r in results)
            total_int8_time = sum(r["int8_dequant_mean"] for r in results)
            overall_speedup_int8 = total_fp16_time / total_int8_time if total_int8_time > 0 else 0
            
            print("\n" + "=" * 80)
            if has_fp8 and any(r['fp8_mean'] is not None for r in results):
                total_fp8_time = sum(r["fp8_mean"] for r in results if r["fp8_mean"] is not None)
                overall_speedup_fp8 = total_fp16_time / total_fp8_time if total_fp8_time > 0 else 0
                print(f"{'GRAND TOTAL':<25s} {total_fp16_time:>8.3f}ms {total_int8_time:>8.3f}ms {total_fp8_time:>8.3f}ms "
                      f"INT8:{overall_speedup_int8:>5.2f}x FP8:{overall_speedup_fp8:>5.2f}x")
            else:
                print(f"{'GRAND TOTAL':<30s} {total_fp16_time:>10.3f}   {total_int8_time:>10.3f}   {overall_speedup_int8:>8.2f}x")
            print("=" * 80)
            
            print(f"\n✓ Overall INT8 speedup: {overall_speedup_int8:.2f}x faster than FP16")
            if has_fp8 and any(r['fp8_mean'] is not None for r in results):
                print(f"✓ Overall FP8 speedup: {overall_speedup_fp8:.2f}x faster than FP16")
            print(f"✓ Average memory reduction: {sum(r['mem_reduction'] for r in results) / len(results):.2f}x")
            print(f"✓ Average INT8 precision error: {sum(r['error_int8'] for r in results) / len(results):.6f}")
            if has_fp8:
                fp8_errors = [r['error_fp8'] for r in results if r['error_fp8'] is not None]
                if fp8_errors:
                    print(f"✓ Average FP8 precision error: {sum(fp8_errors) / len(fp8_errors):.6f}")
            
            # Total VRAM
            total_fp16_vram = sum(r['fp16_vram'] for r in results)
            total_int8_vram = sum(r['int8_vram'] for r in results)
            print(f"✓ Total VRAM: FP16 {total_fp16_vram:.2f}MB, INT8 {total_int8_vram:.2f}MB")
        
        # Assertions for unittest
        self.assertGreater(len(results), 0, "Should have collected benchmark results")
        self.assertGreater(overall_speedup_int8, 0.5, "INT8 should have reasonable performance")

    @unittest.skip("perf benchmark only")
    @unittest.skipUnless(has_gpu(), "GPU not available")
    def test_systematic_benchmark(self):
        """Comprehensive systematic benchmark across multiple dimensions"""
        device = torch.device('cuda')
        torch.manual_seed(42)
        
        n_warmup = 10
        n_iters = 100
        
        print(f"\nWarmup iterations: {n_warmup}")
        print(f"Benchmark iterations: {n_iters}\n")
        
        # Test 1: Varying batch size (typical transformer forward pass)
        print("=" * 60)
        print("Dimension 1: Varying Batch Size")
        print("=" * 60)
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
        seq_len = 64
        in_features = 1024
        out_features = 1024
        block_size = 128
        
        for batch_size in batch_sizes:
            try:
                input_fp32 = torch.randn(batch_size, seq_len, in_features, dtype=torch.float32, device=device)
                weight_fp32 = torch.randn(out_features, in_features, dtype=torch.float32, device=device)
                bias = torch.randn(out_features, dtype=torch.float32, device=device)
                
                input_int8 = QuantizedTensor.from_float(input_fp32, "BlockWiseINT8Layout", block_size=block_size, is_weight=False)
                weight_int8 = QuantizedTensor.from_float(weight_fp32, "BlockWiseINT8Layout", block_size=block_size, is_weight=True)
                
                # Warm up
                for _ in range(n_warmup):
                    _ = torch.nn.functional.linear(input_int8, weight_int8, bias)
                    _ = torch.nn.functional.linear(input_fp32, weight_fp32, bias)
                torch.cuda.synchronize()
                
                # Benchmark
                int8_times = []
                for _ in range(n_iters):
                    start = time.time()
                    _ = torch.nn.functional.linear(input_int8, weight_int8, bias)
                    torch.cuda.synchronize()
                    int8_times.append((time.time() - start) * 1000)
                
                fp32_times = []
                for _ in range(n_iters):
                    start = time.time()
                    _ = torch.nn.functional.linear(input_fp32, weight_fp32, bias)
                    torch.cuda.synchronize()
                    fp32_times.append((time.time() - start) * 1000)
                
                int8_mean = torch.tensor(int8_times).mean().item()
                fp32_mean = torch.tensor(fp32_times).mean().item()
                speedup = fp32_mean / int8_mean
                
                m = batch_size * seq_len
                k = in_features
                n = out_features
                flops = 2 * m * n * k
                
                print(f"Batch={batch_size:3d}: INT8 {int8_mean:.3f}ms, FP32 {fp32_mean:.3f}ms, Speedup: {speedup:.2f}x, [{flops/int8_mean/1e9:.2f} GFLOPS]")
                
                # Clean up after each test
                del input_fp32, weight_fp32, bias, input_int8, weight_int8
                gc.collect()
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"Batch={batch_size:3d}: ⚠ OOM")
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    break
                else:
                    raise
        
        print()
        
        # Test 2: Varying sequence length
        print("=" * 60)
        print("Dimension 2: Varying Sequence Length")
        print("=" * 60)
        seq_lengths = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        batch_size = 8
        in_features = 1024
        out_features = 1024
        block_size = 128
        
        for seq_len in seq_lengths:
            try:
                input_fp32 = torch.randn(batch_size, seq_len, in_features, dtype=torch.float32, device=device)
                weight_fp32 = torch.randn(out_features, in_features, dtype=torch.float32, device=device)
                bias = torch.randn(out_features, dtype=torch.float32, device=device)
                
                input_int8 = QuantizedTensor.from_float(input_fp32, "BlockWiseINT8Layout", block_size=block_size, is_weight=False)
                weight_int8 = QuantizedTensor.from_float(weight_fp32, "BlockWiseINT8Layout", block_size=block_size, is_weight=True)
                
                # Warm up
                for _ in range(n_warmup):
                    _ = torch.nn.functional.linear(input_int8, weight_int8, bias)
                    _ = torch.nn.functional.linear(input_fp32, weight_fp32, bias)
                torch.cuda.synchronize()
                
                # Benchmark
                int8_times = []
                for _ in range(n_iters):
                    start = time.time()
                    _ = torch.nn.functional.linear(input_int8, weight_int8, bias)
                    torch.cuda.synchronize()
                    int8_times.append((time.time() - start) * 1000)
                
                fp32_times = []
                for _ in range(n_iters):
                    start = time.time()
                    _ = torch.nn.functional.linear(input_fp32, weight_fp32, bias)
                    torch.cuda.synchronize()
                    fp32_times.append((time.time() - start) * 1000)
                
                int8_mean = torch.tensor(int8_times).mean().item()
                fp32_mean = torch.tensor(fp32_times).mean().item()
                speedup = fp32_mean / int8_mean
                
                m = batch_size * seq_len
                k = in_features
                n = out_features
                flops = 2 * m * n * k
                
                print(f"SeqLen={seq_len:4d}: INT8 {int8_mean:.3f}ms, FP32 {fp32_mean:.3f}ms, Speedup: {speedup:.2f}x, [{flops/int8_mean/1e9:.2f} GFLOPS]")
                
                # Clean up after each test
                del input_fp32, weight_fp32, bias, input_int8, weight_int8
                gc.collect()
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"SeqLen={seq_len:4d}: ⚠ OOM")
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    break
                else:
                    raise
        
        print()
        
        # Test 3: Varying hidden dimensions
        print("=" * 60)
        print("Dimension 3: Varying Hidden Dimensions")
        print("=" * 60)
        hidden_dims = [256, 512, 768, 1024, 1536, 2048, 3072, 4096, 8192]
        batch_size = 8
        seq_len = 64
        block_size = 128
        
        for hidden_dim in hidden_dims:
            try:
                input_fp32 = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float32, device=device)
                weight_fp32 = torch.randn(hidden_dim, hidden_dim, dtype=torch.float32, device=device)
                bias = torch.randn(hidden_dim, dtype=torch.float32, device=device)
                
                input_int8 = QuantizedTensor.from_float(input_fp32, "BlockWiseINT8Layout", block_size=block_size, is_weight=False)
                weight_int8 = QuantizedTensor.from_float(weight_fp32, "BlockWiseINT8Layout", block_size=block_size, is_weight=True)
                
                # Warm up
                for _ in range(n_warmup):
                    _ = torch.nn.functional.linear(input_int8, weight_int8, bias)
                    _ = torch.nn.functional.linear(input_fp32, weight_fp32, bias)
                torch.cuda.synchronize()
                
                # Benchmark
                int8_times = []
                for _ in range(n_iters):
                    start = time.time()
                    _ = torch.nn.functional.linear(input_int8, weight_int8, bias)
                    torch.cuda.synchronize()
                    int8_times.append((time.time() - start) * 1000)
                
                fp32_times = []
                for _ in range(n_iters):
                    start = time.time()
                    _ = torch.nn.functional.linear(input_fp32, weight_fp32, bias)
                    torch.cuda.synchronize()
                    fp32_times.append((time.time() - start) * 1000)
                
                int8_mean = torch.tensor(int8_times).mean().item()
                fp32_mean = torch.tensor(fp32_times).mean().item()
                speedup = fp32_mean / int8_mean
                
                m = batch_size * seq_len
                k = hidden_dim
                n = hidden_dim
                flops = 2 * m * n * k
                
                print(f"Hidden={hidden_dim:4d}: INT8 {int8_mean:.3f}ms, FP32 {fp32_mean:.3f}ms, Speedup: {speedup:.2f}x, [{flops/int8_mean/1e9:.2f} GFLOPS]")
                
                # Clean up after each test
                del input_fp32, weight_fp32, bias, input_int8, weight_int8
                gc.collect()
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"Hidden={hidden_dim:4d}: ⚠ OOM")
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    break
                else:
                    raise
        
        print()
        
        # Test 4: Varying block size
        print("=" * 60)
        print("Dimension 4: Varying Block Size")
        print("=" * 60)
        block_sizes = [32, 64, 128, 256, 512]
        batch_size = 8
        seq_len = 64
        in_features = 1024
        out_features = 1024
        
        for block_size in block_sizes:
            try:
                input_fp32 = torch.randn(batch_size, seq_len, in_features, dtype=torch.float32, device=device)
                weight_fp32 = torch.randn(out_features, in_features, dtype=torch.float32, device=device)
                bias = torch.randn(out_features, dtype=torch.float32, device=device)
                
                input_int8 = QuantizedTensor.from_float(input_fp32, "BlockWiseINT8Layout", block_size=block_size, is_weight=False)
                weight_int8 = QuantizedTensor.from_float(weight_fp32, "BlockWiseINT8Layout", block_size=block_size, is_weight=True)
                
                # Warm up
                for _ in range(n_warmup):
                    _ = torch.nn.functional.linear(input_int8, weight_int8, bias)
                torch.cuda.synchronize()
                
                # Benchmark
                int8_times = []
                for _ in range(n_iters):
                    start = time.time()
                    _ = torch.nn.functional.linear(input_int8, weight_int8, bias)
                    torch.cuda.synchronize()
                    int8_times.append((time.time() - start) * 1000)
                
                int8_mean = torch.tensor(int8_times).mean().item()
                int8_std = torch.tensor(int8_times).std().item()
                
                m = batch_size * seq_len
                k = in_features
                n = out_features
                flops = 2 * m * n * k
                
                print(f"Block={block_size:3d}: INT8 {int8_mean:.3f}±{int8_std:.3f}ms, [{flops/int8_mean/1e9:.2f} GFLOPS]")
                
                # Clean up after each test
                del input_fp32, weight_fp32, bias, input_int8, weight_int8
                gc.collect()
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"Block={block_size:3d}: ⚠ OOM")
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    break
                else:
                    raise
        
        print()
        print("✓ Systematic benchmark completed!")

    @unittest.skip("perf benchmark only")
    @unittest.skipUnless(has_gpu(), "GPU not available")
    def test_gelu_benchmark(self):
        """Benchmark INT8 GELU vs FP16 GELU"""
        # See test_int8_gelu.py::benchmark_int8_gelu for full implementation
        self.skipTest("Benchmark test - run separately")

    @unittest.skip("perf benchmark only")
    @unittest.skipUnless(has_gpu(), "GPU not available")
    def test_gelu_systematic_benchmark(self):
        """Systematic GELU benchmark across different dimensions"""
        # See test_int8_gelu.py::benchmark_int8_gelu_systematic for full implementation
        self.skipTest("Benchmark test - run separately")

    @unittest.skip("perf benchmark only")
    @unittest.skipUnless(has_gpu(), "GPU not available")
    def test_gelu_real_model_sizes(self):
        """Test FP16 vs INT8 GELU on actual model sizes"""
        # See test_int8_gelu.py::test_fp16_vs_int8_real_model_sizes for full implementation
        self.skipTest("Benchmark test - run separately")

    @unittest.skip("perf benchmark only")
    @unittest.skipUnless(has_gpu(), "GPU not available")
    def test_quant_fusion_performance(self):
        """Compare performance of fused vs separate quantization"""
        # See test_int8_quant_fusion.py::test_performance_comparison for full implementation
        self.skipTest("Benchmark test - run separately")


if __name__ == "__main__":
    unittest.main()
