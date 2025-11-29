import os
import sys
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file

# Add comfy to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

def has_gpu():
    return torch.cuda.is_available()

from comfy.cli_args import args
if not has_gpu():
    args.cpu = True

from comfy.quant_ops import QuantizedTensor, TensorCoreFP8Layout, AWQQuantLayout, SVDQuantLayout
from comfy.ops import mixed_precision_ops
from comfy.svdquant_converter import convert_svdquant_state_dict, convert_awq_state_dict


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


class TestAWQQuantLayout(unittest.TestCase):
    """Test the AWQQuantLayout implementation"""

    def test_awq_layout_creation(self):
        """Test creating an AWQ quantized tensor"""
        # AWQ uses pre-quantized weights loaded from checkpoints
        # Create dummy AWQ quantized weights
        out_features, in_features = 256, 128
        group_size = 64
        
        qweight = torch.randint(0, 255, (out_features // 4, in_features // 2), dtype=torch.int32)
        wscales = torch.randn(in_features // group_size, out_features, dtype=torch.bfloat16)
        wzeros = torch.randn(in_features // group_size, out_features, dtype=torch.bfloat16)
        
        layout_params = {
            'wscales': wscales,
            'wzeros': wzeros,
            'group_size': group_size,
            'orig_dtype': torch.bfloat16,
            'is_weight': True,
        }
        
        qt = QuantizedTensor(qweight, "AWQQuantLayout", layout_params)
        
        self.assertIsInstance(qt, QuantizedTensor)
        self.assertEqual(qt.shape, qweight.shape)
        self.assertEqual(qt.dtype, torch.int32)
        self.assertEqual(qt._layout_type, "AWQQuantLayout")
        self.assertEqual(qt._layout_params['group_size'], group_size)

    def test_awq_quantize_not_supported(self):
        """Test that online quantization raises NotImplementedError for AWQ"""
        # AWQ doesn't support online quantization - weights must be pre-quantized
        float_tensor = torch.randn(32, 64, dtype=torch.float32)
        
        with self.assertRaises(NotImplementedError):
            AWQQuantLayout.quantize(float_tensor, is_weight=True)

    def test_awq_get_plain_tensors(self):
        """Test extracting plain tensors from AWQ quantized tensor"""
        out_features, in_features = 256, 128
        group_size = 64
        
        qweight = torch.randint(0, 255, (out_features // 4, in_features // 2), dtype=torch.int32)
        wscales = torch.randn(in_features // group_size, out_features, dtype=torch.bfloat16)
        wzeros = torch.randn(in_features // group_size, out_features, dtype=torch.bfloat16)
        
        layout_params = {
            'wscales': wscales,
            'wzeros': wzeros,
            'group_size': group_size,
            'orig_dtype': torch.bfloat16,
            'is_weight': True,
        }
        
        qt = QuantizedTensor(qweight, "AWQQuantLayout", layout_params)
        plain_tensors = AWQQuantLayout.get_plain_tensors(qt)
        
        # Verify we can extract all necessary components
        self.assertIsInstance(plain_tensors, dict)
        self.assertIn('qweight', plain_tensors)
        self.assertIn('wscales', plain_tensors)
        self.assertIn('wzeros', plain_tensors)
        self.assertIn('group_size', plain_tensors)
        self.assertTrue(torch.equal(plain_tensors['qweight'], qweight))
        self.assertTrue(torch.equal(plain_tensors['wscales'], wscales))
        self.assertTrue(torch.equal(plain_tensors['wzeros'], wzeros))


class TestSVDQuantLayout(unittest.TestCase):
    """Test the SVDQuantLayout implementation"""

    def test_svdquant_layout_creation(self):
        """Test creating an SVDQuant quantized tensor"""
        # SVDQuant uses pre-quantized weights loaded from checkpoints
        out_features, in_features = 256, 128
        rank = 32
        group_size = 64
        precision = "int4"
        
        # Create dummy SVDQuant quantized weights (int8 range is -128 to 127)
        qweight = torch.randint(-128, 127, (out_features, in_features // 2), dtype=torch.int8)
        wscales = torch.randn(in_features // group_size, out_features, dtype=torch.bfloat16)
        smooth_factor = torch.randn(in_features, dtype=torch.bfloat16)
        smooth_factor_orig = torch.randn(in_features, dtype=torch.bfloat16)
        proj_down = torch.randn(in_features, rank, dtype=torch.bfloat16)
        proj_up = torch.randn(out_features, rank, dtype=torch.bfloat16)
        
        layout_params = {
            'wscales': wscales,
            'smooth_factor': smooth_factor,
            'smooth_factor_orig': smooth_factor_orig,
            'proj_down': proj_down,
            'proj_up': proj_up,
            'group_size': group_size,
            'precision': precision,
            'orig_dtype': torch.bfloat16,
            'is_weight': True,
            'act_unsigned': False,
            'wtscale': None,
            'wcscales': None,
        }
        
        qt = QuantizedTensor(qweight, "SVDQuantLayout", layout_params)
        
        self.assertIsInstance(qt, QuantizedTensor)
        self.assertEqual(qt.shape, qweight.shape)
        self.assertEqual(qt.dtype, torch.int8)
        self.assertEqual(qt._layout_type, "SVDQuantLayout")
        self.assertEqual(qt._layout_params['group_size'], group_size)
        self.assertEqual(qt._layout_params['precision'], precision)

    def test_svdquant_quantize_not_supported(self):
        """Test that online quantization raises NotImplementedError for SVDQuant"""
        # SVDQuant doesn't support online quantization - weights must be pre-quantized
        float_tensor = torch.randn(32, 64, dtype=torch.float32)
        
        with self.assertRaises(NotImplementedError):
            SVDQuantLayout.quantize(float_tensor, is_weight=True)

    def test_svdquant_dequantize_not_supported(self):
        """Test that weight dequantization raises NotImplementedError for SVDQuant"""
        # Full weight dequantization is not supported (complex operation)
        out_features, in_features = 256, 128
        rank = 32
        group_size = 64
        
        qweight = torch.randint(-128, 127, (out_features, in_features // 2), dtype=torch.int8)
        wscales = torch.randn(in_features // group_size, out_features, dtype=torch.bfloat16)
        smooth_factor = torch.randn(in_features, dtype=torch.bfloat16)
        proj_down = torch.randn(in_features, rank, dtype=torch.bfloat16)
        proj_up = torch.randn(out_features, rank, dtype=torch.bfloat16)
        
        with self.assertRaises(NotImplementedError):
            SVDQuantLayout.dequantize(
                qweight,
                is_weight=True,
                wscales=wscales,
                smooth_factor=smooth_factor,
                proj_down=proj_down,
                proj_up=proj_up,
                group_size=group_size,
                precision="int4",
                orig_dtype=torch.bfloat16
            )

    def test_svdquant_get_plain_tensors(self):
        """Test extracting plain tensors from SVDQuant quantized tensor"""
        out_features, in_features = 256, 128
        rank = 32
        group_size = 64
        
        qweight = torch.randint(-128, 127, (out_features, in_features // 2), dtype=torch.int8)
        wscales = torch.randn(in_features // group_size, out_features, dtype=torch.bfloat16)
        smooth_factor = torch.randn(in_features, dtype=torch.bfloat16)
        smooth_factor_orig = torch.randn(in_features, dtype=torch.bfloat16)
        proj_down = torch.randn(in_features, rank, dtype=torch.bfloat16)
        proj_up = torch.randn(out_features, rank, dtype=torch.bfloat16)
        
        layout_params = {
            'wscales': wscales,
            'smooth_factor': smooth_factor,
            'smooth_factor_orig': smooth_factor_orig,
            'proj_down': proj_down,
            'proj_up': proj_up,
            'group_size': group_size,
            'precision': "int4",
            'orig_dtype': torch.bfloat16,
            'is_weight': True,
            'act_unsigned': False,
            'wtscale': None,
            'wcscales': None,
        }
        
        qt = QuantizedTensor(qweight, "SVDQuantLayout", layout_params)
        plain_tensors = SVDQuantLayout.get_plain_tensors(qt)
        
        # Verify we can extract all necessary components
        self.assertIsInstance(plain_tensors, dict)
        self.assertIn('qweight', plain_tensors)
        self.assertIn('wscales', plain_tensors)
        self.assertIn('smooth_factor', plain_tensors)
        self.assertIn('proj_down', plain_tensors)
        self.assertIn('proj_up', plain_tensors)
        self.assertIn('group_size', plain_tensors)
        self.assertIn('precision', plain_tensors)
        self.assertTrue(torch.equal(plain_tensors['qweight'], qweight))
        self.assertTrue(torch.equal(plain_tensors['wscales'], wscales))
        self.assertTrue(torch.equal(plain_tensors['smooth_factor'], smooth_factor))
        self.assertTrue(torch.equal(plain_tensors['proj_down'], proj_down))
        self.assertTrue(torch.equal(plain_tensors['proj_up'], proj_up))


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


class TestAWQConversion(unittest.TestCase):
    """Test AWQ checkpoint conversion"""
    
    def test_awq_single_layer_conversion(self):
        """Test converting a single AWQ layer"""
        in_features, out_features = 128, 256
        group_size = 64
        
        # Create AWQ checkpoint format
        state_dict = {
            "layer.qweight": torch.randint(0, 255, (out_features // 4, in_features // 2), dtype=torch.int32),
            "layer.wscales": torch.randn(in_features // group_size, out_features, dtype=torch.bfloat16),
            "layer.wzeros": torch.randn(in_features // group_size, out_features, dtype=torch.bfloat16),
            "layer.bias": torch.randn(out_features, dtype=torch.bfloat16),
        }
        
        converted = convert_awq_state_dict(state_dict)
        
        # Check that qweight was renamed to weight
        self.assertIn("layer.weight", converted.tensors)
        self.assertNotIn("layer.qweight", converted.tensors)
        
        # Check other parameters preserved
        self.assertIn("layer.wscales", converted.tensors)
        self.assertIn("layer.wzeros", converted.tensors)
        self.assertIn("layer.bias", converted.tensors)
        
        # Check quantization metadata
        self.assertIn("layer", converted.quant_layers)
        self.assertEqual(converted.quant_layers["layer"], "awq_int4")
    
    def test_awq_tensor_shapes(self):
        """Test that converted AWQ tensors have correct shapes"""
        in_features, out_features = 3072, 18432
        group_size = 64
        
        state_dict = {
            "layer.qweight": torch.randint(0, 255, (out_features // 4, in_features // 2), dtype=torch.int32),
            "layer.wscales": torch.randn(in_features // group_size, out_features, dtype=torch.bfloat16),
            "layer.wzeros": torch.randn(in_features // group_size, out_features, dtype=torch.bfloat16),
        }
        
        converted = convert_awq_state_dict(state_dict)
        
        # Check qweight shape (packed 4-bit)
        qweight = converted.tensors["layer.weight"]
        self.assertEqual(qweight.shape, (out_features // 4, in_features // 2))
        self.assertEqual(qweight.dtype, torch.int32)
        
        # Check wscales shape
        wscales = converted.tensors["layer.wscales"]
        self.assertEqual(wscales.shape, (in_features // group_size, out_features))
        self.assertEqual(wscales.dtype, torch.bfloat16)
        
        # Check wzeros shape
        wzeros = converted.tensors["layer.wzeros"]
        self.assertEqual(wzeros.shape, (in_features // group_size, out_features))
        self.assertEqual(wzeros.dtype, torch.bfloat16)


class TestAWQLinearOperation(unittest.TestCase):
    """Test AWQ linear operations with actual nunchaku kernels"""
    
    @unittest.skipUnless(has_gpu(), "GPU required for AWQ operations")
    def test_awq_linear_basic(self):
        """Test basic AWQ linear operation by calling kernel directly"""
        try:
            from nunchaku.ops.gemv import awq_gemv_w4a16_cuda
        except ImportError:
            self.skipTest("nunchaku package not available")
        
        device = torch.device("cuda")
        in_features, out_features = 128, 256
        group_size = 64
        batch_size = 4
        
        # Create AWQ quantized weight tensors
        qweight = torch.randint(0, 255, (out_features // 4, in_features // 2), dtype=torch.int32, device=device)
        wscales = torch.randn(in_features // group_size, out_features, dtype=torch.bfloat16, device=device)
        wzeros = torch.randn(in_features // group_size, out_features, dtype=torch.bfloat16, device=device)
        bias = torch.randn(out_features, dtype=torch.bfloat16, device=device)
        
        # Create layout params
        layout_params = {
            'wscales': wscales,
            'wzeros': wzeros,
            'group_size': group_size,
            'orig_dtype': torch.bfloat16,
            'is_weight': True,
        }
        
        weight = QuantizedTensor(qweight, "AWQQuantLayout", layout_params)
        
        # Check that weight is a QuantizedTensor
        self.assertIsInstance(weight, QuantizedTensor)
        self.assertEqual(weight._layout_type, "AWQQuantLayout")
        
        # Create input
        x = torch.randn(batch_size, in_features, dtype=torch.bfloat16, device=device)
        
        # Call AWQ linear handler directly
        from comfy.quant_ops import awq_linear
        output = awq_linear(torch.ops.aten.linear.default, (x, weight, bias), {})
        
        # Check output shape and dtype
        self.assertEqual(output.shape, (batch_size, out_features))
        self.assertEqual(output.dtype, torch.bfloat16)
    
    @unittest.skipUnless(has_gpu(), "GPU required for AWQ operations")
    def test_awq_linear_2d_input(self):
        """Test AWQ linear with 2D input (batch, features) by calling kernel directly"""
        try:
            from nunchaku.ops.gemv import awq_gemv_w4a16_cuda
        except ImportError:
            self.skipTest("nunchaku package not available")
        
        device = torch.device("cuda")
        in_features, out_features = 128, 256
        group_size = 64
        batch_size = 4
        
        # Create AWQ quantized weight tensors
        qweight = torch.randint(0, 255, (out_features // 4, in_features // 2), dtype=torch.int32, device=device)
        wscales = torch.randn(in_features // group_size, out_features, dtype=torch.bfloat16, device=device)
        wzeros = torch.randn(in_features // group_size, out_features, dtype=torch.bfloat16, device=device)
        
        # Create layout params
        layout_params = {
            'wscales': wscales,
            'wzeros': wzeros,
            'group_size': group_size,
            'orig_dtype': torch.bfloat16,
            'is_weight': True,
        }
        
        weight = QuantizedTensor(qweight, "AWQQuantLayout", layout_params)
        
        # Check that weight is a QuantizedTensor
        self.assertIsInstance(weight, QuantizedTensor)
        self.assertEqual(weight._layout_type, "AWQQuantLayout")
        
        # Create 2D input
        x = torch.randn(batch_size, in_features, dtype=torch.bfloat16, device=device)
        
        # Call AWQ linear handler directly
        from comfy.quant_ops import awq_linear
        output = awq_linear(torch.ops.aten.linear.default, (x, weight, None), {})
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, out_features))
        self.assertEqual(output.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()