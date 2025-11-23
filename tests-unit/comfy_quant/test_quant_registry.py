import unittest
import torch
import sys
import os

# Add comfy to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

def has_gpu():
    return torch.cuda.is_available()

from comfy.cli_args import args
if not has_gpu():
    args.cpu = True

from comfy.quant_ops import QuantizedTensor, TensorCoreFP8Layout


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


if __name__ == "__main__":
    unittest.main()
