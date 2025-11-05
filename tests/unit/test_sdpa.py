import pytest
import torch
import importlib
import sys
from unittest.mock import patch, MagicMock

# For version comparison
from packaging.version import parse as parse_version

# Module under test
import comfy.ops

TORCH_VERSION = parse_version(torch.__version__.split('+')[0])
CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.fixture(autouse=True)
def cleanup_module():
    """Reloads comfy.ops after each test to reset its state."""
    yield
    importlib.reload(comfy.ops)


def test_sdpa_no_cuda():
    """
    Tests that scaled_dot_product_attention falls back to the basic implementation
    when CUDA is not available.
    """
    with patch('torch.cuda.is_available', return_value=False):
        # Reload the module to apply the mock
        importlib.reload(comfy.ops)

        assert comfy.ops.scaled_dot_product_attention is comfy.ops._scaled_dot_product_attention

        # Test functionality
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 8, 16)
        v = torch.randn(2, 4, 8, 16)
        output = comfy.ops.scaled_dot_product_attention(q, k, v)
        assert output.shape == q.shape


def test_sdpa_old_torch_with_cuda():
    """
    Tests that scaled_dot_product_attention falls back and warns
    on older torch versions that have CUDA but lack 'set_priority' in sdpa_kernel.
    """
    # Mock signature object without 'set_priority'
    mock_signature = MagicMock()
    mock_signature.parameters = {}

    # Mock the logger to capture warnings
    mock_logger = MagicMock()

    # Mock the attention module to prevent import errors on non-CUDA builds
    mock_attention_module = MagicMock()
    mock_attention_module.sdpa_kernel = MagicMock()
    mock_attention_module.SDPBackend = MagicMock()

    with patch('torch.cuda.is_available', return_value=True), \
            patch('inspect.signature', return_value=mock_signature), \
            patch('logging.getLogger', return_value=mock_logger), \
            patch.dict('sys.modules', {'torch.nn.attention': mock_attention_module}):
        importlib.reload(comfy.ops)

        assert comfy.ops.scaled_dot_product_attention is comfy.ops._scaled_dot_product_attention
        mock_logger.warning.assert_called_once_with("Torch version too old to set sdpa backend priority, even though you are using CUDA")

        # Test functionality
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 8, 16)
        v = torch.randn(2, 4, 8, 16)
        output = comfy.ops.scaled_dot_product_attention(q, k, v)
        assert output.shape == q.shape


def test_sdpa_import_exception():
    """
    Tests that scaled_dot_product_attention falls back if an exception occurs
    during the SDPA setup.
    """
    mock_logger = MagicMock()
    with patch('torch.cuda.is_available', return_value=True), \
            patch('inspect.signature', side_effect=Exception("Test Exception")), \
            patch('logging.getLogger', return_value=mock_logger):
        # Mock the attention module to prevent import errors on non-CUDA builds
        mock_attention_module = MagicMock()
        mock_attention_module.sdpa_kernel = MagicMock()
        mock_attention_module.SDPBackend = MagicMock()
        with patch.dict('sys.modules', {'torch.nn.attention': mock_attention_module}):
            importlib.reload(comfy.ops)

            assert comfy.ops.scaled_dot_product_attention is comfy.ops._scaled_dot_product_attention
            mock_logger.debug.assert_called()
            # Check that the log message contains the exception info
            assert "Could not set sdpa backend priority." in mock_logger.debug.call_args[0][0]

            # Test functionality
            q = torch.randn(2, 4, 8, 16)
            k = torch.randn(2, 4, 8, 16)
            v = torch.randn(2, 4, 8, 16)
            output = comfy.ops.scaled_dot_product_attention(q, k, v)
            assert output.shape == q.shape


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
@pytest.mark.skipif(TORCH_VERSION < parse_version("2.6.0"), reason="Requires torch version 2.6.0 or greater")
def test_sdpa_with_cuda_and_priority():
    """
    Tests that the prioritized SDPA implementation is used when CUDA is available
    and the torch version is new enough.
    This is a real test and does not use mocks.
    """
    # Reload to ensure the correct version is picked up based on the actual environment
    importlib.reload(comfy.ops)

    # Check that the correct function is assigned
    assert comfy.ops.scaled_dot_product_attention is not comfy.ops._scaled_dot_product_attention
    assert comfy.ops.scaled_dot_product_attention.__name__ == "_scaled_dot_product_attention_sdpa"

    # Create tensors on CUDA device
    device = torch.device("cuda")
    q = torch.randn(2, 4, 8, 16, device=device, dtype=torch.float16)
    k = torch.randn(2, 4, 8, 16, device=device, dtype=torch.float16)
    v = torch.randn(2, 4, 8, 16, device=device, dtype=torch.float16)

    # Execute the function
    output = comfy.ops.scaled_dot_product_attention(q, k, v)

    # Assertions
    assert output.shape == q.shape
    assert output.device.type == device.type
    assert output.dtype == torch.float16
