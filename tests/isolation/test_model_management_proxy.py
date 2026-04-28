"""Unit tests for ModelManagementProxy."""

import pytest
import torch

from comfy.isolation.proxies.model_management_proxy import ModelManagementProxy


class TestModelManagementProxy:
    """Test ModelManagementProxy methods."""

    @pytest.fixture
    def proxy(self):
        """Create a ModelManagementProxy instance for testing."""
        return ModelManagementProxy()

    def test_get_torch_device_returns_device(self, proxy):
        """Verify get_torch_device returns a torch.device object."""
        result = proxy.get_torch_device()
        assert isinstance(result, torch.device), f"Expected torch.device, got {type(result)}"

    def test_get_torch_device_is_valid(self, proxy):
        """Verify get_torch_device returns a valid device (cpu or cuda)."""
        result = proxy.get_torch_device()
        assert result.type in ("cpu", "cuda"), f"Unexpected device type: {result.type}"

    def test_get_torch_device_name_returns_string(self, proxy):
        """Verify get_torch_device_name returns a non-empty string."""
        device = proxy.get_torch_device()
        result = proxy.get_torch_device_name(device)
        assert isinstance(result, str), f"Expected str, got {type(result)}"
        assert len(result) > 0, "Device name is empty"

    def test_get_torch_device_name_with_cpu(self, proxy):
        """Verify get_torch_device_name works with CPU device."""
        cpu_device = torch.device("cpu")
        result = proxy.get_torch_device_name(cpu_device)
        assert isinstance(result, str), f"Expected str, got {type(result)}"
        assert "cpu" in result.lower(), f"Expected 'cpu' in device name, got: {result}"

    def test_get_torch_device_name_with_cuda_if_available(self, proxy):
        """Verify get_torch_device_name works with CUDA device if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        cuda_device = torch.device("cuda:0")
        result = proxy.get_torch_device_name(cuda_device)
        assert isinstance(result, str), f"Expected str, got {type(result)}"
        # Should contain device identifier
        assert len(result) > 0, "CUDA device name is empty"
