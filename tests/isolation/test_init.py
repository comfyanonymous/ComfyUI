"""Unit tests for PyIsolate isolation system initialization."""

import importlib
import sys

from tests.isolation.singleton_boundary_helpers import (
    FakeSingletonRPC,
    reset_forbidden_singleton_modules,
)


def test_log_prefix():
    """Verify LOG_PREFIX constant is correctly defined."""
    from comfy.isolation import LOG_PREFIX
    assert LOG_PREFIX == "]["
    assert isinstance(LOG_PREFIX, str)


def test_module_initialization():
    """Verify module initializes without errors."""
    isolation_pkg = importlib.import_module("comfy.isolation")
    assert hasattr(isolation_pkg, "LOG_PREFIX")
    assert hasattr(isolation_pkg, "initialize_proxies")


class TestInitializeProxies:
    def test_initialize_proxies_runs_without_error(self):
        from comfy.isolation import initialize_proxies
        initialize_proxies()

    def test_initialize_proxies_registers_folder_paths_proxy(self):
        from comfy.isolation import initialize_proxies
        from comfy.isolation.proxies.folder_paths_proxy import FolderPathsProxy
        initialize_proxies()
        proxy = FolderPathsProxy()
        assert proxy is not None
        assert hasattr(proxy, "get_temp_directory")

    def test_initialize_proxies_registers_model_management_proxy(self):
        from comfy.isolation import initialize_proxies
        from comfy.isolation.proxies.model_management_proxy import ModelManagementProxy
        initialize_proxies()
        proxy = ModelManagementProxy()
        assert proxy is not None
        assert hasattr(proxy, "get_torch_device")

    def test_initialize_proxies_can_be_called_multiple_times(self):
        from comfy.isolation import initialize_proxies
        initialize_proxies()
        initialize_proxies()
        initialize_proxies()

    def test_dev_proxies_accessible_when_dev_mode(self, monkeypatch):
        """Verify dev mode does not break core proxy initialization."""
        monkeypatch.setenv("PYISOLATE_DEV", "1")
        from comfy.isolation import initialize_proxies
        from comfy.isolation.proxies.folder_paths_proxy import FolderPathsProxy
        from comfy.isolation.proxies.utils_proxy import UtilsProxy
        initialize_proxies()
        folder_proxy = FolderPathsProxy()
        utils_proxy = UtilsProxy()
        assert folder_proxy is not None
        assert utils_proxy is not None

    def test_sealed_child_safe_initialize_proxies_avoids_real_utils_import(self, monkeypatch):
        monkeypatch.setenv("PYISOLATE_CHILD", "1")
        monkeypatch.setenv("PYISOLATE_IMPORT_TORCH", "0")
        reset_forbidden_singleton_modules()

        from pyisolate._internal import rpc_protocol
        from comfy.isolation import initialize_proxies

        fake_rpc = FakeSingletonRPC()
        monkeypatch.setattr(rpc_protocol, "get_child_rpc_instance", lambda: fake_rpc)

        initialize_proxies()

        assert "comfy.utils" not in sys.modules
        assert "folder_paths" not in sys.modules
        assert "comfy_execution.progress" not in sys.modules
