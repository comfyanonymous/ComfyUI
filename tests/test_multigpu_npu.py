import threading
from types import SimpleNamespace

import pytest

import comfy.model_management as model_management
from comfy.multigpu import MultiGPUThreadPool


class FakeAccelerator:
    def __init__(self, current=0):
        self.current = current
        self.set_calls = []

    def current_device(self):
        return self.current

    def set_device(self, device):
        self.set_calls.append(device)
        self.current = device.index if hasattr(device, "index") else device


def fake_device(device_type, index=None):
    return SimpleNamespace(type=device_type, index=index)


@pytest.mark.parametrize("device_type", ["cuda", "npu"])
def test_device_context_switches_and_restores(monkeypatch, device_type):
    accelerator = FakeAccelerator(current=0)
    monkeypatch.setattr(model_management.torch, device_type, accelerator, raising=False)
    device = fake_device(device_type, 2)

    with model_management.cuda_device_context(device):
        assert accelerator.current_device() == 2

    assert accelerator.current_device() == 0
    assert accelerator.set_calls == [device, 0]


@pytest.mark.parametrize("device_type", ["cuda", "npu"])
def test_device_context_is_noop_for_current_device(monkeypatch, device_type):
    accelerator = FakeAccelerator(current=1)
    monkeypatch.setattr(model_management.torch, device_type, accelerator, raising=False)

    with model_management.cuda_device_context(fake_device(device_type, 1)):
        assert accelerator.current_device() == 1

    assert accelerator.set_calls == []


def test_set_torch_device_supports_npu(monkeypatch):
    accelerator = FakeAccelerator()
    monkeypatch.setattr(model_management.torch, "npu", accelerator, raising=False)
    device = fake_device("npu", 1)

    model_management.set_torch_device(device)

    assert accelerator.set_calls == [device]


def test_multigpu_thread_pool_binds_each_worker(monkeypatch):
    worker_state = threading.local()
    devices = ["npu:0", "npu:1", "npu:2"]
    monkeypatch.setattr(
        model_management,
        "set_torch_device",
        lambda device: setattr(worker_state, "device", device),
    )
    pool = MultiGPUThreadPool(devices)
    try:
        for device in devices:
            pool.submit(device, lambda: worker_state.device)
        for device in devices:
            result, error = pool.get_result(device)
            assert error is None
            assert result == device
    finally:
        pool.shutdown()


def test_multigpu_thread_pool_reports_error_and_keeps_worker_alive(monkeypatch):
    monkeypatch.setattr(model_management, "set_torch_device", lambda device: None)
    pool = MultiGPUThreadPool(["npu:0"])
    try:
        def fail():
            raise RuntimeError("expected failure")

        pool.submit("npu:0", fail)
        result, error = pool.get_result("npu:0")
        assert result is None
        assert isinstance(error, RuntimeError)
        assert str(error) == "expected failure"

        pool.submit("npu:0", lambda: "recovered")
        result, error = pool.get_result("npu:0")
        assert error is None
        assert result == "recovered"
    finally:
        pool.shutdown()
