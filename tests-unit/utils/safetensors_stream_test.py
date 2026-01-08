import os

import pytest
import importlib
import importlib.util

torch = pytest.importorskip("torch")


def _write_safetensors(tmp_path, tensors):
    import safetensors.torch
    path = os.path.join(tmp_path, "test.safetensors")
    safetensors.torch.save_file(tensors, path)
    return path


def test_stream_state_dict_meta_is_lazy(tmp_path, monkeypatch):
    if torch is None:
        pytest.skip("torch not installed")
    import comfy.utils
    path = _write_safetensors(tmp_path, {"a": torch.zeros((2, 3), dtype=torch.float32), "b": torch.ones((4,), dtype=torch.float16)})
    sd = comfy.utils.load_torch_file(path, safe_load=True)
    calls = []

    original = sd._file.read_tensor

    def wrapped(meta, device, dtype, allow_gds, pin_if_cpu):
        calls.append(meta)
        return original(meta, device, dtype, allow_gds, pin_if_cpu)

    monkeypatch.setattr(sd._file, "read_tensor", wrapped)
    meta = sd.meta("a")
    assert meta.shape == (2, 3)
    assert meta.dtype == torch.float32
    assert meta.numel == 6
    assert calls == []


def test_stream_state_dict_getitem_loads_single_tensor(tmp_path, monkeypatch):
    if torch is None:
        pytest.skip("torch not installed")
    import comfy.utils
    path = _write_safetensors(tmp_path, {"a": torch.zeros((2, 3), dtype=torch.float32), "b": torch.ones((4,), dtype=torch.float16)})
    sd = comfy.utils.load_torch_file(path, safe_load=True)
    calls = []

    original = sd._file.read_tensor

    def wrapped(meta, device, dtype, allow_gds, pin_if_cpu):
        calls.append(meta)
        return original(meta, device, dtype, allow_gds, pin_if_cpu)

    monkeypatch.setattr(sd._file, "read_tensor", wrapped)
    _ = sd["a"]
    assert len(calls) == 1
    assert calls[0].shape == (2, 3)


def test_stream_views_do_not_materialize(tmp_path, monkeypatch):
    if torch is None:
        pytest.skip("torch not installed")
    import comfy.utils
    path = _write_safetensors(tmp_path, {"prefix.a": torch.zeros((2, 3)), "other": torch.ones((4,))})
    sd = comfy.utils.load_torch_file(path, safe_load=True)
    calls = []

    original = sd._file.read_tensor

    def wrapped(meta, device, dtype, allow_gds, pin_if_cpu):
        calls.append(meta)
        return original(meta, device, dtype, allow_gds, pin_if_cpu)

    monkeypatch.setattr(sd._file, "read_tensor", wrapped)
    view = comfy.utils.state_dict_prefix_replace(sd, {"prefix.": ""}, filter_keys=True)
    _ = list(view.keys())
    assert calls == []


def test_stream_load_rss_small(tmp_path):
    if torch is None:
        pytest.skip("torch not installed")
    import comfy.utils
    psutil = pytest.importorskip("psutil")
    process = psutil.Process()
    size_elems = 4_000_000  # ~16MB float32
    tensor = torch.zeros((size_elems,), dtype=torch.float32)
    path = _write_safetensors(tmp_path, {"big": tensor})
    rss_before = process.memory_info().rss
    sd = comfy.utils.load_torch_file(path, safe_load=True)
    rss_after = process.memory_info().rss
    expected_size = tensor.numel() * tensor.element_size()
    assert (rss_after - rss_before) < expected_size
    _ = sd.meta("big")


def test_gds_path_errors_without_support(tmp_path, monkeypatch):
    if torch is None:
        pytest.skip("torch not installed")
    import comfy.utils
    path = _write_safetensors(tmp_path, {"a": torch.zeros((2, 3), dtype=torch.float32)})
    sd = comfy.utils.load_torch_file(path, safe_load=True)
    device = torch.device("cuda")

    if importlib.util.find_spec("fastsafetensors") is None:
        fst = None
    else:
        fst = importlib.import_module("fastsafetensors")

    gds_available = False
    if fst is not None and torch.cuda.is_available():
        gds_supported = fst.cpp.is_gds_supported(torch.cuda.current_device())
        gds_available = bool(fst.cpp.is_cufile_found()) and gds_supported == 1

    if not gds_available:
        with pytest.raises(RuntimeError, match="GPUDirect requested"):
            sd.get_tensor("a", device=device, allow_gds=True)
    else:
        def fail_nogds(*args, **kwargs):
            raise AssertionError("nogds path used during GDS request")

        monkeypatch.setattr(sd._file, "_read_tensor_nogds", fail_nogds)
        t = sd.get_tensor("a", device=device, allow_gds=True)
        assert t.device.type == "cuda"


def test_stream_load_without_disk_cache_keeps_cpu_weights(tmp_path):
    if torch is None:
        pytest.skip("torch not installed")
    import comfy.utils
    import comfy.disk_weights

    prev_cache = comfy.disk_weights.CACHE.max_bytes
    prev_gds = comfy.disk_weights.ALLOW_GDS
    prev_pin = comfy.disk_weights.PIN_IF_CPU
    prev_enabled = comfy.disk_weights.DISK_WEIGHTS_ENABLED
    comfy.disk_weights.configure(0, allow_gds=False, pin_if_cpu=False, enabled=True)

    try:
        path = _write_safetensors(tmp_path, {"weight": torch.zeros((4, 4), dtype=torch.float32), "bias": torch.zeros((4,), dtype=torch.float32)})
        sd = comfy.utils.load_torch_file(path, safe_load=True)
        model = torch.nn.Linear(4, 4, bias=True)
        comfy.utils.load_state_dict(model, sd, strict=False)
        assert model.weight.device.type == "cpu"
        assert model.weight.device.type != "meta"
    finally:
        comfy.disk_weights.configure(prev_cache, allow_gds=prev_gds, pin_if_cpu=prev_pin, enabled=prev_enabled)


def test_lazy_disk_weights_loads_on_demand(tmp_path, monkeypatch):
    if importlib.util.find_spec("fastsafetensors") is None:
        pytest.skip("fastsafetensors not installed")
    import comfy.utils
    import comfy.disk_weights

    prev_cache = comfy.disk_weights.CACHE.max_bytes
    prev_gds = comfy.disk_weights.ALLOW_GDS
    prev_pin = comfy.disk_weights.PIN_IF_CPU
    prev_enabled = comfy.disk_weights.DISK_WEIGHTS_ENABLED
    comfy.disk_weights.configure(0, allow_gds=False, pin_if_cpu=False, enabled=True)

    try:
        path = _write_safetensors(tmp_path, {"weight": torch.zeros((4, 4), dtype=torch.float32), "bias": torch.zeros((4,), dtype=torch.float32)})
        sd = comfy.utils.load_torch_file(path, safe_load=True)
        model = torch.nn.Linear(4, 4, bias=True)
        calls = []

        original = sd._file.read_tensor

        def wrapped(meta, device, dtype, allow_gds, pin_if_cpu):
            calls.append(meta)
            return original(meta, device, dtype, allow_gds, pin_if_cpu)

        monkeypatch.setattr(sd._file, "read_tensor", wrapped)
        comfy.utils.load_state_dict(model, sd, strict=True)
        assert model.weight.device.type == "meta"
        assert calls == []

        comfy.disk_weights.ensure_module_materialized(model, torch.device("cpu"))
        assert model.weight.device.type == "cpu"
        assert len(calls) == 2
    finally:
        comfy.disk_weights.configure(prev_cache, allow_gds=prev_gds, pin_if_cpu=prev_pin, enabled=prev_enabled)


def test_lazy_disk_weights_respects_dtype_override(tmp_path):
    if importlib.util.find_spec("fastsafetensors") is None:
        pytest.skip("fastsafetensors not installed")
    import comfy.utils
    import comfy.disk_weights

    prev_cache = comfy.disk_weights.CACHE.max_bytes
    prev_gds = comfy.disk_weights.ALLOW_GDS
    prev_pin = comfy.disk_weights.PIN_IF_CPU
    prev_enabled = comfy.disk_weights.DISK_WEIGHTS_ENABLED
    comfy.disk_weights.configure(0, allow_gds=False, pin_if_cpu=False, enabled=True)

    try:
        path = _write_safetensors(tmp_path, {"weight": torch.zeros((4, 4), dtype=torch.bfloat16), "bias": torch.zeros((4,), dtype=torch.bfloat16)})
        sd = comfy.utils.load_torch_file(path, safe_load=True)
        model = torch.nn.Linear(4, 4, bias=True)
        comfy.utils.load_state_dict(model, sd, strict=True)
        assert model.weight.device.type == "meta"

        comfy.disk_weights.ensure_module_materialized(model, torch.device("cpu"))
        assert model.weight.dtype == torch.bfloat16

        comfy.disk_weights.ensure_module_materialized(model, torch.device("cpu"), dtype_override=torch.float16)
        assert model.weight.dtype == torch.float16
    finally:
        comfy.disk_weights.configure(prev_cache, allow_gds=prev_gds, pin_if_cpu=prev_pin, enabled=prev_enabled)
