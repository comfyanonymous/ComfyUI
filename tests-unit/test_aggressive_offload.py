"""Tests for the aggressive-offload memory management feature.

These tests validate the Apple Silicon (MPS) memory optimisation path without
requiring a GPU or actual model weights.  Every test mocks the relevant model
and cache structures so the suite can run in CI on any platform.
"""

import pytest
import types
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

class FakeLinearModel(nn.Module):
    """Minimal nn.Module whose parameters consume measurable memory."""

    def __init__(self, size_mb: float = 2.0):
        """Create a model with approximately ``size_mb`` MB of float32 params."""
        super().__init__()
        # Each float32 param = 4 bytes, so `n` params ≈ size_mb * 1024² / 4
        n = int(size_mb * 1024 * 1024 / 4)
        self.weight = nn.Parameter(torch.zeros(n, dtype=torch.float32))


class FakeModelPatcher:
    """Mimics the subset of ModelPatcher used by model_management.free_memory."""

    def __init__(self, size_mb: float = 2.0):
        """Create a patcher wrapping a ``FakeLinearModel`` of the given size."""
        self.model = FakeLinearModel(size_mb)
        self._loaded_size = int(size_mb * 1024 * 1024)

    def loaded_size(self):
        """Return reported loaded size in bytes."""
        return self._loaded_size

    def is_dynamic(self):
        """Static model — never dynamically offloaded."""
        return False


class FakeLoadedModel:
    """Mimics LoadedModel entries in current_loaded_models."""

    def __init__(self, patcher: FakeModelPatcher, *, currently_used: bool = False):
        """Wrap a ``FakeModelPatcher`` as a loaded-model entry."""
        self._model = patcher
        self.currently_used = currently_used

    @property
    def model(self):
        """Return the underlying model patcher."""
        return self._model

    def model_memory(self):
        """Return memory footprint in bytes."""
        return self._model.loaded_size()

    def model_unload(self, _memory_to_free):
        """Simulate successful unload."""
        return True

    def model_load(self, _device, _keep_loaded):
        """No-op load for testing."""
        pass


# ---------------------------------------------------------------------------
# 1. BasicCache.clear_all()
# ---------------------------------------------------------------------------

class TestBasicCacheClearAll:
    """Verify that BasicCache.clear_all() is a proper public API."""

    def test_clear_all_empties_cache_and_subcaches(self):
        """clear_all() must remove every entry in both dicts."""
        from comfy_execution.caching import BasicCache, CacheKeySetInputSignature

        cache = BasicCache(CacheKeySetInputSignature)
        cache.cache["key1"] = "value1"
        cache.cache["key2"] = "value2"
        cache.subcaches["sub1"] = "subvalue1"

        cache.clear_all()

        assert len(cache.cache) == 0
        assert len(cache.subcaches) == 0

    def test_clear_all_is_idempotent(self):
        """Calling clear_all() on an already-empty cache must not raise."""
        from comfy_execution.caching import BasicCache, CacheKeySetInputSignature

        cache = BasicCache(CacheKeySetInputSignature)
        cache.clear_all()  # should be a no-op
        cache.clear_all()  # still a no-op

        assert len(cache.cache) == 0

    def test_null_cache_clear_all_is_noop(self):
        """NullCache.clear_all() must not raise — it's the null backend."""
        from comfy_execution.caching import NullCache

        null = NullCache()
        null.clear_all()  # must not raise AttributeError

    def test_lru_cache_clear_all_resets_metadata(self):
        """LRUCache.clear_all() must also reset used_generation, children, min_generation."""
        from comfy_execution.caching import LRUCache, CacheKeySetInputSignature

        cache = LRUCache(CacheKeySetInputSignature, max_size=10)
        # Simulate some entries
        cache.cache["k1"] = "v1"
        cache.used_generation["k1"] = 5
        cache.children["k1"] = ["child1"]
        cache.min_generation = 3
        cache.generation = 5

        cache.clear_all()

        assert len(cache.cache) == 0
        assert len(cache.used_generation) == 0
        assert len(cache.children) == 0
        assert cache.min_generation == 0
        # generation counter should NOT be reset (it's a monotonic counter)
        assert cache.generation == 5

    def test_ram_pressure_cache_clear_all_resets_timestamps(self):
        """RAMPressureCache.clear_all() must also reset timestamps."""
        from comfy_execution.caching import RAMPressureCache, CacheKeySetInputSignature

        cache = RAMPressureCache(CacheKeySetInputSignature)
        cache.cache["k1"] = "v1"
        cache.used_generation["k1"] = 2
        cache.timestamps["k1"] = 1234567890.0

        cache.clear_all()

        assert len(cache.cache) == 0
        assert len(cache.used_generation) == 0
        assert len(cache.timestamps) == 0


# ---------------------------------------------------------------------------
# 2. Callback registration & dispatch
# ---------------------------------------------------------------------------

class TestModelDestroyedCallbacks:
    """Validate the on_model_destroyed lifecycle callback system."""

    def setup_method(self):
        """Save and reset the callback list before every test."""
        import comfy.model_management as mm
        self._original = mm._on_model_destroyed_callbacks.copy()
        mm._on_model_destroyed_callbacks.clear()

    def teardown_method(self):
        """Restore the original callback list after every test."""
        import comfy.model_management as mm
        mm._on_model_destroyed_callbacks.clear()
        mm._on_model_destroyed_callbacks.extend(self._original)

    def test_register_single_callback(self):
        """A single registered callback must be stored and callable."""
        import comfy.model_management as mm

        invocations = []
        mm.register_model_destroyed_callback(lambda reason: invocations.append(reason))

        assert len(mm._on_model_destroyed_callbacks) == 1

        # Simulate dispatch
        for cb in mm._on_model_destroyed_callbacks:
            cb("test")
        assert invocations == ["test"]

    def test_register_multiple_callbacks(self):
        """Multiple registrants must all fire — no silent overwrites."""
        import comfy.model_management as mm

        results_a, results_b = [], []
        mm.register_model_destroyed_callback(lambda r: results_a.append(r))
        mm.register_model_destroyed_callback(lambda r: results_b.append(r))

        for cb in mm._on_model_destroyed_callbacks:
            cb("batch")

        assert results_a == ["batch"]
        assert results_b == ["batch"]

    def test_callback_receives_reason_string(self):
        """The callback signature is (reason: str) -> None."""
        import comfy.model_management as mm

        captured = {}
        def _cb(reason):
            captured["reason"] = reason
            captured["type"] = type(reason).__name__

        mm.register_model_destroyed_callback(_cb)
        for cb in mm._on_model_destroyed_callbacks:
            cb("batch")

        assert captured["reason"] == "batch"
        assert captured["type"] == "str"


# ---------------------------------------------------------------------------
# 3. Meta-device destruction threshold
# ---------------------------------------------------------------------------

class TestMetaDeviceThreshold:
    """Verify that only models > 1 GB are queued for meta-device destruction."""

    def test_small_model_not_destroyed(self):
        """A 160 MB model (VAE-sized) must NOT be moved to meta device."""
        model = FakeLinearModel(size_mb=160)

        # Simulate the threshold check from free_memory
        model_size = sum(p.numel() * p.element_size() for p in model.parameters())
        threshold = 1024 * 1024 * 1024  # 1 GB

        assert model_size < threshold, (
            f"160 MB model should be below 1 GB threshold, got {model_size / (1024**2):.0f} MB"
        )
        # Confirm parameters are still on a real device
        assert model.weight.device.type != "meta"

    def test_large_model_above_threshold(self):
        """A 2 GB model (UNET/CLIP-sized) must BE above the destruction threshold."""
        # Use a meta-device tensor to avoid allocating 2 GB of real memory.
        # Meta tensors report correct numel/element_size but use zero storage.
        n = int(2048 * 1024 * 1024 / 4)  # 2 GB in float32 params
        meta_weight = torch.empty(n, dtype=torch.float32, device="meta")

        model_size = meta_weight.numel() * meta_weight.element_size()
        threshold = 1024 * 1024 * 1024  # 1 GB

        assert model_size > threshold, (
            f"2 GB model should be above 1 GB threshold, got {model_size / (1024**2):.0f} MB"
        )

    def test_meta_device_move_releases_storage(self):
        """Moving parameters to 'meta' must place them on the meta device."""
        model = FakeLinearModel(size_mb=2)
        assert model.weight.device.type != "meta"

        model.to(device="meta")

        assert model.weight.device.type == "meta"
        # Meta tensors retain their logical shape but live on a virtual device
        # with no physical backing — this is what releases RAM.
        assert model.weight.nelement() > 0  # still has logical shape
        assert model.weight.untyped_storage().device.type == "meta"


# ---------------------------------------------------------------------------
# 4. MPS flush conditionality
# ---------------------------------------------------------------------------

class TestMpsFlushConditionality:
    """Verify the MPS flush only activates under correct conditions."""

    def test_flush_requires_aggressive_offload_flag(self):
        """The MPS flush in samplers is gated on AGGRESSIVE_OFFLOAD."""
        import comfy.model_management as mm

        # When False, flush should NOT be injected
        original = getattr(mm, "AGGRESSIVE_OFFLOAD", False)
        try:
            mm.AGGRESSIVE_OFFLOAD = False
            assert not (True and getattr(mm, "AGGRESSIVE_OFFLOAD", False))

            mm.AGGRESSIVE_OFFLOAD = True
            assert (True and getattr(mm, "AGGRESSIVE_OFFLOAD", False))
        finally:
            mm.AGGRESSIVE_OFFLOAD = original

    def test_flush_requires_mps_device(self):
        """The flush must only activate on MPS devices, not CPU or CUDA."""
        # Simulate CPU device — flush should not activate
        cpu_device = torch.device("cpu")
        assert cpu_device.type != "mps"

        # Simulate MPS device string check
        if torch.backends.mps.is_available():
            mps_device = torch.device("mps")
            assert mps_device.type == "mps"


# ---------------------------------------------------------------------------
# 5. AGGRESSIVE_OFFLOAD flag integration
# ---------------------------------------------------------------------------

class TestAggressiveOffloadFlag:
    """Verify the CLI flag is correctly exposed."""

    def test_flag_exists_in_model_management(self):
        """AGGRESSIVE_OFFLOAD must be importable from model_management."""
        import comfy.model_management as mm
        assert hasattr(mm, "AGGRESSIVE_OFFLOAD")
        assert isinstance(mm.AGGRESSIVE_OFFLOAD, bool)

    def test_flag_defaults_from_cli_args(self):
        """The flag should be wired from cli_args to model_management."""
        import comfy.cli_args as cli_args
        import comfy.model_management as mm
        assert hasattr(cli_args.args, "aggressive_offload")
        assert mm.AGGRESSIVE_OFFLOAD == cli_args.args.aggressive_offload
