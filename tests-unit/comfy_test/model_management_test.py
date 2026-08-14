import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.model_management as model_management


class _Patcher:
    def __init__(self, device, dynamic):
        self.model = object()
        self.load_device = device
        self.offload_device = torch.device("cpu")
        self.dynamic = dynamic
        self.detached = False
        self.partial_unloads = []

    def is_dynamic(self):
        return self.dynamic

    def model_patches_models(self):
        return []

    def is_clone(self, other):
        return False

    def loaded_size(self):
        return 100

    def model_size(self):
        return 100

    def partially_unload(self, device, memory_to_free):
        self.partial_unloads.append((device, memory_to_free))
        return memory_to_free


class _LoadedModel:
    def __init__(self, model):
        self.model = model
        self.device = model.load_device
        self.currently_used = True

    def __eq__(self, other):
        return self.model is other.model

    def is_dead(self):
        return False

    def model_memory(self):
        return 100

    def model_memory_required(self, device):
        return 100

    def model_loaded_memory(self):
        return 0

    def model_offloaded_memory(self):
        return 0

    def model_load(self, lowvram_model_memory=0, force_patch_weights=False):
        pass

    def model_unload(self, memory_to_free):
        self.model.detached = True
        return True


def test_loading_static_model_keeps_dynamic_model_ram_cache(monkeypatch):
    device = torch.device("cuda:0")
    dynamic = _Patcher(device, dynamic=True)
    static = _Patcher(device, dynamic=False)
    dynamic_loaded = _LoadedModel(dynamic)

    monkeypatch.setattr(model_management, "LoadedModel", _LoadedModel)
    monkeypatch.setattr(model_management, "current_loaded_models", [dynamic_loaded])
    monkeypatch.setattr(model_management, "cleanup_models_gc", lambda: None)
    monkeypatch.setattr(model_management, "soft_empty_cache", lambda: None)
    monkeypatch.setattr(model_management, "minimum_inference_memory", lambda: 1)
    monkeypatch.setattr(model_management, "extra_reserved_memory", lambda: 0)
    monkeypatch.setattr(model_management, "get_free_memory", lambda device, torch_free_too=False: (0, 0) if torch_free_too else 0)
    monkeypatch.setattr(model_management, "ensure_pin_budget", lambda size: True)
    monkeypatch.setattr(model_management, "ensure_pin_registerable", lambda size: True)
    monkeypatch.setattr(model_management, "lowvram_available", False)

    model_management.load_models_gpu([static])

    assert dynamic_loaded in model_management.current_loaded_models
    assert dynamic.detached is False
    assert dynamic.partial_unloads
