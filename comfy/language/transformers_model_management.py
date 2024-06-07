from __future__ import annotations

import warnings
from typing import Optional, Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase, PretrainedConfig

from ..model_management import unet_offload_device, get_torch_device
from ..model_management_types import ModelManageable


class TransformersManagedModel(ModelManageable):
    def __init__(self, repo_id: str, model: PreTrainedModel, tokenizer: Optional[PreTrainedTokenizerBase] = None, config_dict: Optional[dict] = None):
        self.repo_id = repo_id
        self.model = model
        self.tokenizer = tokenizer
        self._parameter_count = sum(param.nelement() for param in self.model.state_dict().values())
        self._size = sum(param.nelement() * param.element_size() for param in self.model.state_dict().values())
        self.load_device = get_torch_device()
        self.offload_device = unet_offload_device()
        self._config_dict = config_dict
        if model.device != self.offload_device:
            model.to(device=self.offload_device)

    @property
    def config_dict(self) -> dict:
        """
        The original configuration dictionary located in the Transformers model.

        Many models derive from base models and should inherit their settings like a chat template. This
        config_dict will have the base model's name in _name_or_path, enabling a lookup for the valid
        chat template when it is not specified by the derived model (it almost never is).
        :return: the dict value of the config.json in the HuggingFace model
        """
        if self._config_dict is not None:
            return self._config_dict

        return self.model.config.to_dict()

    @property
    def lowvram_patch_counter(self):
        return 0

    @lowvram_patch_counter.setter
    def lowvram_patch_counter(self, value: int):
        warnings.warn("Not supported")
        pass

    load_device: torch.device
    offload_device: torch.device
    model: PreTrainedModel

    @property
    def current_device(self) -> torch.device:
        return self.model.device

    def is_clone(self, other: Any) -> bool:
        return hasattr(other, "model") and self.model is other.model

    def clone_has_same_weights(self, clone: Any) -> bool:
        if not isinstance(clone, TransformersManagedModel):
            return False

        clone: TransformersManagedModel

        if not self.is_clone(clone):
            return False

        return frozenset(self.model.active_adapters()) == frozenset(clone.model.active_adapters())

    def model_size(self) -> int:
        return self._size

    def model_patches_to(self, arg: torch.device | torch.dtype):
        if isinstance(arg, torch.device):
            self.model.to(device=arg)
        else:
            self.model.to(arg)

    def model_dtype(self) -> torch.dtype:
        return self.model.dtype

    def patch_model_lowvram(self, device_to: torch.device, lowvram_model_memory: int, force_patch_weights=False) -> torch.nn.Module:
        warnings.warn("Transformers models do not currently support adapters like LoRAs")
        return self.model.to(device=device_to)

    def patch_model(self, device_to: torch.device, patch_weights: bool) -> torch.nn.Module:
        warnings.warn("Transformers models do not currently support adapters like LoRAs")
        return self.model.to(device=device_to)

    def unpatch_model(self, offload_device: torch.device, unpatch_weights: Optional[bool] = False) -> torch.nn.Module:
        warnings.warn("Transformers models do not currently support adapters like LoRAs")
        return self.model.to(device=offload_device)
