import logging
from typing import Optional

import torch
import comfy.model_management
from .base import WeightAdapterBase, weight_decompose


class GLoRAAdapter(WeightAdapterBase):
    name = "glora"

    def __init__(self, loaded_keys, weights):
        self.loaded_keys = loaded_keys
        self.weights = weights

    @classmethod
    def load(
        cls,
        x: str,
        lora: dict[str, torch.Tensor],
        alpha: float,
        dora_scale: torch.Tensor,
        loaded_keys: set[str] = None,
    ) -> Optional["GLoRAAdapter"]:
        if loaded_keys is None:
            loaded_keys = set()
        a1_name = "{}.a1.weight".format(x)
        a2_name = "{}.a2.weight".format(x)
        b1_name = "{}.b1.weight".format(x)
        b2_name = "{}.b2.weight".format(x)
        if a1_name in lora:
            weights = (lora[a1_name], lora[a2_name], lora[b1_name], lora[b2_name], alpha, dora_scale)
            loaded_keys.add(a1_name)
            loaded_keys.add(a2_name)
            loaded_keys.add(b1_name)
            loaded_keys.add(b2_name)
            return cls(loaded_keys, weights)
        else:
            return None

    def calculate_weight(
        self,
        weight,
        key,
        strength,
        strength_model,
        offset,
        function,
        intermediate_dtype=torch.float32,
        original_weight=None,
    ):
        v = self.weights
        dora_scale = v[5]

        old_glora = False
        if v[3].shape[1] == v[2].shape[0] == v[0].shape[0] == v[1].shape[1]:
            rank = v[0].shape[0]
            old_glora = True

        if v[3].shape[0] == v[2].shape[1] == v[0].shape[1] == v[1].shape[0]:
            if old_glora and v[1].shape[0] == weight.shape[0] and weight.shape[0] == weight.shape[1]:
                pass
            else:
                old_glora = False
                rank = v[1].shape[0]

        a1 = comfy.model_management.cast_to_device(v[0].flatten(start_dim=1), weight.device, intermediate_dtype)
        a2 = comfy.model_management.cast_to_device(v[1].flatten(start_dim=1), weight.device, intermediate_dtype)
        b1 = comfy.model_management.cast_to_device(v[2].flatten(start_dim=1), weight.device, intermediate_dtype)
        b2 = comfy.model_management.cast_to_device(v[3].flatten(start_dim=1), weight.device, intermediate_dtype)

        if v[4] is not None:
            alpha = v[4] / rank
        else:
            alpha = 1.0

        try:
            if old_glora:
                lora_diff = (torch.mm(b2, b1) + torch.mm(torch.mm(weight.flatten(start_dim=1).to(dtype=intermediate_dtype), a2), a1)).reshape(weight.shape) #old lycoris glora
            else:
                if weight.dim() > 2:
                    lora_diff = torch.einsum("o i ..., i j -> o j ...", torch.einsum("o i ..., i j -> o j ...", weight.to(dtype=intermediate_dtype), a1), a2).reshape(weight.shape)
                else:
                    lora_diff = torch.mm(torch.mm(weight.to(dtype=intermediate_dtype), a1), a2).reshape(weight.shape)
                lora_diff += torch.mm(b1, b2).reshape(weight.shape)

            if dora_scale is not None:
                weight = weight_decompose(dora_scale, weight, lora_diff, alpha, strength, intermediate_dtype, function)
            else:
                weight += function(((strength * alpha) * lora_diff).type(weight.dtype))
        except Exception as e:
            logging.error("ERROR {} {} {}".format(self.name, key, e))
        return weight
