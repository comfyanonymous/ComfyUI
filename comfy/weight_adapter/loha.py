import logging
from typing import Optional

import torch
import comfy.model_management
from .base import WeightAdapterBase, weight_decompose


class LoHaAdapter(WeightAdapterBase):
    name = "loha"

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
    ) -> Optional["LoHaAdapter"]:
        if loaded_keys is None:
            loaded_keys = set()

        hada_w1_a_name = "{}.hada_w1_a".format(x)
        hada_w1_b_name = "{}.hada_w1_b".format(x)
        hada_w2_a_name = "{}.hada_w2_a".format(x)
        hada_w2_b_name = "{}.hada_w2_b".format(x)
        hada_t1_name = "{}.hada_t1".format(x)
        hada_t2_name = "{}.hada_t2".format(x)
        if hada_w1_a_name in lora.keys():
            hada_t1 = None
            hada_t2 = None
            if hada_t1_name in lora.keys():
                hada_t1 = lora[hada_t1_name]
                hada_t2 = lora[hada_t2_name]
                loaded_keys.add(hada_t1_name)
                loaded_keys.add(hada_t2_name)

            weights = (lora[hada_w1_a_name], lora[hada_w1_b_name], alpha, lora[hada_w2_a_name], lora[hada_w2_b_name], hada_t1, hada_t2, dora_scale)
            loaded_keys.add(hada_w1_a_name)
            loaded_keys.add(hada_w1_b_name)
            loaded_keys.add(hada_w2_a_name)
            loaded_keys.add(hada_w2_b_name)
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
        w1a = v[0]
        w1b = v[1]
        if v[2] is not None:
            alpha = v[2] / w1b.shape[0]
        else:
            alpha = 1.0

        w2a = v[3]
        w2b = v[4]
        dora_scale = v[7]
        if v[5] is not None: #cp decomposition
            t1 = v[5]
            t2 = v[6]
            m1 = torch.einsum('i j k l, j r, i p -> p r k l',
                                comfy.model_management.cast_to_device(t1, weight.device, intermediate_dtype),
                                comfy.model_management.cast_to_device(w1b, weight.device, intermediate_dtype),
                                comfy.model_management.cast_to_device(w1a, weight.device, intermediate_dtype))

            m2 = torch.einsum('i j k l, j r, i p -> p r k l',
                                comfy.model_management.cast_to_device(t2, weight.device, intermediate_dtype),
                                comfy.model_management.cast_to_device(w2b, weight.device, intermediate_dtype),
                                comfy.model_management.cast_to_device(w2a, weight.device, intermediate_dtype))
        else:
            m1 = torch.mm(comfy.model_management.cast_to_device(w1a, weight.device, intermediate_dtype),
                            comfy.model_management.cast_to_device(w1b, weight.device, intermediate_dtype))
            m2 = torch.mm(comfy.model_management.cast_to_device(w2a, weight.device, intermediate_dtype),
                            comfy.model_management.cast_to_device(w2b, weight.device, intermediate_dtype))

        try:
            lora_diff = (m1 * m2).reshape(weight.shape)
            if dora_scale is not None:
                weight = weight_decompose(dora_scale, weight, lora_diff, alpha, strength, intermediate_dtype, function)
            else:
                weight += function(((strength * alpha) * lora_diff).type(weight.dtype))
        except Exception as e:
            logging.error("ERROR {} {} {}".format(self.name, key, e))
        return weight
