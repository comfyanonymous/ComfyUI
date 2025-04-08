import logging
from typing import Optional

import torch
import comfy.model_management
from .base import WeightAdapterBase, weight_decompose, pad_tensor_to_shape


class LoRAAdapter(WeightAdapterBase):
    name = "lora"

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
    ) -> Optional["LoRAAdapter"]:
        if loaded_keys is None:
            loaded_keys = set()

        reshape_name = "{}.reshape_weight".format(x)
        regular_lora = "{}.lora_up.weight".format(x)
        diffusers_lora = "{}_lora.up.weight".format(x)
        diffusers2_lora = "{}.lora_B.weight".format(x)
        diffusers3_lora = "{}.lora.up.weight".format(x)
        mochi_lora = "{}.lora_B".format(x)
        transformers_lora = "{}.lora_linear_layer.up.weight".format(x)
        A_name = None

        if regular_lora in lora.keys():
            A_name = regular_lora
            B_name = "{}.lora_down.weight".format(x)
            mid_name = "{}.lora_mid.weight".format(x)
        elif diffusers_lora in lora.keys():
            A_name = diffusers_lora
            B_name = "{}_lora.down.weight".format(x)
            mid_name = None
        elif diffusers2_lora in lora.keys():
            A_name = diffusers2_lora
            B_name = "{}.lora_A.weight".format(x)
            mid_name = None
        elif diffusers3_lora in lora.keys():
            A_name = diffusers3_lora
            B_name = "{}.lora.down.weight".format(x)
            mid_name = None
        elif mochi_lora in lora.keys():
            A_name = mochi_lora
            B_name = "{}.lora_A".format(x)
            mid_name = None
        elif transformers_lora in lora.keys():
            A_name = transformers_lora
            B_name = "{}.lora_linear_layer.down.weight".format(x)
            mid_name = None

        if A_name is not None:
            mid = None
            if mid_name is not None and mid_name in lora.keys():
                mid = lora[mid_name]
                loaded_keys.add(mid_name)
            reshape = None
            if reshape_name in lora.keys():
                try:
                    reshape = lora[reshape_name].tolist()
                    loaded_keys.add(reshape_name)
                except:
                    pass
            weights = (lora[A_name], lora[B_name], alpha, mid, dora_scale, reshape)
            loaded_keys.add(A_name)
            loaded_keys.add(B_name)
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
        mat1 = comfy.model_management.cast_to_device(
            v[0], weight.device, intermediate_dtype
        )
        mat2 = comfy.model_management.cast_to_device(
            v[1], weight.device, intermediate_dtype
        )
        dora_scale = v[4]
        reshape = v[5]

        if reshape is not None:
            weight = pad_tensor_to_shape(weight, reshape)

        if v[2] is not None:
            alpha = v[2] / mat2.shape[0]
        else:
            alpha = 1.0

        if v[3] is not None:
            # locon mid weights, hopefully the math is fine because I didn't properly test it
            mat3 = comfy.model_management.cast_to_device(
                v[3], weight.device, intermediate_dtype
            )
            final_shape = [mat2.shape[1], mat2.shape[0], mat3.shape[2], mat3.shape[3]]
            mat2 = (
                torch.mm(
                    mat2.transpose(0, 1).flatten(start_dim=1),
                    mat3.transpose(0, 1).flatten(start_dim=1),
                )
                .reshape(final_shape)
                .transpose(0, 1)
            )
        try:
            lora_diff = torch.mm(
                mat1.flatten(start_dim=1), mat2.flatten(start_dim=1)
            ).reshape(weight.shape)
            if dora_scale is not None:
                weight = weight_decompose(
                    dora_scale,
                    weight,
                    lora_diff,
                    alpha,
                    strength,
                    intermediate_dtype,
                    function,
                )
            else:
                weight += function(((strength * alpha) * lora_diff).type(weight.dtype))
        except Exception as e:
            logging.error("ERROR {} {} {}".format(self.name, key, e))
        return weight
