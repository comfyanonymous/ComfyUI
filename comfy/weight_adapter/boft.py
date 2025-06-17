import logging
from typing import Optional

import torch
import comfy.model_management
from .base import WeightAdapterBase, weight_decompose


class BOFTAdapter(WeightAdapterBase):
    name = "boft"

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
    ) -> Optional["BOFTAdapter"]:
        if loaded_keys is None:
            loaded_keys = set()
        blocks_name = "{}.oft_blocks".format(x)
        rescale_name = "{}.rescale".format(x)

        blocks = None
        if blocks_name in lora.keys():
            blocks = lora[blocks_name]
            if blocks.ndim == 4:
                loaded_keys.add(blocks_name)
            else:
                blocks = None
        if blocks is None:
            return None

        rescale = None
        if rescale_name in lora.keys():
            rescale = lora[rescale_name]
            loaded_keys.add(rescale_name)

        weights = (blocks, rescale, alpha, dora_scale)
        return cls(loaded_keys, weights)

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
        blocks = v[0]
        rescale = v[1]
        alpha = v[2]
        dora_scale = v[3]

        blocks = comfy.model_management.cast_to_device(blocks, weight.device, intermediate_dtype)
        if rescale is not None:
            rescale = comfy.model_management.cast_to_device(rescale, weight.device, intermediate_dtype)

        boft_m, block_num, boft_b, *_ = blocks.shape

        try:
            # Get r
            I = torch.eye(boft_b, device=blocks.device, dtype=blocks.dtype)
            # for Q = -Q^T
            q = blocks - blocks.transpose(-1, -2)
            normed_q = q
            if alpha > 0: # alpha in boft/bboft is for constraint
                q_norm = torch.norm(q) + 1e-8
                if q_norm > alpha:
                    normed_q = q * alpha / q_norm
            # use float() to prevent unsupported type in .inverse()
            r = (I + normed_q) @ (I - normed_q).float().inverse()
            r = r.to(weight)
            inp = org = weight

            r_b = boft_b//2
            for i in range(boft_m):
                bi = r[i]
                g = 2
                k = 2**i * r_b
                if strength != 1:
                    bi = bi * strength + (1-strength) * I
                inp = (
                    inp.unflatten(0, (-1, g, k))
                    .transpose(1, 2)
                    .flatten(0, 2)
                    .unflatten(0, (-1, boft_b))
                )
                inp = torch.einsum("b i j, b j ...-> b i ...", bi, inp)
                inp = (
                    inp.flatten(0, 1).unflatten(0, (-1, k, g)).transpose(1, 2).flatten(0, 2)
                )

            if rescale is not None:
                inp = inp * rescale

            lora_diff = inp - org
            lora_diff = comfy.model_management.cast_to_device(lora_diff, weight.device, intermediate_dtype)
            if dora_scale is not None:
                weight = weight_decompose(dora_scale, weight, lora_diff, alpha, strength, intermediate_dtype, function)
            else:
                weight += function((strength * lora_diff).type(weight.dtype))
        except Exception as e:
            logging.error("ERROR {} {} {}".format(self.name, key, e))
        return weight
