import logging
from typing import Optional

import torch
import comfy.model_management
from .base import WeightAdapterBase, weight_decompose


class OFTAdapter(WeightAdapterBase):
    name = "oft"

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
    ) -> Optional["OFTAdapter"]:
        if loaded_keys is None:
            loaded_keys = set()
        blocks_name = "{}.oft_blocks".format(x)
        rescale_name = "{}.rescale".format(x)

        blocks = None
        if blocks_name in lora.keys():
            blocks = lora[blocks_name]
            if blocks.ndim == 3:
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

        block_num, block_size, *_ = blocks.shape

        try:
            # Get r
            I = torch.eye(block_size, device=blocks.device, dtype=blocks.dtype)
            # for Q = -Q^T
            q = blocks - blocks.transpose(1, 2)
            normed_q = q
            if alpha > 0: # alpha in oft/boft is for constraint
                q_norm = torch.norm(q) + 1e-8
                if q_norm > alpha:
                    normed_q = q * alpha / q_norm
            # use float() to prevent unsupported type in .inverse()
            r = (I + normed_q) @ (I - normed_q).float().inverse()
            r = r.to(weight)
            _, *shape = weight.shape
            lora_diff = torch.einsum(
                "k n m, k n ... -> k m ...",
                (r * strength) - strength * I,
                weight.view(block_num, block_size, *shape),
            ).view(-1, *shape)
            if dora_scale is not None:
                weight = weight_decompose(dora_scale, weight, lora_diff, alpha, strength, intermediate_dtype, function)
            else:
                weight += function((strength * lora_diff).type(weight.dtype))
        except Exception as e:
            logging.error("ERROR {} {} {}".format(self.name, key, e))
        return weight
