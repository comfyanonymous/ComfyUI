import logging
from typing import Optional

import torch
import comfy.model_management
from .base import WeightAdapterBase, WeightAdapterTrainBase, weight_decompose


class LohaDiff(WeightAdapterTrainBase):
    def __init__(self, weights):
        super().__init__()
        # Unpack weights tuple from LoHaAdapter
        w1a, w1b, alpha, w2a, w2b, t1, t2, dora_scale = weights

        # Create trainable parameters
        self.w1a = torch.nn.Parameter(w1a)
        self.w1b = torch.nn.Parameter(w1b)
        self.w2a = torch.nn.Parameter(w2a)
        self.w2b = torch.nn.Parameter(w2b)

        self.use_tucker = False
        if t1 is not None and t2 is not None:
            self.use_tucker = True
            self.t1 = torch.nn.Parameter(t1)
            self.t2 = torch.nn.Parameter(t2)
        else:
            # Keep the attributes for consistent access
            self.t1 = None
            self.t2 = None

        # Store rank and non-trainable alpha
        self.rank = w1b.shape[0]
        self.alpha = torch.nn.Parameter(torch.tensor(alpha), requires_grad=False)
        # dora_scale is not used in the training forward pass

    def __call__(self, w):
        org_dtype = w.dtype

        # Reconstruct the two matrices m1 and m2
        if self.use_tucker:
            # CP/Tucker decomposition case
            m1 = torch.einsum('i j k l, j r, i p -> p r k l', self.t1, self.w1b, self.w1a)
            m2 = torch.einsum('i j k l, j r, i p -> p r k l', self.t2, self.w2b, self.w2a)
        else:
            # Standard Hadmard product case
            m1 = self.w1a @ self.w1b
            m2 = self.w2a @ self.w2b

        # Calculate the final difference via element-wise product
        diff = m1 * m2

        # Apply scaling
        scale = self.alpha / self.rank

        # Add the scaled difference to the original weight
        weight = w + scale * diff.reshape(w.shape)

        return weight.to(org_dtype)

    def passive_memory_usage(self):
        """Calculates memory usage of the trainable parameters."""
        return sum(param.numel() * param.element_size() for param in self.parameters())


class LoHaAdapter(WeightAdapterBase):
    name = "loha"

    def __init__(self, loaded_keys, weights):
        self.loaded_keys = loaded_keys
        self.weights = weights

    @classmethod
    def create_train(cls, weight, rank=1, alpha=1.0):
        out_dim = weight.shape[0]
        in_dim = weight.shape[1:].numel()
        mat1 = torch.empty(out_dim, rank, device=weight.device, dtype=weight.dtype)
        mat2 = torch.empty(rank, in_dim, device=weight.device, dtype=weight.dtype)
        torch.nn.init.kaiming_uniform_(mat1, a=5**0.5)
        torch.nn.init.constant_(mat2, 0.0)
        mat3 = torch.empty(out_dim, rank, device=weight.device, dtype=weight.dtype)
        mat4 = torch.empty(rank, in_dim, device=weight.device, dtype=weight.dtype)
        torch.nn.init.kaiming_uniform_(mat1, a=5**0.5)
        torch.nn.init.kaiming_uniform_(mat2, a=5**0.5)
        return LohaDiff(
            (mat1, mat2, alpha, mat3, mat4, None, None, None)
        )

    def to_train(self):
        return LohaDiff(self.weights)

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
