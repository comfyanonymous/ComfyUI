import logging
from typing import Optional

import torch
import comfy.model_management
from .base import WeightAdapterBase, WeightAdapterTrainBase, weight_decompose


class HadaWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w1u, w1d, w2u, w2d, scale=torch.tensor(1)):
        ctx.save_for_backward(w1d, w1u, w2d, w2u, scale)
        diff_weight = ((w1u @ w1d) * (w2u @ w2d)) * scale
        return diff_weight

    @staticmethod
    def backward(ctx, grad_out):
        (w1d, w1u, w2d, w2u, scale) = ctx.saved_tensors
        grad_out = grad_out * scale
        temp = grad_out * (w2u @ w2d)
        grad_w1u = temp @ w1d.T
        grad_w1d = w1u.T @ temp

        temp = grad_out * (w1u @ w1d)
        grad_w2u = temp @ w2d.T
        grad_w2d = w2u.T @ temp

        del temp
        return grad_w1u, grad_w1d, grad_w2u, grad_w2d, None


class HadaWeightTucker(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t1, w1u, w1d, t2, w2u, w2d, scale=torch.tensor(1)):
        ctx.save_for_backward(t1, w1d, w1u, t2, w2d, w2u, scale)

        rebuild1 = torch.einsum("i j ..., j r, i p -> p r ...", t1, w1d, w1u)
        rebuild2 = torch.einsum("i j ..., j r, i p -> p r ...", t2, w2d, w2u)

        return rebuild1 * rebuild2 * scale

    @staticmethod
    def backward(ctx, grad_out):
        (t1, w1d, w1u, t2, w2d, w2u, scale) = ctx.saved_tensors
        grad_out = grad_out * scale

        temp = torch.einsum("i j ..., j r -> i r ...", t2, w2d)
        rebuild = torch.einsum("i j ..., i r -> r j ...", temp, w2u)

        grad_w = rebuild * grad_out
        del rebuild

        grad_w1u = torch.einsum("r j ..., i j ... -> r i", temp, grad_w)
        grad_temp = torch.einsum("i j ..., i r -> r j ...", grad_w, w1u.T)
        del grad_w, temp

        grad_w1d = torch.einsum("i r ..., i j ... -> r j", t1, grad_temp)
        grad_t1 = torch.einsum("i j ..., j r -> i r ...", grad_temp, w1d.T)
        del grad_temp

        temp = torch.einsum("i j ..., j r -> i r ...", t1, w1d)
        rebuild = torch.einsum("i j ..., i r -> r j ...", temp, w1u)

        grad_w = rebuild * grad_out
        del rebuild

        grad_w2u = torch.einsum("r j ..., i j ... -> r i", temp, grad_w)
        grad_temp = torch.einsum("i j ..., i r -> r j ...", grad_w, w2u.T)
        del grad_w, temp

        grad_w2d = torch.einsum("i r ..., i j ... -> r j", t2, grad_temp)
        grad_t2 = torch.einsum("i j ..., j r -> i r ...", grad_temp, w2d.T)
        del grad_temp
        return grad_t1, grad_w1u, grad_w1d, grad_t2, grad_w2u, grad_w2d, None


class LohaDiff(WeightAdapterTrainBase):
    def __init__(self, weights):
        super().__init__()
        # Unpack weights tuple from LoHaAdapter
        w1a, w1b, alpha, w2a, w2b, t1, t2, _ = weights

        # Create trainable parameters
        self.hada_w1_a = torch.nn.Parameter(w1a)
        self.hada_w1_b = torch.nn.Parameter(w1b)
        self.hada_w2_a = torch.nn.Parameter(w2a)
        self.hada_w2_b = torch.nn.Parameter(w2b)

        self.use_tucker = False
        if t1 is not None and t2 is not None:
            self.use_tucker = True
            self.hada_t1 = torch.nn.Parameter(t1)
            self.hada_t2 = torch.nn.Parameter(t2)
        else:
            # Keep the attributes for consistent access
            self.hada_t1 = None
            self.hada_t2 = None

        # Store rank and non-trainable alpha
        self.rank = w1b.shape[0]
        self.alpha = torch.nn.Parameter(torch.tensor(alpha), requires_grad=False)

    def __call__(self, w):
        org_dtype = w.dtype

        scale = self.alpha / self.rank
        if self.use_tucker:
            diff_weight = HadaWeightTucker.apply(self.hada_t1, self.hada_w1_a, self.hada_w1_b, self.hada_t2, self.hada_w2_a, self.hada_w2_b, scale)
        else:
            diff_weight = HadaWeight.apply(self.hada_w1_a, self.hada_w1_b, self.hada_w2_a, self.hada_w2_b, scale)

        # Add the scaled difference to the original weight
        weight = w.to(diff_weight) + diff_weight.reshape(w.shape)

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
        mat1 = torch.empty(out_dim, rank, device=weight.device, dtype=torch.float32)
        mat2 = torch.empty(rank, in_dim, device=weight.device, dtype=torch.float32)
        torch.nn.init.normal_(mat1, 0.1)
        torch.nn.init.constant_(mat2, 0.0)
        mat3 = torch.empty(out_dim, rank, device=weight.device, dtype=torch.float32)
        mat4 = torch.empty(rank, in_dim, device=weight.device, dtype=torch.float32)
        torch.nn.init.normal_(mat3, 0.1)
        torch.nn.init.normal_(mat4, 0.01)
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
