import logging
from typing import Optional

import torch
import comfy.model_management
from .base import (
    WeightAdapterBase,
    WeightAdapterTrainBase,
    weight_decompose,
    factorization,
)


class LokrDiff(WeightAdapterTrainBase):
    def __init__(self, weights):
        super().__init__()
        (lokr_w1, lokr_w2, alpha, lokr_w1_a, lokr_w1_b, lokr_w2_a, lokr_w2_b, lokr_t2, dora_scale) = weights
        self.use_tucker = False
        if lokr_w1_a is not None:
            _, rank_a = lokr_w1_a.shape[0], lokr_w1_a.shape[1]
            rank_a, _ = lokr_w1_b.shape[0], lokr_w1_b.shape[1]
            self.lokr_w1_a = torch.nn.Parameter(lokr_w1_a)
            self.lokr_w1_b = torch.nn.Parameter(lokr_w1_b)
            self.w1_rebuild = True
            self.ranka = rank_a

        if lokr_w2_a is not None:
            _, rank_b = lokr_w2_a.shape[0], lokr_w2_a.shape[1]
            rank_b, _ = lokr_w2_b.shape[0], lokr_w2_b.shape[1]
            self.lokr_w2_a = torch.nn.Parameter(lokr_w2_a)
            self.lokr_w2_b = torch.nn.Parameter(lokr_w2_b)
            if lokr_t2 is not None:
                self.use_tucker = True
                self.lokr_t2 = torch.nn.Parameter(lokr_t2)
            self.w2_rebuild = True
            self.rankb = rank_b

        if lokr_w1 is not None:
            self.lokr_w1 = torch.nn.Parameter(lokr_w1)
            self.w1_rebuild = False

        if lokr_w2 is not None:
            self.lokr_w2 = torch.nn.Parameter(lokr_w2)
            self.w2_rebuild = False

        self.alpha = torch.nn.Parameter(torch.tensor(alpha), requires_grad=False)

    @property
    def w1(self):
        if self.w1_rebuild:
            return (self.lokr_w1_a @ self.lokr_w1_b) * (self.alpha / self.ranka)
        else:
            return self.lokr_w1

    @property
    def w2(self):
        if self.w2_rebuild:
            if self.use_tucker:
                w2 = torch.einsum(
                    'i j k l, j r, i p -> p r k l',
                    self.lokr_t2,
                    self.lokr_w2_b,
                    self.lokr_w2_a
                )
            else:
                w2 = self.lokr_w2_a @ self.lokr_w2_b
            return w2 * (self.alpha / self.rankb)
        else:
            return self.lokr_w2

    def __call__(self, w):
        diff = torch.kron(self.w1, self.w2)
        return w + diff.reshape(w.shape).to(w)

    def passive_memory_usage(self):
        return sum(param.numel() * param.element_size() for param in self.parameters())


class LoKrAdapter(WeightAdapterBase):
    name = "lokr"

    def __init__(self, loaded_keys, weights):
        self.loaded_keys = loaded_keys
        self.weights = weights

    @classmethod
    def create_train(cls, weight, rank=1, alpha=1.0):
        out_dim = weight.shape[0]
        in_dim = weight.shape[1:].numel()
        out1, out2 = factorization(out_dim, rank)
        in1, in2 = factorization(in_dim, rank)
        mat1 = torch.empty(out1, in1, device=weight.device, dtype=torch.float32)
        mat2 = torch.empty(out2, in2, device=weight.device, dtype=torch.float32)
        torch.nn.init.kaiming_uniform_(mat2, a=5**0.5)
        torch.nn.init.constant_(mat1, 0.0)
        return LokrDiff(
            (mat1, mat2, alpha, None, None, None, None, None, None)
        )

    def to_train(self):
        return LokrDiff(self.weights)

    @classmethod
    def load(
        cls,
        x: str,
        lora: dict[str, torch.Tensor],
        alpha: float,
        dora_scale: torch.Tensor,
        loaded_keys: set[str] = None,
    ) -> Optional["LoKrAdapter"]:
        if loaded_keys is None:
            loaded_keys = set()
        lokr_w1_name = "{}.lokr_w1".format(x)
        lokr_w2_name = "{}.lokr_w2".format(x)
        lokr_w1_a_name = "{}.lokr_w1_a".format(x)
        lokr_w1_b_name = "{}.lokr_w1_b".format(x)
        lokr_t2_name = "{}.lokr_t2".format(x)
        lokr_w2_a_name = "{}.lokr_w2_a".format(x)
        lokr_w2_b_name = "{}.lokr_w2_b".format(x)

        lokr_w1 = None
        if lokr_w1_name in lora.keys():
            lokr_w1 = lora[lokr_w1_name]
            loaded_keys.add(lokr_w1_name)

        lokr_w2 = None
        if lokr_w2_name in lora.keys():
            lokr_w2 = lora[lokr_w2_name]
            loaded_keys.add(lokr_w2_name)

        lokr_w1_a = None
        if lokr_w1_a_name in lora.keys():
            lokr_w1_a = lora[lokr_w1_a_name]
            loaded_keys.add(lokr_w1_a_name)

        lokr_w1_b = None
        if lokr_w1_b_name in lora.keys():
            lokr_w1_b = lora[lokr_w1_b_name]
            loaded_keys.add(lokr_w1_b_name)

        lokr_w2_a = None
        if lokr_w2_a_name in lora.keys():
            lokr_w2_a = lora[lokr_w2_a_name]
            loaded_keys.add(lokr_w2_a_name)

        lokr_w2_b = None
        if lokr_w2_b_name in lora.keys():
            lokr_w2_b = lora[lokr_w2_b_name]
            loaded_keys.add(lokr_w2_b_name)

        lokr_t2 = None
        if lokr_t2_name in lora.keys():
            lokr_t2 = lora[lokr_t2_name]
            loaded_keys.add(lokr_t2_name)

        if (lokr_w1 is not None) or (lokr_w2 is not None) or (lokr_w1_a is not None) or (lokr_w2_a is not None):
            weights = (lokr_w1, lokr_w2, alpha, lokr_w1_a, lokr_w1_b, lokr_w2_a, lokr_w2_b, lokr_t2, dora_scale)
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
        w1 = v[0]
        w2 = v[1]
        w1_a = v[3]
        w1_b = v[4]
        w2_a = v[5]
        w2_b = v[6]
        t2 = v[7]
        dora_scale = v[8]
        dim = None

        if w1 is None:
            dim = w1_b.shape[0]
            w1 = torch.mm(comfy.model_management.cast_to_device(w1_a, weight.device, intermediate_dtype),
                            comfy.model_management.cast_to_device(w1_b, weight.device, intermediate_dtype))
        else:
            w1 = comfy.model_management.cast_to_device(w1, weight.device, intermediate_dtype)

        if w2 is None:
            dim = w2_b.shape[0]
            if t2 is None:
                w2 = torch.mm(comfy.model_management.cast_to_device(w2_a, weight.device, intermediate_dtype),
                                comfy.model_management.cast_to_device(w2_b, weight.device, intermediate_dtype))
            else:
                w2 = torch.einsum('i j k l, j r, i p -> p r k l',
                                    comfy.model_management.cast_to_device(t2, weight.device, intermediate_dtype),
                                    comfy.model_management.cast_to_device(w2_b, weight.device, intermediate_dtype),
                                    comfy.model_management.cast_to_device(w2_a, weight.device, intermediate_dtype))
        else:
            w2 = comfy.model_management.cast_to_device(w2, weight.device, intermediate_dtype)

        if len(w2.shape) == 4:
            w1 = w1.unsqueeze(2).unsqueeze(2)
        if v[2] is not None and dim is not None:
            alpha = v[2] / dim
        else:
            alpha = 1.0

        try:
            lora_diff = torch.kron(w1, w2).reshape(weight.shape)
            if dora_scale is not None:
                weight = weight_decompose(dora_scale, weight, lora_diff, alpha, strength, intermediate_dtype, function)
            else:
                weight += function(((strength * alpha) * lora_diff).type(weight.dtype))
        except Exception as e:
            logging.error("ERROR {} {} {}".format(self.name, key, e))
        return weight
