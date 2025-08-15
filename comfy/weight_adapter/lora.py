import logging
from typing import Optional

import torch
import comfy.model_management
from .base import (
    WeightAdapterBase,
    WeightAdapterTrainBase,
    weight_decompose,
    pad_tensor_to_shape,
    tucker_weight_from_conv,
)


class LoraDiff(WeightAdapterTrainBase):
    def __init__(self, weights):
        super().__init__()
        mat1, mat2, alpha, mid, dora_scale, reshape = weights
        out_dim, rank = mat1.shape[0], mat1.shape[1]
        rank, in_dim = mat2.shape[0], mat2.shape[1]
        if mid is not None:
            convdim = mid.ndim - 2
            layer = (
                torch.nn.Conv1d,
                torch.nn.Conv2d,
                torch.nn.Conv3d
            )[convdim]
        else:
            layer = torch.nn.Linear
        self.lora_up = layer(rank, out_dim, bias=False)
        self.lora_down = layer(in_dim, rank, bias=False)
        self.lora_up.weight.data.copy_(mat1)
        self.lora_down.weight.data.copy_(mat2)
        if mid is not None:
            self.lora_mid = layer(mid, rank, bias=False)
            self.lora_mid.weight.data.copy_(mid)
        else:
            self.lora_mid = None
        self.rank = rank
        self.alpha = torch.nn.Parameter(torch.tensor(alpha), requires_grad=False)

    def __call__(self, w):
        org_dtype = w.dtype
        if self.lora_mid is None:
            diff = self.lora_up.weight @ self.lora_down.weight
        else:
            diff = tucker_weight_from_conv(
                self.lora_up.weight, self.lora_down.weight, self.lora_mid.weight
            )
        scale = self.alpha / self.rank
        weight = w + scale * diff.reshape(w.shape)
        return weight.to(org_dtype)

    def passive_memory_usage(self):
        return sum(param.numel() * param.element_size() for param in self.parameters())


class LoRAAdapter(WeightAdapterBase):
    name = "lora"

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
        return LoraDiff(
            (mat1, mat2, alpha, None, None, None)
        )

    def to_train(self):
        return LoraDiff(self.weights)

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
        qwen_default_lora = "{}.lora_B.default.weight".format(x)
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
        elif qwen_default_lora in lora.keys():
            A_name = qwen_default_lora
            B_name = "{}.lora_A.default.weight".format(x)
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
