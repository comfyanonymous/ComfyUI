import logging
from typing import Callable, Optional

import torch
import torch.nn.functional as F
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
            weights = (
                lora[a1_name],
                lora[a2_name],
                lora[b1_name],
                lora[b2_name],
                alpha,
                dora_scale,
            )
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
            if (
                old_glora
                and v[1].shape[0] == weight.shape[0]
                and weight.shape[0] == weight.shape[1]
            ):
                pass
            else:
                old_glora = False
                rank = v[1].shape[0]

        a1 = comfy.model_management.cast_to_device(
            v[0].flatten(start_dim=1), weight.device, intermediate_dtype
        )
        a2 = comfy.model_management.cast_to_device(
            v[1].flatten(start_dim=1), weight.device, intermediate_dtype
        )
        b1 = comfy.model_management.cast_to_device(
            v[2].flatten(start_dim=1), weight.device, intermediate_dtype
        )
        b2 = comfy.model_management.cast_to_device(
            v[3].flatten(start_dim=1), weight.device, intermediate_dtype
        )

        if v[4] is not None:
            alpha = v[4] / rank
        else:
            alpha = 1.0

        try:
            if old_glora:
                lora_diff = (
                    torch.mm(b2, b1)
                    + torch.mm(
                        torch.mm(
                            weight.flatten(start_dim=1).to(dtype=intermediate_dtype), a2
                        ),
                        a1,
                    )
                ).reshape(
                    weight.shape
                )  # old lycoris glora
            else:
                if weight.dim() > 2:
                    lora_diff = torch.einsum(
                        "o i ..., i j -> o j ...",
                        torch.einsum(
                            "o i ..., i j -> o j ...",
                            weight.to(dtype=intermediate_dtype),
                            a1,
                        ),
                        a2,
                    ).reshape(weight.shape)
                else:
                    lora_diff = torch.mm(
                        torch.mm(weight.to(dtype=intermediate_dtype), a1), a2
                    ).reshape(weight.shape)
                lora_diff += torch.mm(b1, b2).reshape(weight.shape)

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

    def _compute_paths(self, x: torch.Tensor):
        """
        Compute A path and B path outputs for GLoRA bypass.

        GLoRA: f(x) = Wx + WAx + Bx
        - A path: a1(a2(x)) - modifies input to base forward
        - B path: b1(b2(x)) - additive component

        Note:
            Does not access original model weights - bypass mode is designed
            for quantized models where weights may not be accessible.

        Returns: (a_out, b_out)
        """
        v = self.weights
        # v = (a1, a2, b1, b2, alpha, dora_scale)
        a1 = v[0]
        a2 = v[1]
        b1 = v[2]
        b2 = v[3]
        alpha = v[4]

        dtype = x.dtype

        # Cast dtype (weights should already be on correct device from inject())
        a1 = a1.to(dtype=dtype)
        a2 = a2.to(dtype=dtype)
        b1 = b1.to(dtype=dtype)
        b2 = b2.to(dtype=dtype)

        # Determine rank and scale
        # Check for old vs new glora format
        old_glora = False
        if b2.shape[1] == b1.shape[0] == a1.shape[0] == a2.shape[1]:
            rank = a1.shape[0]
            old_glora = True

        if b2.shape[0] == b1.shape[1] == a1.shape[1] == a2.shape[0]:
            if old_glora and a2.shape[0] == x.shape[-1] and x.shape[-1] == x.shape[-1]:
                pass
            else:
                old_glora = False
                rank = a2.shape[0]

        if alpha is not None:
            scale = alpha / rank
        else:
            scale = 1.0

        # Apply multiplier
        multiplier = getattr(self, "multiplier", 1.0)
        scale = scale * multiplier

        # Use module info from bypass injection, not input tensor shape
        is_conv = getattr(self, "is_conv", False)
        conv_dim = getattr(self, "conv_dim", 0)
        kw_dict = getattr(self, "kw_dict", {})

        if is_conv:
            # Conv case - conv_dim is 1/2/3 for conv1d/2d/3d
            conv_fn = (F.conv1d, F.conv2d, F.conv3d)[conv_dim - 1]

            # Get module's stride/padding for spatial dimension handling
            module_stride = kw_dict.get("stride", (1,) * conv_dim)
            module_padding = kw_dict.get("padding", (0,) * conv_dim)
            kernel_size = getattr(self, "kernel_size", (1,) * conv_dim)
            in_channels = getattr(self, "in_channels", None)

            # Ensure weights are in conv shape
            # a1, a2, b1 are always 1x1 kernels
            if a1.ndim == 2:
                a1 = a1.view(*a1.shape, *([1] * conv_dim))
            if a2.ndim == 2:
                a2 = a2.view(*a2.shape, *([1] * conv_dim))
            if b1.ndim == 2:
                b1 = b1.view(*b1.shape, *([1] * conv_dim))
            # b2 has actual kernel_size (like LoRA down)
            if b2.ndim == 2:
                if in_channels is not None:
                    b2 = b2.view(b2.shape[0], in_channels, *kernel_size)
                else:
                    b2 = b2.view(*b2.shape, *([1] * conv_dim))

            # A path: a2(x) -> a1(...) - 1x1 convs, no stride/padding needed, a_out is added to x
            a2_out = conv_fn(x, a2)
            a_out = conv_fn(a2_out, a1) * scale

            # B path: b2(x) with kernel/stride/padding -> b1(...) 1x1
            b2_out = conv_fn(x, b2, stride=module_stride, padding=module_padding)
            b_out = conv_fn(b2_out, b1) * scale
        else:
            # Linear case
            if old_glora:
                # Old format: a1 @ a2 @ x, b2 @ b1
                a_out = F.linear(F.linear(x, a2), a1) * scale
                b_out = F.linear(F.linear(x, b1), b2) * scale
            else:
                # New format: x @ a1 @ a2, b1 @ b2
                a_out = F.linear(F.linear(x, a1), a2) * scale
                b_out = F.linear(F.linear(x, b2), b1) * scale

        return a_out, b_out

    def bypass_forward(
        self,
        org_forward: Callable,
        x: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        GLoRA bypass forward: f(x + a(x)) + b(x)

        Unlike standard adapters, GLoRA modifies the input to the base forward
        AND adds the B path output.

        Note:
            Does not access original model weights - bypass mode is designed
            for quantized models where weights may not be accessible.

        Reference: LyCORIS GLoRAModule._bypass_forward
        """
        a_out, b_out = self._compute_paths(x)

        # Call base forward with modified input
        base_out = org_forward(x + a_out, *args, **kwargs)

        # Add B path
        return base_out + b_out

    def h(self, x: torch.Tensor, base_out: torch.Tensor) -> torch.Tensor:
        """
        For GLoRA, h() returns the B path output.

        Note:
            GLoRA's full bypass requires overriding bypass_forward() since
            it also modifies the input to org_forward. This h() is provided for
            compatibility but bypass_forward() should be used for correct behavior.

            Does not access original model weights - bypass mode is designed
            for quantized models where weights may not be accessible.

        Args:
            x: Input tensor
            base_out: Output from base forward (unused, for API consistency)
        """
        _, b_out = self._compute_paths(x)
        return b_out
