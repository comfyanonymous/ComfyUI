"""
    This file is part of ComfyUI.
    Copyright (C) 2024 Stability AI

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import torch
import comfy.model_management
from comfy.cli_args import args
import comfy.float

cast_to = comfy.model_management.cast_to #TODO: remove once no more references

def cast_to_input(weight, input, non_blocking=False, copy=True):
    return comfy.model_management.cast_to(weight, input.dtype, input.device, non_blocking=non_blocking, copy=copy)

def cast_bias_weight(s, input=None, dtype=None, device=None, bias_dtype=None):
    if input is not None:
        if dtype is None:
            dtype = input.dtype
        if bias_dtype is None:
            bias_dtype = dtype
        if device is None:
            device = input.device

    bias = None
    non_blocking = comfy.model_management.device_supports_non_blocking(device)
    if s.bias is not None:
        has_function = s.bias_function is not None
        bias = comfy.model_management.cast_to(s.bias, bias_dtype, device, non_blocking=non_blocking, copy=has_function)
        if has_function:
            bias = s.bias_function(bias)

    has_function = s.weight_function is not None
    weight = comfy.model_management.cast_to(s.weight, dtype, device, non_blocking=non_blocking, copy=has_function)
    if has_function:
        weight = s.weight_function(weight)
    return weight, bias

class CastWeightBiasOp:
    comfy_cast_weights = False
    weight_function = None
    bias_function = None

class disable_weight_init:
    class Linear(torch.nn.Linear, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.linear(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Conv1d(torch.nn.Conv1d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return self._conv_forward(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Conv2d(torch.nn.Conv2d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return self._conv_forward(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Conv3d(torch.nn.Conv3d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return self._conv_forward(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class GroupNorm(torch.nn.GroupNorm, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.group_norm(input, self.num_groups, weight, bias, self.eps)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)


    class LayerNorm(torch.nn.LayerNorm, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            if self.weight is not None:
                weight, bias = cast_bias_weight(self, input)
            else:
                weight = None
                bias = None
            return torch.nn.functional.layer_norm(input, self.normalized_shape, weight, bias, self.eps)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class ConvTranspose2d(torch.nn.ConvTranspose2d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input, output_size=None):
            num_spatial_dims = 2
            output_padding = self._output_padding(
                input, output_size, self.stride, self.padding, self.kernel_size,
                num_spatial_dims, self.dilation)

            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.conv_transpose2d(
                input, weight, bias, self.stride, self.padding,
                output_padding, self.groups, self.dilation)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class ConvTranspose1d(torch.nn.ConvTranspose1d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input, output_size=None):
            num_spatial_dims = 1
            output_padding = self._output_padding(
                input, output_size, self.stride, self.padding, self.kernel_size,
                num_spatial_dims, self.dilation)

            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.conv_transpose1d(
                input, weight, bias, self.stride, self.padding,
                output_padding, self.groups, self.dilation)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Embedding(torch.nn.Embedding, CastWeightBiasOp):
        def reset_parameters(self):
            self.bias = None
            return None

        def forward_comfy_cast_weights(self, input, out_dtype=None):
            output_dtype = out_dtype
            if self.weight.dtype == torch.float16 or self.weight.dtype == torch.bfloat16:
                out_dtype = None
            weight, bias = cast_bias_weight(self, device=input.device, dtype=out_dtype)
            return torch.nn.functional.embedding(input, weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse).to(dtype=output_dtype)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                if "out_dtype" in kwargs:
                    kwargs.pop("out_dtype")
                return super().forward(*args, **kwargs)

    @classmethod
    def conv_nd(s, dims, *args, **kwargs):
        if dims == 2:
            return s.Conv2d(*args, **kwargs)
        elif dims == 3:
            return s.Conv3d(*args, **kwargs)
        else:
            raise ValueError(f"unsupported dimensions: {dims}")


class manual_cast(disable_weight_init):
    class Linear(disable_weight_init.Linear):
        comfy_cast_weights = True

    class Conv1d(disable_weight_init.Conv1d):
        comfy_cast_weights = True

    class Conv2d(disable_weight_init.Conv2d):
        comfy_cast_weights = True

    class Conv3d(disable_weight_init.Conv3d):
        comfy_cast_weights = True

    class GroupNorm(disable_weight_init.GroupNorm):
        comfy_cast_weights = True

    class LayerNorm(disable_weight_init.LayerNorm):
        comfy_cast_weights = True

    class ConvTranspose2d(disable_weight_init.ConvTranspose2d):
        comfy_cast_weights = True

    class ConvTranspose1d(disable_weight_init.ConvTranspose1d):
        comfy_cast_weights = True

    class Embedding(disable_weight_init.Embedding):
        comfy_cast_weights = True


def fp8_linear(self, input):
    dtype = self.weight.dtype
    if dtype not in [torch.float8_e4m3fn]:
        return None

    tensor_2d = False
    if len(input.shape) == 2:
        tensor_2d = True
        input = input.unsqueeze(1)


    if len(input.shape) == 3:
        w, bias = cast_bias_weight(self, input, dtype=dtype, bias_dtype=input.dtype)
        w = w.t()

        scale_weight = self.scale_weight
        scale_input = self.scale_input
        if scale_weight is None:
            scale_weight = torch.ones((), device=input.device, dtype=torch.float32)
        else:
            scale_weight = scale_weight.to(input.device)

        if scale_input is None:
            scale_input = torch.ones((), device=input.device, dtype=torch.float32)
            inn = input.reshape(-1, input.shape[2]).to(dtype)
        else:
            scale_input = scale_input.to(input.device)
            inn = (input * (1.0 / scale_input).to(input.dtype)).reshape(-1, input.shape[2]).to(dtype)

        if bias is not None:
            o = torch._scaled_mm(inn, w, out_dtype=input.dtype, bias=bias, scale_a=scale_input, scale_b=scale_weight)
        else:
            o = torch._scaled_mm(inn, w, out_dtype=input.dtype, scale_a=scale_input, scale_b=scale_weight)

        if isinstance(o, tuple):
            o = o[0]

        if tensor_2d:
            return o.reshape(input.shape[0], -1)

        return o.reshape((-1, input.shape[1], self.weight.shape[0]))

    return None

class fp8_ops(manual_cast):
    class Linear(manual_cast.Linear):
        def reset_parameters(self):
            self.scale_weight = None
            self.scale_input = None
            return None

        def forward_comfy_cast_weights(self, input):
            out = fp8_linear(self, input)
            if out is not None:
                return out

            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.linear(input, weight, bias)

def scaled_fp8_ops(fp8_matrix_mult=False, scale_input=False, override_dtype=None):
    class scaled_fp8_op(manual_cast):
        class Linear(manual_cast.Linear):
            def __init__(self, *args, **kwargs):
                if override_dtype is not None:
                    kwargs['dtype'] = override_dtype
                super().__init__(*args, **kwargs)

            def reset_parameters(self):
                if not hasattr(self, 'scale_weight'):
                    self.scale_weight = torch.nn.parameter.Parameter(data=torch.ones((), device=self.weight.device, dtype=torch.float32), requires_grad=False)

                if not scale_input:
                    self.scale_input = None

                if not hasattr(self, 'scale_input'):
                    self.scale_input = torch.nn.parameter.Parameter(data=torch.ones((), device=self.weight.device, dtype=torch.float32), requires_grad=False)
                return None

            def forward_comfy_cast_weights(self, input):
                if fp8_matrix_mult:
                    out = fp8_linear(self, input)
                    if out is not None:
                        return out

                weight, bias = cast_bias_weight(self, input)

                if weight.numel() < input.numel(): #TODO: optimize
                    return torch.nn.functional.linear(input, weight * self.scale_weight.to(device=weight.device, dtype=weight.dtype), bias)
                else:
                    return torch.nn.functional.linear(input * self.scale_weight.to(device=weight.device, dtype=weight.dtype), weight, bias)

            def convert_weight(self, weight, inplace=False, **kwargs):
                if inplace:
                    weight *= self.scale_weight.to(device=weight.device, dtype=weight.dtype)
                    return weight
                else:
                    return weight * self.scale_weight.to(device=weight.device, dtype=weight.dtype)

            def set_weight(self, weight, inplace_update=False, seed=None, **kwargs):
                weight = comfy.float.stochastic_rounding(weight / self.scale_weight.to(device=weight.device, dtype=weight.dtype), self.weight.dtype, seed=seed)
                if inplace_update:
                    self.weight.data.copy_(weight)
                else:
                    self.weight = torch.nn.Parameter(weight, requires_grad=False)

    return scaled_fp8_op

def pick_operations(weight_dtype, compute_dtype, load_device=None, disable_fast_fp8=False, fp8_optimizations=False, scaled_fp8=None):
    fp8_compute = comfy.model_management.supports_fp8_compute(load_device)
    if scaled_fp8 is not None:
        return scaled_fp8_ops(fp8_matrix_mult=fp8_compute, scale_input=True, override_dtype=scaled_fp8)

    if fp8_compute and (fp8_optimizations or args.fast) and not disable_fast_fp8:
        return fp8_ops

    if compute_dtype is None or weight_dtype == compute_dtype:
        return disable_weight_init

    return manual_cast
