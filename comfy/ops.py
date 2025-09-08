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
import logging
import comfy.model_management
import comfy.float
import comfy.rmsnorm
import contextlib
from comfy.quant_tensor import Q_TYPES, tensor_quantizer, tensor_dequantizer, dynamic_tensor_quantizer, woq_fwd, quantized_fwd, get_quantizer_with_constraints
import types
import inspect

def scaled_dot_product_attention(q, k, v, *args, **kwargs):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, *args, **kwargs)

try:
    if torch.cuda.is_available():
        from torch.nn.attention import SDPBackend, sdpa_kernel
        import inspect
        if "set_priority" in inspect.signature(sdpa_kernel).parameters:
            SDPA_BACKEND_PRIORITY = [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]

            SDPA_BACKEND_PRIORITY.insert(0, SDPBackend.CUDNN_ATTENTION)

            def scaled_dot_product_attention(q, k, v, *args, **kwargs):
                with sdpa_kernel(SDPA_BACKEND_PRIORITY, set_priority=True):
                    return torch.nn.functional.scaled_dot_product_attention(q, k, v, *args, **kwargs)
        else:
            logging.warning("Torch version too old to set sdpa backend priority.")
except (ModuleNotFoundError, TypeError):
    logging.warning("Could not set sdpa backend priority.")

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

    offload_stream = comfy.model_management.get_offload_stream(device)
    if offload_stream is not None:
        wf_context = offload_stream
    else:
        wf_context = contextlib.nullcontext()

    bias = None
    non_blocking = comfy.model_management.device_supports_non_blocking(device)
    if s.bias is not None:
        has_function = len(s.bias_function) > 0
        bias = comfy.model_management.cast_to(s.bias, bias_dtype, device, non_blocking=non_blocking, copy=has_function, stream=offload_stream)

        if has_function:
            with wf_context:
                for f in s.bias_function:
                    bias = f(bias)

    has_function = len(s.weight_function) > 0
    weight = comfy.model_management.cast_to(s.weight, dtype, device, non_blocking=non_blocking, copy=has_function, stream=offload_stream)
    if has_function:
        with wf_context:
            for f in s.weight_function:
                weight = f(weight)

    comfy.model_management.sync_stream(device, offload_stream)
    return weight, bias

class CastWeightBiasOp:
    comfy_cast_weights = False
    weight_function = []
    bias_function = []
    fp8_compute: bool = False
    use_dynamic_quantizer: bool = False

class disable_weight_init:
    class Linear(torch.nn.Module, CastWeightBiasOp):
        def __init__(
                self,
                in_features: int,
                out_features: int,
                bias: bool = True,
                device=None,
                dtype=None,
        ) -> None:
            factory_kwargs = {"device": device, "dtype": dtype}
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features

            self.device = torch.device("cpu") if device is None else device
            self.compute_dtype = torch.float32 if dtype is None else dtype

            if bias:
                self.bias = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
            else:
                self.register_parameter("bias", None)

        def reset_parameters(self):
            return None

        def _set_quantizer_fn(self, scale_weight, scale_input):
            if scale_weight.ndim != 0 and scale_weight.shape[0] != 1:
                raise ValueError("Blockwise quantization is not supported")
            if scale_input is None and self.use_dynamic_quantizer:
                setattr(self, "quantizer", dynamic_tensor_quantizer)
            else:
                setattr(self, "quantizer", tensor_quantizer)

        def _set_dequantizer_fn(self, scale_weight):
            if scale_weight.ndim != 0 and scale_weight.shape[0] != 1:
                raise ValueError("Blockwise quantization is not supported")
            setattr(self, "dequantizer", tensor_dequantizer)

        def _set_quantized_forward(self):
            if not self.fp8_compute:
                q_fwd = woq_fwd
            else:
                q_fwd = quantized_fwd
            self.forward = types.MethodType(q_fwd, self)

        def _init_parameters_from_sd(self, state_dict, prefix):
            if not state_dict:
                logging.warning(f"No state dict provided for {prefix}.")
                weight = torch.nn.Parameter(
                    torch.empty((self.out_features, self.in_features), dtype=self.compute_dtype, device=self.device)
                )
                self.register_buffer('weight', weight)
                return

            scale_weight = None
            _w = state_dict.pop(f"{prefix}weight")
            if len(self.weight_function):
                _w, scale_weight = self.weight_function[0](_w)
            state_dict[f"{prefix}weight"] = _w
            if scale_weight is not None:
                state_dict[f"{prefix}scale_weight"] = scale_weight

            weight_dtype = _w.dtype
            weight = torch.nn.Parameter(
                torch.empty((self.out_features, self.in_features), device=self.device, dtype=weight_dtype))

            self.register_buffer('weight', weight)
            if weight_dtype not in Q_TYPES:
                return

            scale_weight = state_dict.get(f"{prefix}scale_weight", None)
            if scale_weight is None:
                logging.warning("Using quantized weights without a scale can result in low accuracy.")
                scale_weight = torch.ones(1)
                state_dict[f"{prefix}scale_weight"] = scale_weight
            self.register_buffer('scale_weight', scale_weight.to(device=self.device))

            scale_input = state_dict.get(f"{prefix}scale_input", None)
            if scale_input is not None:
                self.register_buffer('scale_input', scale_input.to(device=self.device))
            elif not self.use_dynamic_quantizer:
                # Fallback to WoQ
                self.fp8_compute = False

            if self.bias is not None:
                # WAR not really nice, but Qwen VL has an input scale but uses f32 intermediates and quantized bias
                self.fp8_compute = self.fp8_compute and (self.bias.dtype in [torch.float16, torch.bfloat16])

            self._set_quantizer_fn(scale_weight, scale_input)
            self._set_dequantizer_fn(scale_weight)
            self._set_quantized_forward()

        def _load_from_state_dict(
                self,
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
        ):
            self._init_parameters_from_sd(state_dict, prefix)

            return super()._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )

        def forward(self, input):
            weight = self.weight
            bias = self.bias
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.linear(input, weight, bias)

    class Conv1d(torch.nn.Conv1d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return self._conv_forward(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
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
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
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
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
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
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
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
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class RMSNorm(comfy.rmsnorm.RMSNorm, CastWeightBiasOp):
        def reset_parameters(self):
            self.bias = None
            return None

        def forward_comfy_cast_weights(self, input):
            if self.weight is not None:
                weight, bias = cast_bias_weight(self, input)
            else:
                weight = None
            return comfy.rmsnorm.rms_norm(input, weight, self.eps)  # TODO: switch to commented out line when old torch is deprecated
            # return torch.nn.functional.rms_norm(input, self.normalized_shape, weight, self.eps)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
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
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
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
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
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
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
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

CUBLAS_IS_AVAILABLE = False
try:
    from cublas_ops import CublasLinear
    CUBLAS_IS_AVAILABLE = True
except ImportError:
    pass

if CUBLAS_IS_AVAILABLE: # TODO check if this is actually faster I call BS
    class cublas_ops(disable_weight_init):
        class Linear(CublasLinear, disable_weight_init.Linear):
            def reset_parameters(self):
                return None

            def forward_comfy_cast_weights(self, input):
                return super().forward(input)

            def forward(self, *args, **kwargs):
                return super().forward(*args, **kwargs)

op_class_list = [
    cls for name, cls in inspect.getmembers(disable_weight_init, inspect.isclass)
    if cls.__module__ == disable_weight_init.__module__ and name != "__class__"
]

def operator_factory(**factory_kwargs):
    class OpSet:
        pass

    for k, v in factory_kwargs.items():
        assert hasattr(CastWeightBiasOp, k)

    for base_class in op_class_list:
        new_class = type(base_class.__name__, (base_class,), factory_kwargs)
        setattr(op_set, base_class.__name__, new_class)

    return op_set

def pick_operations(weight_dtype=None, compute_dtype=None, load_device=None, fast_fp8=False, disable_fast_fp8=False):
    fp8_compute = (comfy.model_management.supports_fp8_compute(load_device) and not disable_fast_fp8)
    use_dynamic_quantizer = fast_fp8
    manual_cast = not((weight_dtype == compute_dtype) or use_dynamic_quantizer or fp8_compute)

    weight_function = []
    if weight_dtype is not None and compute_dtype is not None and not manual_cast:
        weight_function = [get_quantizer_with_constraints(weight_dtype)]
    return operator_factory(comfy_cast_weights=manual_cast, use_dynamic_quantizer=use_dynamic_quantizer, fp8_compute=fp8_compute, weight_function=weight_function)
