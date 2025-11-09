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
from comfy.cli_args import args, PerformanceFeature
import comfy.float
import comfy.rmsnorm
import contextlib

def run_every_op():
    if torch.compiler.is_compiling():
        return

    comfy.model_management.throw_exception_if_processing_interrupted()

def scaled_dot_product_attention(q, k, v, *args, **kwargs):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, *args, **kwargs)


try:
    if torch.cuda.is_available() and comfy.model_management.WINDOWS:
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

NVIDIA_MEMORY_CONV_BUG_WORKAROUND = False
try:
    if comfy.model_management.is_nvidia():
        if torch.backends.cudnn.version() >= 91002 and comfy.model_management.torch_version_numeric >= (2, 9) and comfy.model_management.torch_version_numeric <= (2, 10):
            #TODO: change upper bound version once it's fixed'
            NVIDIA_MEMORY_CONV_BUG_WORKAROUND = True
            logging.info("working around nvidia conv3d memory bug.")
except:
    pass

cast_to = comfy.model_management.cast_to #TODO: remove once no more references

def cast_to_input(weight, input, non_blocking=False, copy=True):
    return comfy.model_management.cast_to(weight, input.dtype, input.device, non_blocking=non_blocking, copy=copy)


def cast_bias_weight(s, input=None, dtype=None, device=None, bias_dtype=None, offloadable=False):
    # NOTE: offloadable=False is a a legacy and if you are a custom node author reading this please pass
    # offloadable=True and call uncast_bias_weight() after your last usage of the weight/bias. This
    # will add async-offload support to your cast and improve performance.
    if input is not None:
        if dtype is None:
            dtype = input.dtype
        if bias_dtype is None:
            bias_dtype = dtype
        if device is None:
            device = input.device

    if offloadable and (device != s.weight.device or
                        (s.bias is not None and device != s.bias.device)):
        offload_stream = comfy.model_management.get_offload_stream(device)
    else:
        offload_stream = None

    if offload_stream is not None:
        wf_context = offload_stream
    else:
        wf_context = contextlib.nullcontext()

    non_blocking = comfy.model_management.device_supports_non_blocking(device)

    weight_has_function = len(s.weight_function) > 0
    bias_has_function = len(s.bias_function) > 0

    weight = comfy.model_management.cast_to(s.weight, None, device, non_blocking=non_blocking, copy=weight_has_function, stream=offload_stream)

    bias = None
    if s.bias is not None:
        bias = comfy.model_management.cast_to(s.bias, bias_dtype, device, non_blocking=non_blocking, copy=bias_has_function, stream=offload_stream)

        if bias_has_function:
            with wf_context:
                for f in s.bias_function:
                    bias = f(bias)

    weight = weight.to(dtype=dtype)
    if weight_has_function:
        with wf_context:
            for f in s.weight_function:
                weight = f(weight)

    comfy.model_management.sync_stream(device, offload_stream)
    if offloadable:
        return weight, bias, offload_stream
    else:
        #Legacy function signature
        return weight, bias


def uncast_bias_weight(s, weight, bias, offload_stream):
    if offload_stream is None:
        return
    if weight is not None:
        device = weight.device
    else:
        if bias is None:
            return
        device = bias.device
    offload_stream.wait_stream(comfy.model_management.current_stream(device))


class CastWeightBiasOp:
    comfy_cast_weights = False
    weight_function = []
    bias_function = []

class disable_weight_init:
    class Linear(torch.nn.Linear, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias, offload_stream = cast_bias_weight(self, input, offloadable=True)
            x = torch.nn.functional.linear(input, weight, bias)
            uncast_bias_weight(self, weight, bias, offload_stream)
            return x

        def forward(self, *args, **kwargs):
            run_every_op()
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Conv1d(torch.nn.Conv1d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias, offload_stream = cast_bias_weight(self, input, offloadable=True)
            x = self._conv_forward(input, weight, bias)
            uncast_bias_weight(self, weight, bias, offload_stream)
            return x

        def forward(self, *args, **kwargs):
            run_every_op()
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Conv2d(torch.nn.Conv2d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias, offload_stream = cast_bias_weight(self, input, offloadable=True)
            x = self._conv_forward(input, weight, bias)
            uncast_bias_weight(self, weight, bias, offload_stream)
            return x

        def forward(self, *args, **kwargs):
            run_every_op()
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Conv3d(torch.nn.Conv3d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def _conv_forward(self, input, weight, bias, *args, **kwargs):
            if NVIDIA_MEMORY_CONV_BUG_WORKAROUND and weight.dtype in (torch.float16, torch.bfloat16):
                out = torch.cudnn_convolution(input, weight, self.padding, self.stride, self.dilation, self.groups, benchmark=False, deterministic=False, allow_tf32=True)
                if bias is not None:
                    out += bias.reshape((1, -1) + (1,) * (out.ndim - 2))
                return out
            else:
                return super()._conv_forward(input, weight, bias, *args, **kwargs)

        def forward_comfy_cast_weights(self, input):
            weight, bias, offload_stream = cast_bias_weight(self, input, offloadable=True)
            x = self._conv_forward(input, weight, bias)
            uncast_bias_weight(self, weight, bias, offload_stream)
            return x

        def forward(self, *args, **kwargs):
            run_every_op()
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class GroupNorm(torch.nn.GroupNorm, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias, offload_stream = cast_bias_weight(self, input, offloadable=True)
            x = torch.nn.functional.group_norm(input, self.num_groups, weight, bias, self.eps)
            uncast_bias_weight(self, weight, bias, offload_stream)
            return x

        def forward(self, *args, **kwargs):
            run_every_op()
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class LayerNorm(torch.nn.LayerNorm, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            if self.weight is not None:
                weight, bias, offload_stream = cast_bias_weight(self, input, offloadable=True)
            else:
                weight = None
                bias = None
                offload_stream = None
            x = torch.nn.functional.layer_norm(input, self.normalized_shape, weight, bias, self.eps)
            uncast_bias_weight(self, weight, bias, offload_stream)
            return x

        def forward(self, *args, **kwargs):
            run_every_op()
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
                weight, bias, offload_stream = cast_bias_weight(self, input, offloadable=True)
            else:
                weight = None
                bias = None
                offload_stream = None
            x = comfy.rmsnorm.rms_norm(input, weight, self.eps)  # TODO: switch to commented out line when old torch is deprecated
            # x = torch.nn.functional.rms_norm(input, self.normalized_shape, weight, self.eps)
            uncast_bias_weight(self, weight, bias, offload_stream)
            return x

        def forward(self, *args, **kwargs):
            run_every_op()
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

            weight, bias, offload_stream = cast_bias_weight(self, input, offloadable=True)
            x = torch.nn.functional.conv_transpose2d(
                input, weight, bias, self.stride, self.padding,
                output_padding, self.groups, self.dilation)
            uncast_bias_weight(self, weight, bias, offload_stream)
            return x

        def forward(self, *args, **kwargs):
            run_every_op()
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

            weight, bias, offload_stream = cast_bias_weight(self, input, offloadable=True)
            x = torch.nn.functional.conv_transpose1d(
                input, weight, bias, self.stride, self.padding,
                output_padding, self.groups, self.dilation)
            uncast_bias_weight(self, weight, bias, offload_stream)
            return x

        def forward(self, *args, **kwargs):
            run_every_op()
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
            weight, bias, offload_stream = cast_bias_weight(self, device=input.device, dtype=out_dtype, offloadable=True)
            x = torch.nn.functional.embedding(input, weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse).to(dtype=output_dtype)
            uncast_bias_weight(self, weight, bias, offload_stream)
            return x


        def forward(self, *args, **kwargs):
            run_every_op()
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

    class RMSNorm(disable_weight_init.RMSNorm):
        comfy_cast_weights = True

    class Embedding(disable_weight_init.Embedding):
        comfy_cast_weights = True


def fp8_linear(self, input):
    """
    Legacy FP8 linear function for backward compatibility.
    Uses QuantizedTensor subclass for dispatch.
    """
    dtype = self.weight.dtype
    if dtype not in [torch.float8_e4m3fn]:
        return None

    input_dtype = input.dtype

    if input.ndim == 3 or input.ndim == 2:
        w, bias, offload_stream = cast_bias_weight(self, input, dtype=dtype, bias_dtype=input_dtype, offloadable=True)

        scale_weight = self.scale_weight
        scale_input = self.scale_input
        if scale_weight is None:
            scale_weight = torch.ones((), device=input.device, dtype=torch.float32)
        else:
            scale_weight = scale_weight.to(input.device)

        if scale_input is None:
            scale_input = torch.ones((), device=input.device, dtype=torch.float32)
            input = torch.clamp(input, min=-448, max=448, out=input)
            layout_params_weight = {'scale': scale_input, 'orig_dtype': input_dtype}
            quantized_input = QuantizedTensor(input.to(dtype).contiguous(), "TensorCoreFP8Layout", layout_params_weight)
        else:
            scale_input = scale_input.to(input.device)
            quantized_input = QuantizedTensor.from_float(input, "TensorCoreFP8Layout", scale=scale_input, dtype=dtype)

        # Wrap weight in QuantizedTensor - this enables unified dispatch
        # Call F.linear - __torch_dispatch__ routes to fp8_linear handler in quant_ops.py!
        layout_params_weight = {'scale': scale_weight, 'orig_dtype': input_dtype}
        quantized_weight = QuantizedTensor(w, "TensorCoreFP8Layout", layout_params_weight)
        o = torch.nn.functional.linear(quantized_input, quantized_weight, bias)

        uncast_bias_weight(self, w, bias, offload_stream)
        return o

    return None

class fp8_ops(manual_cast):
    class Linear(manual_cast.Linear):
        def reset_parameters(self):
            self.scale_weight = None
            self.scale_input = None
            return None

        def forward_comfy_cast_weights(self, input):
            if not self.training:
                try:
                    out = fp8_linear(self, input)
                    if out is not None:
                        return out
                except Exception as e:
                    logging.info("Exception during fp8 op: {}".format(e))

            weight, bias, offload_stream = cast_bias_weight(self, input, offloadable=True)
            x = torch.nn.functional.linear(input, weight, bias)
            uncast_bias_weight(self, weight, bias, offload_stream)
            return x

def scaled_fp8_ops(fp8_matrix_mult=False, scale_input=False, override_dtype=None):
    logging.info("Using scaled fp8: fp8 matrix mult: {}, scale input: {}".format(fp8_matrix_mult, scale_input))
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

                weight, bias, offload_stream = cast_bias_weight(self, input, offloadable=True)

                if weight.numel() < input.numel(): #TODO: optimize
                    x = torch.nn.functional.linear(input, weight * self.scale_weight.to(device=weight.device, dtype=weight.dtype), bias)
                else:
                    x = torch.nn.functional.linear(input * self.scale_weight.to(device=weight.device, dtype=weight.dtype), weight, bias)
                uncast_bias_weight(self, weight, bias, offload_stream)
                return x

            def convert_weight(self, weight, inplace=False, **kwargs):
                if inplace:
                    weight *= self.scale_weight.to(device=weight.device, dtype=weight.dtype)
                    return weight
                else:
                    return weight * self.scale_weight.to(device=weight.device, dtype=weight.dtype)

            def set_weight(self, weight, inplace_update=False, seed=None, return_weight=False, **kwargs):
                weight = comfy.float.stochastic_rounding(weight / self.scale_weight.to(device=weight.device, dtype=weight.dtype), self.weight.dtype, seed=seed)
                if return_weight:
                    return weight
                if inplace_update:
                    self.weight.data.copy_(weight)
                else:
                    self.weight = torch.nn.Parameter(weight, requires_grad=False)

    return scaled_fp8_op

CUBLAS_IS_AVAILABLE = False
try:
    from cublas_ops import CublasLinear
    CUBLAS_IS_AVAILABLE = True
except ImportError:
    pass

if CUBLAS_IS_AVAILABLE:
    class cublas_ops(disable_weight_init):
        class Linear(CublasLinear, disable_weight_init.Linear):
            def reset_parameters(self):
                return None

            def forward_comfy_cast_weights(self, input):
                return super().forward(input)

            def forward(self, *args, **kwargs):
                return super().forward(*args, **kwargs)


# ==============================================================================
# Mixed Precision Operations
# ==============================================================================
from .quant_ops import QuantizedTensor

QUANT_FORMAT_MIXINS = {
    "float8_e4m3fn": {
        "dtype": torch.float8_e4m3fn,
        "layout_type": "TensorCoreFP8Layout",
        "parameters": {
            "weight_scale": torch.nn.Parameter(torch.zeros((), dtype=torch.float32), requires_grad=False),
            "input_scale": torch.nn.Parameter(torch.zeros((), dtype=torch.float32), requires_grad=False),
        }
    }
}

class MixedPrecisionOps(disable_weight_init):
    _layer_quant_config = {}
    _compute_dtype = torch.bfloat16

    class Linear(torch.nn.Module, CastWeightBiasOp):
        def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            device=None,
            dtype=None,
        ) -> None:
            super().__init__()

            self.factory_kwargs = {"device": device, "dtype": MixedPrecisionOps._compute_dtype}
            # self.factory_kwargs = {"device": device, "dtype": dtype}

            self.in_features = in_features
            self.out_features = out_features
            if bias:
                self.bias = torch.nn.Parameter(torch.empty(out_features, **self.factory_kwargs))
            else:
                self.register_parameter("bias", None)

            self.tensor_class = None

        def reset_parameters(self):
            return None

        def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                                  strict, missing_keys, unexpected_keys, error_msgs):

            device = self.factory_kwargs["device"]
            layer_name = prefix.rstrip('.')
            weight_key = f"{prefix}weight"
            weight = state_dict.pop(weight_key, None)
            if weight is None:
                raise ValueError(f"Missing weight for layer {layer_name}")

            manually_loaded_keys = [weight_key]

            if layer_name not in MixedPrecisionOps._layer_quant_config:
                self.weight = torch.nn.Parameter(weight.to(device=device, dtype=MixedPrecisionOps._compute_dtype), requires_grad=False)
            else:
                quant_format = MixedPrecisionOps._layer_quant_config[layer_name].get("format", None)
                if quant_format is None:
                    raise ValueError(f"Unknown quantization format for layer {layer_name}")

                mixin = QUANT_FORMAT_MIXINS[quant_format]
                self.layout_type = mixin["layout_type"]

                scale_key = f"{prefix}weight_scale"
                layout_params = {
                    'scale': state_dict.pop(scale_key, None),
                    'orig_dtype': MixedPrecisionOps._compute_dtype
                }
                if layout_params['scale'] is not None:
                    manually_loaded_keys.append(scale_key)

                self.weight = torch.nn.Parameter(
                    QuantizedTensor(weight.to(device=device, dtype=mixin["dtype"]), self.layout_type, layout_params),
                    requires_grad=False
                )

                for param_name, param_value in mixin["parameters"].items():
                    param_key = f"{prefix}{param_name}"
                    _v = state_dict.pop(param_key, None)
                    if _v is None:
                        continue
                    setattr(self, param_name, torch.nn.Parameter(_v.to(device=device), requires_grad=False))
                    manually_loaded_keys.append(param_key)

            super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

            for key in manually_loaded_keys:
                if key in missing_keys:
                    missing_keys.remove(key)

        def _forward(self, input, weight, bias):
            return torch.nn.functional.linear(input, weight, bias)

        def forward_comfy_cast_weights(self, input):
            weight, bias, offload_stream = cast_bias_weight(self, input, offloadable=True)
            x = self._forward(input, weight, bias)
            uncast_bias_weight(self, weight, bias, offload_stream)
            return x

        def forward(self, input, *args, **kwargs):
            run_every_op()

            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(input, *args, **kwargs)
            if (getattr(self, 'layout_type', None) is not None and
                getattr(self, 'input_scale', None) is not None and
                not isinstance(input, QuantizedTensor)):
                input = QuantizedTensor.from_float(input, self.layout_type, scale=self.input_scale, fp8_dtype=self.weight.dtype)
            return self._forward(input, self.weight, self.bias)


def pick_operations(weight_dtype, compute_dtype, load_device=None, disable_fast_fp8=False, fp8_optimizations=False, scaled_fp8=None, model_config=None):
    if model_config and hasattr(model_config, 'layer_quant_config') and model_config.layer_quant_config:
        MixedPrecisionOps._layer_quant_config = model_config.layer_quant_config
        MixedPrecisionOps._compute_dtype = compute_dtype
        logging.info(f"Using mixed precision operations: {len(model_config.layer_quant_config)} quantized layers")
        return MixedPrecisionOps

    fp8_compute = comfy.model_management.supports_fp8_compute(load_device)
    if scaled_fp8 is not None:
        return scaled_fp8_ops(fp8_matrix_mult=fp8_compute and fp8_optimizations, scale_input=fp8_optimizations, override_dtype=scaled_fp8)

    if (
        fp8_compute and
        (fp8_optimizations or PerformanceFeature.Fp8MatrixMultiplication in args.fast) and
        not disable_fast_fp8
    ):
        return fp8_ops

    if (
        PerformanceFeature.CublasOps in args.fast and
        CUBLAS_IS_AVAILABLE and
        weight_dtype == torch.float16 and
        (compute_dtype == torch.float16 or compute_dtype is None)
    ):
        logging.info("Using cublas ops")
        return cublas_ops

    if compute_dtype is None or weight_dtype == compute_dtype:
        return disable_weight_init

    return manual_cast
