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
import json
import comfy.memory_management
import comfy.pinned_memory
import comfy.utils

import comfy_aimdo.model_vbar
import comfy_aimdo.torch

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
                if q.nelement() < 1024 * 128:  # arbitrary number, for small inputs cudnn attention seems slower
                    return torch.nn.functional.scaled_dot_product_attention(q, k, v, *args, **kwargs)
                with sdpa_kernel(SDPA_BACKEND_PRIORITY, set_priority=True):
                    return torch.nn.functional.scaled_dot_product_attention(q, k, v, *args, **kwargs)
        else:
            logging.warning("Torch version too old to set sdpa backend priority.")
except (ModuleNotFoundError, TypeError):
    logging.warning("Could not set sdpa backend priority.")

NVIDIA_MEMORY_CONV_BUG_WORKAROUND = False
try:
    if comfy.model_management.is_nvidia():
        cudnn_version = torch.backends.cudnn.version()
        if (cudnn_version >= 91002 and cudnn_version < 91500) and comfy.model_management.torch_version_numeric >= (2, 9) and comfy.model_management.torch_version_numeric <= (2, 10):
            #TODO: change upper bound version once it's fixed'
            NVIDIA_MEMORY_CONV_BUG_WORKAROUND = True
            logging.info("working around nvidia conv3d memory bug.")
except:
    pass

cast_to = comfy.model_management.cast_to #TODO: remove once no more references

def cast_to_input(weight, input, non_blocking=False, copy=True):
    return comfy.model_management.cast_to(weight, input.dtype, input.device, non_blocking=non_blocking, copy=copy)


def cast_bias_weight_with_vbar(s, dtype, device, bias_dtype, non_blocking, compute_dtype, want_requant):
    offload_stream = None
    xfer_dest = None

    signature = comfy_aimdo.model_vbar.vbar_fault(s._v)
    resident = comfy_aimdo.model_vbar.vbar_signature_compare(signature, s._v_signature)
    if signature is not None:
        if resident:
            weight = s._v_weight
            bias = s._v_bias
        else:
            xfer_dest = comfy_aimdo.torch.aimdo_to_tensor(s._v, device)

    if not resident:
        cast_geometry = comfy.memory_management.tensors_to_geometries([ s.weight, s.bias ])
        cast_dest = None

        xfer_source = [ s.weight, s.bias ]

        pin = comfy.pinned_memory.get_pin(s)
        if pin is not None:
            xfer_source = [ pin ]

        for data, geometry in zip([ s.weight, s.bias ], cast_geometry):
            if data is None:
                continue
            if data.dtype != geometry.dtype:
                cast_dest = xfer_dest
                if cast_dest is None:
                    cast_dest = torch.empty((comfy.memory_management.vram_aligned_size(cast_geometry),), dtype=torch.uint8, device=device)
                xfer_dest = None
                break

        dest_size = comfy.memory_management.vram_aligned_size(xfer_source)
        offload_stream = comfy.model_management.get_offload_stream(device)
        if xfer_dest is None and offload_stream is not None:
                xfer_dest = comfy.model_management.get_cast_buffer(offload_stream, device, dest_size, s)
                if xfer_dest is None:
                    offload_stream = comfy.model_management.get_offload_stream(device)
                    xfer_dest = comfy.model_management.get_cast_buffer(offload_stream, device, dest_size, s)
        if xfer_dest is None:
            xfer_dest = torch.empty((dest_size,), dtype=torch.uint8, device=device)
            offload_stream = None

        if signature is None and pin is None:
            comfy.pinned_memory.pin_memory(s)
            pin = comfy.pinned_memory.get_pin(s)
        else:
            pin = None

        if pin is not None:
            comfy.model_management.cast_to_gathered(xfer_source, pin)
            xfer_source = [ pin ]
        #send it over
        comfy.model_management.cast_to_gathered(xfer_source, xfer_dest, non_blocking=non_blocking, stream=offload_stream)
        comfy.model_management.sync_stream(device, offload_stream)

        if cast_dest is not None:
            for pre_cast, post_cast in zip(comfy.memory_management.interpret_gathered_like([s.weight, s.bias ], xfer_dest),
                                           comfy.memory_management.interpret_gathered_like(cast_geometry, cast_dest)):
                if post_cast is not None:
                    post_cast.copy_(pre_cast)
            xfer_dest = cast_dest

        params = comfy.memory_management.interpret_gathered_like(cast_geometry, xfer_dest)
        weight = params[0]
        bias = params[1]
        if signature is not None:
            s._v_weight = weight
            s._v_bias = bias
        s._v_signature=signature

    def post_cast(s, param_key, x, dtype, resident, update_weight):
        lowvram_fn = getattr(s, param_key + "_lowvram_function", None)
        fns = getattr(s, param_key + "_function", [])

        orig = x

        def to_dequant(tensor, dtype):
            tensor = tensor.to(dtype=dtype)
            if isinstance(tensor, QuantizedTensor):
                tensor = tensor.dequantize()
            return tensor

        if orig.dtype != dtype or len(fns) > 0:
            x = to_dequant(x, dtype)
        if not resident and lowvram_fn is not None:
            x = to_dequant(x, dtype if compute_dtype is None else compute_dtype)
            x = lowvram_fn(x)
            if (want_requant and len(fns) == 0 or update_weight):
                seed = comfy.utils.string_to_seed(s.seed_key)
                if isinstance(orig, QuantizedTensor):
                    y = QuantizedTensor.from_float(x, s.layout_type, scale="recalculate", stochastic_rounding=seed)
                else:
                    y = comfy.float.stochastic_rounding(x, orig.dtype, seed=seed)
            if want_requant and len(fns) == 0:
                x = y
            if update_weight:
                orig.copy_(y)
        for f in fns:
            x = f(x)
        return x

    update_weight = signature is not None

    weight = post_cast(s, "weight", weight, dtype, resident, update_weight)
    if s.bias is not None:
        bias = post_cast(s, "bias", bias, bias_dtype, resident, update_weight)

    #FIXME: weird offload return protocol
    return weight, bias, (offload_stream, device if signature is not None else None, None)


def cast_bias_weight(s, input=None, dtype=None, device=None, bias_dtype=None, offloadable=False, compute_dtype=None, want_requant=False):
    # NOTE: offloadable=False is a a legacy and if you are a custom node author reading this please pass
    # offloadable=True and call uncast_bias_weight() after your last usage of the weight/bias. This
    # will add async-offload support to your cast and improve performance.
    if input is not None:
        if dtype is None:
            if isinstance(input, QuantizedTensor):
                dtype = input.params.orig_dtype
            else:
                dtype = input.dtype
        if bias_dtype is None:
            bias_dtype = dtype
        if device is None:
            device = input.device

    non_blocking = comfy.model_management.device_supports_non_blocking(device)

    if hasattr(s, "_v"):
        return cast_bias_weight_with_vbar(s, dtype, device, bias_dtype, non_blocking, compute_dtype, want_requant)

    if offloadable and (device != s.weight.device or
                        (s.bias is not None and device != s.bias.device)):
        offload_stream = comfy.model_management.get_offload_stream(device)
    else:
        offload_stream = None

    bias = None
    weight = None

    if offload_stream is not None and not args.cuda_malloc:
        cast_buffer_size = comfy.memory_management.vram_aligned_size([ s.weight, s.bias ])
        cast_buffer = comfy.model_management.get_cast_buffer(offload_stream, device, cast_buffer_size, s)
        #The streams can be uneven in buffer capability and reject us. Retry to get the other stream
        if cast_buffer is None:
            offload_stream = comfy.model_management.get_offload_stream(device)
            cast_buffer = comfy.model_management.get_cast_buffer(offload_stream, device, cast_buffer_size, s)
        params = comfy.memory_management.interpret_gathered_like([ s.weight, s.bias ], cast_buffer)
        weight = params[0]
        bias = params[1]

    weight_has_function = len(s.weight_function) > 0
    bias_has_function = len(s.bias_function) > 0

    weight = comfy.model_management.cast_to(s.weight, None, device, non_blocking=non_blocking, copy=weight_has_function, stream=offload_stream, r=weight)

    if s.bias is not None:
        bias = comfy.model_management.cast_to(s.bias, None, device, non_blocking=non_blocking, copy=bias_has_function, stream=offload_stream, r=bias)

    comfy.model_management.sync_stream(device, offload_stream)

    bias_a = bias
    weight_a = weight

    if s.bias is not None:
        bias = bias.to(dtype=bias_dtype)
        for f in s.bias_function:
            bias = f(bias)

    if weight_has_function or weight.dtype != dtype:
        weight = weight.to(dtype=dtype)
        if isinstance(weight, QuantizedTensor):
            weight = weight.dequantize()
        for f in s.weight_function:
            weight = f(weight)

    if offloadable:
        return weight, bias, (offload_stream, weight_a, bias_a)
    else:
        #Legacy function signature
        return weight, bias


def uncast_bias_weight(s, weight, bias, offload_stream):
    if offload_stream is None:
        return
    os, weight_a, bias_a = offload_stream
    device=None
    #FIXME: This is not good RTTI
    if not isinstance(weight_a, torch.Tensor):
        comfy_aimdo.model_vbar.vbar_unpin(s._v)
        device = weight_a
    if os is None:
        return
    if device is None:
        if weight_a is not None:
            device = weight_a.device
        else:
            if bias_a is None:
                return
            device = bias_a.device
    os.wait_stream(comfy.model_management.current_stream(device))


class CastWeightBiasOp:
    comfy_cast_weights = False
    weight_function = []
    bias_function = []

class disable_weight_init:
    class Linear(torch.nn.Linear, CastWeightBiasOp):

        def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
            if not comfy.model_management.WINDOWS or not comfy.memory_management.aimdo_enabled:
                super().__init__(in_features, out_features, bias, device, dtype)
                return

            # Issue is with `torch.empty` still reserving the full memory for the layer.
            # Windows doesn't over-commit memory so without this, We are momentarily commit
            # charged for the weight even though we might zero-copy it when we load the
            # state dict. If the commit charge exceeds the ceiling we can destabilize the
            # system.
            torch.nn.Module.__init__(self)
            self.in_features = in_features
            self.out_features = out_features
            self.weight = None
            self.bias = None
            self.comfy_need_lazy_init_bias=bias
            self.weight_comfy_model_dtype = dtype
            self.bias_comfy_model_dtype = dtype

        def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                                strict, missing_keys, unexpected_keys, error_msgs):

            if not comfy.model_management.WINDOWS or not comfy.memory_management.aimdo_enabled:
                return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                     missing_keys, unexpected_keys, error_msgs)
            assign_to_params_buffers = local_metadata.get("assign_to_params_buffers", False)
            prefix_len = len(prefix)
            for k,v in state_dict.items():
                if k[prefix_len:] == "weight":
                    if not assign_to_params_buffers:
                        v = v.clone()
                    self.weight = torch.nn.Parameter(v, requires_grad=False)
                elif k[prefix_len:] == "bias" and v is not None:
                    if not assign_to_params_buffers:
                        v = v.clone()
                    self.bias = torch.nn.Parameter(v, requires_grad=False)
                else:
                    unexpected_keys.append(k)

            #Reconcile default construction of the weight if its missing.
            if self.weight is None:
                v = torch.zeros(self.in_features, self.out_features)
                self.weight = torch.nn.Parameter(v, requires_grad=False)
                missing_keys.append(prefix+"weight")
            if self.bias is None and self.comfy_need_lazy_init_bias:
                v = torch.zeros(self.out_features,)
                self.bias = torch.nn.Parameter(v, requires_grad=False)
                missing_keys.append(prefix+"bias")


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

        def _conv_forward(self, input, weight, bias, autopad=None, *args, **kwargs):
            if autopad == "causal_zero":
                weight = weight[:, :, -input.shape[2]:, :, :]
            if NVIDIA_MEMORY_CONV_BUG_WORKAROUND and weight.dtype in (torch.float16, torch.bfloat16):
                out = torch.cudnn_convolution(input, weight, self.padding, self.stride, self.dilation, self.groups, benchmark=False, deterministic=False, allow_tf32=True)
                if bias is not None:
                    out += bias.reshape((1, -1) + (1,) * (out.ndim - 2))
                return out
            else:
                return super()._conv_forward(input, weight, bias, *args, **kwargs)

        def forward_comfy_cast_weights(self, input, autopad=None):
            weight, bias, offload_stream = cast_bias_weight(self, input, offloadable=True)
            x = self._conv_forward(input, weight, bias, autopad=autopad)
            uncast_bias_weight(self, weight, bias, offload_stream)
            return x

        def forward(self, *args, **kwargs):
            run_every_op()
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0 or "autopad" in kwargs:
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

    class RMSNorm(torch.nn.RMSNorm, CastWeightBiasOp):
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
            x = torch.nn.functional.rms_norm(input, self.normalized_shape, weight, self.eps)
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
    input_shape = input.shape
    tensor_3d = input.ndim == 3

    if tensor_3d:
        input = input.reshape(-1, input_shape[2])

    if input.ndim != 2:
        return None
    lora_compute_dtype=comfy.model_management.lora_compute_dtype(input.device)
    w, bias, offload_stream = cast_bias_weight(self, input, dtype=dtype, bias_dtype=input_dtype, offloadable=True, compute_dtype=lora_compute_dtype, want_requant=True)
    scale_weight = torch.ones((), device=input.device, dtype=torch.float32)

    scale_input = torch.ones((), device=input.device, dtype=torch.float32)
    input = torch.clamp(input, min=-448, max=448, out=input)
    input_fp8 = input.to(dtype).contiguous()
    layout_params_input = TensorCoreFP8Layout.Params(scale=scale_input, orig_dtype=input_dtype, orig_shape=tuple(input_fp8.shape))
    quantized_input = QuantizedTensor(input_fp8, "TensorCoreFP8Layout", layout_params_input)

    # Wrap weight in QuantizedTensor - this enables unified dispatch
    # Call F.linear - __torch_dispatch__ routes to fp8_linear handler in quant_ops.py!
    layout_params_weight = TensorCoreFP8Layout.Params(scale=scale_weight, orig_dtype=input_dtype, orig_shape=tuple(w.shape))
    quantized_weight = QuantizedTensor(w, "TensorCoreFP8Layout", layout_params_weight)
    o = torch.nn.functional.linear(quantized_input, quantized_weight, bias)

    uncast_bias_weight(self, w, bias, offload_stream)
    if tensor_3d:
        o = o.reshape((input_shape[0], input_shape[1], w.shape[0]))

    return o

class fp8_ops(manual_cast):
    class Linear(manual_cast.Linear):
        def reset_parameters(self):
            self.scale_weight = None
            self.scale_input = None
            return None

        def forward_comfy_cast_weights(self, input):
            if len(self.weight_function) == 0 and len(self.bias_function) == 0:
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
from .quant_ops import (
    QuantizedTensor,
    QUANT_ALGOS,
    TensorCoreFP8Layout,
    get_layout_class,
)


def mixed_precision_ops(quant_config={}, compute_dtype=torch.bfloat16, full_precision_mm=False, disabled=[]):
    class MixedPrecisionOps(manual_cast):
        _quant_config = quant_config
        _compute_dtype = compute_dtype
        _full_precision_mm = full_precision_mm
        _disabled = disabled

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
                self._full_precision_mm = MixedPrecisionOps._full_precision_mm
                self._full_precision_mm_config = False

            def reset_parameters(self):
                return None

            def _load_scale_param(self, state_dict, prefix, param_name, device, manually_loaded_keys, dtype=None):
                key = f"{prefix}{param_name}"
                value = state_dict.pop(key, None)
                if value is not None:
                    value = value.to(device=device)
                    if dtype is not None:
                        value = value.view(dtype=dtype)
                    manually_loaded_keys.append(key)
                return value

            def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                                    strict, missing_keys, unexpected_keys, error_msgs):

                device = self.factory_kwargs["device"]
                layer_name = prefix.rstrip('.')
                weight_key = f"{prefix}weight"
                weight = state_dict.pop(weight_key, None)
                if weight is None:
                    logging.warning(f"Missing weight for layer {layer_name}")
                    return

                manually_loaded_keys = [weight_key]

                layer_conf = state_dict.pop(f"{prefix}comfy_quant", None)
                if layer_conf is not None:
                    layer_conf = json.loads(layer_conf.numpy().tobytes())

                if layer_conf is None:
                    self.weight = torch.nn.Parameter(weight.to(device=device, dtype=MixedPrecisionOps._compute_dtype), requires_grad=False)
                else:
                    self.quant_format = layer_conf.get("format", None)
                    self._full_precision_mm_config = layer_conf.get("full_precision_matrix_mult", False)
                    if not self._full_precision_mm:
                        self._full_precision_mm = self._full_precision_mm_config

                    if self.quant_format in MixedPrecisionOps._disabled:
                        self._full_precision_mm = True

                    if self.quant_format is None:
                        raise ValueError(f"Unknown quantization format for layer {layer_name}")

                    qconfig = QUANT_ALGOS[self.quant_format]
                    self.layout_type = qconfig["comfy_tensor_layout"]
                    layout_cls = get_layout_class(self.layout_type)

                    # Load format-specific parameters
                    if self.quant_format in ["float8_e4m3fn", "float8_e5m2"]:
                        # FP8: single tensor scale
                        scale = self._load_scale_param(state_dict, prefix, "weight_scale", device, manually_loaded_keys)

                        params = layout_cls.Params(
                            scale=scale,
                            orig_dtype=MixedPrecisionOps._compute_dtype,
                            orig_shape=(self.out_features, self.in_features),
                        )

                    elif self.quant_format == "nvfp4":
                        # NVFP4: tensor_scale (weight_scale_2) + block_scale (weight_scale)
                        tensor_scale = self._load_scale_param(state_dict, prefix, "weight_scale_2", device, manually_loaded_keys)
                        block_scale = self._load_scale_param(state_dict, prefix, "weight_scale", device, manually_loaded_keys,
                                                             dtype=torch.float8_e4m3fn)

                        if tensor_scale is None or block_scale is None:
                            raise ValueError(f"Missing NVFP4 scales for layer {layer_name}")

                        params = layout_cls.Params(
                            scale=tensor_scale,
                            block_scale=block_scale,
                            orig_dtype=MixedPrecisionOps._compute_dtype,
                            orig_shape=(self.out_features, self.in_features),
                        )
                    else:
                        raise ValueError(f"Unsupported quantization format: {self.quant_format}")

                    self.weight = torch.nn.Parameter(
                        QuantizedTensor(weight.to(device=device, dtype=qconfig["storage_t"]), self.layout_type, params),
                        requires_grad=False
                    )

                    for param_name in qconfig["parameters"]:
                        if param_name in {"weight_scale", "weight_scale_2"}:
                            continue  # Already handled above

                        param_key = f"{prefix}{param_name}"
                        _v = state_dict.pop(param_key, None)
                        if _v is None:
                            continue
                        self.register_parameter(param_name, torch.nn.Parameter(_v.to(device=device), requires_grad=False))
                        manually_loaded_keys.append(param_key)

                super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

                for key in manually_loaded_keys:
                    if key in missing_keys:
                        missing_keys.remove(key)

            def state_dict(self, *args, destination=None, prefix="", **kwargs):
                if destination is not None:
                    sd = destination
                else:
                    sd = {}

                if not hasattr(self, 'weight'):
                    logging.warning("Warning: state dict on uninitialized op {}".format(prefix))
                    return sd

                if self.bias is not None:
                    sd["{}bias".format(prefix)] = self.bias

                if isinstance(self.weight, QuantizedTensor):
                    sd_out = self.weight.state_dict("{}weight".format(prefix))
                    for k in sd_out:
                        sd[k] = sd_out[k]

                    quant_conf = {"format": self.quant_format}
                    if self._full_precision_mm_config:
                        quant_conf["full_precision_matrix_mult"] = True
                    sd["{}comfy_quant".format(prefix)] = torch.tensor(list(json.dumps(quant_conf).encode('utf-8')), dtype=torch.uint8)

                    input_scale = getattr(self, 'input_scale', None)
                    if input_scale is not None:
                        sd["{}input_scale".format(prefix)] = input_scale
                else:
                    sd["{}weight".format(prefix)] = self.weight
                return sd

            def _forward(self, input, weight, bias):
                return torch.nn.functional.linear(input, weight, bias)

            def forward_comfy_cast_weights(self, input, compute_dtype=None, want_requant=False):
                weight, bias, offload_stream = cast_bias_weight(self, input, offloadable=True, compute_dtype=compute_dtype, want_requant=want_requant)
                x = self._forward(input, weight, bias)
                uncast_bias_weight(self, weight, bias, offload_stream)
                return x

            def forward(self, input, *args, **kwargs):
                run_every_op()

                input_shape = input.shape
                reshaped_3d = False
                #If cast needs to apply lora, it should be done in the compute dtype
                compute_dtype = input.dtype

                if (getattr(self, 'layout_type', None) is not None and
                    not isinstance(input, QuantizedTensor) and not self._full_precision_mm and
                    not getattr(self, 'comfy_force_cast_weights', False) and
                    len(self.weight_function) == 0 and len(self.bias_function) == 0):

                    # Reshape 3D tensors to 2D for quantization (needed for NVFP4 and others)
                    input_reshaped = input.reshape(-1, input_shape[2]) if input.ndim == 3 else input

                    # Fall back to non-quantized for non-2D tensors
                    if input_reshaped.ndim == 2:
                        reshaped_3d = input.ndim == 3
                        # dtype is now implicit in the layout class
                        scale = getattr(self, 'input_scale', None)
                        if scale is not None:
                            scale = comfy.model_management.cast_to_device(scale, input.device, None)
                        input = QuantizedTensor.from_float(input_reshaped, self.layout_type, scale=scale)

                output = self.forward_comfy_cast_weights(input, compute_dtype, want_requant=isinstance(input, QuantizedTensor))

                # Reshape output back to 3D if input was 3D
                if reshaped_3d:
                    output = output.reshape((input_shape[0], input_shape[1], self.weight.shape[0]))

                return output

            def convert_weight(self, weight, inplace=False, **kwargs):
                if isinstance(weight, QuantizedTensor):
                    return weight.dequantize()
                else:
                    return weight

            def set_weight(self, weight, inplace_update=False, seed=None, return_weight=False, **kwargs):
                if getattr(self, 'layout_type', None) is not None:
                    # dtype is now implicit in the layout class
                    weight = QuantizedTensor.from_float(weight, self.layout_type, scale="recalculate", stochastic_rounding=seed, inplace_ops=True).to(self.weight.dtype)
                else:
                    weight = weight.to(self.weight.dtype)
                if return_weight:
                    return weight

                assert inplace_update is False  # TODO: eventually remove the inplace_update stuff
                self.weight = torch.nn.Parameter(weight, requires_grad=False)

            def _apply(self, fn, recurse=True):  # This is to get torch.compile + moving weights to another device working
                if recurse:
                    for module in self.children():
                        module._apply(fn)

                for key, param in self._parameters.items():
                    if param is None:
                        continue
                    self.register_parameter(key, torch.nn.Parameter(fn(param), requires_grad=False))
                for key, buf in self._buffers.items():
                    if buf is not None:
                        self._buffers[key] = fn(buf)
                return self

    return MixedPrecisionOps

def pick_operations(weight_dtype, compute_dtype, load_device=None, disable_fast_fp8=False, fp8_optimizations=False, model_config=None):
    fp8_compute = comfy.model_management.supports_fp8_compute(load_device) # TODO: if we support more ops this needs to be more granular
    nvfp4_compute = comfy.model_management.supports_nvfp4_compute(load_device)

    if model_config and hasattr(model_config, 'quant_config') and model_config.quant_config:
        logging.info("Using mixed precision operations")
        disabled = set()
        if not nvfp4_compute:
            disabled.add("nvfp4")
        if not fp8_compute:
            disabled.add("float8_e4m3fn")
            disabled.add("float8_e5m2")
        return mixed_precision_ops(model_config.quant_config, compute_dtype, disabled=disabled)

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
