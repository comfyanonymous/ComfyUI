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

@torch.compiler.disable()
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

class disable_weight_init:
    class Linear(torch.nn.Linear, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.linear(input, weight, bias)

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
            weight, bias = cast_bias_weight(self, input)
            return self._conv_forward(input, weight, bias)

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
            weight, bias = cast_bias_weight(self, input)
            return self._conv_forward(input, weight, bias)

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
            weight, bias = cast_bias_weight(self, input)
            return self._conv_forward(input, weight, bias)

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
            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.group_norm(input, self.num_groups, weight, bias, self.eps)

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
                weight, bias = cast_bias_weight(self, input)
            else:
                weight = None
                bias = None
            return torch.nn.functional.layer_norm(input, self.normalized_shape, weight, bias, self.eps)

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
                weight, bias = cast_bias_weight(self, input)
            else:
                weight = None
            return comfy.rmsnorm.rms_norm(input, weight, self.eps)  # TODO: switch to commented out line when old torch is deprecated
            # return torch.nn.functional.rms_norm(input, self.normalized_shape, weight, self.eps)

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

            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.conv_transpose2d(
                input, weight, bias, self.stride, self.padding,
                output_padding, self.groups, self.dilation)

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

            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.conv_transpose1d(
                input, weight, bias, self.stride, self.padding,
                output_padding, self.groups, self.dilation)

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
            weight, bias = cast_bias_weight(self, device=input.device, dtype=out_dtype)
            return torch.nn.functional.embedding(input, weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse).to(dtype=output_dtype)

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
    Legacy FP8 linear function - now uses tensor subclass infrastructure.
    
    This function maintains backward compatibility with existing code while
    routing all FP8 computation through the unified tensor subclass system.
    All actual FP8 matmul logic is handled by the registered operation handlers
    in quant_ops.py via __torch_dispatch__.
    
    Args:
        self: Linear layer with FP8 weight and scale parameters
        input: Input tensor (any dtype)
        
    Returns:
        Output tensor or None if weight is not FP8
    """
    dtype = self.weight.dtype
    if dtype not in [torch.float8_e4m3fn]:
        return None

    tensor_2d = False
    if len(input.shape) == 2:
        tensor_2d = True
        input = input.unsqueeze(1)

    input_shape = input.shape
    input_dtype = input.dtype
    
    if len(input.shape) == 3:
        # Get weight and bias using standard casting
        w, bias = cast_bias_weight(self, input, dtype=dtype, bias_dtype=input_dtype)

        # Get scales (same as before)
        scale_weight = self.scale_weight
        scale_input = self.scale_input
        if scale_weight is None:
            scale_weight = torch.ones((), device=input.device, dtype=torch.float32)
        else:
            scale_weight = scale_weight.to(input.device)

        if scale_input is None:
            scale_input = torch.ones((), device=input.device, dtype=torch.float32)
        else:
            scale_input = scale_input.to(input.device)
        
        # Wrap weight in QuantizedTensorFP8 - this enables unified dispatch
        quantized_weight = QuantizedTensorFP8(w, scale_weight, orig_dtype=input_dtype)
        
        # Handle input quantization and wrapping
        if self.scale_input is None:
            # Clamp input to FP8 range and quantize
            input = torch.clamp(input, min=-448, max=448, out=input)
            input_fp8 = input.reshape(-1, input_shape[2]).to(dtype).contiguous()
        else:
            # Apply inverse scale and quantize
            input_fp8 = (input * (1.0 / scale_input).to(input_dtype)).reshape(-1, input_shape[2]).to(dtype).contiguous()
        
        # Wrap input in QuantizedTensorFP8
        quantized_input = QuantizedTensorFP8(input_fp8, scale_input, orig_dtype=input_dtype)
        
        # Call F.linear - __torch_dispatch__ routes to handle_linear_fp8 in quant_ops.py!
        # This is the key unification: all FP8 computation goes through one path
        o = torch.nn.functional.linear(quantized_input, quantized_weight, bias)
        
        # Reshape output
        if tensor_2d:
            return o.reshape(input_shape[0], -1)
        return o.reshape((-1, input_shape[1], self.weight.shape[0]))

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

            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.linear(input, weight, bias)

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


# Import quantization operations from separate module
from .quant_ops import QuantizedTensorFP8


# ==============================================================================
# Mixed Precision Operations
# ==============================================================================

class MixedPrecisionOps(disable_weight_init):
    """
    Operations class supporting per-layer quantization (mixed precision).
    
    This class enables different layers to use different quantization formats
    within the same model (e.g., some layers FP8, others BF16).
    
    Layer-specific quantization is configured via _layer_quant_config class variable,
    which is set by pick_operations() when a model has mixed precision.
    """
    
    _layer_quant_config = {}  # Class variable set by pick_operations()
    
    class Linear(disable_weight_init.Linear):
        """Linear layer with optional per-layer quantization using tensor subclasses"""
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.quant_format = None
            self.quant_scale = None
            self._quantization_initialized = False
        
        def reset_parameters(self):
            # Don't allocate weights - return None like disable_weight_init
            return None
        
        def _load_from_state_dict(self, state_dict, prefix, local_metadata, 
                                  strict, missing_keys, unexpected_keys, error_msgs):
            """
            Called by PyTorch during load_state_dict.
            Load weight and wrap in QuantizedTensorFP8 if this layer is quantized.
            """
            # Call parent to load weight and bias first
            super()._load_from_state_dict(
                state_dict, prefix, local_metadata,
                strict, missing_keys, unexpected_keys, error_msgs
            )
            
            # After weight is loaded, wrap it if this layer is quantized
            if not self._quantization_initialized:
                # Normalize layer name from prefix
                layer_name = prefix.rstrip('.')
                
                # Strip known model prefixes
                for model_prefix in ["model.diffusion_model.", "model.model.", "net."]:
                    if layer_name.startswith(model_prefix):
                        layer_name = layer_name[len(model_prefix):]
                        break
                
                # Check if this layer has quantization config
                if layer_name in MixedPrecisionOps._layer_quant_config:
                    config = MixedPrecisionOps._layer_quant_config[layer_name]
                    self.quant_format = config.get("format", "fp8_e4m3fn")
                    
                    # Load scale parameter
                    scale_key = f"{prefix}scale_weight"
                    if scale_key in state_dict:
                        self.quant_scale = state_dict[scale_key]
                        
                        # Wrap weight in QuantizedTensorFP8
                        if self.weight is not None and self.weight.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                            try:
                                # Determine original dtype (default to bfloat16)
                                orig_dtype = torch.bfloat16
                                
                                # Wrap weight in quantized tensor subclass
                                quantized_weight = QuantizedTensorFP8(
                                    self.weight.data,
                                    self.quant_scale,
                                    orig_dtype=orig_dtype
                                )
                                
                                # Replace weight parameter with wrapped version
                                self.weight = torch.nn.Parameter(quantized_weight, requires_grad=False)
                                
                                logging.debug(f"Wrapped layer {layer_name} weight in QuantizedTensorFP8 (format: {self.quant_format})")
                            except Exception as e:
                                logging.warning(f"Failed to wrap layer {layer_name} in QuantizedTensorFP8: {e}")
                                self.quant_format = None
                                self.quant_scale = None
                        else:
                            logging.debug(f"Layer {layer_name} has scale but weight dtype is not FP8, skipping quantization")
                            self.quant_format = None
                            self.quant_scale = None
                    else:
                        logging.debug(f"Layer {layer_name} has quant config but no scale_weight in state_dict")
                        self.quant_format = None
                
                self._quantization_initialized = True
        
        def _save_to_state_dict(self, destination, prefix, keep_vars):
            """Save layer parameters including quantization scale"""
            # First unwrap the weight if it's quantized
            if isinstance(self.weight, torch.nn.Parameter) and isinstance(self.weight.data, QuantizedTensorFP8):
                # Temporarily unwrap to save the raw FP8 data
                quantized_tensor = self.weight.data
                raw_fp8_data = quantized_tensor._raw_data
                original_weight = self.weight
                self.weight = torch.nn.Parameter(raw_fp8_data, requires_grad=False)
                
                # Call parent to save unwrapped weight
                super()._save_to_state_dict(destination, prefix, keep_vars)
                
                # Restore the wrapped weight
                self.weight = original_weight
                
                # Save the scale parameter
                if self.quant_scale is not None:
                    destination[f"{prefix}scale_weight"] = self.quant_scale if keep_vars else self.quant_scale.detach()
            else:
                # Standard path for non-quantized weights
                super()._save_to_state_dict(destination, prefix, keep_vars)
        
        def forward_comfy_cast_weights(self, input):
            """
            Forward pass - tensor subclass handles dispatch automatically!
            __torch_dispatch__ will route to registered handlers based on tensor types.
            """
            weight, bias = cast_bias_weight(self, input)
            
            # Call F.linear - if weight is QuantizedTensorFP8, __torch_dispatch__ handles it!
            return torch.nn.functional.linear(input, weight, bias)
        
        def forward(self, *args, **kwargs):
            """Main forward pass"""
            run_every_op()
            # Same logic as disable_weight_init.Linear
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)
    
    @classmethod
    def conv_nd(s, dims, *args, **kwargs):
        """Create Conv layer (same as disable_weight_init)"""
        if dims == 2:
            return s.Conv2d(*args, **kwargs)
        elif dims == 3:
            return s.Conv3d(*args, **kwargs)
        else:
            raise ValueError(f"unsupported dimensions: {dims}")


def pick_operations(weight_dtype, compute_dtype, load_device=None, disable_fast_fp8=False, fp8_optimizations=False, scaled_fp8=None, model_config=None):
    """
    Select appropriate operations class for model.
    
    NEW: If model_config.layer_quant_config exists, returns MixedPrecisionOps (Phase 3).
    LEGACY: All other paths unchanged for backward compatibility.
    
    Args:
        weight_dtype: Weight storage dtype
        compute_dtype: Computation dtype
        load_device: Device for loading
        disable_fast_fp8: Disable fast FP8 paths
        fp8_optimizations: Enable FP8 optimizations
        scaled_fp8: Legacy FP8 dtype marker
        model_config: Model config object (optional, for mixed precision support)
        
    Returns:
        Operations class (e.g., MixedPrecisionOps, fp8_ops, disable_weight_init)
    """
    # NEW: Check for mixed precision
    if model_config and hasattr(model_config, 'layer_quant_config') and model_config.layer_quant_config:
        MixedPrecisionOps._layer_quant_config = model_config.layer_quant_config
        logging.info(f"Using mixed precision operations: {len(model_config.layer_quant_config)} quantized layers")
        return MixedPrecisionOps
    
    # LEGACY paths (unchanged)
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
