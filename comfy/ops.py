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
        w, bias = cast_bias_weight(self, input, dtype=dtype, bias_dtype=input_dtype)
        w = w.t()

        scale_weight = self.scale_weight
        scale_input = self.scale_input
        if scale_weight is None:
            scale_weight = torch.ones((), device=input.device, dtype=torch.float32)
        else:
            scale_weight = scale_weight.to(input.device)

        if scale_input is None:
            scale_input = torch.ones((), device=input.device, dtype=torch.float32)
            input = torch.clamp(input, min=-448, max=448, out=input)
            input = input.reshape(-1, input_shape[2]).to(dtype).contiguous()
        else:
            scale_input = scale_input.to(input.device)
            input = (input * (1.0 / scale_input).to(input_dtype)).reshape(-1, input_shape[2]).to(dtype).contiguous()

        if bias is not None:
            o = torch._scaled_mm(input, w, out_dtype=input_dtype, bias=bias, scale_a=scale_input, scale_b=scale_weight)
        else:
            o = torch._scaled_mm(input, w, out_dtype=input_dtype, scale_a=scale_input, scale_b=scale_weight)

        if isinstance(o, tuple):
            o = o[0]

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


# ==============================================================================
# Quantization Format Registry System
# ==============================================================================

class QuantFormatHandler:
    """
    Base class for all quantization format handlers.
    
    A handler encapsulates the logic for a specific quantization format
    (e.g., FP8 scaled, MX formats) and manages the quantization
    parameters and forward pass for quantized layers.
    """
    
    def __init__(self, layer, **config):
        """
        Initialize handler for a specific layer.
        
        Args:
            layer: The nn.Module layer (Linear, Conv2d, etc.)
            **config: Format-specific configuration
        """
        self.layer = layer
        self.config = config
    
    def setup_parameters(self):
        """
        Initialize quantization parameters on the layer.
        Called during layer construction or load_state_dict.
        
        Subclasses should create parameters like scale_weight, scale_input, etc.
        and attach them to self.layer.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement setup_parameters()")
    
    def forward(self, *args, **kwargs):
        """
        Execute quantized forward pass.
        
        Signature matches the layer's expected forward pass.
        Handler accesses layer parameters via self.layer (weight, bias, etc.)
        
        Args:
            *args: Positional arguments matching layer forward signature
            **kwargs: Keyword arguments matching layer forward signature
            
        Returns:
            Layer output tensor
            
        Examples:
            Linear: forward(input)
            Conv2d: forward(input)
            GroupNorm: forward(input)
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement forward()")
    
    def load_state_dict(self, state_dict, prefix):
        """
        Load quantization parameters from state dict.
        
        Args:
            state_dict: State dictionary
            prefix: Key prefix for this layer (e.g., "model.diffusion_model.layer1.")
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement load_state_dict()")
    
    def state_dict(self, prefix):
        """
        Save quantization parameters to state dict.
        
        Args:
            prefix: Key prefix for this layer
            
        Returns:
            Dictionary of quantization parameters with full keys
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement state_dict()")
    
    def convert_weight(self, weight, inplace=False):
        """
        Convert weight from quantized to full precision (dequantize).
        
        Args:
            weight: Quantized weight tensor
            inplace: Whether to modify in-place
            
        Returns:
            Dequantized weight tensor
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement convert_weight()")
    
    def set_weight(self, weight, inplace_update=False, seed=None, return_weight=False):
        """
        Convert and set weight from full precision to quantized.
        
        Args:
            weight: Full precision weight tensor
            inplace_update: Whether to update layer weight in-place
            seed: Random seed for stochastic rounding
            return_weight: If True, return quantized weight without setting
            
        Returns:
            Quantized weight if return_weight=True, else None
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement set_weight()")


class QuantFormatRegistry:
    """
    Global registry for quantization formats.
    
    Formats are registered with a unique name and handler class.
    Custom formats can be registered by custom nodes.
    """
    
    _formats = {}
    
    @classmethod
    def register(cls, name, handler_class, **default_config):
        """
        Register a new quantization format.
        
        Args:
            name: Unique format identifier (e.g., "fp8_e4m3fn_scaled")
            handler_class: Handler class implementing QuantFormatHandler
            **default_config: Default configuration parameters
            
        Example:
            QuantFormatRegistry.register(
                "fp8_e4m3fn_scaled",
                handler_class=FP8ScaledHandler,
                base_dtype=torch.float8_e4m3fn,
                quantize_activation=False,
                use_fp8_matmul=True,
            )
        """
        if not issubclass(handler_class, QuantFormatHandler):
            raise TypeError(f"handler_class must be a subclass of QuantFormatHandler, got {handler_class}")
        
        cls._formats[name] = {
            "handler": handler_class,
            "config": default_config.copy()
        }
        logging.debug(f"Registered quantization format: {name}")
    
    @classmethod
    def get(cls, name, **override_config):
        """
        Get format info with optional config overrides.
        
        Args:
            name: Format identifier
            **override_config: Configuration overrides
            
        Returns:
            Dict with 'handler' (class) and 'config' (dict) keys
            
        Raises:
            ValueError: If format name not registered
        """
        if name not in cls._formats:
            available = ", ".join(cls._formats.keys()) if cls._formats else "none"
            raise ValueError(f"Unknown quantization format: '{name}'. Available formats: {available}")
        
        format_info = cls._formats[name].copy()
        # Merge override_config into default config
        config = format_info["config"].copy()
        config.update(override_config)
        format_info["config"] = config
        return format_info
    
    @classmethod
    def list_formats(cls):
        """List all registered format names"""
        return list(cls._formats.keys())
    
    @classmethod
    def is_registered(cls, name):
        """Check if a format is registered"""
        return name in cls._formats


class FP8ScaledHandler(QuantFormatHandler):
    """
    Handler for FP8 quantization with per-tensor scaling.
    
    Supports both weight-only and weight+activation quantization.
    Compatible with existing fp8_linear implementation.
    """
    
    def setup_parameters(self):
        """Initialize scale_weight and optionally scale_input"""
        device = self.layer.weight.device
        dtype = torch.float32
        
        # Always have scale_weight for FP8
        if not hasattr(self.layer, 'scale_weight') or self.layer.scale_weight is None:
            self.layer.scale_weight = torch.nn.Parameter(
                torch.ones((), device=device, dtype=dtype),
                requires_grad=False
            )
        
        # scale_input is optional (for activation quantization)
        if self.config.get("quantize_activation", False):
            if not hasattr(self.layer, 'scale_input') or self.layer.scale_input is None:
                self.layer.scale_input = torch.nn.Parameter(
                    torch.ones((), device=device, dtype=dtype),
                    requires_grad=False
                )
        else:
            self.layer.scale_input = None
    
    def forward(self, *args, **kwargs):
        """
        FP8 forward pass with optional activation quantization.
        Supports Linear layers (Conv2d in future).
        """
        # Detect layer type and dispatch
        if isinstance(self.layer, torch.nn.Linear):
            return self._forward_linear(*args, **kwargs)
        else:
            raise NotImplementedError(
                f"FP8ScaledHandler not implemented for {type(self.layer).__name__}"
            )
    
    def _forward_linear(self, input):
        """FP8 forward for Linear layers"""
        # Try fast path with fp8_linear if enabled
        if self.config.get("use_fp8_matmul", False) and not self.layer.training:
            try:
                result = fp8_linear(self.layer, input)
                if result is not None:
                    return result
            except Exception as e:
                logging.debug(f"FP8 matmul failed, falling back to standard path: {e}")
        
        # Standard path: dequantize and compute
        weight, bias = cast_bias_weight(self.layer, input)
        
        # Dequantize weight
        scale = self.layer.scale_weight.to(device=weight.device, dtype=weight.dtype)
        
        # Apply weight functions (LoRA, etc.) - they see dequantized weights
        if hasattr(self.layer, 'weight_function') and len(self.layer.weight_function) > 0:
            weight = weight * scale
            for f in self.layer.weight_function:
                weight = f(weight)
        else:
            weight = weight * scale
        
        if hasattr(self.layer, 'bias_function') and len(self.layer.bias_function) > 0:
            for f in self.layer.bias_function:
                bias = f(bias) if bias is not None else None
        
        # Execute linear operation
        # Optimization: multiply by scale on smaller tensor
        if weight.numel() < input.numel() and len(self.layer.weight_function) == 0:
            return torch.nn.functional.linear(input, weight, bias)
        else:
            return torch.nn.functional.linear(input, weight, bias)
    
    def load_state_dict(self, state_dict, prefix):
        """Load scale parameters from state dict"""
        scale_weight_key = f"{prefix}scale_weight"
        if scale_weight_key in state_dict:
            self.layer.scale_weight.data.copy_(state_dict[scale_weight_key])
        
        scale_input_key = f"{prefix}scale_input"
        if scale_input_key in state_dict and self.layer.scale_input is not None:
            self.layer.scale_input.data.copy_(state_dict[scale_input_key])
    
    def state_dict(self, prefix):
        """Save scale parameters to state dict"""
        result = {f"{prefix}scale_weight": self.layer.scale_weight}
        if self.layer.scale_input is not None:
            result[f"{prefix}scale_input"] = self.layer.scale_input
        return result
    
    def convert_weight(self, weight, inplace=False):
        """Dequantize: multiply by scale"""
        scale = self.layer.scale_weight.to(device=weight.device, dtype=weight.dtype)
        if inplace:
            weight *= scale
            return weight
        return weight * scale
    
    def set_weight(self, weight, inplace_update=False, seed=None, return_weight=False):
        """Quantize: divide by scale with stochastic rounding"""
        scale = self.layer.scale_weight.to(device=weight.device, dtype=weight.dtype)
        quantized = comfy.float.stochastic_rounding(
            weight / scale,
            self.layer.weight.dtype,
            seed=seed
        )
        
        if return_weight:
            return quantized
        
        if inplace_update:
            self.layer.weight.data.copy_(quantized)
        else:
            self.layer.weight = torch.nn.Parameter(quantized, requires_grad=False)


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
        """Linear layer with optional per-layer quantization"""
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.quant_handler = None
            self._handler_initialized = False
        
        def reset_parameters(self):
            # Don't allocate weights - return None like disable_weight_init
            return None
        
        def _load_from_state_dict(self, state_dict, prefix, local_metadata, 
                                  strict, missing_keys, unexpected_keys, error_msgs):
            """
            Called by PyTorch during load_state_dict.
            This is where we initialize the handler since we now know the layer name.
            """
            if not self._handler_initialized:
                # Normalize layer name from prefix
                layer_name = prefix.rstrip('.')
                
                # Strip known model prefixes
                for model_prefix in ["model.diffusion_model.", "model.model.", "net."]:
                    if layer_name.startswith(model_prefix):
                        layer_name = layer_name[len(model_prefix):]
                        break
                
                # Check if this layer has quantization config
                # Access via parent class since _layer_quant_config is a class variable
                if layer_name in MixedPrecisionOps._layer_quant_config:
                    config = MixedPrecisionOps._layer_quant_config[layer_name]
                    try:
                        format_info = QuantFormatRegistry.get(
                            config["format"],
                            **config.get("params", {})
                        )
                        
                        # Initialize handler
                        self.quant_handler = format_info["handler"](self, **format_info["config"])
                        self.quant_handler.setup_parameters()
                        
                        # Let handler load its parameters (scale_weight, etc.)
                        self.quant_handler.load_state_dict(state_dict, prefix)
                        
                        logging.debug(f"Initialized {config['format']} handler for layer {layer_name}")
                    except ValueError as e:
                        # Format not registered - fall back to standard precision
                        logging.warning(
                            f"Quantization format '{config['format']}' not registered for layer {layer_name}. "
                            f"Falling back to standard precision. Error: {e}"
                        )
                        self.quant_handler = None
                    except Exception as e:
                        logging.error(f"Failed to initialize quantization handler for {layer_name}: {e}")
                        self.quant_handler = None
                
                self._handler_initialized = True
            
            # Call parent to load weight and bias
            super()._load_from_state_dict(
                state_dict, prefix, local_metadata,
                strict, missing_keys, unexpected_keys, error_msgs
            )
        
        def _save_to_state_dict(self, destination, prefix, keep_vars):
            """Save layer parameters including quantization metadata"""
            super()._save_to_state_dict(destination, prefix, keep_vars)
            
            # Save handler parameters (scale_weight, etc.)
            if self.quant_handler:
                handler_dict = self.quant_handler.state_dict(prefix)
                destination.update(handler_dict)
        
        def forward_comfy_cast_weights(self, input):
            """Forward pass with optional quantization"""
            if self.quant_handler:
                # Use handler for quantized forward
                return self.quant_handler.forward(input)
            else:
                # Standard path for non-quantized layers
                weight, bias = cast_bias_weight(self, input)
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


# ==============================================================================
# Register built-in quantization formats
# ==============================================================================

# FP8 E4M3FN weight-only quantization
QuantFormatRegistry.register(
    "fp8_e4m3fn_scaled",
    handler_class=FP8ScaledHandler,
    base_dtype=torch.float8_e4m3fn,
    quantize_activation=False,
    use_fp8_matmul=True,
)

# FP8 E4M3FN weight+activation quantization
QuantFormatRegistry.register(
    "fp8_e4m3fn_scaled_dynamic",
    handler_class=FP8ScaledHandler,
    base_dtype=torch.float8_e4m3fn,
    quantize_activation=True,
    use_fp8_matmul=True,
)

# FP8 E5M2 weight-only quantization
QuantFormatRegistry.register(
    "fp8_e5m2_scaled",
    handler_class=FP8ScaledHandler,
    base_dtype=torch.float8_e5m2,
    quantize_activation=False,
    use_fp8_matmul=True,
)
