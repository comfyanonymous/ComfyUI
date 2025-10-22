import torch
import logging

# ==============================================================================
# Global Operation Registry
# ==============================================================================

# Global operation registry: torch operation → handler function
_QUANT_OP_REGISTRY = {}

def register_quant_op(torch_op):
    """
    Decorator to register an operation handler.
    
    Example:
        @register_quant_op(torch.ops.aten.linear.default)
        def handle_linear_fp8(func, args, kwargs):
            # Implementation
            ...
    """
    def decorator(handler_func):
        _QUANT_OP_REGISTRY[torch_op] = handler_func
        return handler_func
    return decorator


def get_quant_handler(torch_op):
    """Get registered handler for an operation"""
    return _QUANT_OP_REGISTRY.get(torch_op)


def list_registered_ops():
    """List all registered quantized operations"""
    return list(_QUANT_OP_REGISTRY.keys())


# ==============================================================================
# comfy_kitchen Integration
# ==============================================================================

try:
    import comfy_kitchen as ck
    ck.disable_backend("cutile")
    _CK_AVAILABLE = True
    logging.info("comfy_kitchen available for optimized quantization kernels")
except ImportError:
    ck = None
    _CK_AVAILABLE = False
    logging.info("comfy_kitchen not available - using PyTorch fallbacks")
except Exception as e:
    ck = None
    _CK_AVAILABLE = False
    logging.warning(f"comfy_kitchen import failed: {e} - using PyTorch fallbacks")


# ==============================================================================
# Quantized Tensor Subclass
# ==============================================================================

class QuantizedTensorFP8(torch.Tensor):
    """
    Tensor subclass for FP8 quantized data.
    Automatically handles operations via __torch_dispatch__.
    """
    
    @staticmethod
    def __new__(cls, tensor, scale, orig_dtype=torch.bfloat16):
        """
        Create a quantized FP8 tensor.
        
        Args:
            tensor: The FP8 tensor data (torch.float8_e4m3fn or e5m2)
            scale: Scale factor for dequantization (scalar tensor)
            orig_dtype: Original dtype before quantization
        """
        return torch.Tensor._make_subclass(cls, tensor, require_grad=False)
    
    def __init__(self, tensor, scale, orig_dtype=torch.bfloat16):
        self._scale = scale
        self._orig_dtype = orig_dtype
        # Store a reference to prevent infinite recursion in dequantize
        self._raw_data = tensor.contiguous()
    
    def __repr__(self):
        return (f"QuantizedTensorFP8(shape={self.shape}, "
                f"scale={self._scale:.4f}, dtype={self._orig_dtype})")
    
    @classmethod
    def quantize(cls, tensor, scale, fp8_dtype=torch.float8_e4m3fn):
        orig_dtype = tensor.dtype
        
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, device=tensor.device, dtype=torch.float32)
        
        tensor_fp8 = None
        if _CK_AVAILABLE:
            try:
                tensor_fp8 = ck.quantize_per_tensor_fp8(tensor, scale, fp8_dtype)
            except Exception as e:
                logging.debug(f"comfy_kitchen quantization failed, using PyTorch: {e}")
        
        if tensor_fp8 is None:
            lp_amax = torch.finfo(fp8_dtype).max
            tensor_scaled = tensor.float() / scale
            torch.clamp(tensor_scaled, min=-lp_amax, max=lp_amax, out=tensor_scaled)
            tensor_fp8 = tensor_scaled.to(fp8_dtype, memory_format=torch.contiguous_format)
        
        return cls(tensor_fp8, scale, orig_dtype=orig_dtype)
    
    @classmethod
    def quantize_dynamic(cls, tensor, strategy="amax", fp8_dtype=torch.float8_e4m3fn):
        if strategy == "amax":
            scale = torch.amax(tensor) / torch.finfo(fp8_dtype).max
            scale = scale.to(tensor.device, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown quantization strategy: {strategy}. "
                           f"Supported: 'amax'")

        return cls.quantize(tensor, scale, fp8_dtype=fp8_dtype)
    
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        
        # Special case: skip dispatch for internal tensor operations
        # that are used for unwrapping (to avoid recursion)
        if func in [torch.ops.aten._to_copy.default, torch.ops.aten.detach.default]:
            # For these ops, use the raw data to avoid recursion, but return QuantizedTensorFP8 for detach
            if func == torch.ops.aten.detach.default and isinstance(args[0], QuantizedTensorFP8):
                # Special handling for detach - return a new QuantizedTensorFP8
                qt = args[0]
                detached_data = qt._raw_data.detach()
                return QuantizedTensorFP8(detached_data, qt._scale, qt._orig_dtype)
            
            # For other ops, just unwrap
            def unwrap(arg):
                if isinstance(arg, QuantizedTensorFP8):
                    return arg._raw_data
                return arg
            new_args = tuple(unwrap(a) if not isinstance(a, (list, tuple, dict)) else a for a in args)
            return func(*new_args, **kwargs)
        
        # Look up registered handler for this operation
        handler = _QUANT_OP_REGISTRY.get(func)
        if handler:
            return handler(func, args, kwargs)
        
        # No handler - dequantize and use standard path
        return cls._dequant_and_fallback(func, args, kwargs)
    
    @classmethod
    def _dequant_and_fallback(cls, func, args, kwargs):
        """Fallback: dequantize all quantized tensors"""
        def dequant_arg(arg):
            if isinstance(arg, QuantizedTensorFP8):
                return arg.dequantize()
            elif isinstance(arg, (list, tuple)):
                return type(arg)(dequant_arg(a) for a in arg)
            return arg
        
        new_args = dequant_arg(args)
        new_kwargs = dequant_arg(kwargs)
        return func(*new_args, **new_kwargs)
    
    def dequantize(self) -> torch.Tensor:
        plain_tensor = torch.ops.aten._to_copy.default(self._raw_data, dtype=self._orig_dtype)
        return plain_tensor * self._scale
    
    def detach(self):
        """Detach returns a new QuantizedTensorFP8 (required for Parameter)"""
        detached_data = self._raw_data.detach()
        return QuantizedTensorFP8(detached_data, self._scale, self._orig_dtype)


# ==============================================================================
# Operation Handlers for Quantized Tensors
# ==============================================================================

@register_quant_op(torch.ops.aten.linear.default)
def handle_linear_fp8(func, args, kwargs):
    """
    Handle F.linear() with quantized inputs.
    
    Supports:
    - QuantizedTensorFP8 input + QuantizedTensorFP8 weight
    - QuantizedTensorFP8 input + regular weight
    - Regular input + QuantizedTensorFP8 weight
    """
    input_tensor = args[0]
    weight = args[1]
    bias = args[2] if len(args) > 2 else None
    out_dtype = kwargs.get("out_dtype", input_tensor._orig_dtype)

    # Case 1: Both input and weight are FP8
    if isinstance(input_tensor, QuantizedTensorFP8) and isinstance(weight, QuantizedTensorFP8):
        # Get plain tensors to avoid dispatch recursion
        plain_input = input_tensor._raw_data
        plain_weight = weight._raw_data
        weight_t = plain_weight.t()  # Keep as column-major for cuBLASLt
        
        try:
            output = torch._scaled_mm(
                plain_input,
                weight_t,
                bias=bias,
                scale_a=input_tensor._scale,
                scale_b=weight._scale,
                out_dtype=out_dtype,
            )
            if isinstance(output, tuple):
                output = output[0]
            
            if output.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                output_scale = input_tensor._scale * weight._scale
                return QuantizedTensorFP8(output, output_scale, input_tensor._orig_dtype) # TODO is this correct? Can't cuBLAS return it
            else:
                return output
        except Exception as e:
            logging.debug(f"FP8 _scaled_mm failed, falling back to dequantization: {e}")

    # Case 2: Only weight is quantized
    if isinstance(weight, QuantizedTensorFP8):
        weight_dq = weight.dequantize()
        input_dq = input_tensor.dequantize() if isinstance(input_tensor, QuantizedTensorFP8) else input_tensor
        return torch.nn.functional.linear(input_dq, weight_dq, bias)
    
    # Case 3: Only input is quantized
    elif isinstance(input_tensor, QuantizedTensorFP8):
        input_dq = input_tensor.dequantize()
        return torch.nn.functional.linear(input_dq, weight, bias)
    
    # Case 4: Neither is quantized (shouldn't happen, but handle it)
    else:
        return torch.nn.functional.linear(input_tensor, weight, bias)

