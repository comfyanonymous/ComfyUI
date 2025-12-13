import torch
import logging
from typing import Tuple, Dict
import comfy.float

_LAYOUT_REGISTRY = {}
_GENERIC_UTILS = {}

# Try to import Triton-based INT8 kernels
try:
    from .int8_kernels import (
        act_quant as triton_act_quant,
        act_dequant as triton_act_dequant,
        weight_quant as triton_weight_quant,
        weight_dequant as triton_weight_dequant,
        int8_gemm as triton_int8_gemm,
        int8_addmm as triton_int8_addmm,
        int8_gemm_quant as triton_int8_gemm_quant,
        int8_addmm_quant as triton_int8_addmm_quant
    )
    _HAS_TRITON_INT8 = True
except ImportError:
    _HAS_TRITON_INT8 = False
    logging.warning("Triton INT8 kernels not available, using PyTorch fallback")


def register_layout_op(torch_op, layout_type):
    """
    Decorator to register a layout-specific operation handler.
    Args:
        torch_op: PyTorch operation (e.g., torch.ops.aten.linear.default)
        layout_type: Layout class (e.g., TensorCoreFP8Layout)
    Example:
        @register_layout_op(torch.ops.aten.linear.default, TensorCoreFP8Layout)
        def fp8_linear(func, args, kwargs):
            # FP8-specific linear implementation
            ...
    """
    def decorator(handler_func):
        if torch_op not in _LAYOUT_REGISTRY:
            _LAYOUT_REGISTRY[torch_op] = {}
        _LAYOUT_REGISTRY[torch_op][layout_type] = handler_func
        return handler_func
    return decorator


def register_generic_util(torch_op):
    """
    Decorator to register a generic utility that works for all layouts.
    Args:
        torch_op: PyTorch operation (e.g., torch.ops.aten.detach.default)

    Example:
        @register_generic_util(torch.ops.aten.detach.default)
        def generic_detach(func, args, kwargs):
            # Works for any layout
            ...
    """
    def decorator(handler_func):
        _GENERIC_UTILS[torch_op] = handler_func
        return handler_func
    return decorator


def _get_layout_from_args(args):
    for arg in args:
        if isinstance(arg, QuantizedTensor):
            return arg._layout_type
        elif isinstance(arg, (list, tuple)):
            for item in arg:
                if isinstance(item, QuantizedTensor):
                    return item._layout_type
    return None


def _move_layout_params_to_device(params, device):
    new_params = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            new_params[k] = v.to(device=device)
        else:
            new_params[k] = v
    return new_params


def _copy_layout_params(params):
    new_params = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            new_params[k] = v.clone()
        else:
            new_params[k] = v
    return new_params

def _copy_layout_params_inplace(src, dst, non_blocking=False):
    for k, v in src.items():
        if isinstance(v, torch.Tensor):
            dst[k].copy_(v, non_blocking=non_blocking)
        else:
            dst[k] = v

class QuantizedLayout:
    """
    Base class for quantization layouts.

    A layout encapsulates the format-specific logic for quantization/dequantization
    and provides a uniform interface for extracting raw tensors needed for computation.

    New quantization formats should subclass this and implement the required methods.
    """
    @classmethod
    def quantize(cls, tensor, **kwargs) -> Tuple[torch.Tensor, Dict]:
        raise NotImplementedError(f"{cls.__name__} must implement quantize()")

    @staticmethod
    def dequantize(qdata, **layout_params) -> torch.Tensor:
        raise NotImplementedError("TensorLayout must implement dequantize()")

    @classmethod
    def get_plain_tensors(cls, qtensor) -> torch.Tensor:
        raise NotImplementedError(f"{cls.__name__} must implement get_plain_tensors()")


class QuantizedTensor(torch.Tensor):
    """
    Universal quantized tensor that works with any layout.

    This tensor subclass uses a pluggable layout system to support multiple
    quantization formats (FP8, INT4, INT8, etc.) without code duplication.

    The layout_type determines format-specific behavior, while common operations
    (detach, clone, to) are handled generically.

    Attributes:
        _qdata: The quantized tensor data
        _layout_type: Layout class (e.g., TensorCoreFP8Layout)
        _layout_params: Dict with layout-specific params (scale, zero_point, etc.)
    """

    @staticmethod
    def __new__(cls, qdata, layout_type, layout_params):
        """
        Create a quantized tensor.

        Args:
            qdata: The quantized data tensor
            layout_type: Layout class (subclass of QuantizedLayout)
            layout_params: Dict with layout-specific parameters
        """
        return torch.Tensor._make_wrapper_subclass(cls, qdata.shape, device=qdata.device, dtype=qdata.dtype, requires_grad=False)

    def __init__(self, qdata, layout_type, layout_params):
        self._qdata = qdata
        self._layout_type = layout_type
        self._layout_params = layout_params

    def __repr__(self):
        layout_name = self._layout_type
        param_str = ", ".join(f"{k}={v}" for k, v in list(self._layout_params.items())[:2])
        return f"QuantizedTensor(shape={self.shape}, layout={layout_name}, {param_str})"

    @property
    def layout_type(self):
        return self._layout_type

    def __tensor_flatten__(self):
        """
        Tensor flattening protocol for proper device movement.
        """
        inner_tensors = ["_qdata"]
        ctx = {
            "layout_type": self._layout_type,
        }

        tensor_params = {}
        non_tensor_params = {}
        for k, v in self._layout_params.items():
            if isinstance(v, torch.Tensor):
                tensor_params[k] = v
            else:
                non_tensor_params[k] = v

        ctx["tensor_param_keys"] = list(tensor_params.keys())
        ctx["non_tensor_params"] = non_tensor_params

        for k, v in tensor_params.items():
            attr_name = f"_layout_param_{k}"
            object.__setattr__(self, attr_name, v)
            inner_tensors.append(attr_name)

        return inner_tensors, ctx

    @staticmethod
    def __tensor_unflatten__(inner_tensors, ctx, outer_size, outer_stride):
        """
        Tensor unflattening protocol for proper device movement.
        Reconstructs the QuantizedTensor after device movement.
        """
        layout_type = ctx["layout_type"]
        layout_params = dict(ctx["non_tensor_params"])

        for key in ctx["tensor_param_keys"]:
            attr_name = f"_layout_param_{key}"
            layout_params[key] = inner_tensors[attr_name]

        return QuantizedTensor(inner_tensors["_qdata"], layout_type, layout_params)

    @classmethod
    def from_float(cls, tensor, layout_type, **quantize_kwargs) -> 'QuantizedTensor':
        qdata, layout_params = LAYOUTS[layout_type].quantize(tensor, **quantize_kwargs)
        return cls(qdata, layout_type, layout_params)

    def dequantize(self) -> torch.Tensor:
        return LAYOUTS[self._layout_type].dequantize(self._qdata, **self._layout_params)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        # Step 1: Check generic utilities first (detach, clone, to, etc.)
        if func in _GENERIC_UTILS:
            return _GENERIC_UTILS[func](func, args, kwargs)

        # Step 2: Check layout-specific handlers (linear, matmul, etc.)
        layout_type = _get_layout_from_args(args)
        if layout_type and func in _LAYOUT_REGISTRY:
            handler = _LAYOUT_REGISTRY[func].get(layout_type)
            if handler:
                return handler(func, args, kwargs)

        # Step 3: Fallback to dequantization
        #if isinstance(args[0] if args else None, QuantizedTensor):
            #logging.info(f"QuantizedTensor: Unhandled operation {func}, falling back to dequantization. kwargs={kwargs}, args={args}")

        to_return = cls._dequant_and_fallback(func, args, kwargs)

        return to_return
 
    def data_ptr(self):
        return self._qdata.data_ptr()

    def is_pinned(self):
        return self._qdata.is_pinned()

    def is_contiguous(self):
        return self._qdata.is_contiguous()

    @classmethod
    def _dequant_and_fallback(cls, func, args, kwargs):
        def dequant_arg(arg):
            if isinstance(arg, QuantizedTensor):
                return arg.dequantize()
            elif isinstance(arg, (list, tuple)):
                return type(arg)(dequant_arg(a) for a in arg)
            return arg

        new_args = dequant_arg(args)
        new_kwargs = dequant_arg(kwargs)
        return func(*new_args, **new_kwargs)


# ==============================================================================
# Generic Utilities (Layout-Agnostic Operations)
# ==============================================================================

def _create_transformed_qtensor(qt, transform_fn):
    new_data = transform_fn(qt._qdata)
    new_params = _copy_layout_params(qt._layout_params)
    return QuantizedTensor(new_data, qt._layout_type, new_params)


def _handle_device_transfer(qt, target_device, target_dtype=None, target_layout=None, op_name="to"):
    if target_layout is not None and target_layout != torch.strided:
        logging.warning(
            f"QuantizedTensor: layout change requested to {target_layout}, "
            f"but not supported. Ignoring layout."
        )

    # Handle device transfer
    current_device = qt._qdata.device
    if target_device is not None:
        # Normalize device for comparison
        if isinstance(target_device, str):
            target_device = torch.device(target_device)
        if isinstance(current_device, str):
            current_device = torch.device(current_device)

        if target_device != current_device:
            logging.debug(f"QuantizedTensor.{op_name}: Moving from {current_device} to {target_device}")
            new_q_data = qt._qdata.to(device=target_device)
            new_params = _move_layout_params_to_device(qt._layout_params, target_device)
            if target_dtype is not None:
                new_params["orig_dtype"] = target_dtype
            new_qt = QuantizedTensor(new_q_data, qt._layout_type, new_params)
            logging.debug(f"QuantizedTensor.{op_name}: Created new tensor on {target_device}")
            return new_qt

    logging.debug(f"QuantizedTensor.{op_name}: No device change needed, returning original")
    return qt


@register_generic_util(torch.ops.aten.detach.default)
def generic_detach(func, args, kwargs):
    """Detach operation - creates a detached copy of the quantized tensor."""
    qt = args[0]
    if isinstance(qt, QuantizedTensor):
        return _create_transformed_qtensor(qt, lambda x: x.detach())
    return func(*args, **kwargs)


@register_generic_util(torch.ops.aten.clone.default)
def generic_clone(func, args, kwargs):
    """Clone operation - creates a deep copy of the quantized tensor."""
    qt = args[0]
    if isinstance(qt, QuantizedTensor):
        return _create_transformed_qtensor(qt, lambda x: x.clone())
    return func(*args, **kwargs)


@register_generic_util(torch.ops.aten._to_copy.default)
def generic_to_copy(func, args, kwargs):
    """Device/dtype transfer operation - handles .to(device) calls."""
    qt = args[0]
    if isinstance(qt, QuantizedTensor):
        return _handle_device_transfer(
            qt,
            target_device=kwargs.get('device', None),
            target_dtype=kwargs.get('dtype', None),
            op_name="_to_copy"
        )
    return func(*args, **kwargs)


@register_generic_util(torch.ops.aten.to.dtype_layout)
def generic_to_dtype_layout(func, args, kwargs):
    """Handle .to(device) calls using the dtype_layout variant."""
    qt = args[0]
    if isinstance(qt, QuantizedTensor):
        return _handle_device_transfer(
            qt,
            target_device=kwargs.get('device', None),
            target_dtype=kwargs.get('dtype', None),
            target_layout=kwargs.get('layout', None),
            op_name="to"
        )
    return func(*args, **kwargs)

@register_generic_util(torch.ops.aten.to.dtype)
def generic_to_dtype(func, args, kwargs):
    """Handle .to(dtype) calls - dtype conversion only."""
    src = args[0]
    if isinstance(src, QuantizedTensor):
        # For dtype-only conversion, just change the orig_dtype, no real cast is needed
        target_dtype = args[1] if len(args) > 1 else kwargs.get('dtype')
        src._layout_params["orig_dtype"] = target_dtype
        return src
    return func(*args, **kwargs)


@register_generic_util(torch.ops.aten.copy_.default)
def generic_copy_(func, args, kwargs):
    qt_dest = args[0]
    src = args[1]
    non_blocking = args[2] if len(args) > 2 else False
    if isinstance(qt_dest, QuantizedTensor):
        if isinstance(src, QuantizedTensor):
            # Copy from another quantized tensor
            qt_dest._qdata.copy_(src._qdata, non_blocking=non_blocking)
            qt_dest._layout_type = src._layout_type
            orig_dtype = qt_dest._layout_params["orig_dtype"]
            _copy_layout_params_inplace(src._layout_params, qt_dest._layout_params, non_blocking=non_blocking)
            qt_dest._layout_params["orig_dtype"] = orig_dtype
        else:
            # Copy from regular tensor - just copy raw data
            qt_dest._qdata.copy_(src)
        return qt_dest
    return func(*args, **kwargs)


@register_generic_util(torch.ops.aten._has_compatible_shallow_copy_type.default)
def generic_has_compatible_shallow_copy_type(func, args, kwargs):
    return True


@register_generic_util(torch.ops.aten.empty_like.default)
def generic_empty_like(func, args, kwargs):
    """Empty_like operation - creates an empty tensor with the same quantized structure."""
    qt = args[0]
    if isinstance(qt, QuantizedTensor):
        # Create empty tensor with same shape and dtype as the quantized data
        hp_dtype = kwargs.pop('dtype', qt._layout_params["orig_dtype"])
        new_qdata = torch.empty_like(qt._qdata, **kwargs)

        # Handle device transfer for layout params
        target_device = kwargs.get('device', new_qdata.device)
        new_params = _move_layout_params_to_device(qt._layout_params, target_device)

        # Update orig_dtype if dtype is specified
        new_params['orig_dtype'] = hp_dtype

        return QuantizedTensor(new_qdata, qt._layout_type, new_params)
    return func(*args, **kwargs)

# ==============================================================================
# FP8 Layout + Operation Handlers
# ==============================================================================
class TensorCoreFP8Layout(QuantizedLayout):
    """
    Storage format:
    - qdata: FP8 tensor (torch.float8_e4m3fn or torch.float8_e5m2)
    - scale: Scalar tensor (float32) for dequantization
    - orig_dtype: Original dtype before quantization (for casting back)
    """
    @classmethod
    def quantize(cls, tensor, scale=None, dtype=torch.float8_e4m3fn, stochastic_rounding=0, inplace_ops=False):
        orig_dtype = tensor.dtype

        if isinstance(scale, str) and scale == "recalculate":
            scale = torch.amax(tensor.abs()).to(dtype=torch.float32) / torch.finfo(dtype).max
            if tensor.dtype not in [torch.float32, torch.bfloat16]:  # Prevent scale from being too small
                tensor_info = torch.finfo(tensor.dtype)
                scale = (1.0 / torch.clamp((1.0 / scale), min=tensor_info.min, max=tensor_info.max))

        if scale is not None:
            if not isinstance(scale, torch.Tensor):
                scale = torch.tensor(scale)
            scale = scale.to(device=tensor.device, dtype=torch.float32)

            if inplace_ops:
                tensor *= (1.0 / scale).to(tensor.dtype)
            else:
                tensor = tensor * (1.0 / scale).to(tensor.dtype)
        else:
            scale = torch.ones((), device=tensor.device, dtype=torch.float32)

        if stochastic_rounding > 0:
            tensor = comfy.float.stochastic_rounding(tensor, dtype=dtype, seed=stochastic_rounding)
        else:
            lp_amax = torch.finfo(dtype).max
            torch.clamp(tensor, min=-lp_amax, max=lp_amax, out=tensor)
            tensor = tensor.to(dtype, memory_format=torch.contiguous_format)

        layout_params = {
            'scale': scale,
            'orig_dtype': orig_dtype
        }
        return tensor, layout_params

    @staticmethod
    def dequantize(qdata, scale, orig_dtype, **kwargs):
        plain_tensor = torch.ops.aten._to_copy.default(qdata, dtype=orig_dtype)
        plain_tensor.mul_(scale)
        return plain_tensor

    @classmethod
    def get_plain_tensors(cls, qtensor):
        return qtensor._qdata, qtensor._layout_params['scale']


# ==============================================================================
# Block-Wise INT8 Layout + Operation Handlers
# ==============================================================================
class BlockWiseINT8Layout(QuantizedLayout):
    """
    Block-wise INT8 quantization layout.
    
    Storage format:
    - qdata: INT8 tensor (torch.int8)
    - scale: Per-block scaling factors (float32)
    - block_size: Size of quantization blocks (default 128)
    - orig_dtype: Original dtype before quantization (for casting back)
    - is_weight: Whether this is a weight tensor (affects blocking dimension)
    
    Asymmetric blocking:
    - Weights: blocks partition along first dimension (M) and second dimension (N)
              scale shape: (M//block_size, N//block_size)
    - Activations: blocks partition along last dimension (K)
                  scale shape: (*batch_dims, K//block_size)
    """
    
    @classmethod
    def quantize(cls, tensor, scale=None, block_size=128, is_weight=False, **kwargs):
        """
        Quantize a tensor to INT8 with block-wise scaling.
        
        Args:
            tensor: Input tensor to quantize
            scale: Optional pre-computed scaling factors
            block_size: Size of quantization blocks (default 128)
            is_weight: If True, block along both dimensions (for weights)
                      If False, block along last dimension only (for activations)
        
        Returns:
            Tuple of (quantized_data, layout_params)
        """
        orig_dtype = tensor.dtype
        
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        
        if is_weight:
            # Weight quantization: block-wise along both M and N dimensions
            # Expected shape: (M, N)
            assert tensor.dim() == 2, f"Weight tensor must be 2D, got shape {tensor.shape}"
            M, N = tensor.shape
            assert M % block_size == 0 and N % block_size == 0, \
                f"Dimensions must be divisible by block_size={block_size}, got shape {tensor.shape}"
            
            # Use Triton kernel if available AND tensor is on CUDA
            if _HAS_TRITON_INT8 and scale is None and tensor.is_cuda:
                try:
                    qdata, scale = triton_weight_quant(tensor, block_size=block_size)
                except Exception as e:
                    # don't fall back, raise, for easier debugging
                    logging.warning(f"Triton weight_quant failed: {e}, falling back to PyTorch")
                    raise e
                    # qdata, scale = cls._weight_quantize_pytorch(tensor, block_size)
            else:
                qdata, scale = cls._weight_quantize_pytorch(tensor, block_size, scale)
            
        else:
            # Activation quantization: block-wise along last dimension (K)
            # Can handle any shape: (*batch_dims, K)
            K = tensor.shape[-1]
            assert K % block_size == 0, \
                f"Last dimension must be divisible by block_size={block_size}, got {K}"
            
            # Use Triton kernel if available AND tensor is on CUDA
            # ignore input scale for now
            # TODO: why do we need input scale?
            if _HAS_TRITON_INT8 and tensor.is_cuda:
                try:
                    qdata, scale = triton_act_quant(tensor, block_size=block_size)
                except Exception as e:
                    logging.warning(f"Triton act_quant failed: {e}, falling back to PyTorch")
                    qdata, scale = cls._activation_quantize_pytorch(tensor, block_size)
            else:
                qdata, scale = cls._activation_quantize_pytorch(tensor, block_size, scale)
        
        layout_params = {
            'scale': scale.to(torch.float32),
            'block_size': block_size,
            'is_weight': is_weight,
            'orig_dtype': orig_dtype
        }
        
        return qdata, layout_params
    
    @staticmethod
    def _weight_quantize_pytorch(tensor, block_size, scale=None):
        """PyTorch fallback for weight quantization"""
        M, N = tensor.shape
        # Reshape to (M//block_size, block_size, N//block_size, block_size)
        tensor_blocked = tensor.reshape(M // block_size, block_size, N // block_size, block_size)
        # Permute to (M//block_size, N//block_size, block_size, block_size)
        tensor_blocked = tensor_blocked.permute(0, 2, 1, 3)
        
        if scale is None:
            # Compute per-block absolute maximum
            amax = tensor_blocked.abs().amax(dim=(-2, -1))
            scale = amax / 127.0
            scale = torch.maximum(scale, torch.tensor(1e-8, device=scale.device, dtype=scale.dtype))
        
        # Broadcast scale for division: (M//block_size, N//block_size, 1, 1)
        scale_broadcast = scale.unsqueeze(-1).unsqueeze(-1)
        tensor_scaled = tensor_blocked / scale_broadcast
        
        # Clamp and convert to int8
        tensor_scaled = torch.clamp(tensor_scaled, -127.0, 127.0)
        qdata = tensor_scaled.to(torch.int8)
        
        # Reshape back to original shape
        qdata = qdata.permute(0, 2, 1, 3).reshape(M, N)
        return qdata, scale
    
    @staticmethod
    def _activation_quantize_pytorch(tensor, block_size, scale=None):
        """PyTorch fallback for activation quantization"""
        K = tensor.shape[-1]
        batch_shape = tensor.shape[:-1]
        tensor_blocked = tensor.reshape(*batch_shape, K // block_size, block_size)
        
        if scale is None:
            # Compute per-block absolute maximum
            amax = tensor_blocked.abs().amax(dim=-1)
            scale = amax / 127.0
            scale = torch.maximum(scale, torch.tensor(1e-8, device=scale.device, dtype=scale.dtype))
        
        # Broadcast scale for division
        scale_broadcast = scale.unsqueeze(-1)
        tensor_scaled = tensor_blocked / scale_broadcast
        
        # Clamp and convert to int8
        tensor_scaled = torch.clamp(tensor_scaled, -127.0, 127.0)
        qdata = tensor_scaled.to(torch.int8)
        
        # Reshape back to original shape
        qdata = qdata.reshape(tensor.shape)
        return qdata, scale
    
    @staticmethod
    def dequantize(qdata, scale, block_size, is_weight=False, orig_dtype=None, output_dtype=None, **kwargs):
        """
        Dequantize INT8 tensor back to original precision.
        
        Args:
            qdata: Quantized INT8 tensor
            scale: Per-block scaling factors
            block_size: Size of quantization blocks
            is_weight: Whether this is a weight tensor
            orig_dtype: Target dtype for dequantization
        
        Returns:
            Dequantized tensor in orig_dtype
        """
        if not qdata.is_contiguous():
            qdata = qdata.contiguous()
        if not scale.is_contiguous():
            scale = scale.contiguous()
        
        if is_weight:
            # Weight dequantization
            if _HAS_TRITON_INT8 and qdata.dim() == 2 and qdata.is_cuda:
                try:
                    dequant = triton_weight_dequant(qdata, scale, block_size=block_size, output_dtype=output_dtype if output_dtype is not None else orig_dtype)
                    return dequant
                except Exception as e:
                    logging.warning(f"Triton weight_dequant failed: {e}, falling back to PyTorch")
                    raise e
            
            # PyTorch fallback
            M, N = qdata.shape
            # Ensure scale has the correct shape for weight dequantization
            expected_scale_shape = (M // block_size, N // block_size)
            if scale.shape != expected_scale_shape:
                expected_numel = (M // block_size) * (N // block_size)
                if scale.numel() == expected_numel:
                    scale = scale.reshape(expected_scale_shape)
                else:
                    raise RuntimeError(
                        f"Weight dequant scale shape mismatch: scale.shape={scale.shape}, expected {expected_scale_shape}"
                    )
            qdata_blocked = qdata.reshape(M // block_size, block_size, N // block_size, block_size)
            qdata_blocked = qdata_blocked.permute(0, 2, 1, 3)
            scale_broadcast = scale.unsqueeze(-1).unsqueeze(-1)
            dequant = qdata_blocked.to(orig_dtype) * scale_broadcast
            dequant = dequant.permute(0, 2, 1, 3).reshape(M, N)
        else:
            # Activation dequantization
            if _HAS_TRITON_INT8 and qdata.is_cuda:
                try:
                    dequant = triton_act_dequant(qdata, scale, block_size=block_size, output_dtype=output_dtype if output_dtype is not None else orig_dtype)
                    return dequant
                except Exception as e:
                    logging.warning(f"Triton act_dequant failed: {e}, falling back to PyTorch")
                    raise e
            
            # PyTorch fallback
            batch_shape = qdata.shape[:-1]
            K = qdata.shape[-1]
            # Ensure scale has the correct shape for activation dequantization
            expected_scale_shape = (*batch_shape, K // block_size)
            if scale.shape != expected_scale_shape:
                expected_numel = 1
                for dim in expected_scale_shape:
                    expected_numel *= dim
                if scale.numel() == expected_numel:
                    scale = scale.reshape(expected_scale_shape)
                else:
                    raise RuntimeError(
                        f"Activation dequant scale shape mismatch: scale.shape={scale.shape}, expected {expected_scale_shape}"
                    )
            qdata_blocked = qdata.reshape(*batch_shape, K // block_size, block_size)
            scale_broadcast = scale.unsqueeze(-1)
            dequant = qdata_blocked.to(orig_dtype) * scale_broadcast
            dequant = dequant.reshape(qdata.shape)
        
        return dequant
    
    @classmethod
    def get_plain_tensors(cls, qtensor):
        """
        Extract raw tensors for computation.
        
        Returns:
            Tuple of (qdata, scale, block_size, is_weight)
        """
        return (
            qtensor._qdata,
            qtensor._layout_params['scale'],
            qtensor._layout_params['block_size'],
            qtensor._layout_params['is_weight']
        )


QUANT_ALGOS = {
    "float8_e4m3fn": {
        "storage_t": torch.float8_e4m3fn,
        "parameters": {"weight_scale", "input_scale"},
        "comfy_tensor_layout": "TensorCoreFP8Layout",
    },
    "int8_blockwise": {
        "storage_t": torch.int8,
        "parameters": {"weight_scale", "input_scale"},
        "comfy_tensor_layout": "BlockWiseINT8Layout",
        "group_size": 128,  # Default block size,
        "asymmetric_layout": True,
    },
}

LAYOUTS = {
    "TensorCoreFP8Layout": TensorCoreFP8Layout,
    "BlockWiseINT8Layout": BlockWiseINT8Layout,
}


@register_layout_op(torch.ops.aten.linear.default, "TensorCoreFP8Layout")
def fp8_linear(func, args, kwargs):
    input_tensor = args[0]
    weight = args[1]
    bias = args[2] if len(args) > 2 else None

    if isinstance(input_tensor, QuantizedTensor) and isinstance(weight, QuantizedTensor):
        plain_input, scale_a = TensorCoreFP8Layout.get_plain_tensors(input_tensor)
        plain_weight, scale_b = TensorCoreFP8Layout.get_plain_tensors(weight)

        out_dtype = kwargs.get("out_dtype")
        if out_dtype is None:
            out_dtype = input_tensor._layout_params['orig_dtype']

        weight_t = plain_weight.t()

        tensor_2d = False
        if len(plain_input.shape) == 2:
            tensor_2d = True
            plain_input = plain_input.unsqueeze(1)

        input_shape = plain_input.shape
        if len(input_shape) != 3:
            return None

        try:
            output = torch._scaled_mm(
                plain_input.reshape(-1, input_shape[2]).contiguous(),
                weight_t,
                bias=bias,
                scale_a=scale_a,
                scale_b=scale_b,
                out_dtype=out_dtype,
            )

            if isinstance(output, tuple):  # TODO: remove when we drop support for torch 2.4
                output = output[0]

            if not tensor_2d:
                output = output.reshape((-1, input_shape[1], weight.shape[0]))

            if output.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                output_scale = scale_a * scale_b
                output_params = {
                    'scale': output_scale,
                    'orig_dtype': input_tensor._layout_params['orig_dtype']
                }
                return QuantizedTensor(output, "TensorCoreFP8Layout", output_params)
            else:
                return output

        except Exception as e:
            raise RuntimeError(f"FP8 _scaled_mm failed, falling back to dequantization: {e}")

    # Case 2: DQ Fallback
    if isinstance(weight, QuantizedTensor):
        weight = weight.dequantize()
    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()

    return torch.nn.functional.linear(input_tensor, weight, bias)

def fp8_mm_(input_tensor, weight, bias=None, out_dtype=None):
    if out_dtype is None:
        out_dtype = input_tensor._layout_params['orig_dtype']

    plain_input, scale_a = TensorCoreFP8Layout.get_plain_tensors(input_tensor)
    plain_weight, scale_b = TensorCoreFP8Layout.get_plain_tensors(weight)

    output = torch._scaled_mm(
        plain_input.contiguous(),
        plain_weight,
        bias=bias,
        scale_a=scale_a,
        scale_b=scale_b,
        out_dtype=out_dtype,
    )

    if isinstance(output, tuple):  # TODO: remove when we drop support for torch 2.4
        output = output[0]
    return output

@register_layout_op(torch.ops.aten.addmm.default, "TensorCoreFP8Layout")
def fp8_addmm(func, args, kwargs):
    input_tensor = args[1]
    weight = args[2]
    bias = args[0]

    if isinstance(input_tensor, QuantizedTensor) and isinstance(weight, QuantizedTensor):
        return fp8_mm_(input_tensor, weight, bias=bias, out_dtype=kwargs.get("out_dtype", None))

    a = list(args)
    if isinstance(args[0], QuantizedTensor):
        a[0] = args[0].dequantize()
    if isinstance(args[1], QuantizedTensor):
        a[1] = args[1].dequantize()
    if isinstance(args[2], QuantizedTensor):
        a[2] = args[2].dequantize()

    return func(*a, **kwargs)

@register_layout_op(torch.ops.aten.mm.default, "TensorCoreFP8Layout")
def fp8_mm(func, args, kwargs):
    input_tensor = args[0]
    weight = args[1]

    if isinstance(input_tensor, QuantizedTensor) and isinstance(weight, QuantizedTensor):
        return fp8_mm_(input_tensor, weight, bias=None, out_dtype=kwargs.get("out_dtype", None))

    a = list(args)
    if isinstance(args[0], QuantizedTensor):
        a[0] = args[0].dequantize()
    if isinstance(args[1], QuantizedTensor):
        a[1] = args[1].dequantize()
    return func(*a, **kwargs)

@register_layout_op(torch.ops.aten.view.default, "TensorCoreFP8Layout")
@register_layout_op(torch.ops.aten.t.default, "TensorCoreFP8Layout")
def fp8_func(func, args, kwargs):
    input_tensor = args[0]
    if isinstance(input_tensor, QuantizedTensor):
        plain_input, scale_a = TensorCoreFP8Layout.get_plain_tensors(input_tensor)
        ar = list(args)
        ar[0] = plain_input
        return QuantizedTensor(func(*ar, **kwargs), "TensorCoreFP8Layout", input_tensor._layout_params)
    return func(*args, **kwargs)


# ==============================================================================
# Block-Wise INT8 Operation Handlers
# ==============================================================================

def _int8_gemm_pytorch_fallback(a_int8, a_scale, b_int8, b_scale, block_size, bias=None):
    """
    PyTorch fallback for INT8 matrix multiplication: dequantize and use standard matmul.
    
    Args:
        a_int8: INT8 activations, shape (*batch, K)
        a_scale: Activation scales, shape (*batch, K//block_size)
        b_int8: INT8 weights, shape (N, K) - standard PyTorch weight format
        b_scale: Weight scales, shape (N//block_size, K//block_size)
        block_size: Block size for quantization
        bias: Optional bias vector, shape (N,)
    
    Returns:
        Output in float32, shape (*batch, N)
    """
    K = a_int8.shape[-1]
    batch_shape = a_int8.shape[:-1]
    N = b_int8.shape[0]
    
    # Dequantize activations
    # Ensure a_scale has the correct shape - it should be (*batch_shape, K // block_size)
    expected_scale_shape = (*batch_shape, K // block_size)
    if a_scale.shape != expected_scale_shape:
        # Try to reshape if the number of elements matches
        expected_numel = 1
        for dim in expected_scale_shape:
            expected_numel *= dim
        if a_scale.numel() == expected_numel:
            a_scale = a_scale.reshape(expected_scale_shape)
        else:
            raise RuntimeError(
                f"Scale shape mismatch: a_scale.shape={a_scale.shape}, expected {expected_scale_shape}. " +
                f"a_int8.shape={a_int8.shape}, K={K}, block_size={block_size}"
            )
    
    a_blocked = a_int8.reshape(*batch_shape, K // block_size, block_size)
    a_scale_broadcast = a_scale.unsqueeze(-1)
    a_fp32 = a_blocked.to(torch.float32) * a_scale_broadcast
    a_fp32 = a_fp32.reshape(*batch_shape, K)
    
    # Dequantize weights
    # b_int8 is in (N, K) format (standard weight format), b_scale is in (N//block_size, K//block_size) format
    expected_weight_scale_shape = (N // block_size, K // block_size)
    if b_scale.shape != expected_weight_scale_shape:
        # Try to reshape if the number of elements matches
        expected_weight_numel = (N // block_size) * (K // block_size)
        if b_scale.numel() == expected_weight_numel:
            b_scale = b_scale.reshape(expected_weight_scale_shape)
        else:
            raise RuntimeError(
                f"Weight scale shape mismatch: b_scale.shape={b_scale.shape}, expected {expected_weight_scale_shape}. " +
                f"b_int8.shape={b_int8.shape}, N={N}, K={K}, block_size={block_size}"
            )
    
    # Dequantize weight: (N, K) -> blocks -> dequantize -> (N, K)
    b_blocked = b_int8.reshape(N // block_size, block_size, K // block_size, block_size)
    b_blocked = b_blocked.permute(0, 2, 1, 3)  # (N//bs, K//bs, bs, bs)
    b_scale_broadcast = b_scale.unsqueeze(-1).unsqueeze(-1)
    b_fp32 = b_blocked.to(torch.float32) * b_scale_broadcast
    b_fp32 = b_fp32.permute(0, 2, 1, 3).reshape(N, K)  # Back to (N, K)
    
    output = torch.nn.functional.linear(a_fp32, b_fp32, bias)
    return output


def _int8_gemm_triton_or_fallback(a_int8, a_scale, b_int8, b_scale, block_size, bias=None, out_quant=False):
    """
    INT8 matrix multiplication with optional fused bias using Triton kernels or PyTorch fallback.
    
    Args:
        a_int8: INT8 activations, shape (*batch, K)
        a_scale: Activation scales, shape (*batch, K//block_size)
        b_int8: INT8 weights, shape (N, K) - standard PyTorch weight format
        b_scale: Weight scales, shape (N//block_size, K//block_size)
        block_size: Block size for quantization
        bias: Optional bias vector, shape (N,)
        out_quant: If True, return quantized output (INT8 + scales) instead of float
    
    Returns:
        If out_quant=False: Output in float16/float32, shape (*batch, N)
        If out_quant=True: Tuple of (output_int8, output_scale)
    """
    K = a_int8.shape[-1]
    batch_shape = a_int8.shape[:-1]
    # b_int8 is weight in (N, K) format (standard PyTorch weight format)
    N = b_int8.shape[0]
    assert b_int8.shape[1] == K, f"Weight shape mismatch: expected b_int8.shape[1]={K}, got {b_int8.shape[1]}"
    
    # Try Triton kernel first (only if tensors are on CUDA)
    if _HAS_TRITON_INT8 and a_int8.is_cuda:
        try:
            # int8_gemm/int8_addmm expects: (a, a_s, b, b_s, [bias])
            # a: (*batch, K), a_s: (*batch, K//block_size)
            # b: (N, K), b_s: (N//block_size, K//block_size)
            # Triton kernels transpose b internally
            
            # Reshape activations to 2D for int8_gemm
            a_2d = a_int8.reshape(-1, K).contiguous()
            a_scale_2d = a_scale.reshape(-1, a_scale.shape[-1]).contiguous()
            
            # Ensure weight tensors are contiguous
            b_int8_c = b_int8.contiguous()
            b_scale_c = b_scale.contiguous()
            
            # Call appropriate Triton kernel based on out_quant flag
            if out_quant:
                # Use fused matmul + quantization kernels
                if bias is not None:
                    # Fused addmm + quantization
                    output_2d, output_scale_2d = triton_int8_addmm_quant(
                        a_2d, a_scale_2d, b_int8_c, b_scale_c, bias, out_block_size=block_size
                    )
                else:
                    # Fused gemm + quantization
                    output_2d, output_scale_2d = triton_int8_gemm_quant(
                        a_2d, a_scale_2d, b_int8_c, b_scale_c, out_block_size=block_size
                    )
                
                # Reshape back to original batch shape
                output = output_2d.reshape(*batch_shape, N)
                output_scale = output_scale_2d.reshape(*batch_shape, N // block_size)
                return output, output_scale
            else:
                # Standard float output
                if bias is not None:
                    # Use fused addmm kernel
                    output_2d = triton_int8_addmm(a_2d, a_scale_2d, b_int8_c, b_scale_c, bias)
                else:
                    # Use standard gemm kernel
                    output_2d = triton_int8_gemm(a_2d, a_scale_2d, b_int8_c, b_scale_c)
                
                # Reshape back to original batch shape
                output = output_2d.reshape(*batch_shape, N)
                return output
        except Exception as e:
            logging.warning(f"Triton int8_gemm/addmm failed: {e}, falling back to PyTorch")
            raise e
    
    # Use PyTorch fallback
    fallback_output = _int8_gemm_pytorch_fallback(a_int8, a_scale, b_int8, b_scale, block_size, bias)
    
    # If out_quant is requested, quantize the fallback output
    if out_quant:
        # Use PyTorch activation quantization on the output
        from .int8_kernels import act_quant
        try:
            output_int8, output_scale = act_quant(fallback_output, block_size=block_size)
            return output_int8, output_scale
        except:
            # Fallback to CPU quantization if Triton not available
            output_int8, output_scale = BlockWiseINT8Layout._activation_quantize_pytorch(
                fallback_output, block_size
            )
            return output_int8, output_scale
    
    return fallback_output


@register_layout_op(torch.ops.aten.linear.default, "BlockWiseINT8Layout")
def int8_linear(func, args, kwargs):
    """
    Block-wise INT8 linear operation handler with fused Triton kernel support.
    
    Supports:
    - Both quantized input and weight (uses Triton int8_addmm with fused bias)
    - Mixed precision (quantized weight, float input)
    - Optional quantized output via out_dtype and out_quant parameters
    """
    input_tensor = args[0]
    weight = args[1]
    bias = args[2] if len(args) > 2 else None
    
    # Case 1: Both input and weight are quantized
    if isinstance(input_tensor, QuantizedTensor) and isinstance(weight, QuantizedTensor):

        # Extract quantized data
        a_int8, a_scale, a_block_size, a_is_weight = BlockWiseINT8Layout.get_plain_tensors(input_tensor)
        b_int8, b_scale, b_block_size, b_is_weight = BlockWiseINT8Layout.get_plain_tensors(weight)
        
        # Verify configurations
        assert not a_is_weight, "Input tensor should not be marked as weight"
        assert b_is_weight, "Weight tensor should be marked as weight"
        assert a_block_size == b_block_size, f"Block sizes must match: {a_block_size} vs {b_block_size}"
        
        orig_dtype = input_tensor._layout_params['orig_dtype']
        out_dtype = kwargs.get('out_dtype', orig_dtype)
        out_quant = kwargs.get('out_quant', False)  # Whether to return quantized output
        
        # Weight is already in (N, K) format (standard PyTorch weight format)
        # Pass out_quant to _int8_gemm_triton_or_fallback for fused matmul+quant
        result = _int8_gemm_triton_or_fallback(
            a_int8, a_scale, b_int8, b_scale, a_block_size, 
            bias=bias, out_quant=out_quant
        )
        
        # Handle quantized vs float output
        if out_quant:
            # Result is (output_int8, output_scale) tuple
            output_int8, output_scale = result
            
            # Wrap in QuantizedTensor
            layout_params = {
                'scale': output_scale,
                'block_size': a_block_size,
                'is_weight': False,
                'orig_dtype': out_dtype
            }
            return QuantizedTensor(output_int8, "BlockWiseINT8Layout", layout_params)
        else:
            # Result is float tensor
            output = result
            # Convert to target dtype if needed
            if output.dtype != out_dtype:
                output = output.to(out_dtype)
            return output
    
    # Case 2: Fallback - dequantize and use standard linear
    if isinstance(weight, QuantizedTensor):
        weight = weight.dequantize()
    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()
    
    return torch.nn.functional.linear(input_tensor, weight, bias)


@register_layout_op(torch.ops.aten.mm.default, "BlockWiseINT8Layout")
def int8_mm(func, args, kwargs):
    """Block-wise INT8 matrix multiplication handler with Triton kernel support."""
    input_tensor = args[0]
    weight = args[1]
    
    if isinstance(input_tensor, QuantizedTensor) and isinstance(weight, QuantizedTensor):
        a_int8, a_scale, a_block_size, a_is_weight = BlockWiseINT8Layout.get_plain_tensors(input_tensor)
        b_int8, b_scale, b_block_size, b_is_weight = BlockWiseINT8Layout.get_plain_tensors(weight)
        
        assert a_block_size == b_block_size, f"Block sizes must match: {a_block_size} vs {b_block_size}"
        
        # Note: For mm, we expect both to be 2D
        # If input is marked as weight (2D blocking), we need different logic
        # For simplicity, dequantize if configurations don't match expected pattern
        if a_is_weight or not b_is_weight:
            logging.warning("INT8 mm: Unexpected tensor configurations, falling back to dequantization")
            return func(input_tensor.dequantize(), weight.dequantize())
        
        orig_dtype = input_tensor._layout_params['orig_dtype']
        out_dtype = kwargs.get('out_dtype', orig_dtype)
        out_quant = kwargs.get('out_quant', False)  # Whether to return quantized output (default: True)
        
        # Check if weight needs to be transposed to (N, K) format
        # For mm: input is (M, K), weight should be (N, K) for the kernel
        K = a_int8.shape[-1]
        if b_int8.shape[0] == K and b_int8.shape[1] != K:
            # Weight is in (K, N) format (transposed), transpose back to (N, K)
            b_int8 = b_int8.t().contiguous()
            b_scale = b_scale.t().contiguous()
        
        result = _int8_gemm_triton_or_fallback(
            a_int8, a_scale, b_int8, b_scale, a_block_size, 
            bias=None, out_quant=out_quant
        )
        
        # Handle quantized vs float output
        if out_quant:
            # Result is (output_int8, output_scale) tuple
            output_int8, output_scale = result
            
            # Wrap in QuantizedTensor
            layout_params = {
                'scale': output_scale,
                'block_size': a_block_size,
                'is_weight': False,
                'orig_dtype': out_dtype
            }
            return QuantizedTensor(output_int8, "BlockWiseINT8Layout", layout_params)
        else:
            # Result is float tensor
            output = result
            # Convert to target dtype if needed
            if output.dtype != out_dtype:
                output = output.to(out_dtype)
            return output
    
    # Fallback
    a = list(args)
    if isinstance(args[0], QuantizedTensor):
        a[0] = args[0].dequantize()
    if isinstance(args[1], QuantizedTensor):
        a[1] = args[1].dequantize()
    return func(*a, **kwargs)


@register_layout_op(torch.ops.aten.addmm.default, "BlockWiseINT8Layout")
def int8_addmm(func, args, kwargs):
    """
    Block-wise INT8 addmm operation handler with fused Triton kernel support.
    addmm: out = beta * input + alpha * (mat1 @ mat2)
    
    This uses the fused int8_addmm kernel which combines matmul and bias addition
    in a single pass for better performance.
    
    Args:
        args[0]: bias tensor
        args[1]: mat1 (input)
        args[2]: mat2 (weight)
    """
    bias = args[0]
    input_tensor = args[1]
    weight = args[2]
    
    # Case 1: Both input and weight are quantized
    if isinstance(input_tensor, QuantizedTensor) and isinstance(weight, QuantizedTensor):
        # Extract quantized data
        a_int8, a_scale, a_block_size, a_is_weight = BlockWiseINT8Layout.get_plain_tensors(input_tensor)
        b_int8, b_scale, b_block_size, b_is_weight = BlockWiseINT8Layout.get_plain_tensors(weight)
        
        # Verify configurations
        assert a_block_size == b_block_size, f"Block sizes must match: {a_block_size} vs {b_block_size}"
        
        orig_dtype = input_tensor._layout_params['orig_dtype']
        out_dtype = kwargs.get('out_dtype', orig_dtype)
        out_quant = kwargs.get('out_quant', False)  # Whether to return quantized output
        
        # PyTorch's F.linear internally calls addmm(bias, input, weight.t())
        # So weight arrives in (K, N) format (transposed), need to transpose back to (N, K)
        # Check if weight is transposed by comparing dimensions with input
        K = a_int8.shape[-1]
        if b_is_weight and b_int8.shape[0] == K:
            # Weight is in (K, N) format (transposed), transpose back to (N, K)
            # The transpose handler also transposed the scale, so we need to transpose it back too
            b_int8 = b_int8.t().contiguous()
            b_scale = b_scale.t().contiguous()
        
        # Use fused Triton kernel (combines matmul + bias + optional quant)
        result = _int8_gemm_triton_or_fallback(
            a_int8, a_scale, b_int8, b_scale, a_block_size, 
            bias=bias, out_quant=out_quant
        )
        
        # Handle quantized vs float output
        if out_quant:
            # Result is (output_int8, output_scale) tuple
            output_int8, output_scale = result
            
            # Wrap in QuantizedTensor
            layout_params = {
                'scale': output_scale,
                'block_size': a_block_size,
                'is_weight': False,
                'orig_dtype': out_dtype
            }
            return QuantizedTensor(output_int8, "BlockWiseINT8Layout", layout_params)
        else:
            # Result is float tensor
            output = result
            # Convert to target dtype if needed
            if output.dtype != out_dtype:
                output = output.to(out_dtype)
            return output
    
    # Fallback: dequantize and use standard addmm
    a = list(args)
    if isinstance(args[0], QuantizedTensor):
        a[0] = args[0].dequantize()
    if isinstance(args[1], QuantizedTensor):
        a[1] = args[1].dequantize()
    if isinstance(args[2], QuantizedTensor):
        a[2] = args[2].dequantize()
    
    return func(*a, **kwargs)


@register_layout_op(torch.ops.aten.view.default, "BlockWiseINT8Layout")
def int8_view(func, args, kwargs):
    """Handle view operations for INT8 tensors."""
    input_tensor = args[0]
    if isinstance(input_tensor, QuantizedTensor):
        # For view, we need to be careful with block structure
        # For safety, we'll allow these ops but note that they might break block alignment
        plain_input = input_tensor._qdata
        ar = list(args)
        ar[0] = plain_input
        transformed = func(*ar, **kwargs)
        
        # Return new QuantizedTensor with same layout params
        # Note: This assumes the transformation preserves block structure
        return QuantizedTensor(transformed, "BlockWiseINT8Layout", input_tensor._layout_params)
    return func(*args, **kwargs)


@register_layout_op(torch.ops.aten.t.default, "BlockWiseINT8Layout")
def int8_transpose(func, args, kwargs):
    """Handle transpose operations for INT8 tensors."""
    input_tensor = args[0]
    if isinstance(input_tensor, QuantizedTensor):
        # Transpose the quantized data
        plain_input = input_tensor._qdata
        ar = list(args)
        ar[0] = plain_input
        transformed = func(*ar, **kwargs)
        
        # For weight tensors, we need to transpose the scale tensor as well
        new_layout_params = input_tensor._layout_params.copy()
        if new_layout_params.get('is_weight', False):
            # Transpose the scale tensor to match the transposed weight
            new_layout_params['scale'] = new_layout_params['scale'].t().contiguous()
        
        # Return new QuantizedTensor with updated layout params
        return QuantizedTensor(transformed, "BlockWiseINT8Layout", new_layout_params)
    return func(*args, **kwargs)


@register_layout_op(torch.ops.aten.transpose.int, "BlockWiseINT8Layout")
def int8_transpose_int(func, args, kwargs):
    """
    Handle general transpose operations for INT8 tensors.
    
    torch.transpose(input, dim0, dim1) swaps two dimensions.
    
    For BlockWiseINT8Layout:
    - Activations: quantized along last dimension, scale shape is (*batch_dims, K//block_size)
      If we swap the last dimension, we need to adjust scale handling
    - Weights: quantized in 2D blocks (M, N), scale shape is (M//block_size, N//block_size)
      If we swap dimensions on a 2D weight, transpose the scale tensor too
    """
    input_tensor = args[0]
    dim0 = args[1] if len(args) > 1 else kwargs.get('dim0', 0)
    dim1 = args[2] if len(args) > 2 else kwargs.get('dim1', 1)
    
    if isinstance(input_tensor, QuantizedTensor):
        # Transpose the quantized data
        plain_input = input_tensor._qdata
        ar = list(args)
        ar[0] = plain_input
        transformed = func(*ar, **kwargs)
        
        # Copy layout params
        new_layout_params = input_tensor._layout_params.copy()
        is_weight = new_layout_params.get('is_weight', False)
        
        # Normalize dimensions to positive indices
        ndim = plain_input.ndim
        if dim0 < 0:
            dim0 = ndim + dim0
        if dim1 < 0:
            dim1 = ndim + dim1
        
        # Handle scale tensor transposition
        if is_weight:
            # For weight tensors (2D with block-wise quantization in both dims)
            # If we're transposing the two dimensions of a 2D tensor, transpose scales too
            if ndim == 2 and set([dim0, dim1]) == {0, 1}:
                # Transposing a 2D weight tensor (M, N) -> (N, M)
                # Scale goes from (M//block_size, N//block_size) -> (N//block_size, M//block_size)
                new_layout_params['scale'] = new_layout_params['scale'].t().contiguous()
            else:
                # For higher dimensional weight tensors or partial transposes,
                # we may need more complex scale handling
                # For now, log a warning as this is an uncommon case
                logging.warning(
                    f"Transpose on weight tensor with dims ({dim0}, {dim1}) and shape {plain_input.shape}. "
                    f"Scale tensor may need adjustment for correct behavior."
                )
        else:
            # For activation tensors, block-wise quantization is along last dimension
            # If we're swapping the last dimension, this changes the quantization structure
            last_dim = ndim - 1
            if dim0 == last_dim or dim1 == last_dim:
                # The last dimension is being moved, which affects quantization blocks
                # This is a complex case - for safety, we could:
                # 1. Dequantize, transpose, requantize (safest but slower)
                # 2. Try to adjust scale tensor (complex, error-prone)
                # For now, log a warning and proceed with transposing the scale tensor
                # The scale tensor dimensions follow the input dimensions except the last
                # which is divided by block_size
                
                # Determine how to transpose the scale tensor
                # Scale shape is (*batch_dims, K//block_size) where K is the last dim of input
                # When we transpose input dims, we need to transpose scale dims accordingly
                # But the last scale dim always corresponds to the quantization blocks
                
                # Simple heuristic: if transposing involves last dim and input has 3+ dims,
                # we transpose the corresponding scale dimensions
                scale = new_layout_params['scale']
                if scale.ndim >= 2:
                    # Map input dimensions to scale dimensions
                    # Scale has shape (*batch_dims, K//block_size)
                    # If input has shape (*batch_dims, K), scale maps batch_dims directly
                    # and last dim is K//block_size
                    
                    # For transpose, if we swap dims d0 and d1 in input:
                    # - If d1 is last_dim (K), then in scale it's still last (K//block_size)
                    # - If d0 is last_dim, same applies
                    # - If neither is last_dim, transpose applies to batch dimensions
                    
                    if dim1 == last_dim:
                        # Swapping some batch dim with the last dim
                        # In scale, this means swapping that batch dim with last scale dim
                        scale_dim0 = dim0  # Same batch dimension
                        scale_dim1 = scale.ndim - 1  # Last dim of scale (K//block_size)
                        new_layout_params['scale'] = scale.transpose(scale_dim0, scale_dim1).contiguous()
                    elif dim0 == last_dim:
                        # Swapping last dim with some batch dim
                        scale_dim0 = scale.ndim - 1  # Last dim of scale
                        scale_dim1 = dim1  # Same batch dimension
                        new_layout_params['scale'] = scale.transpose(scale_dim0, scale_dim1).contiguous()
                    else:
                        # Swapping two batch dimensions (not involving last dim)
                        # Transpose the same dimensions in scale
                        new_layout_params['scale'] = scale.transpose(dim0, dim1).contiguous()
                else:
                    logging.warning(
                        f"Transpose involves last dimension but scale tensor has shape {scale.shape}. "
                        f"Scale tensor may need adjustment."
                    )
            else:
                # Transposing batch dimensions that don't affect the quantized dimension
                # Transpose the same dimensions in scale tensor
                scale = new_layout_params['scale']
                if scale.ndim > max(dim0, dim1):
                    new_layout_params['scale'] = scale.transpose(dim0, dim1).contiguous()
        
        # Return new QuantizedTensor with updated layout params
        return QuantizedTensor(transformed, "BlockWiseINT8Layout", new_layout_params)
    
    return func(*args, **kwargs)


@register_layout_op(torch.ops.aten.gelu.default, "BlockWiseINT8Layout")
def int8_gelu(func, args, kwargs):
    """
    Block-wise INT8 GELU activation handler with fused Triton kernel support.
    
    Supports quantized input -> GELU -> quantized output in a single fused kernel.
    This avoids materializing full-precision intermediate results.
    """
    input_tensor = args[0]
    
    # Case 1: Input is quantized - use fused kernel
    if isinstance(input_tensor, QuantizedTensor):
        # Extract quantized data
        qdata, scale, block_size, is_weight = BlockWiseINT8Layout.get_plain_tensors(input_tensor)
        
        orig_dtype = input_tensor._layout_params['orig_dtype']
        
        # Determine if we should use Triton kernel
        if _HAS_TRITON_INT8 and qdata.is_cuda:
            try:
                # Import the Triton kernel
                from .int8_kernels import int8_gelu as triton_int8_gelu
                
                # Call fused kernel
                output_qdata, output_scale = triton_int8_gelu(qdata, scale, block_size=block_size)
                
                # Wrap result in QuantizedTensor
                layout_params = {
                    'scale': output_scale.to(torch.float32),
                    'block_size': block_size,
                    'is_weight': False,  # Output is always activation format
                    'orig_dtype': orig_dtype
                }
                return QuantizedTensor(output_qdata, "BlockWiseINT8Layout", layout_params)
            
            except Exception as e:
                logging.warning(f"Triton int8_gelu failed: {e}, falling back to dequantization")
                # Fall through to dequantization fallback
        
        # Fallback: dequantize, apply GELU, quantize
        fp_input = input_tensor.dequantize()
        fp_output = torch.nn.functional.gelu(fp_input)
        
        # Quantize output
        output_qdata, output_layout_params = BlockWiseINT8Layout.quantize(
            fp_output, 
            block_size=block_size, 
            is_weight=False
        )
        output_layout_params['orig_dtype'] = orig_dtype
        
        return QuantizedTensor(output_qdata, "BlockWiseINT8Layout", output_layout_params)
    
    # Case 2: Input is not quantized - use standard GELU
    return func(*args, **kwargs)


@register_layout_op(torch.ops.aten.add_.Tensor, "BlockWiseINT8Layout")
def int8_add_(func, args, kwargs):
    """
    Block-wise INT8 in-place addition handler for LoRA application.
    
    This operation is typically used when applying LoRA to weight matrices.
    Since speed is not critical for this operation:
    - If target is a weight: dequantize, add, then requantize as weight
    - Otherwise: dequantize and fallback to regular addition
    
    Args:
        args[0]: Target tensor (self) to be modified in-place
        args[1]: Tensor to add
    """
    target = args[0]
    
    if isinstance(target, QuantizedTensor):
        # Extract quantization parameters
        _, _, block_size, is_weight = BlockWiseINT8Layout.get_plain_tensors(target)
        
        # Only handle the weight case specially
        if is_weight:
            other = args[1]
            orig_dtype = target._layout_params['orig_dtype']
            
            # Dequantize target
            target_fp = target.dequantize()
            
            # Dequantize other if it's also quantized
            if isinstance(other, QuantizedTensor):
                other_fp = other.dequantize()
            else:
                other_fp = other
            
            # Perform addition
            result_fp = target_fp + other_fp
            
            # Requantize as weight
            result_qdata, result_layout_params = BlockWiseINT8Layout.quantize(
                result_fp,
                block_size=block_size,
                is_weight=True
            )
            result_layout_params['orig_dtype'] = orig_dtype
            
            # Update target in-place by copying the new quantized data
            target._qdata.copy_(result_qdata)
            target._layout_params['scale'].copy_(result_layout_params['scale'])
            return target
    
    # For non-weight tensors or non-quantized tensors, use standard fallback
    return QuantizedTensor._dequant_and_fallback(func, args, kwargs)


@register_layout_op(torch.ops.aten.to.dtype, "BlockWiseINT8Layout")
def int8_to_dtype(func, args, kwargs):
    """
    Block-wise INT8 dtype conversion handler.
    
    This operation handles .to(dtype) calls on quantized tensors.
    - If converting to torch.int8, do nothing (already in INT8 format)
    - Otherwise, dequantize and fallback
    
    Args:
        args[0]: Input tensor
        args[1]: Target dtype
    """
    input_tensor = args[0]
    target_dtype = args[1] if len(args) > 1 else kwargs.get('dtype', None)
    
    if isinstance(input_tensor, QuantizedTensor):
        # If target dtype is int8, the tensor is already in INT8 format
        if target_dtype == torch.int8:
            # No conversion needed, return as-is
            return input_tensor
    
    # For any other dtype or non-quantized tensors, use standard fallback
    return QuantizedTensor._dequant_and_fallback(func, args, kwargs)
