import torch
import logging
from typing import Tuple, Dict
import comfy.float

_LAYOUT_REGISTRY = {}
_GENERIC_UTILS = {}


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
    def _extract_layout(obj):
        if isinstance(obj, QuantizedTensor):
            return obj._layout_type
        # For torch.nn.Parameter wrapping QuantizedTensor, check the data attribute
        if isinstance(obj, torch.nn.Parameter):
            if isinstance(obj.data, QuantizedTensor):
                return obj.data._layout_type
            if hasattr(obj.data, "_layout_type"):
                return getattr(obj.data, "_layout_type", None)
        if hasattr(obj, "_layout_type"):
            return getattr(obj, "_layout_type", None)
        return None

    for arg in args:
        layout = _extract_layout(arg)
        if layout is not None:
            return layout
        if isinstance(arg, (list, tuple)):
            for item in arg:
                layout = _extract_layout(item)
                if layout is not None:
                    return layout
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
        if isinstance(args[0] if args else None, QuantizedTensor):
            logging.info(f"QuantizedTensor: Unhandled operation {func}, falling back to dequantization. kwargs={kwargs}")
        return cls._dequant_and_fallback(func, args, kwargs)

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

    def data_ptr(self):
        return self._qdata.data_ptr()

    def is_pinned(self):
        return self._qdata.is_pinned()

    def is_contiguous(self, *arg, **kwargs):
        return self._qdata.is_contiguous(*arg, **kwargs)

# ==============================================================================
# Generic Utilities (Layout-Agnostic Operations)
# ==============================================================================

def _create_transformed_qtensor(qt, transform_fn):
    new_data = transform_fn(qt._qdata)
    new_params = _copy_layout_params(qt._layout_params)
    return QuantizedTensor(new_data, qt._layout_type, new_params)


def _handle_device_transfer(qt, target_device, target_dtype=None, target_layout=None, op_name="to"):
    if target_dtype is not None and target_dtype != qt.dtype:
        logging.warning(
            f"QuantizedTensor: dtype conversion requested to {target_dtype}, "
            f"but not supported for quantized tensors. Ignoring dtype."
        )

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
            _copy_layout_params_inplace(src._layout_params, qt_dest._layout_params, non_blocking=non_blocking)
        else:
            # Copy from regular tensor - just copy raw data
            qt_dest._qdata.copy_(src)
        return qt_dest
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

        if scale is None:
            scale = torch.amax(tensor.abs()) / torch.finfo(dtype).max

        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale)
        scale = scale.to(device=tensor.device, dtype=torch.float32)

        if inplace_ops:
            tensor *= (1.0 / scale).to(tensor.dtype)
        else:
            tensor = tensor * (1.0 / scale).to(tensor.dtype)

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

QUANT_ALGOS = {
    "float8_e4m3fn": {
        "storage_t": torch.float8_e4m3fn,
        "parameters": {"weight_scale", "input_scale"},
        "comfy_tensor_layout": "TensorCoreFP8Layout",
    },
    "svdquant_int4": {
        "storage_t": torch.int8,  # Packed 4-bit stored in int8
        "parameters": {
            "wscales",
            "smooth_factor", 
            "smooth_factor_orig",
            "proj_down",
            "proj_up",
        },
        "custom_layer_params_keys": ["wscales", "smooth_factor", "smooth_factor_orig", "proj_down", "proj_up"],
        "comfy_tensor_layout": "SVDQuantLayout",
        "group_size": 64,
        "precision": "int4",
    },
    "svdquant_nvfp4": {
        "storage_t": torch.int8,  # Packed 4-bit stored in int8
        "parameters": {
            "wscales",
            "smooth_factor",
            "smooth_factor_orig", 
            "proj_down",
            "proj_up",
            "wtscale",
            "wcscales",
        },
        "custom_layer_params_keys": ["wscales", "smooth_factor", "smooth_factor_orig", "proj_down", "proj_up", "wtscale", "wcscales"],
        "comfy_tensor_layout": "SVDQuantLayout",
        "group_size": 16,
        "precision": "nvfp4",
    },
    "awq_int4": {
        "storage_t": torch.int32,  # Packed 4-bit stored in int32
        "parameters": {
            "wscales",
            "wzeros",
        },
        "custom_layer_params_keys": ["wscales", "wzeros"],
        "comfy_tensor_layout": "AWQQuantLayout",
        "group_size": 64,
    },
}

LAYOUTS = {
    "TensorCoreFP8Layout": TensorCoreFP8Layout,
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
# SVDQuant Layout + Operation Handlers
# ==============================================================================

class SVDQuantLayout(QuantizedLayout):
    """
    SVDQuant W4A4 quantization layout.
    
    SVDQuant decomposes linear operations as:
    X*W = X * proj_up * proj_down + quantize(X) * quantize(R)
    
    Where:
    - proj_up, proj_down: Low-rank factorization of weights
    - R: Residual weights (quantized to 4-bit)
    - quantize(): 4-bit quantization with smoothing factors
    
    Storage format:
    For weights (is_weight=True):
        - qdata: Packed quantized residual weights (out_features, in_features // 2), int8
        - wscales: Weight quantization scales
        - smooth_factor: Smoothing factors for inputs
        - proj_down: Low-rank down projection
        - proj_up: Low-rank up projection
        - group_size: Quantization group size (64 for int4, 16 for nvfp4)
        - precision: 'int4' or 'nvfp4'
        - rank: SVD rank
        - wtscale: Global weight scale (nvfp4 only)
        - wcscales: Channel-wise weight scales (nvfp4 only)
        - act_unsigned: Whether activations are unsigned (int4 only)
        - orig_dtype: Original dtype before quantization
    
    For activations (is_weight=False):
        - qdata: Original activation tensor (not quantized yet)
        - orig_dtype: Original dtype
        - is_weight: False marker
    """
    
    @classmethod
    def quantize(cls, tensor, is_weight=True, **kwargs):
        """
        For SVDQuant, we don't perform online quantization.
        - Weights are pre-quantized offline and loaded from checkpoint
        - Activations are stored as-is and quantized during forward pass
        """
        orig_dtype = tensor.dtype
        
        if is_weight:
            # This shouldn't be called for weights as they're loaded pre-quantized
            raise NotImplementedError(
                "SVDQuant weights should be loaded pre-quantized from checkpoint, "
                "not quantized on-the-fly"
            )
        else:
            # For activations, just store the tensor as-is
            # It will be quantized during the linear operation
            layout_params = {
                'orig_dtype': orig_dtype,
                'is_weight': False
            }
            return tensor, layout_params
    
    @staticmethod
    def dequantize(qdata, is_weight=True, orig_dtype=None, **kwargs):
        """
        Dequantization for SVDQuant.
        - Activations: return as-is (not actually quantized)
        - Weights: full dequantization not supported (would need to reconstruct from SVD + residual)
        """
        if not is_weight:
            # Activations aren't actually quantized, just return them
            return qdata.to(orig_dtype) if orig_dtype else qdata
        else:
            # Full weight dequantization is complex and not typically needed
            # Would require: proj_down @ proj_up.T + dequantize(qweight)
            raise NotImplementedError(
                "Full dequantization of SVDQuant weights is not supported. "
                "Use the quantized forward pass instead."
            )
    
    @classmethod
    def get_plain_tensors(cls, qtensor):
        """Extract the raw tensors needed for SVDQuant computation."""
        if qtensor._layout_params.get('is_weight', True):
            # For weights, return all the necessary components
            return {
                'qweight': qtensor._qdata,
                'wscales': qtensor._layout_params.get('wscales'),
                'smooth_factor': qtensor._layout_params.get('smooth_factor'),
                'proj_down': qtensor._layout_params.get('proj_down'),
                'proj_up': qtensor._layout_params.get('proj_up'),
                'group_size': qtensor._layout_params.get('group_size'),
                'precision': qtensor._layout_params.get('precision', 'int4'),
                'wtscale': qtensor._layout_params.get('wtscale'),
                'wcscales': qtensor._layout_params.get('wcscales'),
                'act_unsigned': qtensor._layout_params.get('act_unsigned', False),
            }
        else:
            # For activations, just return the tensor
            return qtensor._qdata


@register_layout_op(torch.ops.aten.addmm.default, "SVDQuantLayout")
@register_layout_op(torch.ops.aten.linear.default, "SVDQuantLayout")
def svdquant_linear(func, args, kwargs):
    """
    SVDQuant linear operation handler.
    
    Implements: X*W = X * proj_up * proj_down + quantize(X) * quantize(R)
    
    Handles both aten.linear and aten.addmm (which linear decomposes into).
    """
    # Handle both linear and addmm calling conventions
    if func == torch.ops.aten.addmm.default:
        # addmm(bias, input, weight.t()) -> out
        bias = args[0] if len(args) > 0 else None
        input_tensor = args[1] if len(args) > 1 else None
        weight = args[2] if len(args) > 2 else None
        # Weight comes transposed in addmm, but SVDQuant stores it non-transposed
        # So we need to transpose it back
        need_transpose = True
    else:
        # linear(input, weight, bias) -> out
        input_tensor = args[0]
        weight = args[1]
        bias = args[2] if len(args) > 2 else None
        need_transpose = False
    
    # Unwrap Parameter if necessary
    if isinstance(weight, torch.nn.Parameter):
        weight = weight.data
    
    # Check if weight is SVDQuant quantized
    if not isinstance(weight, QuantizedTensor) or weight._layout_type != "SVDQuantLayout":
        # Fallback to standard linear
        if isinstance(weight, QuantizedTensor):
            weight = weight.dequantize()
        if isinstance(input_tensor, QuantizedTensor):
            input_tensor = input_tensor.dequantize()
        if func == torch.ops.aten.addmm.default:
            return torch.addmm(bias, input_tensor, weight)
        else:
            return torch.nn.functional.linear(input_tensor, weight, bias)
    
    # Extract weight parameters
    weight_params = SVDQuantLayout.get_plain_tensors(weight)
    qweight = weight_params['qweight']
    wscales = weight_params['wscales']
    smooth_factor = weight_params['smooth_factor']
    proj_down = weight_params['proj_down']
    proj_up = weight_params['proj_up']
    group_size = weight_params['group_size']
    precision = weight_params['precision']
    wtscale = weight_params['wtscale']
    wcscales = weight_params['wcscales']
    act_unsigned = weight_params['act_unsigned']
    
    # Get activation tensor (dequantize if it's a QuantizedTensor)
    if isinstance(input_tensor, QuantizedTensor):
        if input_tensor._layout_type == "SVDQuantLayout":
            x = SVDQuantLayout.get_plain_tensors(input_tensor)
        else:
            x = input_tensor.dequantize()
    else:
        x = input_tensor
    
    # Import nunchaku operations
    try:
        from nunchaku.ops.quantize import svdq_quantize_w4a4_act_fuse_lora_cuda
        from nunchaku.ops.gemm import svdq_gemm_w4a4_cuda
    except ImportError:
        raise ImportError(
            "SVDQuant requires the nunchaku library. "
            "Install it with: pip install nunchaku"
        )
    
    # Handle batch dimensions
    original_shape = x.shape
    if len(original_shape) == 2:
        batch_size, channels = original_shape
        seq_len = 1
        x = x.view(batch_size, seq_len, channels)
    elif len(original_shape) == 3:
        batch_size, seq_len, channels = original_shape
    else:
        raise ValueError(f"SVDQuant linear expects 2D or 3D input, got {len(original_shape)}D")
    
    # Reshape to 2D for computation
    x_2d = x.reshape(batch_size * seq_len, channels)
    original_batch_size = x_2d.shape[0]  # Track original size before padding
    
    # Step 1: Quantize activations and compute low-rank hidden states
    # Output: quantized_x, ascales, lora_act_out
    quantized_x, ascales, lora_act_out = svdq_quantize_w4a4_act_fuse_lora_cuda(
        x_2d,
        lora_down=proj_down,
        smooth=smooth_factor,
        fp4=(precision == "nvfp4"),
        pad_size=256
    )
    
    # Step 2: Compute quantized GEMM with low-rank residual
    # Output shape: (N_padded, out_features) where N_padded may be larger due to padding
    out_features = qweight.shape[0]
    output = torch.empty(
        quantized_x.shape[0],
        out_features,
        dtype=proj_up.dtype,
        device=x.device
    )
    
    svdq_gemm_w4a4_cuda(
        act=quantized_x,
        wgt=qweight,
        out=output,
        ascales=ascales,
        wscales=wscales,
        lora_act_in=lora_act_out,
        lora_up=proj_up,
        bias=bias,
        fp4=(precision == "nvfp4"),
        alpha=wtscale,
        wcscales=wcscales,
        act_unsigned=act_unsigned,
    )
    
    # Slice to remove padding and reshape back to original batch dimensions
    output = output[:original_batch_size, :]  # Remove padding
    if len(original_shape) == 2:
        output = output.view(batch_size, out_features)
    else:
        output = output.view(batch_size, seq_len, out_features)
    
    return output


# ==============================================================================
# AWQ Layout + Operation Handlers
# ==============================================================================

class AWQQuantLayout(QuantizedLayout):
    """
    AWQ W4A16 quantization layout.
    
    AWQ (Activation-aware Weight Quantization) quantizes weights to 4-bit
    while keeping activations in 16-bit precision (float16/bfloat16).
    
    Storage format:
    For weights (is_weight=True):
        - qdata: Packed quantized weights (out_features // 4, in_features // 2), int32
        - wscales: Weight quantization scales (in_features // group_size, out_features)
        - wzeros: Weight zero points (in_features // group_size, out_features)
        - group_size: Quantization group size (default 64)
        - orig_dtype: Original dtype before quantization
    
    For activations (is_weight=False):
        - qdata: Original activation tensor (not quantized)
        - orig_dtype: Original dtype
        - is_weight: False marker
    """
    
    @classmethod
    def quantize(cls, tensor, is_weight=True, **kwargs):
        """
        For AWQ, we don't perform online quantization.
        - Weights are pre-quantized offline and loaded from checkpoint
        - Activations remain in 16-bit precision
        """
        orig_dtype = tensor.dtype
        
        if is_weight:
            # This shouldn't be called for weights as they're loaded pre-quantized
            raise NotImplementedError(
                "AWQ weights should be loaded pre-quantized from checkpoint, "
                "not quantized on-the-fly"
            )
        else:
            # For activations, just store the tensor as-is
            layout_params = {
                'orig_dtype': orig_dtype,
                'is_weight': False
            }
            return tensor, layout_params
    
    @staticmethod
    def dequantize(qdata, is_weight=True, orig_dtype=None, wscales=None, wzeros=None, group_size=64, **kwargs):
        """
        Dequantization for AWQ.
        - Activations: return as-is (not quantized)
        - Weights: unpack and dequantize from 4-bit
        """
        if not is_weight:
            # Activations aren't quantized, just return them
            return qdata.to(orig_dtype) if orig_dtype else qdata
        else:
            # Dequantize 4-bit weights
            # qdata shape: (out_features // 4, in_features // 2), dtype int32
            # Output shape should be: (out_features, in_features)
            
            # This is a complex operation that requires unpacking 4-bit values
            # For now, we'll raise an error and rely on the quantized forward pass
            raise NotImplementedError(
                "Full dequantization of AWQ weights is not yet supported. "
                "Use the quantized forward pass instead."
            )
    
    @classmethod
    def get_plain_tensors(cls, qtensor):
        """Extract the raw tensors needed for AWQ computation."""
        if qtensor._layout_params.get('is_weight', True):
            # For weights, return all the necessary components
            return {
                'qweight': qtensor._qdata,
                'wscales': qtensor._layout_params.get('wscales'),
                'wzeros': qtensor._layout_params.get('wzeros'),
                'group_size': qtensor._layout_params.get('group_size', 64),
            }
        else:
            # For activations, just return the tensor
            return qtensor._qdata


@register_layout_op(torch.ops.aten.addmm.default, "AWQQuantLayout")
@register_layout_op(torch.ops.aten.linear.default, "AWQQuantLayout")
def awq_linear(func, args, kwargs):
    """
    AWQ linear operation handler.
    
    Implements W4A16 quantized linear using AWQ format.
    
    Handles both aten.linear and aten.addmm (which linear decomposes into).
    """
    # Handle both linear and addmm calling conventions
    if func == torch.ops.aten.addmm.default:
        # addmm(bias, input, weight.t()) -> out
        bias = args[0] if len(args) > 0 else None
        input_tensor = args[1] if len(args) > 1 else None
        weight = args[2] if len(args) > 2 else None
    else:
        # linear(input, weight, bias) -> out
        input_tensor = args[0]
        weight = args[1]
        bias = args[2] if len(args) > 2 else None
    
    # Unwrap Parameter if necessary
    if isinstance(weight, torch.nn.Parameter):
        weight = weight.data
    
    # Check if weight is AWQ quantized
    if not isinstance(weight, QuantizedTensor) or weight._layout_type != "AWQQuantLayout":
        # Fallback to standard linear
        if isinstance(weight, QuantizedTensor):
            weight = weight.dequantize()
        if isinstance(input_tensor, QuantizedTensor):
            input_tensor = input_tensor.dequantize()
        if func == torch.ops.aten.addmm.default:
            return torch.addmm(bias, input_tensor, weight)
        else:
            return torch.nn.functional.linear(input_tensor, weight, bias)
    
    # Extract weight parameters
    weight_params = AWQQuantLayout.get_plain_tensors(weight)
    qweight = weight_params['qweight']
    wscales = weight_params['wscales']
    wzeros = weight_params['wzeros']
    group_size = weight_params['group_size']
    
    # Get activation tensor (dequantize if it's a QuantizedTensor)
    if isinstance(input_tensor, QuantizedTensor):
        if input_tensor._layout_type == "AWQQuantLayout":
            x = AWQQuantLayout.get_plain_tensors(input_tensor)
        else:
            x = input_tensor.dequantize()
    else:
        x = input_tensor
    
    # Import nunchaku AWQ operation
    try:
        from nunchaku.ops.gemv import awq_gemv_w4a16_cuda
    except ImportError:
        raise ImportError(
            "AWQ requires the nunchaku library. "
            "Install it with: pip install nunchaku"
        )
    
    # Calculate output dimensions from packed weight shape
    # qweight shape: (out_features // 4, in_features // 2)
    out_features = qweight.shape[0] * 4
    in_features = qweight.shape[1] * 2

    
    # Handle batch dimensions - preserve original shape
    # Important: nunchaku expects 2D input only, so we reshape 3D to 2D
    original_shape = x.shape
    if len(original_shape) == 2:
        # (batch_size, in_features)
        batch_size = original_shape[0]
        x_2d = x
    #elif len(original_shape) == 3:
    #    # (batch_size, seq_len, in_features) -> (batch_size * seq_len, in_features)
    #    batch_size, seq_len, _ = original_shape
    #    x_2d = x.reshape(batch_size * seq_len, in_features)
    else:
        raise ValueError(f"AWQ linear expects 2D or 3D input, got {len(original_shape)}D")
    
    # Ensure input is contiguous (required by CUDA kernel)
    # Only create a contiguous copy if absolutely necessary
    #if not x_2d.is_contiguous():
    #    x_2d = x_2d.contiguous()
    
    output = awq_gemv_w4a16_cuda(
        in_feats=x_2d,
        kernel=qweight,
        scaling_factors=wscales,
        zeros=wzeros,
        m=x_2d.shape[0],
        n=out_features,
        k=in_features,
        group_size=group_size,
    )
    
    # Add bias if present
    if bias is not None:
        view_shape = [1] * (output.ndim - 1) + [-1]
        output = output + bias.view(view_shape)
    
    # Reshape back to original batch dimensions
    #if len(original_shape) == 3:
    #    output = output.view(batch_size, seq_len, out_features)
    
    return output


LAYOUTS["SVDQuantLayout"] = SVDQuantLayout
LAYOUTS["AWQQuantLayout"] = AWQQuantLayout