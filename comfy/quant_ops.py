import torch
import logging
from typing import Tuple, Dict

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

    if isinstance(qt_dest, QuantizedTensor):
        if isinstance(src, QuantizedTensor):
            # Copy from another quantized tensor
            qt_dest._qdata.copy_(src._qdata)
            qt_dest._layout_type = src._layout_type
            qt_dest._layout_params = _copy_layout_params(src._layout_params)
        else:
            # Copy from regular tensor - just copy raw data
            qt_dest._qdata.copy_(src)
        return qt_dest
    return func(*args, **kwargs)


@register_generic_util(torch.ops.aten._has_compatible_shallow_copy_type.default)
def generic_has_compatible_shallow_copy_type(func, args, kwargs):
    return True

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
    def quantize(cls, tensor, scale=None, dtype=torch.float8_e4m3fn):
        orig_dtype = tensor.dtype

        if scale is None:
            scale = torch.amax(tensor.abs()) / torch.finfo(dtype).max

        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale)
        scale = scale.to(device=tensor.device, dtype=torch.float32)

        tensor_scaled = tensor * (1.0 / scale).to(tensor.dtype)
        # TODO: uncomment this if it's actually needed because the clamp has a small performance penality'
        # lp_amax = torch.finfo(dtype).max
        # torch.clamp(tensor_scaled, min=-lp_amax, max=lp_amax, out=tensor_scaled)
        qdata = tensor_scaled.to(dtype, memory_format=torch.contiguous_format)

        layout_params = {
            'scale': scale,
            'orig_dtype': orig_dtype
        }
        return qdata, layout_params

    @staticmethod
    def dequantize(qdata, scale, orig_dtype, **kwargs):
        plain_tensor = torch.ops.aten._to_copy.default(qdata, dtype=orig_dtype)
        return plain_tensor * scale

    @classmethod
    def get_plain_tensors(cls, qtensor):
        return qtensor._qdata, qtensor._layout_params['scale']


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
