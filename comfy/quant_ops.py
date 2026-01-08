import torch
import logging

try:
    import comfy_kitchen as ck
    from comfy_kitchen.tensor import (
        QuantizedTensor,
        QuantizedLayout,
        TensorCoreFP8Layout as _CKFp8Layout,
        TensorCoreNVFP4Layout,  # Direct import, no wrapper needed
        register_layout_op,
        register_layout_class,
        get_layout_class,
    )
    _CK_AVAILABLE = True
    if torch.version.cuda is None:
        ck.registry.disable("cuda")
    else:
        cuda_version = tuple(map(int, str(torch.version.cuda).split('.')))
        if cuda_version < (13,):
            ck.registry.disable("cuda")
            logging.warning("WARNING: You need pytorch with cu130 or higher to use optimized CUDA operations.")

    ck.registry.disable("triton")
    for k, v in ck.list_backends().items():
        logging.info(f"Found comfy_kitchen backend {k}: {v}")
except ImportError as e:
    logging.error(f"Failed to import comfy_kitchen, Error: {e}, fp8 and fp4 support will not be available.")
    _CK_AVAILABLE = False

    class ck_dummy:
        @staticmethod
        def quantize_per_tensor_fp8(tensor, scale, dtype):
            return (tensor / scale.to(tensor.device)).to(dtype)
    ck = ck_dummy

    class QuantizedTensor:
        def __init__(self, qdata, layout_type, layout_params):
            self._qdata = qdata
            self._layout_type = layout_type
            self._layout_params = layout_params
            self.device = qdata.device
            self.dtype = qdata.dtype

        @classmethod
        def from_float(cls, tensor, layout_type, **kwargs):
            layout_cls = get_layout_class(layout_type)
            if layout_cls is None:
                raise ValueError(f"Unknown layout type: {layout_type}")
            qdata, params = layout_cls.quantize(tensor, **kwargs)
            return cls(qdata, layout_type, params)

        def dequantize(self):
            layout_cls = get_layout_class(self._layout_type)
            if layout_cls is None:
                return self._qdata
            return layout_cls.dequantize(self._qdata, **self._layout_params.__dict__)

        def to(self, *args, **kwargs):
            device = kwargs.get("device", None)
            if device is None and len(args) > 0:
                if isinstance(args[0], (torch.device, str)):
                    device = args[0]
            
            new_qdata = self._qdata.to(*args, **kwargs)
            new_params = self._layout_params.copy()
            if device is not None:
                for k, v in new_params.__dict__.items():
                    if isinstance(v, torch.Tensor):
                        new_params.__dict__[k] = v.to(device=device)
            
            return type(self)(new_qdata, self._layout_type, new_params)

        def detach(self):
            return type(self)(self._qdata.detach(), self._layout_type, self._layout_params.copy())

        def clone(self):
            return type(self)(self._qdata.clone(), self._layout_type, self._layout_params.copy())

        def requires_grad_(self, requires_grad=True):
            self._qdata.requires_grad_(requires_grad)
            return self

        def __getattr__(self, name):
            if name == "params":
                return self._layout_params
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            return NotImplemented

    class QuantizedLayout:
        class Params:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
            
            def copy(self):
                return type(self)(**self.__dict__)

    class _CKFp8Layout(QuantizedLayout):
        pass

    class TensorCoreNVFP4Layout(QuantizedLayout):
        pass

    _LOCAL_LAYOUT_REGISTRY = {}

    def register_layout_class(name, cls):
        _LOCAL_LAYOUT_REGISTRY[name] = cls

    def get_layout_class(name):
        return _LOCAL_LAYOUT_REGISTRY.get(name)

    def register_layout_op(torch_op, layout_type):
        def decorator(handler_func):
            return handler_func
        return decorator


import comfy.float
import comfy.mps_ops

# ==============================================================================
# FP8 Layouts with Comfy-Specific Extensions
# ==============================================================================

class _TensorCoreFP8LayoutBase(_CKFp8Layout):
    FP8_DTYPE = None  # Must be overridden in subclass
    
    """
    Storage format:
    - qdata: FP8 tensor (torch.float8_e4m3fn or torch.float8_e5m2)
    - scale: Scalar tensor (float32) for dequantization
    - orig_dtype: Original dtype before quantization (for casting back)
    """
    @classmethod
    def quantize(cls, tensor, scale=None, stochastic_rounding=0, inplace_ops=False):
        if cls.FP8_DTYPE is None:
            raise NotImplementedError(f"{cls.__name__} must define FP8_DTYPE")

        orig_dtype = tensor.dtype
        orig_shape = tuple(tensor.shape)

        if isinstance(scale, str) and scale == "recalculate":
            scale = torch.amax(tensor.abs()).to(dtype=torch.float32) / torch.finfo(cls.FP8_DTYPE).max
            if tensor.dtype not in [torch.float32, torch.bfloat16]:  # Prevent scale from being too small
                tensor_info = torch.finfo(tensor.dtype)
                scale = (1.0 / torch.clamp((1.0 / scale), min=tensor_info.min, max=tensor_info.max))

        if scale is None:
            scale = torch.ones((), device=tensor.device, dtype=torch.float32)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, device=tensor.device, dtype=torch.float32)

        if stochastic_rounding > 0:
            if inplace_ops:
                tensor *= (1.0 / scale).to(tensor.dtype)
            else:
                tensor = tensor * (1.0 / scale).to(tensor.dtype)
            qdata = comfy.float.stochastic_rounding(tensor, dtype=cls.FP8_DTYPE, seed=stochastic_rounding)
        else:
            qdata = ck.quantize_per_tensor_fp8(tensor, scale, cls.FP8_DTYPE)

        params = cls.Params(scale=scale.float(), orig_dtype=orig_dtype, orig_shape=orig_shape)
        return qdata, params

    @staticmethod
    def dequantize(qdata, scale, orig_dtype, **kwargs):
        if qdata.device.type == "mps":
             if qdata.dtype == torch.uint8:
                 return comfy.mps_ops.mps_dequantize(qdata, scale, orig_dtype, kwargs.get("mps_float8_dtype", torch.float8_e4m3fn))
             elif qdata.is_floating_point() and qdata.element_size() == 1:
                 # It is MPS Float8. View as uint8.
                 return comfy.mps_ops.mps_dequantize(qdata.view(torch.uint8), scale, orig_dtype, qdata.dtype)

        plain_tensor = torch.ops.aten._to_copy.default(qdata, dtype=orig_dtype)
        plain_tensor.mul_(scale)
        return plain_tensor


class TensorCoreFP8E4M3Layout(_TensorCoreFP8LayoutBase):
    FP8_DTYPE = torch.float8_e4m3fn


class TensorCoreFP8E5M2Layout(_TensorCoreFP8LayoutBase):
    FP8_DTYPE = torch.float8_e5m2


# Backward compatibility alias - default to E4M3
TensorCoreFP8Layout = TensorCoreFP8E4M3Layout


# ==============================================================================
# Registry
# ==============================================================================

register_layout_class("TensorCoreFP8Layout", TensorCoreFP8Layout)
register_layout_class("TensorCoreFP8E4M3Layout", TensorCoreFP8E4M3Layout)
register_layout_class("TensorCoreFP8E5M2Layout", TensorCoreFP8E5M2Layout)
register_layout_class("TensorCoreNVFP4Layout", TensorCoreNVFP4Layout)

QUANT_ALGOS = {
    "float8_e4m3fn": {
        "storage_t": torch.float8_e4m3fn,
        "parameters": {"weight_scale", "input_scale"},
        "comfy_tensor_layout": "TensorCoreFP8E4M3Layout",
    },
    "float8_e5m2": {
        "storage_t": torch.float8_e5m2,
        "parameters": {"weight_scale", "input_scale"},
        "comfy_tensor_layout": "TensorCoreFP8E5M2Layout",
    },
    "nvfp4": {
        "storage_t": torch.uint8,
        "parameters": {"weight_scale", "weight_scale_2", "input_scale"},
        "comfy_tensor_layout": "TensorCoreNVFP4Layout",
        "group_size": 16,
    },
}


# ==============================================================================
# Re-exports for backward compatibility
# ==============================================================================

__all__ = [
    "QuantizedTensor",
    "QuantizedLayout",
    "TensorCoreFP8Layout",
    "TensorCoreFP8E4M3Layout",
    "TensorCoreFP8E5M2Layout",
    "TensorCoreNVFP4Layout",
    "QUANT_ALGOS",
    "register_layout_op",
]
