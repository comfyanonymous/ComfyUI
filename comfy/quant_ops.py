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
    ck.registry.disable("triton")
    for k, v in ck.list_backends().items():
        logging.info(f"Found comfy_kitchen backend {k}: {v}")
except ImportError as e:
    logging.error(f"Failed to import comfy_kitchen, Error: {e}, fp8 and fp4 support will not be available.")
    _CK_AVAILABLE = False

    class QuantizedTensor:
        pass

    class _CKFp8Layout:
        pass

    class TensorCoreNVFP4Layout:
        pass

    def register_layout_class(name, cls):
        pass

    def get_layout_class(name):
        return None

import comfy.float

# ==============================================================================
# FP8 Layouts with Comfy-Specific Extensions
# ==============================================================================

class _TensorCoreFP8LayoutBase(_CKFp8Layout):
    FP8_DTYPE = None  # Must be overridden in subclass

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
