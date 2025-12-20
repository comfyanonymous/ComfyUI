import torch
import logging
import dataclasses
from typing import Dict

try:
    import comfy_kitchen as ck
    from comfy_kitchen.tensor import (
        QuantizedTensor as _CKQuantizedTensor,
        QuantizedLayout,
        TensorCoreFP8Layout as _CKFp8Layout,
        TensorCoreNVFP4Layout,  # Direct import, no wrapper needed
        register_layout_op,
    )
    _CK_AVAILABLE = True
    for k, v in ck.list_backends().items():
        logging.info(f"Found comfy_kitchen backend {k}: {v}")
except ImportError as e:
    logging.info(f"Failed to import comfy_kitchen, falling back to torch ops. Error: {e}")
    _CK_AVAILABLE = False
    raise ImportError(f"comfy_kitchen is required but not available: {e}")

import comfy.float


# ==============================================================================
# Backward Compatibility Layer
# ==============================================================================

class QuantizedTensor(_CKQuantizedTensor):
    @staticmethod
    def __new__(cls, qdata, layout_cls, params):
        # Backward compat: Convert string layout names and dict params before __new__
        if isinstance(layout_cls, str):
            layout_cls = LAYOUTS[layout_cls]

        if isinstance(params, dict):
            params = layout_cls.Params(**params)

        return _CKQuantizedTensor.__new__(cls, qdata, layout_cls, params)

    def __init__(self, qdata, layout_cls, params):
        super().__init__(qdata, layout_cls, params)

    @property
    def _layout_params(self) -> Dict:
        return dataclasses.asdict(self._params)

    @property
    def _layout_type(self) -> str:
        return self._layout_cls.__name__

    @property
    def layout_type(self) -> str:
        """Backward compatibility alias for _layout_type."""
        return self._layout_type

    def _copy_with(self, qdata=None, params=None, clone_params=True):
        if params is None:
            params = self._params.clone() if clone_params else self._params
        return type(self)(
            qdata if qdata is not None else self._qdata,
            self._layout_cls,
            params,
        )


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
            scale = torch.amax(tensor.abs()) / torch.finfo(cls.FP8_DTYPE).max

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

        params = cls.Params(scale=scale, orig_dtype=orig_dtype, orig_shape=orig_shape)
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

LAYOUTS = {
    "TensorCoreFP8Layout": TensorCoreFP8Layout,  # Backward compat alias (E4M3)
    "TensorCoreFP8E4M3Layout": TensorCoreFP8E4M3Layout,
    "TensorCoreFP8E5M2Layout": TensorCoreFP8E5M2Layout,
    "TensorCoreNVFP4Layout": TensorCoreNVFP4Layout,  # Direct from comfy_kitchen
}

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
    "LAYOUTS",
    "QUANT_ALGOS",
    "register_layout_op",
]
