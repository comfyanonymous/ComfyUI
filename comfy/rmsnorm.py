import torch
import comfy.model_management
import numbers

RMSNorm = None

try:
    rms_norm_torch = torch.nn.functional.rms_norm
    RMSNorm = torch.nn.RMSNorm
except:
    rms_norm_torch = None


def rms_norm(x, weight=None, eps=1e-6):
    if rms_norm_torch is not None and not (torch.jit.is_tracing() or torch.jit.is_scripting()):
        if weight is None:
            return rms_norm_torch(x, (x.shape[-1],), eps=eps)
        else:
            return rms_norm_torch(x, weight.shape, weight=comfy.model_management.cast_to(weight, dtype=x.dtype, device=x.device), eps=eps)
    else:
        r = x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
        if weight is None:
            return r
        else:
            return r * comfy.model_management.cast_to(weight, dtype=x.dtype, device=x.device)


if RMSNorm is None:
    class RMSNorm(torch.nn.Module):
        def __init__(
            self,
            normalized_shape,
            eps=1e-6,
            elementwise_affine=True,
            device=None,
            dtype=None,
        ):
            factory_kwargs = {"device": device, "dtype": dtype}
            super().__init__()
            if isinstance(normalized_shape, numbers.Integral):
                # mypy error: incompatible types in assignment
                normalized_shape = (normalized_shape,)  # type: ignore[assignment]
            self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
            self.eps = eps
            self.elementwise_affine = elementwise_affine
            if self.elementwise_affine:
                self.weight = torch.nn.Parameter(
                    torch.empty(self.normalized_shape, **factory_kwargs)
                )
            else:
                self.register_parameter("weight", None)
            self.bias = None

        def forward(self, x):
            return rms_norm(x, self.weight, self.eps)
