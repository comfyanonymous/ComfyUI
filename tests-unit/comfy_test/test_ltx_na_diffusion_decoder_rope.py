import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.ldm.lightricks.vae.na_diffusion_decoder as na_diffusion_decoder


def test_rope_inv_freqs_never_allocates_float64_on_target_device(monkeypatch):
    # Stands in for a device (like MPS) that cannot materialize float64 tensors.
    device = torch.device("cpu")
    real_arange = torch.arange
    real_tensor = torch.tensor

    def guarded_arange(*args, **kwargs):
        if kwargs.get("dtype") == torch.float64 and kwargs.get("device") is device:
            raise TypeError("Cannot convert a MPS Tensor to float64 dtype")
        return real_arange(*args, **kwargs)

    def guarded_tensor(*args, **kwargs):
        if kwargs.get("dtype") == torch.float64 and kwargs.get("device") is device:
            raise TypeError("Cannot convert a MPS Tensor to float64 dtype")
        return real_tensor(*args, **kwargs)

    monkeypatch.setattr(torch, "arange", guarded_arange)
    monkeypatch.setattr(torch, "tensor", guarded_tensor)

    result = na_diffusion_decoder.rope_inv_freqs(16, device=device)

    assert result.dtype == torch.float32
    assert result.device == device
