import inspect

import torch
import torch.nn as nn

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.ldm.seedvr.vae as vae_mod  # noqa: E402

VideoAutoencoderKLWrapper = vae_mod.VideoAutoencoderKLWrapper


_INPUT_SHAPE = (1, 3, 5, 16, 16)
_POSTERIOR_SHAPE = (1, 16, 1, 2, 2)
_DECODE_OUT_SHAPE = (1, 3, 5, 16, 16)


def _build_wrapper_standin() -> VideoAutoencoderKLWrapper:
    wrapper = VideoAutoencoderKLWrapper.__new__(VideoAutoencoderKLWrapper)
    nn.Module.__init__(wrapper)
    return wrapper


def test_wrapper_forward_returns_tensor_triple(monkeypatch):
    wrapper = _build_wrapper_standin()
    wrapper.original_image_video = torch.zeros(*_INPUT_SHAPE)
    wrapper.img_dims = (16, 16)
    wrapper.freeze_encoder = True

    posterior = torch.full(_POSTERIOR_SHAPE, 7.0)
    decode_out = torch.full(_DECODE_OUT_SHAPE, 13.0)

    def stub_encode(self, x, orig_dims=None):
        return posterior.squeeze(2), posterior

    def stub_decode(self, z):
        return decode_out

    monkeypatch.setattr(VideoAutoencoderKLWrapper, "encode", stub_encode)
    monkeypatch.setattr(VideoAutoencoderKLWrapper, "decode", stub_decode)

    x = torch.zeros(*_INPUT_SHAPE)
    result = wrapper.forward(x)

    assert isinstance(result, tuple)
    assert len(result) == 3
    x_out, z, p = result
    assert type(x_out) is torch.Tensor
    assert type(z) is torch.Tensor
    assert type(p) is torch.Tensor
    assert x_out.shape == decode_out.shape
    assert z.shape == posterior.squeeze(2).shape
    assert torch.equal(x_out, decode_out)
    assert torch.equal(z, posterior.squeeze(2))
    assert p is posterior


def test_wrapper_forward_source_has_no_sample_access():
    src = inspect.getsource(VideoAutoencoderKLWrapper.forward)
    assert ".sample" not in src
