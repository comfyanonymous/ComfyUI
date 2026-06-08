"""Regression: ``comfy.ldm.seedvr.vae.VideoAutoencoderKL.forward`` must
honor the actual tensor/tuple return contract of ``encode()`` and
``decode_()`` and must NOT dereference diffusers-style ``.latent_dist``
or ``.sample`` attributes on those returns.

The pre-fix body raised ``AttributeError: 'Tensor' object has no
attribute 'latent_dist'`` for ``mode in {"encode", "all"}`` and
``AttributeError: 'VideoAutoencoderKL' object has no attribute 'decode'``
for ``mode == "decode"`` (the class only defines ``decode_`` with a
trailing underscore). The post-fix body unwraps the optional one-element
tuple shape that ``return_dict=False`` produces and returns the tensor
directly.

Tests construct a stub subclass of ``VideoAutoencoderKL`` that bypasses
the heavy ``__init__`` via ``torch.nn.Module.__init__(self)`` and
overrides ``encode``/``decode_`` with known tensors so the contract can
be probed without loading any real VAE weights.
"""

import torch
import torch.nn as nn

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy.ldm.seedvr.vae import VideoAutoencoderKL  # noqa: E402


_LATENT_SHAPE = (1, 16, 2, 2, 2)
_DECODED_SHAPE = (1, 3, 5, 16, 16)
_INPUT_ENCODE_SHAPE = (1, 3, 5, 16, 16)
_INPUT_DECODE_SHAPE = (1, 16, 2, 2, 2)


class _StubVAE(VideoAutoencoderKL):
    def __init__(self):
        nn.Module.__init__(self)
        self._encode_out = torch.zeros(*_LATENT_SHAPE)
        self._decode_out = torch.zeros(*_DECODED_SHAPE)

    def encode(self, x, return_dict=True):
        return self._encode_out

    def decode_(self, z, return_dict=True):
        return self._decode_out


def test_forward_encode_returns_tensor():
    vae = _StubVAE()
    x = torch.zeros(*_INPUT_ENCODE_SHAPE)
    result = vae.forward(x, mode="encode")
    assert type(result) is torch.Tensor
    assert result.shape == torch.Size(_LATENT_SHAPE)


def test_forward_decode_returns_tensor():
    vae = _StubVAE()
    z = torch.zeros(*_INPUT_DECODE_SHAPE)
    result = vae.forward(z, mode="decode")
    assert type(result) is torch.Tensor
    assert result.shape == torch.Size(_DECODED_SHAPE)


class _TupleReturningStubVAE(VideoAutoencoderKL):
    """Stub variant whose ``encode``/``decode_`` return the
    ``(tensor,)`` one-element tuple shape ``return_dict=False`` produces
    in the parent class. Exercises the unwrap branch of
    ``VideoAutoencoderKL.forward``.
    """

    def __init__(self):
        nn.Module.__init__(self)
        self._encode_tensor = torch.zeros(*_LATENT_SHAPE)
        self._decode_tensor = torch.zeros(*_DECODED_SHAPE)

    def encode(self, x, return_dict=True):
        return (self._encode_tensor,)

    def decode_(self, z, return_dict=True):
        return (self._decode_tensor,)


def test_forward_all_unwraps_one_tuple_at_each_step():
    vae = _TupleReturningStubVAE()
    x = torch.zeros(*_INPUT_ENCODE_SHAPE)
    result = vae.forward(x, mode="all")
    assert type(result) is torch.Tensor
    assert result.shape == torch.Size(_DECODED_SHAPE)
