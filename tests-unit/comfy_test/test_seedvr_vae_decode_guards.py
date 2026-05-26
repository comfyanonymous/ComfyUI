from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.ldm.seedvr.vae as vae_mod  # noqa: E402


class _Wrapper(vae_mod.VideoAutoencoderKLWrapper):
    def __init__(self):
        nn.Module.__init__(self)
        self.calls = []

    def parameters(self):
        return iter([torch.nn.Parameter(torch.zeros(()))])

def _decode_stub(self, latent):
    self.calls.append(tuple(latent.shape))
    return torch.zeros(latent.shape[0], 3, latent.shape[2], latent.shape[3] * 8, latent.shape[4] * 8)


def test_seedvr2_wrapper_decode_accepts_5d_channel_first_latents_without_preprocessor_state():
    wrapper = _Wrapper()

    with patch.object(vae_mod.VideoAutoencoderKL, "decode_", _decode_stub):
        out = wrapper.decode(torch.zeros(1, 16, 2, 4, 5))

    assert tuple(out.shape) == (1, 3, 2, 32, 40)
    assert wrapper.calls == [(1, 16, 2, 4, 5)]


def test_seedvr2_wrapper_decode_accepts_collapsed_4d_latents_without_preprocessor_state():
    wrapper = _Wrapper()

    with patch.object(vae_mod.VideoAutoencoderKL, "decode_", _decode_stub):
        out = wrapper.decode(torch.zeros(1, 32, 4, 5))

    assert tuple(out.shape) == (1, 3, 2, 32, 40)
    assert wrapper.calls == [(1, 16, 2, 4, 5)]


def test_seedvr2_wrapper_decode_accepts_noncontiguous_collapsed_4d_latents():
    wrapper = _Wrapper()
    latent = torch.zeros(1, 4, 5, 32).permute(0, 3, 1, 2)

    with patch.object(vae_mod.VideoAutoencoderKL, "decode_", _decode_stub):
        out = wrapper.decode(latent)

    assert not latent.is_contiguous()
    assert tuple(out.shape) == (1, 3, 2, 32, 40)
    assert wrapper.calls == [(1, 16, 2, 4, 5)]


def test_seedvr2_wrapper_decode_rejects_non_dict_tiling_options():
    wrapper = _Wrapper()

    with pytest.raises(RuntimeError, match="seedvr2_tiling.*dict"):
        wrapper.decode(torch.zeros(1, 16, 2, 4, 5), seedvr2_tiling=True)


def test_seedvr2_wrapper_decode_rejects_wrong_5d_channel_count():
    wrapper = _Wrapper()

    with pytest.raises(RuntimeError, match="5-D latent input must have 16 channels"):
        wrapper.decode(torch.zeros(1, 8, 2, 4, 5))


def test_seedvr2_wrapper_decode_rejects_misaligned_collapsed_4d_latents():
    wrapper = _Wrapper()

    with pytest.raises(RuntimeError, match=r"4-D latent input must use collapsed channel layout"):
        wrapper.decode(torch.zeros(1, 17, 4, 5))


def test_seedvr2_wrapper_decode_rejects_wrong_rank_latents():
    wrapper = _Wrapper()

    with pytest.raises(RuntimeError, match=r"latent input must be 4-D collapsed .* or 5-D"):
        wrapper.decode(torch.zeros(1, 16, 4))
