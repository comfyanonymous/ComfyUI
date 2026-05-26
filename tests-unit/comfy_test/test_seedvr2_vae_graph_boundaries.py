from unittest.mock import MagicMock

import torch
import torch.nn as nn

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.ldm.seedvr.vae as seedvr_vae_mod  # noqa: E402
import comfy.sd as sd_mod  # noqa: E402
import nodes as nodes_mod  # noqa: E402


class _Patcher:
    def get_free_memory(self, device):
        return 1024 * 1024 * 1024


class _EncodeWrapper(seedvr_vae_mod.VideoAutoencoderKLWrapper):
    def __init__(self, encoded):
        nn.Module.__init__(self)
        self.encoded = encoded
        self.spatial_downsample_factor = 8
        self.temporal_downsample_factor = 4
        self.seen = []

    def encode(self, x):
        self.seen.append(tuple(x.shape))
        return self.encoded.to(device=x.device, dtype=x.dtype)


class _DecodeWrapper(seedvr_vae_mod.VideoAutoencoderKLWrapper):
    def __init__(self):
        nn.Module.__init__(self)
        self.spatial_downsample_factor = 8
        self.temporal_downsample_factor = 4
        self.calls = []

    def decode(self, z, seedvr2_tiling=None):
        self.calls.append({"shape": tuple(z.shape), "seedvr2_tiling": seedvr2_tiling})
        if z.ndim == 4:
            b, tc, h, w = z.shape
            t = tc // 16
        else:
            b, _, t, h, w = z.shape
        return torch.zeros(b, 3, t, h * 8, w * 8, dtype=z.dtype, device=z.device)


def _make_vae(wrapper):
    vae = sd_mod.VAE.__new__(sd_mod.VAE)
    vae.first_stage_model = wrapper
    vae.device = torch.device("cpu")
    vae.output_device = torch.device("cpu")
    vae.vae_dtype = torch.float32
    vae.latent_channels = 16
    vae.latent_dim = 3
    vae.downscale_ratio = (lambda a: max(0, (a + 3) // 4), 8, 8)
    vae.upscale_ratio = (lambda a: max(0, a * 4 - 3), 8, 8)
    vae.output_channels = 3
    vae.disable_offload = True
    vae.extra_1d_channel = None
    vae.crop_input = False
    vae.not_video = False
    vae.patcher = _Patcher()
    vae.process_input = lambda image: image
    vae.process_output = lambda image: image.add(1.0).div(2.0).clamp(0.0, 1.0)
    vae.vae_output_dtype = lambda: torch.float32
    vae.memory_used_encode = lambda shape, dtype: 1
    vae.memory_used_decode = lambda shape, dtype: 1
    vae.throw_exception_if_invalid = lambda: None
    vae.vae_encode_crop_pixels = lambda pixels: pixels
    vae.spacial_compression_decode = lambda: 8
    vae.temporal_compression_decode = lambda: 4
    return vae


def test_seedvr2_encode_and_encode_tiled_preserve_native_latent_contract(monkeypatch):
    monkeypatch.setattr(sd_mod.model_management, "load_models_gpu", lambda *a, **k: None)

    encoded = torch.full((1, 16, 2, 4, 5), 2.0)
    vae = _make_vae(_EncodeWrapper(encoded))
    pixels = torch.zeros(1, 5, 32, 40, 3)

    node_output = nodes_mod.VAEEncode().encode(vae, pixels)[0]
    node_latent = node_output["samples"]
    assert set(node_output) == {"samples"}
    assert tuple(node_latent.shape) == (1, 16, 2, 4, 5)
    assert node_latent.dtype == torch.float32
    assert node_latent.stride()[-1] == 1
    assert torch.equal(node_latent, torch.full_like(node_latent, 2.0 * 0.9152))

    tiled = torch.full((1, 16, 2, 4, 5), 3.0)
    monkeypatch.setattr(seedvr_vae_mod, "tiled_vae", MagicMock(return_value=tiled))
    tiled_output = nodes_mod.VAEEncodeTiled().encode(
        vae,
        pixels,
        tile_size=512,
        overlap=64,
        temporal_size=16,
        temporal_overlap=4,
    )[0]
    tiled_latent = tiled_output["samples"]
    assert set(tiled_output) == {"samples"}
    assert tuple(tiled_latent.shape) == (1, 16, 2, 4, 5)
    assert tiled_latent.dtype == torch.float32
    assert torch.equal(tiled_latent, torch.full_like(tiled_latent, 3.0 * 0.9152))


def test_seedvr2_decode_and_decode_tiled_do_not_require_preprocessor_state(monkeypatch):
    monkeypatch.setattr(sd_mod.model_management, "load_models_gpu", lambda *a, **k: None)
    vae = _make_vae(_DecodeWrapper())

    latent = {"samples": torch.zeros(1, 32, 4, 5)}
    decoded = nodes_mod.VAEDecode().decode(vae, latent)[0]
    assert tuple(decoded.shape) == (2, 32, 40, 3)

    tiled = nodes_mod.VAEDecodeTiled().decode(
        vae,
        {"samples": torch.zeros(1, 16, 2, 4, 5)},
        tile_size=512,
        overlap=64,
        temporal_size=16,
        temporal_overlap=4,
    )[0]
    assert tuple(tiled.shape) == (2, 32, 40, 3)


def test_seedvr2_vaedecode_does_not_repair_latent_layout(monkeypatch):
    monkeypatch.setattr(sd_mod.model_management, "load_models_gpu", lambda *a, **k: None)
    vae = _make_vae(_DecodeWrapper())

    latent = {"samples": torch.zeros(1, 2, 4, 5, 16)}
    nodes_mod.VAEDecode().decode(vae, latent)

    assert vae.first_stage_model.calls == [{"shape": (1, 2, 4, 5, 16), "seedvr2_tiling": None}]


def test_seedvr2_vaedecode_keeps_public_channel_first_width_16_latents(monkeypatch):
    monkeypatch.setattr(sd_mod.model_management, "load_models_gpu", lambda *a, **k: None)
    vae = _make_vae(_DecodeWrapper())

    nodes_mod.VAEDecode().decode(
        vae,
        {"samples": torch.zeros(1, 16, 4, 5, 16)},
    )

    assert vae.first_stage_model.calls == [{"shape": (1, 16, 4, 5, 16), "seedvr2_tiling": None}]


def test_seedvr2_direct_decode_preserves_channel_first_width_16(monkeypatch):
    monkeypatch.setattr(sd_mod.model_management, "load_models_gpu", lambda *a, **k: None)
    vae = _make_vae(_DecodeWrapper())

    vae.decode(torch.zeros(1, 16, 2, 4, 16))

    assert vae.first_stage_model.calls == [{"shape": (1, 16, 2, 4, 16), "seedvr2_tiling": None}]


def test_seedvr2_decode_tiled_preserves_direct_channel_first_width_16(monkeypatch):
    monkeypatch.setattr(sd_mod.model_management, "load_models_gpu", lambda *a, **k: None)
    vae = _make_vae(_DecodeWrapper())

    vae.decode_tiled_seedvr2(torch.zeros(1, 16, 2, 4, 16))

    assert vae.first_stage_model.calls[0]["shape"] == (1, 16, 2, 4, 16)


def test_seedvr2_vaedecode_tiled_keeps_public_channel_first_width_16_latents(monkeypatch):
    monkeypatch.setattr(sd_mod.model_management, "load_models_gpu", lambda *a, **k: None)
    vae = _make_vae(_DecodeWrapper())

    nodes_mod.VAEDecodeTiled().decode(
        vae,
        {"samples": torch.zeros(1, 16, 4, 5, 16)},
        tile_size=512,
        overlap=64,
        temporal_size=16,
        temporal_overlap=4,
    )

    assert vae.first_stage_model.calls[0]["shape"] == (1, 16, 4, 5, 16)


def test_vaedecode_tiled_visible_inputs_are_seedvr2_decode_tiling_authority(monkeypatch):
    monkeypatch.setattr(sd_mod.model_management, "load_models_gpu", lambda *a, **k: None)
    vae = _make_vae(_DecodeWrapper())

    nodes_mod.VAEDecodeTiled().decode(
        vae,
        {"samples": torch.zeros(1, 16, 2, 4, 5)},
        tile_size=512,
        overlap=64,
        temporal_size=16,
        temporal_overlap=4,
    )

    assert vae.first_stage_model.calls == [
        {
            "shape": (1, 16, 2, 4, 5),
            "seedvr2_tiling": {
                "enable_tiling": True,
                "tile_size": (512, 512),
                "tile_overlap": (64, 64),
                "temporal_size": 16,
                "temporal_overlap": 4,
            },
        }
    ]
