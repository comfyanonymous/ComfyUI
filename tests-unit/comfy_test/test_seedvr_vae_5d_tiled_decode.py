from unittest.mock import patch

import torch
import torch.nn as nn

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.ldm.seedvr.vae as vae_mod  # noqa: E402
import comfy.sd as sd_mod  # noqa: E402
import nodes as nodes_mod  # noqa: E402


def _lab_color_passthrough(content, style):
    return content


def _decode_fingerprint(self, z, return_dict=True):
    b, _, t, h, w = z.shape
    out = torch.empty(b, 3, t, h * 8, w * 8, dtype=z.dtype, device=z.device)
    for batch_idx in range(b):
        out[batch_idx].fill_(float(batch_idx + 1))
    return out


def _make_wrapper(b=2, t=3, enable_tiling=False):
    wrapper = vae_mod.VideoAutoencoderKLWrapper.__new__(
        vae_mod.VideoAutoencoderKLWrapper
    )
    nn.Module.__init__(wrapper)
    wrapper.tiled_args = {"enable_tiling": enable_tiling}
    wrapper.original_image_video = torch.zeros(b, 3, t, 16, 16)
    wrapper.img_dims = (16, 16)
    return wrapper


def test_seedvr2_decode_accepts_5d_bcthw_latents_and_preserves_batch_time_axes():
    wrapper = _make_wrapper(b=2, t=3, enable_tiling=False)
    latent = torch.zeros(2, 16, 3, 2, 2)

    with patch.object(vae_mod.VideoAutoencoderKL, "decode_", _decode_fingerprint), \
         patch.object(vae_mod, "lab_color_transfer", _lab_color_passthrough):
        out = wrapper.decode(latent)

    assert tuple(out.shape) == (2, 3, 3, 16, 16)
    assert out[0, 0, 0, 0, 0].item() == 1.0
    assert out[1, 0, 0, 0, 0].item() == 2.0


class _SeedVR2DecodeStub(vae_mod.VideoAutoencoderKLWrapper):
    def __init__(self):
        nn.Module.__init__(self)
        self.tiled_args = {}
        self.calls = []
        self.original_image_video = torch.zeros(1, 3, 12, 16, 16)
        self.spatial_downsample_factor = 8
        self.temporal_downsample_factor = 4

    def decode(self, z, seedvr2_tiling=None):
        self.calls.append({"seedvr2_tiling": seedvr2_tiling, "shape": tuple(z.shape)})
        return z


def test_vae_decode_tiled_allows_zero_temporal_controls_and_passes_them_through():
    input_types = nodes_mod.VAEDecodeTiled.INPUT_TYPES()["required"]
    assert input_types["temporal_size"][1]["min"] == 0
    assert input_types["temporal_overlap"][1]["min"] == 0
    assert "SeedVR2 allows 0" in input_types["temporal_size"][1]["tooltip"]

    class _DecodeRecorder:
        def __init__(self):
            self.calls = []

        def temporal_compression_decode(self):
            return 4

        def spacial_compression_decode(self):
            return 8

        def decode_tiled(self, samples, **kwargs):
            self.calls.append({"shape": tuple(samples.shape), **kwargs})
            return torch.zeros(1, 8, 8, 3)

    recorder = _DecodeRecorder()
    node = nodes_mod.VAEDecodeTiled()

    node.decode(
        recorder,
        {"samples": torch.zeros(1, 16, 3, 32, 32)},
        tile_size=256,
        overlap=64,
        temporal_size=0,
        temporal_overlap=0,
    )

    assert recorder.calls == [
        {
            "shape": (1, 16, 3, 32, 32),
            "tile_x": 32,
            "tile_y": 32,
            "overlap": 8,
            "tile_t": 0,
            "overlap_t": 0,
        }
    ]


def test_vae_decode_tiled_preserves_positive_overlap_after_temporal_compression():
    class _DecodeRecorder:
        def __init__(self):
            self.calls = []

        def temporal_compression_decode(self):
            return 8

        def spacial_compression_decode(self):
            return 8

        def decode_tiled(self, samples, **kwargs):
            self.calls.append(kwargs)
            return torch.zeros(1, 8, 8, 3)

    recorder = _DecodeRecorder()

    nodes_mod.VAEDecodeTiled().decode(
        recorder,
        {"samples": torch.zeros(1, 16, 3, 32, 32)},
        tile_size=256,
        overlap=64,
        temporal_size=64,
        temporal_overlap=4,
    )

    assert recorder.calls[0]["tile_t"] == 8
    assert recorder.calls[0]["overlap_t"] == 1


def test_seedvr2_decode_tiled_uses_seedvr2_path_not_generic_3d_tiler(monkeypatch):
    vae = sd_mod.VAE.__new__(sd_mod.VAE)
    vae.first_stage_model = _SeedVR2DecodeStub()
    vae.vae_dtype = torch.float32
    vae.device = torch.device("cpu")
    vae.output_device = torch.device("cpu")
    vae.disable_offload = True
    vae.extra_1d_channel = None
    vae.memory_used_decode = lambda shape, dtype: 1
    vae.process_output = lambda x: x
    vae.patcher = object()

    monkeypatch.setattr(sd_mod.model_management, "load_models_gpu", lambda *a, **k: None)
    monkeypatch.setattr(sd_mod.VAE, "decode_tiled_3d", lambda *a, **k: (_ for _ in ()).throw(AssertionError("generic decode_tiled_3d called")))

    latent = torch.zeros(1, 16, 3, 2, 2)
    out = vae.decode_tiled(latent, tile_x=2, tile_y=2, overlap=1, tile_t=16, overlap_t=4)

    assert tuple(out.shape) == (1, 3, 2, 2, 16)
    assert vae.first_stage_model.calls == [
        {
            "shape": (1, 16, 3, 2, 2),
            "seedvr2_tiling": {
                "enable_tiling": True,
                "tile_size": (16, 16),
                "tile_overlap": (8, 8),
                "temporal_size": 64,
                "temporal_overlap": 16,
            },
        }
    ]


def test_seedvr2_decode_tiled_explicit_args_override_stale_tiled_args():
    vae = sd_mod.VAE.__new__(sd_mod.VAE)
    vae.first_stage_model = _SeedVR2DecodeStub()
    vae.first_stage_model.tiled_args = {
        "enable_tiling": False,
        "tile_size": (384, 384),
        "tile_overlap": (128, 128),
        "temporal_size": 16,
        "temporal_overlap": 4,
        "preserved": "metadata",
    }
    vae.vae_dtype = torch.float32
    vae.device = torch.device("cpu")
    vae.output_device = torch.device("cpu")
    vae.disable_offload = True
    vae.extra_1d_channel = None
    vae.memory_used_decode = lambda shape, dtype: 1
    vae.process_output = lambda x: x
    vae.patcher = object()

    latent = torch.zeros(1, 16, 3, 2, 2)
    vae.decode_tiled_seedvr2(
        latent,
        tile_x=32,
        tile_y=32,
        overlap=8,
        tile_t=0,
        overlap_t=0,
    )

    captured = vae.first_stage_model.calls[0]["seedvr2_tiling"]
    assert captured["enable_tiling"] is True
    assert captured["tile_size"] == (256, 256)
    assert captured["tile_overlap"] == (64, 64)
    assert captured["temporal_size"] == 0
    assert captured["temporal_overlap"] == 0
    assert "preserved" not in captured
    assert vae.first_stage_model.tiled_args == {
        "enable_tiling": False,
        "tile_size": (384, 384),
        "tile_overlap": (128, 128),
        "temporal_size": 16,
        "temporal_overlap": 4,
        "preserved": "metadata",
    }


def test_seedvr2_decode_preserves_requested_spatial_tile_above_512(monkeypatch):
    wrapper = vae_mod.VideoAutoencoderKLWrapper.__new__(
        vae_mod.VideoAutoencoderKLWrapper
    )
    nn.Module.__init__(wrapper)

    captured = {}

    def fake_tiled_vae(latent, model, **kwargs):
        captured.update(kwargs)
        return torch.zeros(1, 3, 1, 16, 16)

    monkeypatch.setattr(vae_mod, "tiled_vae", fake_tiled_vae)

    wrapper.decode(
        torch.zeros(1, 16, 1, 2, 2),
        seedvr2_tiling={
            "enable_tiling": True,
            "tile_size": (1024, 768),
            "tile_overlap": (800, 800),
            "temporal_size": 0,
            "temporal_overlap": 0,
        },
    )

    assert captured["tile_size"] == (1024, 768)
    assert captured["tile_overlap"] == (800, 760)


def test_seedvr2_decode_tiled_preserves_ambiguous_channel_first_latents(monkeypatch):
    vae = sd_mod.VAE.__new__(sd_mod.VAE)
    vae.first_stage_model = _SeedVR2DecodeStub()
    vae.vae_dtype = torch.float32
    vae.device = torch.device("cpu")
    vae.output_device = torch.device("cpu")
    vae.disable_offload = True
    vae.extra_1d_channel = None
    vae.latent_channels = 16
    vae.memory_used_decode = lambda shape, dtype: 1
    vae.process_output = lambda x: x
    vae.patcher = object()

    monkeypatch.setattr(sd_mod.model_management, "load_models_gpu", lambda *a, **k: None)
    monkeypatch.setattr(sd_mod.VAE, "decode_tiled_3d", lambda *a, **k: (_ for _ in ()).throw(AssertionError("generic decode_tiled_3d called")))

    latent = torch.zeros(1, 16, 8, 8, 16)
    vae.decode_tiled(latent, tile_x=2, tile_y=2, overlap=1, tile_t=16, overlap_t=4)

    assert vae.first_stage_model.calls[0]["shape"] == (1, 16, 8, 8, 16)


def test_seedvr2_decode_tiled_does_not_repair_latent_layout(monkeypatch):
    vae = sd_mod.VAE.__new__(sd_mod.VAE)
    vae.first_stage_model = _SeedVR2DecodeStub()
    vae.vae_dtype = torch.float32
    vae.device = torch.device("cpu")
    vae.output_device = torch.device("cpu")
    vae.disable_offload = True
    vae.extra_1d_channel = None
    vae.latent_channels = 16
    vae.memory_used_decode = lambda shape, dtype: 1
    vae.process_output = lambda x: x
    vae.patcher = object()

    monkeypatch.setattr(sd_mod.model_management, "load_models_gpu", lambda *a, **k: None)
    monkeypatch.setattr(sd_mod.VAE, "decode_tiled_3d", lambda *a, **k: (_ for _ in ()).throw(AssertionError("generic decode_tiled_3d called")))

    latent = torch.zeros(1, 9, 8, 8, 16)
    vae.decode_tiled(latent, tile_x=2, tile_y=2, overlap=1, tile_t=16, overlap_t=4)

    assert vae.first_stage_model.calls[0]["shape"] == (1, 9, 8, 8, 16)


def test_seedvr2_decode_tiled_routes_collapsed_latents_to_seedvr2_tiler(monkeypatch):
    vae = sd_mod.VAE.__new__(sd_mod.VAE)
    vae.first_stage_model = _SeedVR2DecodeStub()
    vae.vae_dtype = torch.float32
    vae.device = torch.device("cpu")
    vae.output_device = torch.device("cpu")
    vae.disable_offload = True
    vae.extra_1d_channel = None
    vae.latent_channels = 16
    vae.memory_used_decode = lambda shape, dtype: 1
    vae.process_output = lambda x: x
    vae.patcher = object()

    monkeypatch.setattr(sd_mod.model_management, "load_models_gpu", lambda *a, **k: None)
    monkeypatch.setattr(sd_mod.VAE, "decode_tiled_", lambda *a, **k: (_ for _ in ()).throw(AssertionError("generic decode_tiled_ called")))

    latent = torch.zeros(1, 48, 2, 2)
    vae.decode_tiled(latent, tile_x=2, tile_y=2, overlap=1, tile_t=16, overlap_t=4)

    assert vae.first_stage_model.calls[0]["shape"] == (1, 48, 2, 2)
    assert vae.first_stage_model.calls[0]["seedvr2_tiling"]["temporal_overlap"] == 16


class _TemporalChunkRecorder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.device = "cpu"
        self.spatial_downsample_factor = 1
        self.temporal_downsample_factor = 4
        self.chunks = []

    def decode_(self, z):
        self.chunks.append([int(v) for v in z[0, 0, :, 0, 0].tolist()])
        pieces = [z[:, :1, :1]]
        if z.shape[2] > 1:
            pieces.append(z[:, :1, 1:].repeat_interleave(4, dim=2))
        return torch.cat(pieces, dim=2)


def test_seedvr2_tiled_vae_decode_uses_single_slicing_call_per_spatial_tile():
    """After the temporal-stitching fix, run_temporal_chunks delegates to
    the wrapper's slicing path with a single decode_ call per spatial tile
    (rather than the old hand-rolled outer temporal chunking that reset
    causal cache between chunks). Validate the new contract: recorder sees
    one call covering the full temporal axis, output shape and value
    pattern are equivalent to what the temporal-overlap path produced.
    """
    recorder = _TemporalChunkRecorder()
    latent = torch.arange(6, dtype=torch.float32).view(1, 1, 6, 1, 1)

    out = vae_mod.tiled_vae(
        latent,
        recorder,
        tile_size=(1, 1),
        tile_overlap=(0, 0),
        temporal_size=16,
        temporal_overlap=4,
        encode=False,
    )

    assert recorder.chunks == [[0, 1, 2, 3, 4, 5]]
    assert tuple(out.shape) == (1, 1, 21, 1, 1)
    assert [int(v) for v in out[0, 0, [0, 1, 5, 9, 13, 17], 0, 0].tolist()] == [0, 1, 2, 3, 4, 5]
