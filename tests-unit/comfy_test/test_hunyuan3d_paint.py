"""Unit tests for the native Hunyuan3D 2.1 PBR paint model port.

These construct a small randomly-initialized UNet2p5DConditionModel (no weights
required) and verify architecture, state_dict/detection parity, the DDIM
v-prediction schedule, and the multiview diffusion driver's output shapes.
"""

from __future__ import annotations

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.ops as comfy_ops  # noqa: E402
from comfy.ldm.hunyuan3d.paint.unet import UNet2p5DConditionModel  # noqa: E402
from comfy.ldm.hunyuan3d.paint.loader import detect_paint_config, load_paint_unet, _HEAD_DIM  # noqa: E402
from comfy.ldm.hunyuan3d.paint.sampler import (  # noqa: E402
    DDIMVScheduler,
    generate_multiview,
    _cam_mapping,
)

OPS = comfy_ops.disable_weight_init

# Small config using the real model's SD-2 head dim (64) so detection round-trips.
SMALL = dict(
    in_channels=12, ref_in_channels=4, out_channels=4,
    block_out_channels=(64, 64, 128, 128), layers_per_block=2, cross_attention_dim=64,
    num_attention_heads=(1, 1, 2, 2), transformer_layers_per_block=1, norm_num_groups=32,
    pbr_setting=("albedo", "mr"), pbr_token_channels=7, dino_embeddings_dim=32, use_dino=True,
)
CROSS = SMALL["cross_attention_dim"]
DINO_DIM = SMALL["dino_embeddings_dim"]
TOKENS = SMALL["pbr_token_channels"]


def _build(dtype=torch.float32, device="cpu", **over):
    cfg = {**SMALL, **over}
    return UNet2p5DConditionModel(dtype=dtype, device=device, operations=OPS, **cfg), cfg


def _inputs(B, n_pbr, V, H, dtype=torch.float32, dino=True, position=True):
    sample = torch.randn(B, n_pbr, V, 4, H, H, dtype=dtype)
    enc = torch.randn(B, n_pbr, TOKENS, CROSS, dtype=dtype)
    ref = torch.randn(B, 1, 4, H, H, dtype=dtype)
    normal = torch.randn(B, V, 4, H, H, dtype=dtype)
    pos_embed = torch.randn(B, V, 4, H, H, dtype=dtype)
    pos_map = torch.rand(B, V, 3, H, H, dtype=dtype) if position else None
    dino_h = torch.randn(B, 5, DINO_DIM, dtype=dtype) if dino else None
    return sample, enc, ref, normal, pos_embed, pos_map, dino_h


def test_forward_output_shape_and_dtype():
    model, _ = _build()
    model.eval()
    B, n_pbr, V, H = 1, 2, 3, 16
    sample, enc, ref, normal, pos_embed, pos_map, dino_h = _inputs(B, n_pbr, V, H)
    with torch.no_grad():
        out = model(sample, torch.tensor([500]), enc, dino_hidden_states=dino_h, ref_latents=ref,
                    embeds_normal=normal, embeds_position=pos_embed, position_maps=pos_map)
    assert tuple(out.shape) == (B * n_pbr * V, 4, H, H)
    assert out.dtype == torch.float32


def test_forward_without_dino_or_position():
    model, _ = _build()
    model.eval()
    B, n_pbr, V, H = 1, 2, 3, 16
    sample, enc, ref, normal, pos_embed, _, _ = _inputs(B, n_pbr, V, H, dino=False, position=False)
    with torch.no_grad():
        out = model(sample, torch.tensor([10]), enc, dino_hidden_states=None, ref_latents=ref,
                    embeds_normal=normal, embeds_position=pos_embed, position_maps=None)
    assert tuple(out.shape) == (B * n_pbr * V, 4, H, H)


def test_forward_cfg_batch_with_tensor_ref_scale():
    """Reference pipeline batches uncond/ref/full along B with a per-batch ref_scale."""
    model, _ = _build()
    model.eval()
    B, n_pbr, V, H = 3, 2, 2, 16
    sample, enc, ref, normal, pos_embed, pos_map, dino_h = _inputs(B, n_pbr, V, H)
    ref_scale = torch.tensor([0.0, 1.0, 1.0])
    with torch.no_grad():
        out = model(sample, torch.tensor([500]), enc, dino_hidden_states=dino_h, ref_latents=ref,
                    embeds_normal=normal, embeds_position=pos_embed, position_maps=pos_map,
                    ref_scale=ref_scale)
    assert tuple(out.shape) == (B * n_pbr * V, 4, H, H)


def test_bfloat16_forward():
    model, _ = _build(dtype=torch.bfloat16)
    model.eval()
    B, n_pbr, V, H = 1, 2, 2, 16
    sample, enc, ref, normal, pos_embed, pos_map, dino_h = _inputs(B, n_pbr, V, H, dtype=torch.bfloat16)
    with torch.no_grad():
        out = model(sample, torch.tensor([500]), enc, dino_hidden_states=dino_h, ref_latents=ref,
                    embeds_normal=normal, embeds_position=pos_embed, position_maps=pos_map)
    assert out.dtype == torch.bfloat16
    assert tuple(out.shape) == (B * n_pbr * V, 4, H, H)


def test_detect_config_roundtrip():
    model, cfg = _build()
    detected = detect_paint_config(model.state_dict())
    assert detected is not None
    for key in ("in_channels", "ref_in_channels", "out_channels", "block_out_channels",
                "layers_per_block", "cross_attention_dim", "transformer_layers_per_block",
                "pbr_setting", "pbr_token_channels", "dino_embeddings_dim", "use_dino"):
        assert detected[key] == cfg[key], (key, detected[key], cfg[key])
    # heads are recovered assuming SD-2 head dim
    assert detected["num_attention_heads"] == tuple(c // _HEAD_DIM for c in cfg["block_out_channels"])


def test_detect_returns_none_for_non_paint_state_dict():
    assert detect_paint_config({"foo.weight": torch.zeros(1)}) is None
    assert detect_paint_config({}) is None


def test_loader_rejects_ldm_checkpoint_with_family_diagnosis():
    """A wrong-family checkpoint must fail with a clear "this looks like X, expected
    hunyuan3d-paintpbr-v2-1" ValueError - never a KeyError."""
    sd = {"model.diffusion_model.input_blocks.0.0.weight": torch.zeros(320, 4, 3, 3),
          "model.diffusion_model.out.2.weight": torch.zeros(4, 320, 3, 3)}
    with pytest.raises(ValueError, match=r"looks like a ComfyUI/LDM diffusion checkpoint"):
        load_paint_unet(sd)


def test_loader_rejects_plain_diffusers_unet_with_family_diagnosis():
    sd = {"conv_in.weight": torch.zeros(320, 4, 3, 3),
          "down_blocks.0.resnets.0.conv1.weight": torch.zeros(320, 320, 3, 3)}
    with pytest.raises(ValueError, match="plain diffusers UNet"):
        load_paint_unet(sd)


def test_loader_rejects_single_stream_checkpoint():
    """Paint-style keys without the dual-stream reference UNet are named as such."""
    model, _ = _build()
    sd = {k: v for k, v in model.state_dict().items() if not k.startswith("unet_dual.")}
    with pytest.raises(ValueError, match="dual-stream reference UNet"):
        load_paint_unet(sd)


def test_loader_rejects_truncated_checkpoint_not_keyerror():
    """A paint-family checkpoint missing derivation keys raises an informative
    ValueError (truncated), not a KeyError from a raw dict lookup."""
    model, _ = _build()
    sd = dict(model.state_dict())
    del sd["unet.conv_out.weight"]
    with pytest.raises(ValueError, match="truncated"):
        detect_paint_config(sd)


def test_loader_rejects_checkpoint_with_missing_weights():
    """Detection can pass while weights are still missing; the load must hard-fail
    with a truncation message instead of silently leaving random weights."""
    model, _ = _build()
    sd = dict(model.state_dict())
    # not read by detect_paint_config, but required by the model
    del sd["unet.mid_block.resnets.0.conv1.weight"]
    with pytest.raises(ValueError, match="missing 1 weights"):
        load_paint_unet(sd, model_options={"dtype": torch.float32})


def test_detected_config_rebuilds_with_strict_key_parity():
    """Detection -> config -> rebuild must reproduce the exact state_dict keys/shapes,
    and loading the weights must give a numerically identical model."""
    torch.manual_seed(0)
    model, _ = _build()
    model.eval()
    sd = model.state_dict()

    detected = detect_paint_config(sd)
    twin = UNet2p5DConditionModel(
        dtype=torch.float32, device="cpu", operations=OPS,
        in_channels=detected["in_channels"], ref_in_channels=detected["ref_in_channels"],
        out_channels=detected["out_channels"], block_out_channels=detected["block_out_channels"],
        layers_per_block=detected["layers_per_block"], cross_attention_dim=detected["cross_attention_dim"],
        num_attention_heads=detected["num_attention_heads"],
        transformer_layers_per_block=detected["transformer_layers_per_block"],
        norm_num_groups=detected["norm_num_groups"], pbr_setting=detected["pbr_setting"],
        pbr_token_channels=detected["pbr_token_channels"], dino_embeddings_dim=detected["dino_embeddings_dim"],
        use_dino=detected["use_dino"])
    twin.eval()

    missing, unexpected = twin.load_state_dict(sd, strict=False)
    assert missing == [], missing
    assert unexpected == [], unexpected

    B, n_pbr, V, H = 1, 2, 2, 16
    sample, enc, ref, normal, pos_embed, pos_map, dino_h = _inputs(B, n_pbr, V, H)
    with torch.no_grad():
        a = model(sample, torch.tensor([500]), enc, dino_hidden_states=dino_h, ref_latents=ref,
                  embeds_normal=normal, embeds_position=pos_embed, position_maps=pos_map)
        b = twin(sample, torch.tensor([500]), enc, dino_hidden_states=dino_h, ref_latents=ref,
                 embeds_normal=normal, embeds_position=pos_embed, position_maps=pos_map)
    # The rebuilt model must reproduce the reference exactly (equal_nan: random-init
    # deep nets can overflow, but both models must overflow identically).
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4, equal_nan=True)


def test_load_paint_unet_returns_patcher_and_config():
    """The core loader path builds a ModelPatcher-wrapped model with a detected config."""
    from comfy.model_patcher import ModelPatcher
    model, cfg = _build()
    sd = model.state_dict()
    patcher, config = load_paint_unet(sd, model_options={"dtype": torch.float32})
    assert isinstance(patcher, ModelPatcher)
    assert isinstance(patcher.model, UNet2p5DConditionModel)
    assert config["block_out_channels"] == cfg["block_out_channels"]
    assert config["pbr_setting"] == cfg["pbr_setting"]
    assert set(patcher.model.state_dict().keys()) == set(sd.keys())


def test_state_dict_has_expected_special_keys():
    model, _ = _build()
    keys = set(model.state_dict().keys())
    assert "unet.conv_in.weight" in keys
    assert "unet_dual.conv_in.weight" in keys
    assert "unet.learned_text_clip_albedo" in keys
    assert "unet.learned_text_clip_mr" in keys
    assert "unet.learned_text_clip_ref" in keys
    assert "unet.image_proj_model_dino.proj.weight" in keys
    # material and reference PBR heads
    assert any(k.endswith("transformer.attn1.processor.to_q_mr.weight") for k in keys)
    assert any("attn_refview.processor.to_v_mr.weight" in k for k in keys)
    assert any("attn_multiview.to_q.weight" in k for k in keys)
    assert any("attn_dino.to_k.weight" in k for k in keys)
    # conv_in is 12 channels on the main stream, 4 on the dual/reference stream
    assert model.state_dict()["unet.conv_in.weight"].shape[1] == 12
    assert model.state_dict()["unet_dual.conv_in.weight"].shape[1] == 4


def test_ddim_scheduler_zero_terminal_snr_and_trailing():
    sched = DDIMVScheduler()
    ts = sched.set_timesteps(15)
    assert len(ts) == 15
    assert int(ts[0]) == 999            # trailing spacing starts at the last train step
    assert torch.all(ts[:-1] > ts[1:])  # strictly decreasing
    assert sched.alphas_cumprod[-1].item() < 1e-6  # zero terminal SNR


def test_ddim_schedule_matches_core_model_sampling():
    """Rewire-parity contract: the bespoke DDIM v-pred/zero-SNR schedule must stay
    interchangeable with core's ModelSamplingDiscrete(zsnr=True).

    Same betas (scaled-linear 0.00085..0.012), same zero-terminal-SNR rescale; the
    only intended difference is the terminal step, where core clamps
    alpha_cumprod[-1] to 4.897e-8 (finite sigma_max ~4519) while the reference
    scheduler keeps an exact 0 (infinite sigma). A full 15-step DDIM loop under
    either convention agrees to ~2e-4 (see PR notes)."""
    import comfy.model_sampling as cms

    ms = cms.ModelSamplingDiscrete(model_config=None, zsnr=True)
    sched = DDIMVScheduler()
    ac = sched.alphas_cumprod
    sigmas = ((1 - ac[:-1]) / ac[:-1]) ** 0.5
    torch.testing.assert_close(ms.sigmas[:999].double(), sigmas, rtol=1e-6, atol=1e-4)
    # documented divergence at the terminal step
    assert float(ac[-1]) == 0.0
    assert float(ms.sigmas[-1]) > 4000.0
    # the trailing timesteps round-trip exactly through core's sigma<->timestep maps
    ts = sched.set_timesteps(15).tolist()
    rt = ms.timestep(ms.sigma(torch.tensor(ts, dtype=torch.float32))).tolist()
    assert rt == ts


def test_cam_mapping_view_scale():
    assert _cam_mapping(0) == pytest.approx(1.0)
    assert _cam_mapping(90) == pytest.approx(2.0)
    assert _cam_mapping(180) == pytest.approx(2.0)
    assert _cam_mapping(360) == pytest.approx(1.0)


def test_generate_multiview_end_to_end_shapes():
    model, cfg = _build()
    model.eval()
    V, H = 4, 16
    ref_latent = torch.randn(1, 4, H, H)
    normal_latents = torch.randn(V, 4, H, H)
    position_latents = torch.randn(V, 4, H, H)
    position_maps = torch.rand(V, 3, H, H)
    dino = torch.randn(1, 5, DINO_DIM)
    out = generate_multiview(model, cfg, ref_latent, normal_latents, position_latents, position_maps,
                             dino_features=dino, camera_azims=[0, 90, 180, 270],
                             num_inference_steps=3, guidance_scale=3.0, seed=0, device="cpu")
    assert set(out.keys()) == {"albedo", "mr"}
    for v in out.values():
        assert tuple(v.shape) == (V, 4, H, H)


def test_generate_multiview_without_cfg_or_dino():
    model, cfg = _build()
    model.eval()
    V, H = 2, 16
    ref_latent = torch.randn(1, 4, H, H)
    normal_latents = torch.randn(V, 4, H, H)
    position_latents = torch.randn(V, 4, H, H)
    position_maps = torch.rand(V, 3, H, H)
    out = generate_multiview(model, cfg, ref_latent, normal_latents, position_latents, position_maps,
                             dino_features=None, camera_azims=[0, 180],
                             num_inference_steps=2, guidance_scale=1.0, seed=1, device="cpu")
    for v in out.values():
        assert tuple(v.shape) == (V, 4, H, H)
