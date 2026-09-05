import pytest
import torch

from comfy.cli_args import args

args.cpu = True

import comfy.diffusers_convert
import comfy.model_management
import comfy.sd


def test_flux2_vae_load_encode_decode_matches_llada_reference():
    diffusers = pytest.importorskip("diffusers")
    torch.manual_seed(172)
    # Pinned LLaDA-Image VAE configuration; random weights isolate the adapter.
    reference = diffusers.AutoencoderKLFlux2(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(128, 256, 512, 512),
        layers_per_block=2,
        latent_channels=32,
        norm_num_groups=32,
        sample_size=1024,
        patch_size=(2, 2),
        batch_norm_eps=1e-4,
        batch_norm_momentum=0.1,
        use_quant_conv=True,
        use_post_quant_conv=True,
    ).eval()
    reference.bn.running_mean.copy_(torch.linspace(-0.3, 0.3, 128))
    reference.bn.running_var.copy_(torch.linspace(0.5, 1.5, 128))
    source_state = reference.state_dict()
    vae = comfy.sd.VAE(
        sd=dict(source_state), device=torch.device("cpu"), dtype=torch.float32
    )
    converted = comfy.diffusers_convert.convert_vae_state_dict(dict(source_state))
    actual_state = vae.first_stage_model.state_dict()
    assert set(actual_state) == set(converted)
    for key, expected in converted.items():
        assert torch.equal(actual_state[key], expected), key
    assert vae.latent_channels == 128
    assert vae.downscale_ratio == vae.upscale_ratio == 16

    image = torch.rand(1, 32, 64, 3)
    try:
        with torch.inference_mode():
            unpatched = reference.encode(image.movedim(-1, 1) * 2 - 1).latent_dist.mode()
            batch, channels, height, width = unpatched.shape
            patched = (
                unpatched.reshape(batch, channels, height // 2, 2, width // 2, 2)
                .permute(0, 1, 3, 5, 2, 4)
                .reshape(batch, channels * 4, height // 2, width // 2)
            )
            mean = reference.bn.running_mean.view(1, -1, 1, 1)
            std = (reference.bn.running_var.view(1, -1, 1, 1) + 1e-4).sqrt()
            expected_latent = (patched - mean) / std
            actual_latent = vae.encode(image)
            torch.testing.assert_close(actual_latent, expected_latent, rtol=1e-4, atol=1e-5)

            # Decode identical normalized latents to isolate the reverse adapter.
            normalized = torch.randn_like(expected_latent)
            patched = normalized * std + mean
            unpatched = (
                patched.reshape(batch, channels, 2, 2, height // 2, width // 2)
                .permute(0, 1, 4, 2, 5, 3)
                .reshape(batch, channels, height, width)
            )
            expected_image = reference.decode(unpatched).sample
            expected_image = ((expected_image + 1) / 2).clamp(0, 1).movedim(1, -1)
            actual_image = vae.decode(normalized)
            torch.testing.assert_close(actual_image, expected_image, rtol=1e-4, atol=1e-5)
    finally:
        comfy.model_management.unload_all_models()
