import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.ldm.minimax.model as minimax_model


def test_cond_video_rows_pads_odd_spatial_grid():
    model = minimax_model.MiniMaxH3Model.__new__(minimax_model.MiniMaxH3Model)
    model.patch_size = (1, 2, 2)

    # odd latent grid (e.g. from a source width/height whose VAE-cropped
    # size divides down to an odd value) must not crash patchify_video
    z = torch.randn(1, 24, 1, 31, 33)
    payload = {"cond_video_latents": [z], "visual_cond_noise_aug": 1.0}

    rows = model._cond_video_rows(payload, torch.device("cpu"))

    padded_t, padded_h, padded_w = 1, 32, 34
    expected_rows = padded_t * (padded_h // 2) * (padded_w // 2)
    expected_dim = 24 * 1 * 2 * 2
    assert rows.shape == (expected_rows, expected_dim)
