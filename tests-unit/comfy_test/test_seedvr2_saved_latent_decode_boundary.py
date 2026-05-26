import io

import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy_extras import nodes_seedvr  # noqa: E402
import nodes as nodes_mod  # noqa: E402


class _DecodeOnlyVAE:
    def __init__(self):
        self.decode_calls = 0

    def decode(self, latent):
        self.decode_calls += 1
        b, tc, h, w = latent.shape
        t = tc // 16
        return torch.full((b, t, h * 8, w * 8, 3), 0.25)


def test_saved_loaded_seedvr2_latent_decode_boundary_does_not_rerun_preprocessing():
    latent = {"samples": torch.zeros(1, 32, 4, 5)}
    buffer = io.BytesIO()
    torch.save(latent["samples"], buffer)
    buffer.seek(0)
    loaded = {"samples": torch.load(buffer, weights_only=True)}

    vae = _DecodeOnlyVAE()
    decoded = nodes_mod.VAEDecode().decode(vae, loaded)[0]
    original = torch.full((1, 2, 32, 40, 3), 0.75)
    output = nodes_seedvr.SeedVR2PostProcessing.execute(decoded, original, 32, "none").result[0]

    assert vae.decode_calls == 1
    assert tuple(output.shape) == (2, 32, 40, 3)
