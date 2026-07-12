import torch
from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy_extras.nodes_custom_sampler import SplitSigmasDenoise


class TestSplitSigmasDenoise:
    def test_low_denoise_keeps_full_high_sigmas(self):
        # denoise small enough that round(steps * denoise) == 0. sigmas[:-0] == sigmas[:0]
        # collapsed high_sigmas to empty; it should stay the full schedule.
        sigmas = torch.linspace(10, 0, 21)  # 20 steps
        high, low = SplitSigmasDenoise.execute(sigmas, 0.02)
        assert high.shape[-1] == 21
        assert low.shape[-1] == 1

    def test_normal_denoise_split(self):
        sigmas = torch.linspace(10, 0, 21)
        high, low = SplitSigmasDenoise.execute(sigmas, 0.5)  # total_steps = 10
        assert high.shape[-1] == 11
        assert low.shape[-1] == 11
