import torch
from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy.samplers import kl_optimal_scheduler


class TestKLOptimalScheduler:
    def test_single_step_is_not_nan(self):
        # steps=1 is a valid KSampler input; div by (n - 1) made the first sigma NaN.
        sigmas = kl_optimal_scheduler(1, 0.0291675, 14.614642)
        assert not torch.isnan(sigmas).any()
        assert sigmas.shape == (2,)
        # the single step should start at sigma_max and end at 0
        assert torch.isclose(sigmas[0], torch.tensor(14.614642), atol=1e-3)
        assert sigmas[-1] == 0

    def test_multi_step_unchanged(self):
        sigmas = kl_optimal_scheduler(20, 0.0291675, 14.614642)
        assert not torch.isnan(sigmas).any()
        assert sigmas.shape == (21,)
        assert torch.isclose(sigmas[0], torch.tensor(14.614642), atol=1e-3)
        assert sigmas[-1] == 0
