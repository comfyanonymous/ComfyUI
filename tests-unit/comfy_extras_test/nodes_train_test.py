import torch
from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy_extras.nodes_train import LossGraphNode  # noqa: E402


class TestLossGraphNode:
    def test_single_step_does_not_raise(self):
        # A one-step run has exactly one loss value, so min == max and the
        # old normalization divided by zero.
        LossGraphNode.execute({"loss": [3.1607]}, "loss_graph")

    def test_constant_loss_series_does_not_raise(self):
        LossGraphNode.execute({"loss": [1.0, 1.0, 1.0]}, "loss_graph")

    def test_varying_loss_series_still_scales(self):
        LossGraphNode.execute({"loss": [1.0, 0.5, 0.0]}, "loss_graph")
