import torch

import node_helpers


class TestImageAlphaFix:
    def test_pads_rgb_to_rgba(self):
        destination = torch.zeros(1, 2, 2, 4)
        source = torch.full((1, 2, 2, 3), 0.5)
        _, source = node_helpers.image_alpha_fix(destination, source)
        assert source.shape[-1] == 4
        assert torch.all(source[..., :3] == 0.5)  # original channels preserved
        assert torch.all(source[..., 3] == 1.0)   # added alpha

    def test_pads_more_than_one_channel(self):
        # channel gap > 1 (grayscale source, RGBA destination) must still match up.
        destination = torch.zeros(1, 2, 2, 4)
        source = torch.zeros(1, 2, 2, 1)
        _, source = node_helpers.image_alpha_fix(destination, source)
        assert source.shape[-1] == 4
        assert torch.all(source[..., 1:] == 1.0)

    def test_truncates_when_source_has_more_channels(self):
        destination = torch.zeros(1, 2, 2, 3)
        source = torch.ones(1, 2, 2, 4)
        _, source = node_helpers.image_alpha_fix(destination, source)
        assert source.shape[-1] == 3
