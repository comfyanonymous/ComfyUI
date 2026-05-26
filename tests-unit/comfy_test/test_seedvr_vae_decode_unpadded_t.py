import pytest
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy_extras import nodes_seedvr  # noqa: E402


def _t_padded(t_in: int) -> int:
    if t_in == 1:
        return 1
    if t_in <= 4:
        return 5
    if (t_in - 1) % 4 == 0:
        return t_in
    return t_in + (4 - ((t_in - 1) % 4))


@pytest.mark.parametrize("t_in", [1, 2, 3, 4, 5, 6, 7, 8])
def test_t_padded_matches_cut_videos(t_in):
    dummy = torch.zeros(1, t_in, 1, 1, 1)
    assert nodes_seedvr.cut_videos(dummy).shape[1] == _t_padded(t_in)


@pytest.mark.parametrize("t_in", [1, 2, 3, 4, 5, 6, 7, 8])
def test_post_processing_trims_decoded_video_to_explicit_reference_frames(t_in):
    decoded = torch.zeros(1, _t_padded(t_in), 32, 32, 3)
    original = torch.zeros(1, t_in, 32, 32, 3)

    output = nodes_seedvr.SeedVR2PostProcessing.execute(decoded, original, 32, "none").result[0]

    assert tuple(output.shape) == (1, t_in, 32, 32, 3)
