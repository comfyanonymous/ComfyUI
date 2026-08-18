import pytest
import torch
import torch.nn as nn

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy.model_patcher import ModelPatcher
from comfy_extras.nodes_model_merging import CLIPMergeSimple


class TinyModel(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = nn.Parameter(weight)


class TinyClip:
    def __init__(self, patcher):
        self.patcher = patcher

    def clone(self):
        return TinyClip(self.patcher.clone())

    def get_key_patches(self):
        return self.patcher.get_key_patches()

    def add_patches(self, patches, strength_patch, strength_model):
        return self.patcher.add_patches(patches, strength_patch, strength_model)


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [
        (0.0, torch.tensor([1.0, 3.0])),
        (0.25, torch.tensor([2.0, 4.0])),
        (1.0, torch.tensor([5.0, 7.0])),
    ],
)
def test_clip_merge_ratio_blends_from_clip1_to_clip2(ratio, expected):
    clip1 = TinyClip(_make_patcher(torch.tensor([1.0, 3.0])))
    clip2 = TinyClip(_make_patcher(torch.tensor([5.0, 7.0])))

    merged = CLIPMergeSimple().merge(clip1, clip2, ratio)[0]
    weight = merged.patcher.patch_weight_to_device("weight", return_weight=True)

    torch.testing.assert_close(weight, expected)


def _make_patcher(weight):
    device = torch.device("cpu")
    return ModelPatcher(TinyModel(weight), load_device=device, offload_device=device)
