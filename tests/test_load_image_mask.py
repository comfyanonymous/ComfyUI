import sys
from types import SimpleNamespace

from PIL import Image
import pytest
import torch

import comfy.options


# LoadImage imports ComfyUI's device management, which must be initialized in
# CPU mode on CPU-only test environments.
comfy.options.args_parsing = True
pytest_argv = sys.argv
sys.argv = [sys.argv[0], "--cpu"]

import nodes

sys.argv = pytest_argv
comfy.options.args_parsing = False


class EmptyVideoComponents:
    def get_components(self):
        return SimpleNamespace(images=torch.empty(0), alpha=None)


@pytest.mark.parametrize("use_video_loader", [True, False])
def test_empty_mask_matches_image_dimensions(tmp_path, monkeypatch, use_video_loader):
    image_path = tmp_path / "rgb.png"
    Image.new("RGB", (37, 23), (128, 64, 32)).save(image_path)

    if not use_video_loader:
        monkeypatch.setattr(nodes.InputImpl, "VideoFromFile", lambda _: EmptyVideoComponents())
    monkeypatch.setattr(nodes.folder_paths, "get_annotated_filepath", lambda _: str(image_path))

    image, mask = nodes.LoadImage().load_image("rgb.png")

    assert image.shape == (1, 23, 37, 3)
    assert mask.shape == (1, 23, 37)
    assert torch.equal(mask, torch.zeros_like(mask))
