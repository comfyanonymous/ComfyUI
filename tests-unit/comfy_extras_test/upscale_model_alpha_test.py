import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.model_management  # noqa: E402
from comfy_extras.nodes_compositing import JoinImageWithAlpha  # noqa: E402
from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel  # noqa: E402


class StubUpscaleModel:
    """Stands in for a spandrel ImageModelDescriptor that only accepts RGB."""

    scale = 2
    input_channels = 3

    class patcher:
        load_device = torch.device("cpu")

    def __init__(self):
        self.seen_channels = None

    def __call__(self, image):
        self.seen_channels = image.shape[1]
        assert image.shape[1] == self.input_channels, "model was handed a non-RGB tensor"
        return torch.nn.functional.interpolate(image, scale_factor=self.scale, mode="nearest")


class GrayscaleUpscaleModel(StubUpscaleModel):
    """A model that wants a single channel. It must not be fed a colour channel as alpha."""

    input_channels = 1


def rgba_image():
    """16x16 RGBA where the top half is transparent and the bottom half opaque."""
    image = torch.zeros(1, 16, 16, 4)
    image[..., :3] = 0.5
    image[0, 8:, :, 3] = 1.0
    return image


def upscale(image, monkeypatch, model=None):
    monkeypatch.setattr(comfy.model_management, "load_models_gpu", lambda *args, **kwargs: None)
    model = model or StubUpscaleModel()
    return model, ImageUpscaleWithModel.execute(model, image)


def test_rgba_input_does_not_reach_the_model_and_alpha_is_upscaled(monkeypatch):
    model, out = upscale(rgba_image(), monkeypatch)
    image, mask = out.result

    assert model.seen_channels == 3
    assert image.shape == (1, 32, 32, 3)
    assert mask.shape == (1, 32, 32)
    # inverted convention: transparent -> 1, opaque -> 0
    assert mask[0, :14].min() > 0.9
    assert mask[0, 18:].max() < 0.1


def test_rgb_input_reports_a_fully_opaque_mask(monkeypatch):
    model, out = upscale(torch.zeros(1, 16, 16, 3), monkeypatch)
    image, mask = out.result

    assert model.seen_channels == 3
    assert image.shape == (1, 32, 32, 3)
    assert mask.shape == (1, 32, 32)
    assert mask.max() == 0.0


def test_rgb_input_is_untouched_by_a_model_that_wants_fewer_channels(monkeypatch):
    """The alpha split keys off the image being RGBA, never off the model's channel count.

    A single channel model must still receive all three colour channels and fail loudly,
    rather than having the green channel silently reinterpreted as alpha.
    """
    image = torch.zeros(1, 16, 16, 3)
    image[..., 1] = 0.7

    with pytest.raises(AssertionError):
        upscale(image, monkeypatch, model=GrayscaleUpscaleModel())


def test_mask_round_trips_through_join_image_with_alpha(monkeypatch):
    _, out = upscale(rgba_image(), monkeypatch)
    image, mask = out.result

    rgba = JoinImageWithAlpha.execute(image, mask).result[0]

    assert rgba.shape == (1, 32, 32, 4)
    assert rgba[0, :14, :, 3].max() < 0.1
    assert rgba[0, 18:, :, 3].min() > 0.9
