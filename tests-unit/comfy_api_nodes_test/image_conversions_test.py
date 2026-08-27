from io import BytesIO

import pytest
import torch
from PIL import Image

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy_api_nodes.util.conversions import bytesio_to_image_tensor, pad_images_to_common_channels  # noqa: E402


def encode(image: Image.Image, image_format: str = "PNG") -> BytesIO:
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    buffer.seek(0)
    return buffer


def test_rgb_png_stays_three_channels():
    tensor = bytesio_to_image_tensor(encode(Image.new("RGB", (4, 4), (10, 20, 30))))
    assert tensor.shape == (1, 4, 4, 3)


def test_jpeg_stays_three_channels():
    tensor = bytesio_to_image_tensor(encode(Image.new("RGB", (4, 4), (10, 20, 30)), "JPEG"))
    assert tensor.shape == (1, 4, 4, 3)


def test_grayscale_is_expanded_to_rgb():
    tensor = bytesio_to_image_tensor(encode(Image.new("L", (4, 4), 128)))
    assert tensor.shape == (1, 4, 4, 3)


def test_rgba_png_keeps_its_alpha():
    tensor = bytesio_to_image_tensor(encode(Image.new("RGBA", (4, 4), (10, 20, 30, 0))))
    assert tensor.shape == (1, 4, 4, 4)
    assert tensor[..., 3].max() == 0.0


def test_palette_png_with_transparency_keeps_its_alpha():
    image = Image.new("P", (4, 4), 1)
    image.putpalette([0, 0, 0, 255, 255, 255])
    image.info["transparency"] = 0
    image.putpixel((0, 0), 0)
    tensor = bytesio_to_image_tensor(encode(image))
    assert tensor.shape == (1, 4, 4, 4)
    assert tensor[0, 0, 0, 3] == 0.0
    assert tensor[0, 1, 1, 3] == 1.0


@pytest.mark.parametrize("mode,channels", [("RGB", 3), ("RGBA", 4)])
def test_explicit_mode_is_respected(mode, channels):
    tensor = bytesio_to_image_tensor(encode(Image.new("RGBA", (4, 4), (10, 20, 30, 128))), mode=mode)
    assert tensor.shape == (1, 4, 4, channels)


def test_pad_mixed_channels_concatenates():
    rgb = torch.rand(1, 4, 4, 3)
    rgba = torch.rand(2, 4, 4, 4)
    padded = pad_images_to_common_channels([rgb, rgba])
    result = torch.cat(padded, dim=0)
    assert result.shape == (3, 4, 4, 4)


def test_pad_adds_opaque_alpha_and_keeps_rgb_values():
    rgb = torch.rand(1, 4, 4, 3)
    rgba = torch.rand(1, 4, 4, 4)
    padded_rgb, padded_rgba = pad_images_to_common_channels([rgb, rgba])
    assert torch.equal(padded_rgb[..., :3], rgb)
    assert padded_rgb[..., 3].min() == 1.0
    assert padded_rgba is rgba


def test_pad_leaves_homogeneous_channels_unchanged():
    images = [torch.rand(1, 4, 4, 3), torch.rand(2, 4, 4, 3)]
    padded = pad_images_to_common_channels(images)
    assert all(p is i for p, i in zip(padded, images))
