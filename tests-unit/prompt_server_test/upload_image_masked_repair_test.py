"""Tests for the mask editor painted-masked upload repair.

The mask editor uploads each save as four sibling files named
clipspace-{mask,paint,painted,painted-masked}-<timestamp>.png. Frontends that
serialize the masked layer from the browser canvas bitmap upload
premultiplied alpha, which zeroes the RGB of fully masked pixels; the server
rebuilds the file from the opaque painted sibling so the original image
survives under the painted mask (#16139).
"""

import io
import os

import numpy as np
import pytest
from PIL import Image

from server import repair_clipspace_painted_masked


class FakeUpload:
    def __init__(self, png_bytes, filename):
        self.file = io.BytesIO(png_bytes)
        self.filename = filename


def make_rgba(rgb_array, alpha_array):
    rgba = np.dstack([rgb_array, alpha_array]).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def gradient():
    h, w = 8, 8
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[..., 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    rgb[..., 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    rgb[..., 2] = 128
    return rgb


def test_rebuilds_rgb_from_painted_sibling(tmp_path, gradient):
    # Premultiplied upload: masked (top) half lost its RGB, like a blob
    # serialized from a browser canvas.
    alpha = np.full((8, 8), 255, dtype=np.uint8)
    alpha[:4] = 0
    premultiplied = gradient * (alpha[..., None] // 255)
    masked_bytes = make_rgba(premultiplied, alpha)
    with open(os.path.join(tmp_path, "clipspace-painted-42.png"), "wb") as f:
        f.write(make_rgba(gradient, np.full((8, 8), 255, dtype=np.uint8)))

    upload = FakeUpload(masked_bytes, "clipspace-painted-masked-42.png")
    data = repair_clipspace_painted_masked(upload, upload.filename, str(tmp_path))

    assert data is not None
    rebuilt = np.array(Image.open(io.BytesIO(data)))
    assert np.array_equal(rebuilt[..., :3], gradient)  # original pixels restored
    assert np.array_equal(rebuilt[..., 3], alpha)  # painted mask preserved

    # The upload stream is rewound so the raw write fallback still works.
    upload.file.seek(0)
    assert upload.file.read() == masked_bytes


def test_straight_alpha_upload_is_unchanged_in_content(tmp_path, gradient):
    # A frontend that encodes straight alpha keeps its RGB; rebuilding from
    # the identical painted sibling must not alter any pixel.
    alpha = np.full((8, 8), 255, dtype=np.uint8)
    alpha[:4] = 0
    masked_bytes = make_rgba(gradient, alpha)
    with open(os.path.join(tmp_path, "clipspace-painted-42.png"), "wb") as f:
        f.write(make_rgba(gradient, np.full((8, 8), 255, dtype=np.uint8)))

    upload = FakeUpload(masked_bytes, "clipspace-painted-masked-42.png")
    data = repair_clipspace_painted_masked(upload, upload.filename, str(tmp_path))

    rebuilt = np.array(Image.open(io.BytesIO(data)))
    assert np.array_equal(rebuilt[..., :3], gradient)
    assert np.array_equal(rebuilt[..., 3], alpha)


def test_returns_none_without_sibling(tmp_path):
    alpha = np.full((4, 4), 0, dtype=np.uint8)
    upload = FakeUpload(make_rgba(np.zeros((4, 4, 3), np.uint8), alpha),
                        "clipspace-painted-masked-42.png")
    assert repair_clipspace_painted_masked(upload, upload.filename, str(tmp_path)) is None


def test_returns_none_for_other_filenames(tmp_path, gradient):
    with open(os.path.join(tmp_path, "clipspace-painted-42.png"), "wb") as f:
        f.write(make_rgba(gradient, np.full((8, 8), 255, dtype=np.uint8)))
    for filename in ("clipspace-mask-42.png", "clipspace-painted-42.png", "example.png"):
        upload = FakeUpload(make_rgba(gradient, np.full((8, 8), 255, dtype=np.uint8)), filename)
        assert repair_clipspace_painted_masked(upload, filename, str(tmp_path)) is None


def test_returns_none_on_size_mismatch(tmp_path, gradient):
    alpha = np.full((8, 8), 0, dtype=np.uint8)
    with open(os.path.join(tmp_path, "clipspace-painted-42.png"), "wb") as f:
        f.write(make_rgba(gradient[:4], np.full((4, 8), 255, dtype=np.uint8)))

    upload = FakeUpload(make_rgba(gradient, alpha), "clipspace-painted-masked-42.png")
    assert repair_clipspace_painted_masked(upload, upload.filename, str(tmp_path)) is None


def test_returns_none_without_alpha_channel(tmp_path, gradient):
    with open(os.path.join(tmp_path, "clipspace-painted-42.png"), "wb") as f:
        f.write(make_rgba(gradient, np.full((8, 8), 255, dtype=np.uint8)))

    buffer = io.BytesIO()
    Image.fromarray(gradient, "RGB").save(buffer, format="PNG")
    upload = FakeUpload(buffer.getvalue(), "clipspace-painted-masked-42.png")
    assert repair_clipspace_painted_masked(upload, upload.filename, str(tmp_path)) is None
