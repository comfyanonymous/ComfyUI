import io

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import folder_paths
import nodes
from PIL import Image


def _ftyp_box(major, *brands):
    body = major + b"\x00\x00\x00\x00" + b"".join(brands)
    return (8 + len(body)).to_bytes(4, "big") + b"ftyp" + body


def test_is_heif_detects_major_and_compatible_brands(tmp_path):
    major_heic = tmp_path / "major.heic"
    major_heic.write_bytes(_ftyp_box(b"heic", b"mif1"))
    assert nodes.is_heif(str(major_heic))

    compat_heic = tmp_path / "compat.heic"
    compat_heic.write_bytes(_ftyp_box(b"mif1", b"mif1", b"heic", b"hevc"))
    assert nodes.is_heif(str(compat_heic))


def test_is_heif_rejects_other_containers(tmp_path):
    avif = tmp_path / "image.avif"
    avif.write_bytes(_ftyp_box(b"avif", b"avif", b"mif1", b"miaf"))
    assert not nodes.is_heif(str(avif))

    mp4 = tmp_path / "video.mp4"
    mp4.write_bytes(_ftyp_box(b"isom", b"isom", b"iso2", b"avc1"))
    assert not nodes.is_heif(str(mp4))

    png = tmp_path / "image.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    assert not nodes.is_heif(str(png))

    truncated = tmp_path / "broken.heic"
    truncated.write_bytes(b"\x00\x00\x00\x1cftyp")
    assert not nodes.is_heif(str(truncated))


@pytest.fixture
def _load_image_file(tmp_path, monkeypatch):
    def _set(name, content):
        path = tmp_path / name
        path.write_bytes(content)
        monkeypatch.setattr(folder_paths, "get_annotated_filepath", lambda image: str(path))

    return _set


def test_load_image_uses_pillow_for_heif(monkeypatch, _load_image_file):
    _load_image_file("image.heic", _ftyp_box(b"mif1", b"heic") + b"\x00" * 64)

    def _fail(*args, **kwargs):
        raise AssertionError("HEIF images should not be loaded with pyav")

    monkeypatch.setattr(nodes.InputImpl, "VideoFromFile", _fail)
    with pytest.raises(OSError):
        nodes.LoadImage().load_image("image.heic")


def test_load_image_still_uses_pyav_for_other_formats(monkeypatch, _load_image_file):
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), (255, 0, 0)).save(buffer, format="PNG")
    _load_image_file("image.png", buffer.getvalue())

    calls = []
    original = nodes.InputImpl.VideoFromFile

    def _spy(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(nodes.InputImpl, "VideoFromFile", _spy)

    images, _ = nodes.LoadImage().load_image("image.png")
    assert len(calls) == 1
    assert images.shape[1:3] == (4, 4)
