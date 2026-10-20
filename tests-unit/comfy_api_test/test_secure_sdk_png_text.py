import asyncio
import json

import numpy as np
import pytest
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from comfy.cli_args import args
from comfy_api.latest import _sdk


PROMPT = {"7": {"class_type": "Save", "inputs": {"enabled": False}}}
EXTRA = {"workflow": {"nodes": []}, "custom": {"values": [None, float("nan"), float("inf"), -float("inf")]}}


def context():
    plan = _sdk.ExecutionPlan("png-text", "7", "Save", prompt=PROMPT, extra_pnginfo=EXTRA)
    return _sdk.InProcessCtxProvider().build(plan)


@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("channels", [2, 3])
def test_explicit_text_matches_pillow_bytes_with_nonfinite_json(tmp_path, monkeypatch, disabled, channels):
    import folder_paths

    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    monkeypatch.setattr(args, "disable_metadata", disabled)
    pixels = torch.linspace(-0.2, 1.2, 4 * 5 * channels).reshape(1, 4, 5, channels)
    text = {"prompt": json.dumps(PROMPT), **{key: json.dumps(value) for key, value in EXTRA.items()}}
    text["parameters"] = "literal text, not another JSON string"

    async def run():
        refs = _sdk.InProcessRefResolver()
        ctx = context()
        with _sdk.bind_runtime(refs, ctx, _sdk.InProcessOps()):
            image = _sdk.ImageRef._wrap(await refs.create("IMAGE", pixels))
            return await ctx.output.save_images(image, filenames=["actual.png"], png_text=text)

    result = asyncio.run(run())
    metadata = PngInfo()
    for key, value in text.items():
        metadata.add_text(key, value)
    expected = np.clip(255.0 * pixels[0].numpy(), 0, 255).astype(np.uint8)
    Image.fromarray(expected).save(tmp_path / "expected.png", pnginfo=metadata, compress_level=4)
    assert (tmp_path / "actual.png").read_bytes() == (tmp_path / "expected.png").read_bytes()
    assert result == {"images": [{"filename": "actual.png", "subfolder": "", "type": "output"}]}
    with Image.open(tmp_path / "actual.png") as saved:
        assert saved.mode == ("LA" if channels == 2 else "RGB")
        assert saved.info == text
        assert saved.size == (5, 4)
        assert np.array_equal(np.asarray(saved), expected)


def test_existing_metadata_defaults_and_strict_extra_json_are_unchanged(tmp_path, monkeypatch):
    import folder_paths

    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))

    async def run():
        refs = _sdk.InProcessRefResolver()
        ctx = context()
        with _sdk.bind_runtime(refs, ctx, _sdk.InProcessOps()):
            image = _sdk.ImageRef._wrap(await refs.create("IMAGE", torch.zeros((1, 2, 2, 3))))
            for disabled in (False, True):
                monkeypatch.setattr(args, "disable_metadata", disabled)
                await ctx.output.save_images(image, filenames=[f"default-{disabled}.png"])
            await ctx.output.save_images(image, filenames=["empty.png"], png_text={})
            with pytest.raises(ValueError, match="JSON-compatible"):
                await ctx.output.save_images(image, filenames=["bad.png"], extra_metadata={"value": float("nan")})

    asyncio.run(run())
    with Image.open(tmp_path / "default-False.png") as saved:
        assert saved.info == {"prompt": json.dumps(PROMPT), **{key: json.dumps(value) for key, value in EXTRA.items()}}
    for filename in ("default-True.png", "empty.png"):
        with Image.open(tmp_path / filename) as saved:
            assert saved.info == {}
    assert not (tmp_path / "bad.png").exists()


@pytest.mark.parametrize("text,other,error", [
    ((), {}, TypeError),
    ({1: "value"}, {}, TypeError),
    ({"value": 1}, {}, TypeError),
    ({"value": None}, {}, TypeError),
    ({"": "value"}, {}, ValueError),
    ({"bad\x00key": "value"}, {}, ValueError),
    ({"k" * 80: "value"}, {}, ValueError),
    ({"key": "x" * (1024 * 1024)}, {}, ValueError),
    ({"key": "\\" * (600 * 1024)}, {}, ValueError),
    ({"key": "value"}, {"image_format": "jpeg"}, ValueError),
    ({"key": "value"}, {"extra_metadata": {}}, ValueError),
    ({"key": "value"}, {"a1111_parameters": "text"}, ValueError),
])
def test_invalid_text_refuses_before_creating_output_paths(tmp_path, monkeypatch, text, other, error):
    import folder_paths

    output = tmp_path / "not-created"
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(output))

    async def run():
        refs = _sdk.InProcessRefResolver()
        ctx = context()
        with _sdk.bind_runtime(refs, ctx, _sdk.InProcessOps()):
            image = _sdk.ImageRef._wrap(await refs.create("IMAGE", torch.zeros((1, 2, 2, 3))))
            with pytest.raises(error):
                await ctx.output.save_images(image, filenames=["nested/fail.png"], png_text=text, **other)

    asyncio.run(run())
    assert not output.exists()


def test_la_remains_unsupported_for_other_codecs(tmp_path, monkeypatch):
    import folder_paths

    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))

    async def run():
        refs = _sdk.InProcessRefResolver()
        ctx = context()
        with _sdk.bind_runtime(refs, ctx, _sdk.InProcessOps()):
            image = _sdk.ImageRef._wrap(await refs.create("IMAGE", torch.zeros((1, 2, 2, 2))))
            with pytest.raises(ValueError, match="1, 3, or 4 channels"):
                await ctx.output.save_images(image, filenames=["fail.jpg"], image_format="jpeg")

    asyncio.run(run())
    assert not list(tmp_path.iterdir())


def _ordered_png_chunks(data):
    """Read actual uncompressed Pillow tEXt/iTXt chunks without dict collapse."""
    import struct
    import zlib

    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    offset, result = 8, []
    while offset < len(data):
        length = struct.unpack(">I", data[offset:offset + 4])[0]
        kind = data[offset + 4:offset + 8]
        value = data[offset + 8:offset + 8 + length]
        crc = struct.unpack(">I", data[offset + 8 + length:offset + 12 + length])[0]
        assert crc == zlib.crc32(kind + value) & 0xffffffff
        if kind == b"tEXt":
            key, text = value.split(b"\x00", 1)
            result.append((key.decode("latin-1"), text.decode("latin-1")))
        elif kind == b"iTXt":
            key, remaining = value.split(b"\x00", 1)
            assert remaining[:2] == b"\x00\x00"
            language, translated, text = remaining[2:].split(b"\x00", 2)
            assert language == translated == b""
            result.append((key.decode("latin-1"), text.decode("utf-8")))
        offset += length + 12
    assert offset == len(data)
    return result


def test_empty_ordered_png_text_is_valid(tmp_path, monkeypatch):
    import folder_paths

    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))

    async def run():
        refs, ctx = _sdk.InProcessRefResolver(), context()
        with _sdk.bind_runtime(refs, ctx, _sdk.InProcessOps()):
            image = _sdk.ImageRef._wrap(await refs.create("IMAGE", torch.zeros((1, 2, 2, 3))))
            return await ctx.output.save_images(image, filenames=["empty-list.png"], png_text=[])

    assert asyncio.run(run()) == {"images": [{"filename": "empty-list.png", "subfolder": "", "type": "output"}]}
    assert _ordered_png_chunks((tmp_path / "empty-list.png").read_bytes()) == []
    with Image.open(tmp_path / "empty-list.png") as actual:
        assert actual.info == {} and actual.size == (2, 2)


@pytest.mark.parametrize("channels", [2, 3])
@pytest.mark.parametrize("entries", [
    [("parameters", "first"), ("parameters", "second")],
    [["prompt", "first"], ["workflow", "{}"], ["prompt", "last"]],
    [("parameters", "caf\u00e9 \u65e5\u672c\u8a9e"), ["prompt", "one"], ("parameters", "last"), ["prompt", "two"]],
])
def test_ordered_duplicate_png_text_matches_pillow_chunks_and_bytes(tmp_path, monkeypatch, channels, entries):
    import folder_paths

    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    pixels = torch.linspace(0, 1, 3 * 5 * channels).reshape(1, 3, 5, channels)

    async def run():
        refs, ctx = _sdk.InProcessRefResolver(), context()
        with _sdk.bind_runtime(refs, ctx, _sdk.InProcessOps()):
            image = _sdk.ImageRef._wrap(await refs.create("IMAGE", pixels))
            await ctx.output.save_images(image, filenames=["ordered.png"], png_text=entries)

    asyncio.run(run())
    expected = PngInfo()
    for key, value in entries:
        expected.add_text(key, value)
    array = np.clip(255.0 * pixels[0].numpy(), 0, 255).astype(np.uint8)
    Image.fromarray(array).save(tmp_path / "expected.png", pnginfo=expected, compress_level=4)
    actual = (tmp_path / "ordered.png").read_bytes()
    assert actual == (tmp_path / "expected.png").read_bytes()
    assert _ordered_png_chunks(actual) == [tuple(entry) for entry in entries]
    # Independent wrong-result controls exercise chunk order and duplicate loss.
    assert _ordered_png_chunks(actual) != list(dict(entries).items())
    assert _ordered_png_chunks(actual) != [tuple(entry) for entry in reversed(entries)]


@pytest.mark.parametrize("limit", ["entries", "json-bytes"])
def test_ordered_png_text_accepts_exact_limits(tmp_path, monkeypatch, limit):
    import folder_paths

    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    if limit == "entries":
        entries = [("k", "")] * 4096
    else:
        overhead = len(json.dumps([["k", ""]]).encode("utf-8"))
        entries = [["k", "x" * (1024 * 1024 - overhead)]]
        assert len(json.dumps(entries).encode("utf-8")) == 1024 * 1024

    async def run():
        refs, ctx = _sdk.InProcessRefResolver(), context()
        with _sdk.bind_runtime(refs, ctx, _sdk.InProcessOps()):
            image = _sdk.ImageRef._wrap(await refs.create("IMAGE", torch.zeros((1, 1, 1, 3))))
            await ctx.output.save_images(image, filenames=["limit.png"], png_text=entries)

    asyncio.run(run())
    assert _ordered_png_chunks((tmp_path / "limit.png").read_bytes()) == [tuple(entry) for entry in entries]


@pytest.mark.parametrize("entries,error", [
    (["not-a-pair"], TypeError),
    ([["k"]], TypeError),
    ([["k", "v", "extra"]], TypeError),
    ([[True, "v"]], TypeError),
    ([["k", True]], TypeError),
    ([["k", None]], TypeError),
    ([["", "v"]], ValueError),
    ([["bad\x00key", "v"]], ValueError),
    ([["k" * 80, "v"]], ValueError),
    ([["\u03a9", "v"]], UnicodeEncodeError),
    ([("k", "")] * 4097, ValueError),
    ([["k", "x" * (1024 * 1024)]], ValueError),
    ([["k", "\\" * (600 * 1024)]], ValueError),
])
def test_invalid_ordered_png_text_refuses_before_paths(tmp_path, monkeypatch, entries, error):
    import folder_paths

    output = tmp_path / "not-created"
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(output))

    async def run():
        refs, ctx = _sdk.InProcessRefResolver(), context()
        with _sdk.bind_runtime(refs, ctx, _sdk.InProcessOps()):
            image = _sdk.ImageRef._wrap(await refs.create("IMAGE", torch.zeros((1, 1, 1, 3))))
            with pytest.raises(error):
                await ctx.output.save_images(image, filenames=["nested/fail.png"], png_text=entries)

    asyncio.run(run())
    assert not output.exists()


@pytest.mark.parametrize("other", [
    {"image_format": "jpeg"}, {"extra_metadata": {}}, {"a1111_parameters": "text"},
])
def test_ordered_png_text_keeps_format_and_metadata_exclusions(tmp_path, monkeypatch, other):
    import folder_paths

    output = tmp_path / "not-created"
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(output))

    async def run():
        refs, ctx = _sdk.InProcessRefResolver(), context()
        with _sdk.bind_runtime(refs, ctx, _sdk.InProcessOps()):
            image = _sdk.ImageRef._wrap(await refs.create("IMAGE", torch.zeros((1, 1, 1, 3))))
            with pytest.raises(ValueError):
                await ctx.output.save_images(image, filenames=["nested/fail.png"], png_text=[["k", "v"]], **other)

    asyncio.run(run())
    assert not output.exists()
