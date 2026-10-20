"""Closed JPEG encoder options against real Pillow; no model or provider fake."""
import asyncio

import numpy as np
import pytest
import torch
from PIL import Image

from comfy_api.latest import _sdk


def test_public_encoder_options_keep_keyword_only_closed_defaults():
    import inspect
    from comfy_api.latest import sdk

    signature = inspect.signature(sdk.OutputDomain.save_images)
    jpeg, png = [signature.parameters[name] for name in ("jpeg_subsampling", "png_text")]
    assert jpeg.kind is png.kind is inspect.Parameter.KEYWORD_ONLY
    assert type(jpeg.default) is int and jpeg.default == 0 and jpeg.annotation in (int, "int")
    assert png.default is None and "list[tuple[str, str] | list[str]]" in str(png.annotation)


def _chroma_checkerboard():
    checker = np.indices((16, 16)).sum(axis=0) % 2
    array = np.where(checker[..., None], [255, 0, 0], [0, 0, 255]).astype(np.uint8)
    return array, torch.from_numpy(array.copy()).to(torch.float32).unsqueeze(0) / 255.0


async def _save(pixels, filename, **options):
    refs = _sdk.InProcessRefResolver()
    ctx = _sdk.InProcessCtxProvider().build(_sdk.ExecutionPlan("encoder", "7", "Save"))
    with _sdk.bind_runtime(refs, ctx, _sdk.InProcessOps()):
        image = _sdk.ImageRef._wrap(await refs.create("IMAGE", pixels))
        try:
            return await ctx.output.save_images(image, filenames=[filename], save_metadata=False, **options)
        finally:
            await refs.release(image)
            assert refs.clear() == 0


@pytest.mark.parametrize("image_format", ["jpeg", "jpg"])
@pytest.mark.parametrize("quality", [45, 92, 100])
@pytest.mark.parametrize("subsampling", [-1, 0, 1, 2])
def test_jpeg_subsampling_matches_actual_pillow(tmp_path, monkeypatch, image_format, quality, subsampling):
    import folder_paths

    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    array, pixels = _chroma_checkerboard()
    asyncio.run(_save(pixels, "actual.jpg", image_format=image_format, quality=quality, optimize=True, jpeg_subsampling=subsampling))
    kwargs = {"quality": quality, "optimize": True}
    if subsampling != -1:
        kwargs["subsampling"] = subsampling
    Image.fromarray(array).save(tmp_path / "expected.jpg", format="JPEG", **kwargs)
    assert (tmp_path / "actual.jpg").read_bytes() == (tmp_path / "expected.jpg").read_bytes()
    with Image.open(tmp_path / "actual.jpg") as actual, Image.open(tmp_path / "expected.jpg") as expected:
        assert actual.mode == expected.mode == "RGB" and actual.size == expected.size == (16, 16)
        assert np.array_equal(np.asarray(actual), np.asarray(expected))
    if subsampling == -1:
        # Ignoring -1 and forcing old0 must fail for this actual chroma fixture.
        Image.fromarray(array).save(tmp_path / "wrong-zero.jpg", format="JPEG", quality=quality, optimize=True, subsampling=0)
        with Image.open(tmp_path / "actual.jpg") as actual, Image.open(tmp_path / "wrong-zero.jpg") as wrong:
            assert not np.array_equal(np.asarray(actual), np.asarray(wrong))


@pytest.mark.parametrize("image_format", ["jpeg", "jpg"])
def test_jpeg_omitted_and_explicit_zero_keep_old_default_bytes(tmp_path, monkeypatch, image_format):
    import folder_paths

    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    array, pixels = _chroma_checkerboard()
    asyncio.run(_save(pixels, "omitted.jpg", image_format=image_format, quality=92))
    asyncio.run(_save(pixels, "zero.jpg", image_format=image_format, quality=92, jpeg_subsampling=0))
    Image.fromarray(array).save(tmp_path / "old-default.jpg", format="JPEG", quality=92, optimize=False, subsampling=0)
    assert (tmp_path / "omitted.jpg").read_bytes() == (tmp_path / "zero.jpg").read_bytes() == (tmp_path / "old-default.jpg").read_bytes()


@pytest.mark.parametrize("subsampling", [-2, 3, True, False, "0", None, 1.0, []])
def test_jpeg_subsampling_rejects_non_enum_exact_integers_before_paths(tmp_path, monkeypatch, subsampling):
    import folder_paths

    output = tmp_path / "not-created"
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(output))
    _, pixels = _chroma_checkerboard()
    with pytest.raises(ValueError, match="JPEG subsampling must be -1, 0, 1, or 2"):
        asyncio.run(_save(pixels, "nested/fail.jpg", image_format="jpeg", jpeg_subsampling=subsampling))
    assert not output.exists()


@pytest.mark.parametrize("image_format", ["png", "webp"])
@pytest.mark.parametrize("subsampling", [-1, 1, 2])
def test_nondefault_jpeg_subsampling_requires_jpeg_before_paths(tmp_path, monkeypatch, image_format, subsampling):
    import folder_paths

    output = tmp_path / "not-created"
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(output))
    _, pixels = _chroma_checkerboard()
    with pytest.raises(ValueError, match="non-default JPEG subsampling requires JPEG output"):
        asyncio.run(_save(pixels, "nested/fail." + image_format, image_format=image_format, jpeg_subsampling=subsampling))
    assert not output.exists()
