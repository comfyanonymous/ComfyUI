import asyncio
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import comfy_extras.image_formats
from app.assets import previews


@pytest.mark.parametrize(("limit", "skipped"), [(576, False), (575, True)])
def test_upload_decodes_are_bounded(tmp_path, limit, skipped):
    """A decode holds every channel of every part, and running out of memory can't be caught."""
    import OpenEXR

    def part(name, width, height, channels):
        header = {"type": OpenEXR.scanlineimage, "displayWindow": ((0, 0), (15, 7))}
        return OpenEXR.Part(header, {c: np.zeros((height, width), np.float32) for c in channels}, name=name)

    # 8x4 with 6 channels plus 16x8 with 3: 576 samples, counted from the headers before decoding.
    path = str(tmp_path / "layers.exr")
    OpenEXR.File([part("beauty", 8, 4, ["R", "G", "B", "A", "depth.Z", "N.X"]), part("aov", 16, 8, "RGB")]).write(path)
    with (
        patch.object(previews, "PREVIEW_MAX_SOURCE_SAMPLES", limit),
        patch.dict(comfy_extras.image_formats.LOADERS, {".exr": MagicMock(side_effect=RuntimeError("decoded"))}),
    ):
        decode = comfy_extras.image_formats.LOADERS[".exr"]
        with pytest.raises(previews.PreviewSkipped if skipped else RuntimeError):
            previews._decode_for_preview(path)
        assert decode.called is not skipped

    # At most two decode at once, and the rest still get their preview.
    running, peak, lock = 0, 0, threading.Lock()

    def make(path):
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        time.sleep(0.05)
        with lock:
            running -= 1
        return b"webp", 2, 2

    async def uploads():
        # A semaphore made inside this loop, at the module's limit, so the module's stays unbound.
        with patch.object(previews, "_DECODE_SLOTS", asyncio.Semaphore(previews._DECODE_SLOTS._value)):
            await asyncio.gather(*(previews.generate_upload_preview(f"id{i}", str(tmp_path / f"{i}.exr"), None) for i in range(6)))

    with patch.object(previews, "_make_preview", make), patch.object(previews, "_store_and_link", return_value="linked") as store:
        asyncio.run(uploads())
    assert peak == 2 and store.call_count == 6


def test_a_piz_float32_exr_previews_its_pixels(tmp_path):
    """FFmpeg decodes some PIZ float32 EXRs to all zeros; OpenEXR decodes them."""
    import OpenEXR

    path = str(tmp_path / "piz.exr")
    rng = np.random.default_rng(0)
    # Fine noise around 0.5 at this width is one FFmpeg decodes to zeros.
    channels = {c: (0.5 + rng.normal(0, 0.001, (32, 512))).astype(np.float32) for c in "RGB"}
    OpenEXR.File({"compression": OpenEXR.PIZ_COMPRESSION, "type": OpenEXR.scanlineimage}, channels).write(path)

    preview = np.asarray(previews._decode_for_preview(path))
    assert preview.shape == (32, 512, 3)
    assert abs(int(preview.mean()) - 188) <= 1, "0.5 linear is 188 in sRGB"


def test_without_openexr_an_upload_gets_no_preview(tmp_path, caplog):
    # The real emit, which under pytest raises on a reason outside the event vocabulary.
    with patch.dict(comfy_extras.image_formats.LOADERS, clear=True), caplog.at_level("INFO"):
        assert asyncio.run(previews.generate_upload_preview("id", str(tmp_path / "a.exr"), None)) is None
    assert "previews.generation_failed" in caplog.text and "reason=decoder_unavailable" in caplog.text
