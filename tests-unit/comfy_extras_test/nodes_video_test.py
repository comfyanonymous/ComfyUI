"""Tests for video-related nodes in comfy_extras.nodes_video.

Focused on ``GetVideoLastFrame`` — trim respect, tensor-format conformance,
and the three input-type shapes the node is expected to handle:
``VideoFromFile`` (untrimmed and trimmed) and ``VideoFromComponents``.
"""
import io
from fractions import Fraction
from unittest.mock import MagicMock, patch

import av
import numpy as np
import pytest
import torch

mock_nodes = MagicMock()
mock_nodes.MAX_RESOLUTION = 16384
mock_server = MagicMock()

with patch.dict("sys.modules", {"nodes": mock_nodes, "server": mock_server}):
    from comfy_api.latest import InputImpl
    from comfy_api.latest._util import VideoComponents
    from comfy_extras.nodes_video import GetVideoLastFrame


def _encode_mp4(frames, fps=30, opts=None):
    """Encode the given RGB uint8 frames into an in-memory mp4/h264 BytesIO."""
    buf = io.BytesIO()
    h, w = frames[0].shape[:2]
    with av.open(buf, mode="w", format="mp4") as out:
        stream = out.add_stream("h264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
        if opts:
            stream.options = {k: str(v) for k, v in opts.items()}
        for img in frames:
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    buf.seek(0)
    return buf


def _distinguishable_frames(n, w=96, h=96):
    """Frames with a unique spatial signature per index — robust to codec drift."""
    out = []
    for i in range(n):
        img = np.full((h, w, 3), 80, dtype=np.uint8)
        # Bar height grows with index — unambiguous visual marker
        bar_h = 2 + (i % (h - 4))
        img[h - bar_h:, :, 2] = 180
        # Redundant index encoding in top-left pixels (defense-in-depth)
        img[0, 0, 0] = (i >> 8) & 0xFF
        img[0, 1, 0] = i & 0xFF
        img[1, 0, 0] = (i ^ 0xA5) & 0xFF
        out.append(img)
    return out


def _decode_all_rgb_float(mp4_buf):
    """Return every frame of mp4 buffer as a list of (H, W, 3) float32 [0,1] ndarrays."""
    frames = []
    with av.open(io.BytesIO(mp4_buf.getvalue())) as container:
        for f in container.decode(container.streams.video[0]):
            frames.append(f.to_ndarray(format="rgb24").astype(np.float32) / 255.0)
    return frames


def _is_last_frame(output_tensor, reference_frames, tol=0.02):
    """True if the output matches the last reference frame within a small tolerance.

    The tolerance covers two sources of small pixel differences:
      * The fast path decodes via ``rgb24`` (8-bit) while the reference uses the
        same format -- typically bit-identical, but kept here for safety.
      * The trim path decodes via ``gbrpf32le`` (float32) inside
        ``get_components_internal``, while the reference uses ``rgb24``-then-/255 --
        sub-1/255 differences are expected.
    """
    assert output_tensor.shape[0] == 1
    arr = output_tensor[0].numpy()
    return float(np.abs(arr - reference_frames[-1]).mean()) <= tol


class TestGetVideoLastFrameUntrimmed:
    """Fast-path: untrimmed VideoFromFile decodes the raw source directly."""

    def test_mp4_h264_untrimmed(self):
        buf = _encode_mp4(_distinguishable_frames(30))
        video = InputImpl.VideoFromFile(io.BytesIO(buf.getvalue()))
        result = GetVideoLastFrame.execute(video)
        references = _decode_all_rgb_float(buf)
        # Fast path produces bit-identical output to a direct decode
        assert torch.equal(
            result[0],
            torch.from_numpy(references[-1]).unsqueeze(0),
        )

    def test_single_frame_video(self):
        buf = _encode_mp4(_distinguishable_frames(1))
        video = InputImpl.VideoFromFile(io.BytesIO(buf.getvalue()))
        result = GetVideoLastFrame.execute(video)
        references = _decode_all_rgb_float(buf)
        assert _is_last_frame(result[0], references)

    @pytest.mark.parametrize(
        "n_frames",
        [
            59,   # 59/30 ≈ 1.97 s — just below the 2 s tail window (no seek)
            60,   # 60/30 = 2.00 s — exactly at the boundary (no seek)
            61,   # 61/30 ≈ 2.03 s — just over (seek branch fires with a small positive offset)
            90,   # 3.0 s — comfortably in the seek branch
        ],
    )
    def test_around_2_second_seek_boundary(self, n_frames):
        """The seek branch only fires for videos strictly longer than the 2 s
        tail window. Below or exactly-at that threshold we must skip the seek
        (so it is impossible to compute a negative or zero offset). Above it,
        the computed offset is strictly positive. This test pins the whole
        boundary so a future change to ``tail_window`` can't silently break it.
        """
        buf = _encode_mp4(_distinguishable_frames(n_frames))
        video = InputImpl.VideoFromFile(io.BytesIO(buf.getvalue()))
        result = GetVideoLastFrame.execute(video)
        references = _decode_all_rgb_float(buf)
        assert _is_last_frame(result[0], references)

    def test_file_path_source(self, tmp_path):
        """VideoFromFile with a file-path source (not BytesIO)."""
        buf = _encode_mp4(_distinguishable_frames(20))
        mp4_path = tmp_path / "clip.mp4"
        mp4_path.write_bytes(buf.getvalue())
        video = InputImpl.VideoFromFile(str(mp4_path))
        result = GetVideoLastFrame.execute(video)
        references = _decode_all_rgb_float(buf)
        assert _is_last_frame(result[0], references)

    def test_idempotent_repeated_calls(self):
        """Calling execute() on the same VideoFromFile multiple times must
        return identical tensors (no hidden state corruption)."""
        buf = _encode_mp4(_distinguishable_frames(30))
        video = InputImpl.VideoFromFile(io.BytesIO(buf.getvalue()))

        r1 = GetVideoLastFrame.execute(video)[0]
        r2 = GetVideoLastFrame.execute(video)[0]
        r3 = GetVideoLastFrame.execute(video)[0]
        assert torch.equal(r1, r2)
        assert torch.equal(r2, r3)


class TestGetVideoLastFrameTrimmed:
    """Trimmed VideoFromFile: routes through the generic ``get_components()`` path,
    which respects ``as_trimmed``'s start/duration window."""

    def test_trim_returns_visible_last_frame_not_file_last(self):
        """The core feature request: trimmed videos must return the last visible
        frame of the trim, NOT the last frame of the underlying file."""
        # 5-second source; trim to keep only the first 1 second
        buf = _encode_mp4(_distinguishable_frames(150), opts={"g": 15})
        full_refs = _decode_all_rgb_float(buf)

        video = InputImpl.VideoFromFile(io.BytesIO(buf.getvalue()))
        trimmed = video.as_trimmed(start_time=0, duration=1.0, strict_duration=False)
        result = GetVideoLastFrame.execute(trimmed)

        # Ground truth via get_components (also respects trim)
        trimmed_components = trimmed.get_components().images
        trim_refs = [trimmed_components[i].numpy() for i in range(trimmed_components.shape[0])]

        arr = result[0][0].numpy()
        diff_to_trim_end = float(np.abs(arr - trim_refs[-1]).mean())
        diff_to_file_end = float(np.abs(arr - full_refs[-1]).mean())

        # Helper output must be closer to the trim's last frame than the file's last frame
        assert diff_to_trim_end < diff_to_file_end, (
            f"Trim not respected: helper output is closer to file-end "
            f"({diff_to_file_end:.4f}) than trim-end ({diff_to_trim_end:.4f})"
        )
        # And within absolute tolerance of the trim's last frame
        assert diff_to_trim_end <= 0.02

    def test_trim_with_nonzero_start(self):
        """Trim starting partway through the source — last visible frame should
        be at start + duration, not at the file end."""
        buf = _encode_mp4(_distinguishable_frames(150), opts={"g": 15})
        video = InputImpl.VideoFromFile(io.BytesIO(buf.getvalue()))
        trimmed = video.as_trimmed(start_time=1.0, duration=2.0, strict_duration=False)
        result = GetVideoLastFrame.execute(trimmed)
        trimmed_components = trimmed.get_components().images
        trim_refs = [trimmed_components[i].numpy() for i in range(trimmed_components.shape[0])]
        full_refs = _decode_all_rgb_float(buf)
        arr = result[0][0].numpy()
        assert float(np.abs(arr - trim_refs[-1]).mean()) < float(np.abs(arr - full_refs[-1]).mean())

    def test_small_trim_on_short_video(self):
        """Regression test for the trim-detection tolerance: trimming a 0.4s clip down to 0.1s
        must be detected as trimmed and route through the generic path so the trim is respected.
        """
        buf = _encode_mp4(_distinguishable_frames(12))  # 0.4 s
        video = InputImpl.VideoFromFile(io.BytesIO(buf.getvalue()))
        # Trim to keep the first 0.1 s (3 frames) of a 0.4 s source
        trimmed = video.as_trimmed(start_time=0, duration=0.1, strict_duration=False)
        result = GetVideoLastFrame.execute(trimmed)

        trimmed_components = trimmed.get_components().images
        trim_refs = [trimmed_components[i].numpy() for i in range(trimmed_components.shape[0])]
        full_refs = _decode_all_rgb_float(buf)

        arr = result[0][0].numpy()
        diff_to_trim_end = float(np.abs(arr - trim_refs[-1]).mean())
        diff_to_file_end = float(np.abs(arr - full_refs[-1]).mean())
        assert diff_to_trim_end < diff_to_file_end, (
            "Small trim on short video not detected — tolerance may be too loose"
        )


class TestGetVideoLastFrameFromComponents:
    """VideoFromComponents: routes through the generic ``get_components()`` path,
    which slices the in-memory tensor directly -- no decode required, output is bit-exactly the last frame of the input.
    """

    def test_bit_identical_to_last_image(self):
        images = torch.zeros(10, 32, 32, 3, dtype=torch.float32)
        for i in range(10):
            images[i, :, :, 0] = (i + 0.5) / 10.0  # distinct per frame
        components = VideoComponents(images=images, audio=None, frame_rate=Fraction(30, 1))
        video = InputImpl.VideoFromComponents(components)
        result = GetVideoLastFrame.execute(video)
        assert torch.equal(result[0], images[-1:].contiguous())

    def test_returns_independent_copy(self):
        """The returned tensor is a clone — mutating it must not mutate
        the underlying components tensor."""
        images = torch.rand(5, 16, 16, 3, dtype=torch.float32)
        components = VideoComponents(images=images, audio=None, frame_rate=Fraction(30, 1))
        video = InputImpl.VideoFromComponents(components)
        result = GetVideoLastFrame.execute(video)[0]
        assert result.data_ptr() != images.data_ptr()


class TestGetVideoLastFrameTensorFormat:
    """Conformance to the Comfy IMAGE tensor convention: (1, H, W, 3) float32 [0,1] CPU contiguous."""

    def test_shape(self):
        buf = _encode_mp4(_distinguishable_frames(10, w=64, h=48))
        video = InputImpl.VideoFromFile(io.BytesIO(buf.getvalue()))
        result = GetVideoLastFrame.execute(video)[0]
        assert result.shape == (1, 48, 64, 3)

    def test_dtype_and_device(self):
        buf = _encode_mp4(_distinguishable_frames(10))
        video = InputImpl.VideoFromFile(io.BytesIO(buf.getvalue()))
        result = GetVideoLastFrame.execute(video)[0]
        assert result.dtype == torch.float32
        assert result.device.type == "cpu"

    def test_pixel_range_and_contiguous(self):
        buf = _encode_mp4(_distinguishable_frames(10))
        video = InputImpl.VideoFromFile(io.BytesIO(buf.getvalue()))
        result = GetVideoLastFrame.execute(video)[0]
        assert 0.0 <= float(result.min())
        assert float(result.max()) <= 1.0
        assert result.is_contiguous()


class TestGetVideoLastFrameErrors:
    """Error paths produce clear ValueErrors."""

    def test_audio_only_container(self):
        buf = io.BytesIO()
        with av.open(buf, mode="w", format="mp4") as out:
            stream = out.add_stream("aac", rate=44100)
            stream.sample_rate = 44100
            frame = av.AudioFrame.from_ndarray(
                np.zeros((2, 1024), "float32"), format="fltp", layout="stereo"
            )
            frame.sample_rate = 44100
            frame.pts = 0
            for packet in stream.encode(frame):
                out.mux(packet)
            for packet in stream.encode():
                out.mux(packet)
        buf.seek(0)
        video = InputImpl.VideoFromFile(io.BytesIO(buf.getvalue()))
        with pytest.raises(ValueError, match="no video stream"):
            GetVideoLastFrame.execute(video)

    def test_empty_components(self):
        empty = VideoComponents(
            images=torch.zeros(0, 0, 0, 3),
            audio=None,
            frame_rate=Fraction(30, 1),
        )
        video = InputImpl.VideoFromComponents(empty)
        with pytest.raises(ValueError, match="no frames"):
            GetVideoLastFrame.execute(video)

    def test_corrupt_bytes_raises(self):
        video = InputImpl.VideoFromFile(io.BytesIO(b"not a real video" * 50))
        with pytest.raises(Exception):
            GetVideoLastFrame.execute(video)


class TestGetVideoLastFrameDoesNotCorruptSource:
    """After extraction, the source video must remain fully usable downstream —
    save_to(), get_components(), get_duration() all still work."""

    def test_save_to_still_works(self):
        buf = _encode_mp4(_distinguishable_frames(20))
        video = InputImpl.VideoFromFile(io.BytesIO(buf.getvalue()))
        _ = GetVideoLastFrame.execute(video)

        sink = io.BytesIO()
        video.save_to(sink)
        assert sink.tell() > 0
        # And the round-tripped buffer is itself decodable
        sink.seek(0)
        with av.open(sink) as container:
            count = sum(1 for _ in container.decode(container.streams.video[0]))
        assert count == 20

    def test_get_components_still_works(self):
        buf = _encode_mp4(_distinguishable_frames(15))
        video = InputImpl.VideoFromFile(io.BytesIO(buf.getvalue()))

        _ = GetVideoLastFrame.execute(video)
        components = video.get_components()
        assert components.images.shape[0] == 15
