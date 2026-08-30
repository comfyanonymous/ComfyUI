import asyncio
import os
import tempfile

import av
import numpy as np
import pytest

from comfy_api.latest import io
from comfy_api.latest._input_impl.video_types import VideoConcatenated, VideoFromFile
from comfy_extras.nodes_video import VideoConcat, VideoExtension


def create_h264_video(width=64, height=64, frames=30, fps=30):
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    with av.open(tmp.name, mode="w") as container:
        stream = container.add_stream("h264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for i in range(frames):
            frame = av.VideoFrame.from_ndarray(np.full((height, width, 3), (i * 7) % 256, dtype=np.uint8), format="rgb24")
            container.mux(stream.encode(frame.reformat(format="yuv420p")))
        container.mux(stream.encode(None))
    return tmp.name


@pytest.fixture
def two_clips():
    a = create_h264_video(frames=30)
    b = create_h264_video(frames=45)
    yield VideoFromFile(a), VideoFromFile(b)
    os.unlink(a)
    os.unlink(b)


def test_video_concat_schema():
    schema = VideoConcat.define_schema()
    assert schema.node_id == "VideoConcat"
    assert schema.essentials_category == "Video Tools"
    assert "combine videos" in schema.search_aliases
    videos = schema.inputs[0]
    assert isinstance(videos, io.Autogrow.Input)
    assert videos.id == "videos"
    assert videos.template.prefix == "video"
    assert videos.template.min == 2
    assert videos.template.names[:2] == ["video0", "video1"]
    resize = next(i for i in schema.inputs if i.id == "resize")
    assert resize.options == ["fit", "error"]
    assert resize.default == "fit"
    assert isinstance(schema.outputs[0], io.Video.Output)


def test_video_concat_registered():
    nodes = asyncio.run(VideoExtension().get_node_list())
    assert VideoConcat in nodes


def test_video_concat_execute_in_slot_order(two_clips):
    a, b = two_clips
    result = VideoConcat.execute({"video0": a, "video1": b}, "fit").args[0]
    assert isinstance(result, VideoConcatenated)
    assert result._sources == [a, b]
    assert result.get_duration() == pytest.approx(2.5)
    assert result.get_frame_count() == 75


def test_video_concat_execute_skips_empty_slots(two_clips):
    a, b = two_clips
    result = VideoConcat.execute({"video0": a, "video1": None, "video2": b}, "fit").args[0]
    assert result._sources == [a, b]


def test_video_concat_single_video_passthrough(two_clips):
    a, _ = two_clips
    assert VideoConcat.execute({"video0": a}, "fit").args[0] is a


def test_video_concat_no_videos_raises():
    with pytest.raises(ValueError):
        VideoConcat.execute({}, "fit")


def test_video_concat_resize_error_rejects_mismatched_resolution(two_clips):
    a, _ = two_clips
    small = create_h264_video(width=32, height=64, frames=5)
    try:
        with pytest.raises(ValueError, match="32x64"):
            VideoConcat.execute({"video0": a, "video1": VideoFromFile(small)}, "error")
        assert isinstance(VideoConcat.execute({"video0": a, "video1": VideoFromFile(small)}, "fit").args[0], VideoConcatenated)
    finally:
        os.unlink(small)
