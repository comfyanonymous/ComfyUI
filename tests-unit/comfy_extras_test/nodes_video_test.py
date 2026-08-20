from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from comfy_api.latest import Types
from comfy_extras import nodes_video


@pytest.fixture
def save_video(monkeypatch, tmp_path):
    monkeypatch.setattr(nodes_video.folder_paths, "get_output_directory", lambda: str(tmp_path))
    monkeypatch.setattr(
        nodes_video.folder_paths,
        "get_save_image_path",
        lambda *args: (str(tmp_path), "output", 1, "", args[0]),
    )
    monkeypatch.setattr(nodes_video.SaveVideo, "hidden", SimpleNamespace(prompt=None, extra_pnginfo=None), raising=False)
    video = Mock()
    video.get_dimensions.return_value = (64, 64)
    return video


def test_save_video_accepts_legacy_codec_string(save_video):
    nodes_video.SaveVideo.execute(save_video, "video", "mp4", "h264")

    kwargs = save_video.save_to.call_args.kwargs
    assert kwargs["format"] == Types.VideoContainer.MP4
    assert kwargs["codec"] == Types.VideoCodec.H264
    assert "color_space" not in kwargs


def test_save_video_forwards_nested_encoding_options(save_video):
    nodes_video.SaveVideo.execute(
        save_video,
        "video",
        {
            "format": "webm",
            "codec": {
                "codec": "av1",
                "encoding": {"crf": 30, "color_space": "HDR"},
            },
        },
    )

    kwargs = save_video.save_to.call_args.kwargs
    assert kwargs["format"] == Types.VideoContainer.WEBM
    assert kwargs["codec"] == Types.VideoCodec.AV1
    assert kwargs["crf"] == 30
    assert kwargs["color_space"] == "HDR"
