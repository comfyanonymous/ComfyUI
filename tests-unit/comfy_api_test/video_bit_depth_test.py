import pytest
import torch
import av
import logging
import numpy as np
from fractions import Fraction
from comfy_api.input_impl.video_types import VideoFromFile, VideoFromComponents
from comfy_api.latest._input_impl.video_types import apply_video_input_accepts
from comfy_api.util.video_types import VideoComponents
from comfy_api.latest._util.video_types import VideoBitDepth

DECLARED = {"accepts": {"depth": 10}}


@pytest.fixture(scope="module")
def gradient_components():
    """Narrow horizontal ramp (0.25..0.30) that needs more than 8 bits to stay smooth"""
    width, height, frames = 64, 64, 3
    ramp = torch.linspace(0.25, 0.30, width).view(1, 1, width, 1).expand(frames, height, width, 3)
    return VideoComponents(images=ramp.contiguous(), frame_rate=Fraction(30))


@pytest.fixture(scope="module")
def src8(gradient_components, tmp_path_factory):
    """8-bit h264 mp4 source file"""
    path = str(tmp_path_factory.mktemp("video") / "src8.mp4")
    VideoFromComponents(gradient_components).save_to(path)
    return path


@pytest.fixture(scope="module")
def src10(gradient_components, tmp_path_factory):
    """10-bit h264 mp4 source file"""
    path = str(tmp_path_factory.mktemp("video") / "src10.mp4")
    VideoFromComponents(gradient_components).save_to(path, bit_depth=VideoBitDepth.BIT_10)
    return path


def probe(path):
    """Return (codec, pix_fmt, bit_depth) of the first video stream"""
    with av.open(path) as container:
        stream = container.streams.video[0]
        return (
            stream.codec.name,
            stream.format.name,
            max(component.bits for component in stream.format.components),
        )


def decoded_levels(path):
    """Unique tonal levels in the first decoded frame (banding measure)"""
    with av.open(path) as container:
        frame = next(container.decode(container.streams.video[0]))
        return len(np.unique(frame.to_ndarray(format="gbrpf32le")[..., 0]))


def video_packet_bytes(path):
    """Raw video packet payloads; identical to the source's only for a true remux"""
    with av.open(path) as container:
        return [bytes(packet) for packet in container.demux(container.streams.video[0]) if packet.size]


def test_components_save_bit_depths(src8, src10):
    """Default save stays 8-bit h264; 10-bit keeps h264 and clearly reduces banding"""
    assert probe(src8) == ("h264", "yuv420p", 8)
    assert probe(src10) == ("h264", "yuv420p10le", 10)
    assert decoded_levels(src10) > 2 * decoded_levels(src8)


def test_components_unsupported_codec_raises(gradient_components, tmp_path):
    with pytest.raises(ValueError, match="H264"):
        VideoFromComponents(gradient_components).save_to(str(tmp_path / "x.mp4"), codec="vp9")


def test_bit_depth_enum():
    assert VideoBitDepth.as_input() == ["auto", "8-bit", "10-bit"]
    assert [d.bits() for d in VideoBitDepth] == [None, 8, 10]


def test_10bit_source_remuxes_untouched(src10, tmp_path):
    """auto and a cap of 10 both keep a 10-bit stream untouched"""
    for name, video in [("auto", VideoFromFile(src10)), ("cap10", VideoFromFile(src10).with_bit_depth_cap(10))]:
        path = str(tmp_path / f"{name}.mp4")
        video.save_to(path)
        assert probe(path) == ("h264", "yuv420p10le", 10)
        assert video_packet_bytes(path) == video_packet_bytes(src10)


def test_8bit_source_remuxes_on_8bit_request(src8, tmp_path):
    """Neither explicit 8-bit nor a cap of 8 re-encodes an already 8-bit source"""
    for name, save in [
        ("explicit", lambda p: VideoFromFile(src8).save_to(p, bit_depth="8-bit")),
        ("capped", lambda p: VideoFromFile(src8).with_bit_depth_cap(8).save_to(p)),
    ]:
        path = str(tmp_path / f"{name}.mp4")
        save(path)
        assert video_packet_bytes(path) == video_packet_bytes(src8)


def test_trim_keeps_source_depth(src10, tmp_path):
    """A re-encode forced by trimming preserves the source's 10-bit depth"""
    path = str(tmp_path / "trim.mp4")
    VideoFromFile(src10).as_trimmed(start_time=0, duration=1 / 30, strict_duration=False).save_to(path)
    assert probe(path) == ("h264", "yuv420p10le", 10)


def test_explicit_depth_mismatch_forces_reencode(src8, src10, tmp_path):
    """An explicit depth that differs from the source's re-encodes instead of remuxing"""
    down = str(tmp_path / "down8.mp4")
    VideoFromFile(src10).save_to(down, bit_depth=VideoBitDepth.BIT_8)
    assert probe(down) == ("h264", "yuv420p", 8)

    up = str(tmp_path / "up10.mp4")
    VideoFromFile(src8).save_to(up, bit_depth=VideoBitDepth.BIT_10)
    assert probe(up) == ("h264", "yuv420p10le", 10)


def test_bit_depth_cap(src10, tmp_path):
    """A cap of 8 makes saves default to 8-bit (also through as_trimmed), but an
    explicit request wins, and tensor access keeps full precision"""
    capped = VideoFromFile(src10).with_bit_depth_cap(8)

    path = str(tmp_path / "capped.mp4")
    capped.save_to(path)
    assert probe(path) == ("h264", "yuv420p", 8)

    trimmed = str(tmp_path / "trimmed.mp4")
    capped.as_trimmed(0, 1 / 30, strict_duration=False).save_to(trimmed)
    assert probe(trimmed) == ("h264", "yuv420p", 8)

    explicit = str(tmp_path / "explicit10.mp4")
    capped.save_to(explicit, bit_depth=VideoBitDepth.BIT_10)
    assert probe(explicit) == ("h264", "yuv420p10le", 10)

    images = capped.get_components().images
    assert images.dtype == torch.float32
    assert len(torch.unique(images[0, :, :, 0])) > 30  # ~13 levels if quantized to 8-bit


def test_accepts_binding_policy(gradient_components, src10, tmp_path):
    """Undeclared inputs get an 8-bit-capped copy of file videos; declared inputs
    get uncapped videos; everything else passes through untouched"""
    video = VideoFromFile(src10)

    # undeclared input: capped copy that saves 8-bit
    [capped] = apply_video_input_accepts([video], {"tooltip": "x"})
    assert type(capped) is VideoFromFile and capped is not video
    bound = str(tmp_path / "bound.mp4")
    capped.save_to(bound)
    assert probe(bound) == ("h264", "yuv420p", 8)

    # declared input: original passes through; a cap from an earlier binding is lifted
    assert apply_video_input_accepts([video], DECLARED)[0] is video
    [lifted] = apply_video_input_accepts([capped], DECLARED)
    lifted_path = str(tmp_path / "lifted.mp4")
    lifted.save_to(lifted_path)
    assert probe(lifted_path) == ("h264", "yuv420p10le", 10)

    # declaring depth 8 is the same as not declaring
    assert apply_video_input_accepts([video], {"accepts": {"depth": 8}})[0] is not video

    # subclasses, component videos, custom implementations, and non-videos pass through
    from comfy_api.latest._input import VideoInput as VideoInputABC

    class SubVideo(VideoFromFile):
        pass

    class CustomVideo(VideoInputABC):
        def get_components(self):
            raise NotImplementedError

        def save_to(self, path, format=None, codec=None, metadata=None):
            raise NotImplementedError

        def as_trimmed(self, start_time=None, duration=None, strict_duration=False):
            return self

    passthrough = [SubVideo(src10), VideoFromComponents(gradient_components), CustomVideo(), "not a video", None]
    assert apply_video_input_accepts(passthrough, None) == passthrough


def test_accepts_declaration():
    """Video.Input validates and serializes accepts; SaveVideo and VideoSlice declare it"""
    from comfy_api.latest import io
    import comfy_extras.nodes_video as nv
    from comfy_execution.graph import get_input_info

    assert io.Video.Input("video", accepts={"depth": 10}).as_dict()["accepts"] == {"depth": 10}
    assert "accepts" not in io.Video.Input("video").as_dict()
    with pytest.raises(ValueError, match="Unsupported keys"):
        io.Video.Input("video", accepts={"codec": "h264"})
    with pytest.raises(ValueError, match="must be 8 or 10"):
        io.Video.Input("video", accepts={"depth": 12})

    for node in (nv.SaveVideo, nv.VideoSlice):
        _, _, info = get_input_info(node, "video", node.INPUT_TYPES())
        assert info.get("accepts") == {"depth": 10}, node


def test_save_video_node_bit_depth_handling(tmp_path, monkeypatch, caplog):
    """SaveVideo forwards bit_depth to a source that accepts it (the file is really 10-bit),
    and for a legacy save_to that predates the parameter it warns and saves anyway instead of raising TypeError"""
    import comfy_extras.nodes_video as nv
    from comfy_api.latest._io import HiddenHolder
    from comfy_api.latest._input import VideoInput as VideoInputABC

    monkeypatch.setattr(nv.folder_paths, "get_output_directory", lambda: str(tmp_path))
    monkeypatch.setattr(nv.SaveVideo, "hidden", HiddenHolder.from_dict(None))

    class LegacyVideo(VideoInputABC):
        def get_dimensions(self):
            return 16, 16

        def get_components(self):
            raise NotImplementedError

        def save_to(self, path, format=None, codec=None, metadata=None):  # no bit_depth
            with open(path, "wb") as f:
                f.write(b"data")

        def as_trimmed(self, start_time=None, duration=None, strict_duration=False):
            return self

    # legacy source: an explicit 10-bit request must not crash; it warns and still saves
    with caplog.at_level(logging.WARNING):
        nv.SaveVideo.execute(LegacyVideo(), "legacy", "auto", "auto", bit_depth="10-bit")
    assert "does not support bit_depth" in caplog.text
    assert list(tmp_path.glob("legacy*"))

    # supporting source: bit_depth reaches save_to, so the file really is 10-bit
    ramp = torch.linspace(0.25, 0.30, 64).view(1, 1, 64, 1).expand(3, 64, 64, 3).contiguous()
    nv.SaveVideo.execute(
        VideoFromComponents(VideoComponents(images=ramp, frame_rate=Fraction(30))),
        "supported", "auto", "auto", bit_depth="10-bit",
    )
    outs = list(tmp_path.glob("supported*.mp4"))
    assert len(outs) == 1
    assert probe(str(outs[0])) == ("h264", "yuv420p10le", 10)
