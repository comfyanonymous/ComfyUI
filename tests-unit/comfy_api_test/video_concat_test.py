import io
import json
import os
import tempfile
from fractions import Fraction

import av
import numpy as np
import pytest
import torch
from av.video.reformatter import ColorRange, ColorTrc

# comfy_api.input_impl must come before comfy_api.input (circular import otherwise)
from comfy_api.input_impl.video_types import VideoConcatenated, VideoFromComponents, VideoFromFile
from comfy_api.util.video_types import VideoCodec, VideoComponents, VideoContainer
from comfy_api.input.basic_types import AudioInput
from comfy_api_test.video_types_test import create_hdr_av1_video, create_transcode_source, transcode_and_probe
from comfy_api.latest._input_impl.video_types import _match_audio_channels


def create_av_source(
    width=64, height=64, frames=30, fps=30, sample_rate=44100, channels=1, tone_hz=None,
    container_format="mp4", audio=True, level=None, metadata=None, audio_seconds=None,
):
    """h264 video plus (optionally) an AAC track carrying a sine tone or silence.
    ``audio_seconds`` makes the track shorter/longer than the video (default: same length)."""
    tmp = tempfile.NamedTemporaryFile(suffix=f".{container_format}", delete=False)
    tmp.close()
    layout = {1: "mono", 2: "stereo"}[channels]
    with av.open(tmp.name, mode="w") as container:
        if metadata:
            for key, value in metadata.items():
                container.metadata[key] = value
        video_stream = container.add_stream("h264", rate=fps)
        video_stream.width = width
        video_stream.height = height
        video_stream.pix_fmt = "yuv420p"
        audio_stream = container.add_stream("aac", rate=sample_rate, layout=layout) if audio else None
        for i in range(frames):
            value = (i * 7) % 256 if level is None else level
            frame = av.VideoFrame.from_ndarray(np.full((height, width, 3), value, dtype=np.uint8), format="rgb24")
            container.mux(video_stream.encode(frame.reformat(format="yuv420p")))
        if audio_stream is not None:
            total = sample_rate * frames // fps if audio_seconds is None else int(sample_rate * audio_seconds)
            t = np.arange(total, dtype=np.float32) / sample_rate
            signal = 0.5 * np.sin(2 * np.pi * tone_hz * t) if tone_hz else np.zeros(total, dtype=np.float32)
            data = np.tile(signal.astype(np.float32), (channels, 1))
            for offset in range(0, total, 1024):
                audio_frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(data[:, offset:offset + 1024]), format="fltp", layout=layout)
                audio_frame.sample_rate = sample_rate
                audio_frame.pts = offset
                container.mux(audio_stream.encode(audio_frame))
        for stream in (video_stream, audio_stream):
            if stream is not None:
                container.mux(stream.encode(None))
    return tmp.name


@pytest.fixture
def cleanup():
    paths = []
    yield paths
    for path in paths:
        os.unlink(path)


def save_mp4(video, **kwargs):
    buffer = io.BytesIO()
    video.save_to(buffer, format=kwargs.pop("format", VideoContainer.MP4), codec=kwargs.pop("codec", VideoCodec.H264), **kwargs)
    buffer.seek(0)
    return buffer


def video_packets(buffer):
    buffer.seek(0)
    with av.open(buffer) as container:
        stream = container.streams.video[0]
        return stream.time_base, [(p.pts, p.dts, p.duration) for p in container.demux(stream) if p.pts is not None]


def decode_audio(buffer):
    buffer.seek(0)
    with av.open(buffer) as container:
        stream = container.streams.audio[0]
        chunks = [frame.to_ndarray() for frame in container.decode(stream)]
        return stream.rate, stream.layout.name, np.concatenate(chunks, axis=1)


def loud_windows(samples, rate, window=0.01, threshold=0.1):
    """Start times of ``window``-second slices whose RMS exceeds ``threshold``."""
    size = int(rate * window)
    mono = samples[0]
    starts = []
    for offset in range(0, mono.shape[0] - size, size):
        if np.sqrt(np.mean(mono[offset:offset + size] ** 2)) > threshold:
            starts.append(offset / rate)
    return starts


def decode_frame(buffer, index):
    buffer.seek(0)
    with av.open(buffer) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i == index:
                return frame.to_ndarray(format="rgb24")
    raise IndexError(index)


def test_concat_sums_duration_frames_and_audio(cleanup):
    a, b = create_av_source(frames=30), create_av_source(frames=45)
    cleanup += [a, b]
    video = VideoConcatenated([VideoFromFile(a), VideoFromFile(b)])
    assert video.get_duration() == pytest.approx(2.5, abs=0.01)
    assert video.get_frame_count() == 75
    assert video.get_dimensions() == (64, 64)
    assert video.get_frame_rate() == Fraction(30)
    assert video.get_active_trim_window() == (0.0, 0.0)

    result = transcode_and_probe(video)
    assert result["codec"] == "h264"
    assert result["frames"] == 75
    assert result["first_pts"] == 0
    assert result["video_seconds"] == pytest.approx(2.5, abs=0.01)
    assert result["audio_seconds"] == pytest.approx(2.5, abs=0.05)
    assert result["audio_codecs"] == ["aac"]


def test_concat_pts_strictly_increasing_and_audio_contiguous(cleanup):
    a, b = create_av_source(frames=30), create_av_source(frames=45)
    cleanup += [a, b]
    buffer = save_mp4(VideoConcatenated([VideoFromFile(a), VideoFromFile(b)]))
    _, packets = video_packets(buffer)
    pts = [p[0] for p in packets]
    dts = [p[1] for p in packets]
    assert len(pts) == 75
    assert all(later > earlier for earlier, later in zip(pts, pts[1:]))
    assert all(later > earlier for earlier, later in zip(dts, dts[1:]))
    rate, _, samples = decode_audio(buffer)
    assert rate == 44100
    assert samples.shape[1] == pytest.approx(2.5 * 44100, abs=2048)


def test_concat_audio_stays_aligned_across_boundaries(cleanup):
    """tone / silence / tone: the second tone must start exactly where the third clip starts.
    The middle clip's audio is shorter than its video, so merely appending tracks would put the
    second tone at 1.6 s instead of 2.0 s."""
    a = create_av_source(frames=30, tone_hz=440)
    b = create_av_source(frames=30, tone_hz=None, audio_seconds=0.6)
    c = create_av_source(frames=30, tone_hz=440)
    cleanup += [a, b, c]
    buffer = save_mp4(VideoConcatenated([VideoFromFile(a), VideoFromFile(b), VideoFromFile(c)]))
    rate, _, samples = decode_audio(buffer)
    loud = loud_windows(samples, rate)
    assert loud, "no tone decoded"
    first_end = max(t for t in loud if t < 1.5)
    second_start = min(t for t in loud if t > 1.5)
    assert first_end == pytest.approx(1.0, abs=0.06)
    assert second_start == pytest.approx(2.0, abs=0.06)
    assert samples.shape[1] == pytest.approx(3.0 * rate, abs=2048)


def test_concat_video_only_then_audio_clip_leads_with_silence(cleanup):
    a = create_av_source(frames=30, audio=False)
    b = create_av_source(frames=45, tone_hz=440)
    cleanup += [a, b]
    buffer = save_mp4(VideoConcatenated([VideoFromFile(a), VideoFromFile(b)]))
    result = transcode_and_probe(VideoConcatenated([VideoFromFile(a), VideoFromFile(b)]))
    assert result["audio_codecs"] == ["aac"]
    assert result["audio_seconds"] == pytest.approx(result["video_seconds"], abs=0.05)
    rate, _, samples = decode_audio(buffer)
    assert np.abs(samples[:, :int(0.9 * rate)]).max() < 0.01
    assert min(loud_windows(samples, rate)) == pytest.approx(1.0, abs=0.06)


def test_concat_audio_clip_then_video_only_pads_trailing_silence(cleanup):
    a = create_av_source(frames=30, tone_hz=440)
    b = create_av_source(frames=45, audio=False)
    cleanup += [a, b]
    buffer = save_mp4(VideoConcatenated([VideoFromFile(a), VideoFromFile(b)]))
    result = transcode_and_probe(VideoConcatenated([VideoFromFile(a), VideoFromFile(b)]))
    assert result["video_seconds"] == pytest.approx(2.5, abs=0.01)
    assert result["audio_seconds"] == pytest.approx(2.5, abs=0.05)
    rate, _, samples = decode_audio(buffer)
    assert samples.shape[1] == pytest.approx(2.5 * rate, abs=2048)
    assert np.abs(samples[:, int(1.1 * rate):]).max() < 0.01


def test_concat_no_audio_anywhere(cleanup):
    a, b = create_av_source(frames=10, audio=False), create_av_source(frames=10, audio=False)
    cleanup += [a, b]
    result = transcode_and_probe(VideoConcatenated([VideoFromFile(a), VideoFromFile(b)]))
    assert result["frames"] == 20
    assert result["audio_codecs"] == []
    assert result["audio_seconds"] is None


def test_concat_trimmed_sources(cleanup):
    a, b = create_av_source(frames=30), create_av_source(frames=45)
    cleanup += [a, b]
    video = VideoConcatenated([
        VideoFromFile(a).as_trimmed(0.5, 0.5),
        VideoFromFile(b, start_time=0.2, duration=0.3),
    ])
    assert video.get_duration() == pytest.approx(0.8, abs=0.01)
    result = transcode_and_probe(video)
    assert result["frames"] == pytest.approx(24, abs=2)
    assert result["first_pts"] == 0
    assert result["video_seconds"] == pytest.approx(0.8, abs=0.05)
    assert result["audio_seconds"] == pytest.approx(0.8, abs=0.05)


def test_concat_mixed_time_bases_and_frame_rates(cleanup):
    """mp4 (1/15360) after mkv (1/1000): every frame is rescaled into one output time base and
    stamped with it, so mixed frame rates keep their true timing (PyAV rescales frame.pts by
    frame.time_base on encode; a stale time base would blow the timeline up by orders of magnitude)."""
    a = create_av_source(frames=30, fps=30, audio=False)
    b = create_av_source(frames=15, fps=15, audio=False, container_format="mkv")
    cleanup += [a, b]
    with av.open(a) as ca, av.open(b) as cb:
        assert ca.streams.video[0].time_base != cb.streams.video[0].time_base

    video = VideoConcatenated([VideoFromFile(a), VideoFromFile(b)])
    buffer = save_mp4(video)
    time_base, packets = video_packets(buffer)
    assert time_base == Fraction(1, 360000)
    pts = [p[0] for p in packets]
    assert len(pts) == 45
    assert all(later > earlier for earlier, later in zip(pts, pts[1:]))
    assert pts[:30] == [i * 12000 for i in range(30)]
    assert pts[30] == 360000  # second clip starts exactly where the first ends
    for i, value in enumerate(pts[30:]):
        assert value == pytest.approx(360000 + i * 24000, abs=400)  # mkv stores ms, so +-1 ms
    buffer.seek(0)
    with av.open(buffer) as container:
        stream = container.streams.video[0]
        assert float(stream.duration * stream.time_base) == pytest.approx(2.0, abs=0.005)


def test_concat_resize_error_rejects_mismatched_resolution(cleanup):
    a, b = create_av_source(frames=5), create_av_source(width=32, height=64, frames=5)
    cleanup += [a, b]
    with pytest.raises(ValueError, match="video 1 is 32x64; expected 64x64"):
        VideoConcatenated([VideoFromFile(a), VideoFromFile(b)], resize="error")
    with pytest.raises(ValueError):
        VideoConcatenated([VideoFromFile(a), VideoFromFile(b)], resize="stretch")


def test_concat_fit_letterboxes_into_first_clip(cleanup):
    a = create_av_source(frames=30, level=200, audio=False)
    b = create_av_source(width=32, height=64, frames=10, level=200, audio=False)
    cleanup += [a, b]
    video = VideoConcatenated([VideoFromFile(a), VideoFromFile(b)])
    assert video.get_dimensions() == (64, 64)
    buffer = save_mp4(video)
    buffer.seek(0)
    with av.open(buffer) as container:
        stream = container.streams.video[0]
        assert (stream.codec_context.width, stream.codec_context.height) == (64, 64)
    assert transcode_and_probe(video)["frames"] == 40
    frame = decode_frame(buffer, 35)
    assert frame.shape == (64, 64, 3)
    assert frame[:, :16].mean() < 20 and frame[:, 48:].mean() < 20  # pillarbox
    assert frame[:, 16:48].mean() > 150


def test_concat_mismatched_audio_params_conform_to_best_source(cleanup):
    """44.1 kHz mono followed by 48 kHz stereo: nothing is degraded, the mono clip is upmixed/resampled."""
    a = create_av_source(frames=30, sample_rate=44100, channels=1, tone_hz=440)
    b = create_av_source(frames=45, sample_rate=48000, channels=2, tone_hz=440)
    cleanup += [a, b]
    buffer = save_mp4(VideoConcatenated([VideoFromFile(a), VideoFromFile(b)]))
    rate, layout, samples = decode_audio(buffer)
    assert rate == 48000
    assert layout == "stereo"
    assert samples.shape[1] == pytest.approx(2.5 * 48000, abs=2048)
    loud = loud_windows(samples, rate)
    assert min(loud) < 0.1 and max(loud) > 2.3  # both clips' tones are present
    # reversed order gives the same output format
    buffer = save_mp4(VideoConcatenated([VideoFromFile(b), VideoFromFile(a)]))
    rate, layout, _ = decode_audio(buffer)
    assert (rate, layout) == (48000, "stereo")


def test_concat_bit_depth_is_max_of_sources(cleanup, tmp_path):
    ramp = torch.linspace(0.25, 0.30, 64).view(1, 1, 64, 1).expand(6, 64, 64, 3).contiguous()
    src8 = str(tmp_path / "src8.mp4")
    src10 = str(tmp_path / "src10.mp4")
    VideoFromComponents(VideoComponents(images=ramp, frame_rate=Fraction(30))).save_to(src8)
    VideoFromComponents(VideoComponents(images=ramp, frame_rate=Fraction(30)), bit_depth=10).save_to(src10)
    video = VideoConcatenated([VideoFromFile(src8), VideoFromFile(src10)])
    assert video.get_bit_depth() == 10
    buffer = save_mp4(video)
    buffer.seek(0)
    with av.open(buffer) as container:
        stream = container.streams.video[0]
        assert max(component.bits for component in stream.format.components) == 10
    assert transcode_and_probe(video)["frames"] == 12


def test_concat_color_space_mismatch_raises(cleanup, tmp_path):
    hdr = str(tmp_path / "hdr.mkv")
    create_hdr_av1_video(hdr, ColorTrc.SMPTE2084, ColorRange.MPEG)
    sdr = create_av_source(frames=3, audio=False)
    cleanup.append(sdr)
    assert VideoFromFile(hdr).get_color_space() == "HDR PQ"
    with pytest.raises(ValueError, match="color space"):
        VideoConcatenated([VideoFromFile(sdr), VideoFromFile(hdr)])


def test_concat_get_components(cleanup):
    a = create_av_source(frames=30, tone_hz=440)
    b = create_av_source(width=32, height=64, frames=15, fps=15, audio=False)
    cleanup += [a, b]
    video = VideoConcatenated([VideoFromFile(a), VideoFromFile(b)])
    components = video.get_components()
    assert components.frame_rate == Fraction(30)
    # the 15 fps clip is retimed to 30 fps (each frame repeated), so the 2 s duration is kept
    assert components.images.shape == (60, 64, 64, 3)
    assert video.get_frame_count() == 60  # the same constant-frame-rate view
    assert torch.equal(components.images[30], components.images[31])
    assert video.get_duration() == pytest.approx(components.images.shape[0] / 30, abs=0.01)
    # the fitted clip is pillarboxed: black columns on both sides
    assert components.images[40, :, :16].max() < 0.05
    source_audio = VideoFromFile(a).get_components().audio
    assert components.audio["sample_rate"] == source_audio["sample_rate"]
    assert components.audio["waveform"].shape[1] == source_audio["waveform"].shape[1]
    assert components.audio["waveform"].shape[2] == round(2.0 * source_audio["sample_rate"])
    assert components.audio["waveform"][..., int(1.1 * source_audio["sample_rate"]):].abs().max() == 0


def test_concat_get_components_keeps_sub_frame_clips(cleanup):
    """A clip shorter than one frame at the first clip's rate still contributes a frame."""
    slow = create_av_source(frames=2, fps=1, audio=False)
    short = create_av_source(frames=5, fps=30, audio=False)
    cleanup += [slow, short]
    video = VideoConcatenated([VideoFromFile(slow), VideoFromFile(short)])
    assert video.get_components().images.shape[0] == 3
    assert video.get_frame_count() == 3
    assert transcode_and_probe(video)["frames"] == 7


def test_match_audio_channels_downmixes():
    stereo = torch.zeros(1, 2, 100)
    stereo[:, 1] = 1.0
    assert torch.allclose(_match_audio_channels(stereo, 1), torch.full((1, 1, 100), 0.5))
    assert _match_audio_channels(torch.ones(1, 1, 10), 2).shape == (1, 2, 10)
    assert _match_audio_channels(torch.ones(1, 2, 10), 6).shape == (1, 6, 10)
    assert _match_audio_channels(torch.ones(1, 6, 10), 2).shape == (1, 2, 10)


def test_concat_rotated_source_uses_display_dimensions(cleanup):
    """A 64x32 source with a 90-degree display matrix shows as 32x64; resize=error must compare
    displayed sizes, and get_dimensions() must report what gets encoded."""
    rotated = create_transcode_source(width=64, height=32, frames=6, rotation=True)
    plain = create_av_source(width=32, height=64, frames=6, audio=False)
    square = create_av_source(frames=6, audio=False)
    cleanup += [rotated, plain, square]
    assert VideoFromFile(rotated).get_dimensions() == (64, 32)

    video = VideoConcatenated([VideoFromFile(rotated), VideoFromFile(plain)], resize="error")
    assert video.get_dimensions() == (32, 64)
    result = transcode_and_probe(video)
    assert (result["width"], result["height"]) == (32, 64)
    assert result["frames"] == 12

    with pytest.raises(ValueError, match="video 1 is 32x64; expected 64x64"):
        VideoConcatenated([VideoFromFile(square), VideoFromFile(rotated)], resize="error")
    assert VideoConcatenated([VideoFromFile(plain), VideoFromFile(rotated)]).get_dimensions() == (32, 64)


def test_concat_get_components_resamples_audio_to_best_source():
    images = torch.zeros(3, 8, 8, 3)
    first = VideoFromComponents(VideoComponents(images=images, frame_rate=Fraction(30), audio=AudioInput({"waveform": torch.rand(1, 1, 4410), "sample_rate": 44100})))
    second = VideoFromComponents(VideoComponents(images=images, frame_rate=Fraction(30), audio=AudioInput({"waveform": torch.rand(1, 2, 4800), "sample_rate": 48000})))
    components = VideoConcatenated([first, second]).get_components()
    assert components.audio["sample_rate"] == 48000
    assert components.audio["waveform"].shape == (1, 2, 2 * 4800)
    assert components.audio["waveform"][..., 4800:].abs().sum() > 0


def test_concat_as_trimmed(cleanup):
    a, b = create_av_source(frames=30), create_av_source(frames=45)
    cleanup += [a, b]
    video = VideoConcatenated([VideoFromFile(a), VideoFromFile(b)])

    middle = video.as_trimmed(0.5, 1.0)
    assert isinstance(middle, VideoConcatenated)
    assert middle.get_duration() == pytest.approx(1.0, abs=0.01)
    assert transcode_and_probe(middle)["frames"] == pytest.approx(30, abs=2)

    tail = video.as_trimmed(1.0, 0)  # duration 0 means "to the end" (Trim Video passes 0)
    assert tail.get_duration() == pytest.approx(1.5, abs=0.01)
    assert video.as_trimmed(-1.0, 0).get_duration() == pytest.approx(1.0, abs=0.01)

    assert video.as_trimmed(2.0, 1.0, strict_duration=True) is None
    assert video.as_trimmed(2.0, 1.0, strict_duration=False).get_duration() == pytest.approx(0.5, abs=0.01)
    assert video.as_trimmed(3.0, 1.0, strict_duration=False) is None


def test_concat_as_trimmed_exact_boundaries_and_tensor_sources(cleanup):
    """Float rounding must neither drop a clip (a source's as_trimmed returning None) nor append
    a zero-length sliver past an exact clip boundary."""
    files = [create_av_source(frames=n, audio=False) for n in (7, 11, 13)]
    cleanup += files
    clips = [VideoFromFile(path) for path in files]
    video = VideoConcatenated(clips)
    first_two = clips[0].get_duration() + clips[1].get_duration()  # 0.6000000000000001
    trimmed = video.as_trimmed(0, first_two, strict_duration=True)
    assert len(trimmed._sources) == 2
    assert trimmed.get_frame_count() == 18
    assert transcode_and_probe(trimmed)["frames"] == 18

    tensor_clip = VideoFromComponents(VideoComponents(images=torch.rand(11, 64, 64, 3), frame_rate=Fraction(30)))
    video = VideoConcatenated([tensor_clip, clips[2]])
    start = 0.04270229789540167
    assert start + (tensor_clip.get_duration() - start) > tensor_clip.get_duration()  # the ulp overshoot
    trimmed = video.as_trimmed(start, 0.723964368771265, strict_duration=False)
    assert len(trimmed._sources) == 2
    assert trimmed.get_duration() == pytest.approx(0.724, abs=0.01)
    assert video.as_trimmed(start, 0.723964368771265, strict_duration=True) is not None


def test_concat_trimmed_sources_keep_their_windows_when_retrimmed(cleanup):
    """Trim Video -> Concatenate -> Trim Video: re-trimming must stay inside each source's own window."""
    a, b = create_av_source(frames=30), create_av_source(frames=30)
    cleanup += [a, b]
    video = VideoConcatenated([VideoFromFile(a, start_time=0.2, duration=0.5), VideoFromFile(b, duration=0.5)])
    trimmed = video.as_trimmed(0.1, 0.9, strict_duration=True)
    assert trimmed.get_duration() == pytest.approx(0.9, abs=0.01)
    assert trimmed.get_frame_count() == 27
    assert transcode_and_probe(trimmed)["frames"] == 27
    tail = video.as_trimmed(-0.4, 0)
    assert tail.get_duration() == pytest.approx(0.4, abs=0.01)
    assert tail.get_frame_count() == 12


def test_concat_trimmed_clip_audio_does_not_bleed_past_its_window(cleanup):
    """A trim whose sample cap lands exactly on a decoded audio frame boundary (0.064 s = 3 x 1024
    samples at 48 kHz) must not leak the clip's queued audio into the next clip."""
    a = create_av_source(frames=30, sample_rate=48000, tone_hz=440)
    b = create_av_source(frames=60, sample_rate=48000, tone_hz=None)
    cleanup += [a, b]
    video = VideoConcatenated([VideoFromFile(a).as_trimmed(0, 0.064), VideoFromFile(b)])
    rate, _, samples = decode_audio(save_mp4(video))
    loud = loud_windows(samples, rate)
    assert loud and max(loud) < 0.08
    assert np.abs(samples[:, int(0.1 * rate):]).max() < 0.01
    assert samples.shape[1] == pytest.approx(2.064 * rate, abs=2048)


def test_concat_flattens_nested(cleanup):
    a, b, c = create_av_source(frames=6), create_av_source(frames=6), create_av_source(frames=6)
    cleanup += [a, b, c]
    inner = VideoConcatenated([VideoFromFile(a), VideoFromFile(b)])
    outer = VideoConcatenated([inner, VideoFromFile(c)])
    assert len(outer._sources) == 3
    assert outer.get_frame_count() == 18
    assert transcode_and_probe(outer)["frames"] == 18


def test_concat_nested_keeps_inner_resize_mode(cleanup):
    """An inner 'fit' concat already presents one frame size, so an outer 'error' concat accepts it."""
    a = create_av_source(frames=6, audio=False)
    small = create_av_source(width=32, height=32, frames=6, audio=False)
    c = create_av_source(frames=6, audio=False)
    cleanup += [a, small, c]
    inner = VideoConcatenated([VideoFromFile(a), VideoFromFile(small)], resize="fit")
    outer = VideoConcatenated([inner, VideoFromFile(c)], resize="error")
    assert len(outer._sources) == 2 and outer._sources[0] is inner
    result = transcode_and_probe(outer)
    assert result["frames"] == 18
    assert (result["width"], result["height"]) == (64, 64)


def test_concat_tensor_source(cleanup):
    a = create_av_source(frames=30)
    cleanup.append(a)
    components = VideoFromComponents(VideoComponents(images=torch.rand(12, 64, 64, 3), frame_rate=Fraction(30)))
    result = transcode_and_probe(VideoConcatenated([VideoFromFile(a), components]))
    assert result["frames"] == 42
    assert result["video_seconds"] == pytest.approx(1.4, abs=0.01)
    assert result["audio_seconds"] == pytest.approx(1.4, abs=0.05)


def test_concat_same_bytesio_source_twice(cleanup):
    a = create_av_source(frames=30)
    cleanup.append(a)
    with open(a, "rb") as f:
        source = io.BytesIO(f.read())
    clip = VideoFromFile(source)
    result = transcode_and_probe(VideoConcatenated([clip, clip]))
    assert result["frames"] == 60
    assert result["video_seconds"] == pytest.approx(2.0, abs=0.01)
    assert result["audio_seconds"] == pytest.approx(2.0, abs=0.05)


def test_concat_writes_only_caller_metadata(cleanup):
    a = create_av_source(frames=3, metadata={"prompt": "from clip one"})
    b = create_av_source(frames=3)
    cleanup += [a, b]
    buffer = save_mp4(VideoConcatenated([VideoFromFile(a), VideoFromFile(b)]), metadata={"workflow": {"nodes": 2}})
    buffer.seek(0)
    with av.open(buffer) as container:
        assert json.loads(container.metadata["workflow"]) == {"nodes": 2}
        assert "prompt" not in container.metadata


def test_concat_stream_source_is_cached_mp4(cleanup):
    a, b = create_av_source(frames=6), create_av_source(frames=6)
    cleanup += [a, b]
    video = VideoConcatenated([VideoFromFile(a), VideoFromFile(b)])
    assert video.get_container_format() == "mp4"
    source = video.get_stream_source()
    assert isinstance(source, io.BytesIO)
    assert video.get_stream_source() is source
    assert source.tell() == 0
    with av.open(source) as container:
        assert container.format.name.startswith("mov,mp4")
        assert len(container.streams.video) == 1 and len(container.streams.audio) == 1
        assert container.streams.video[0].frames == 12


def test_concat_webm_output(cleanup):
    a, b = create_av_source(frames=6), create_av_source(frames=6)
    cleanup += [a, b]
    buffer = save_mp4(VideoConcatenated([VideoFromFile(a), VideoFromFile(b)]), format=VideoContainer.WEBM, codec=VideoCodec.AV1)
    with av.open(buffer) as container:
        assert container.streams.video[0].codec.canonical_name == "av1"
        assert container.streams.audio[0].codec.canonical_name == "opus"
        assert container.streams.audio[0].rate == 48000
    rate, _, samples = decode_audio(buffer)
    assert samples.shape[1] == pytest.approx(0.4 * rate, abs=4096)


def test_concat_requires_a_source():
    with pytest.raises(ValueError):
        VideoConcatenated([])
