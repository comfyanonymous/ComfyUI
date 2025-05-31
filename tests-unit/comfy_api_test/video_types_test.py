import pytest
import torch
import tempfile
import os
import av
import io
from fractions import Fraction
from comfy_api.input_impl.video_types import VideoFromFile, VideoFromComponents
from comfy_api.util.video_types import VideoComponents
from comfy_api.input.basic_types import AudioInput
from av.error import InvalidDataError

EPSILON = 0.0001


@pytest.fixture
def sample_images():
    """3-frame 2x2 RGB video tensor"""
    return torch.rand(3, 2, 2, 3)


@pytest.fixture
def sample_audio():
    """Stereo audio with 44.1kHz sample rate"""
    return AudioInput(
        {
            "waveform": torch.rand(1, 2, 1000),
            "sample_rate": 44100,
        }
    )


@pytest.fixture
def video_components(sample_images, sample_audio):
    """VideoComponents with images, audio, and metadata"""
    return VideoComponents(
        images=sample_images,
        audio=sample_audio,
        frame_rate=Fraction(30),
        metadata={"test": "metadata"},
    )


def create_test_video(width=4, height=4, frames=3, fps=30):
    """Helper to create a temporary video file"""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    with av.open(tmp.name, mode="w") as container:
        stream = container.add_stream("h264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"

        for i in range(frames):
            frame = av.VideoFrame.from_ndarray(
                torch.ones(height, width, 3, dtype=torch.uint8).numpy() * (i * 85),
                format="rgb24",
            )
            frame = frame.reformat(format="yuv420p")
            packet = stream.encode(frame)
            container.mux(packet)

        # Flush
        packet = stream.encode(None)
        container.mux(packet)

    return tmp.name


@pytest.fixture
def simple_video_file():
    """4x4 video with 3 frames at 30fps"""
    file_path = create_test_video()
    yield file_path
    os.unlink(file_path)


def test_video_from_components_get_duration(video_components):
    """Duration calculated correctly from frame count and frame rate"""
    video = VideoFromComponents(video_components)
    duration = video.get_duration()

    expected_duration = 3.0 / 30.0
    assert duration == pytest.approx(expected_duration)


def test_video_from_components_get_duration_different_frame_rates(sample_images):
    """Duration correct for different frame rates including fractional"""
    # Test with 60 fps
    components_60fps = VideoComponents(images=sample_images, frame_rate=Fraction(60))
    video_60fps = VideoFromComponents(components_60fps)
    assert video_60fps.get_duration() == pytest.approx(3.0 / 60.0)

    # Test with fractional frame rate (23.976fps)
    components_frac = VideoComponents(
        images=sample_images, frame_rate=Fraction(24000, 1001)
    )
    video_frac = VideoFromComponents(components_frac)
    expected_frac = 3.0 / (24000.0 / 1001.0)
    assert video_frac.get_duration() == pytest.approx(expected_frac)


def test_video_from_components_get_duration_empty_video():
    """Duration is zero for empty video"""
    empty_components = VideoComponents(
        images=torch.zeros(0, 2, 2, 3), frame_rate=Fraction(30)
    )
    video = VideoFromComponents(empty_components)
    assert video.get_duration() == 0.0


def test_video_from_components_get_dimensions(video_components):
    """Dimensions returned correctly from image tensor shape"""
    video = VideoFromComponents(video_components)
    width, height = video.get_dimensions()
    assert width == 2
    assert height == 2


def test_video_from_file_get_duration(simple_video_file):
    """Duration extracted from file metadata"""
    video = VideoFromFile(simple_video_file)
    duration = video.get_duration()
    assert duration == pytest.approx(0.1, abs=0.01)


def test_video_from_file_get_dimensions(simple_video_file):
    """Dimensions read from stream without decoding frames"""
    video = VideoFromFile(simple_video_file)
    width, height = video.get_dimensions()
    assert width == 4
    assert height == 4


def test_video_from_file_bytesio_input():
    """VideoFromFile works with BytesIO input"""
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        stream = container.add_stream("h264", rate=30)
        stream.width = 2
        stream.height = 2
        stream.pix_fmt = "yuv420p"

        frame = av.VideoFrame.from_ndarray(
            torch.zeros(2, 2, 3, dtype=torch.uint8).numpy(), format="rgb24"
        )
        frame = frame.reformat(format="yuv420p")
        packet = stream.encode(frame)
        container.mux(packet)
        packet = stream.encode(None)
        container.mux(packet)

    buffer.seek(0)
    video = VideoFromFile(buffer)

    assert video.get_dimensions() == (2, 2)
    assert video.get_duration() == pytest.approx(1 / 30, abs=0.01)


def test_video_from_file_invalid_file_error():
    """InvalidDataError raised for non-video files"""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        tmp.write(b"not a video file")
        tmp.flush()
        tmp_name = tmp.name

    try:
        with pytest.raises(InvalidDataError):
            video = VideoFromFile(tmp_name)
            video.get_dimensions()
    finally:
        os.unlink(tmp_name)


def test_video_from_file_audio_only_error():
    """ValueError raised for audio-only files"""
    with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as tmp:
        tmp_name = tmp.name

    try:
        with av.open(tmp_name, mode="w") as container:
            stream = container.add_stream("aac", rate=44100)
            stream.sample_rate = 44100
            stream.format = "fltp"

            audio_data = torch.zeros(1, 1024).numpy()
            audio_frame = av.AudioFrame.from_ndarray(
                audio_data, format="fltp", layout="mono"
            )
            audio_frame.sample_rate = 44100
            audio_frame.pts = 0
            packet = stream.encode(audio_frame)
            container.mux(packet)

            for packet in stream.encode(None):
                container.mux(packet)

        with pytest.raises(ValueError, match="No video stream found"):
            video = VideoFromFile(tmp_name)
            video.get_dimensions()
    finally:
        os.unlink(tmp_name)


def test_single_frame_video():
    """Single frame video has correct duration"""
    components = VideoComponents(
        images=torch.rand(1, 10, 10, 3), frame_rate=Fraction(1)
    )
    video = VideoFromComponents(components)
    assert video.get_duration() == 1.0


@pytest.mark.parametrize(
    "frame_rate,expected_fps",
    [
        (Fraction(24000, 1001), 24000 / 1001),
        (Fraction(30000, 1001), 30000 / 1001),
        (Fraction(25, 1), 25.0),
        (Fraction(50, 2), 25.0),
    ],
)
def test_fractional_frame_rates(frame_rate, expected_fps):
    """Duration calculated correctly for various fractional frame rates"""
    components = VideoComponents(images=torch.rand(100, 4, 4, 3), frame_rate=frame_rate)
    video = VideoFromComponents(components)
    duration = video.get_duration()
    expected_duration = 100.0 / expected_fps
    assert duration == pytest.approx(expected_duration)


def test_duration_consistency(video_components):
    """get_duration() consistent with manual calculation from components"""
    video = VideoFromComponents(video_components)

    duration = video.get_duration()
    components = video.get_components()
    manual_duration = float(components.images.shape[0] / components.frame_rate)

    assert duration == pytest.approx(manual_duration)
