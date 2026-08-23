import av
import pytest
import torch

from comfy_api.latest._io import FolderType
from comfy_api.latest._ui import AudioSaveHelper


@pytest.mark.parametrize("channels", [1, 2, 6])
def test_flac_preserves_audio_channels_and_duration(channels, tmp_path, monkeypatch):
    sample_rate = 48000
    duration = 0.1
    samples = int(sample_rate * duration)
    waveform = torch.zeros((1, channels, samples), dtype=torch.float32)
    audio = {"waveform": waveform, "sample_rate": sample_rate}

    monkeypatch.setattr(
        "comfy_api.latest._ui.folder_paths.get_save_image_path",
        lambda *args: (str(tmp_path), "preview", 1, "", "preview"),
    )

    result = AudioSaveHelper.save_audio(
        audio,
        filename_prefix="preview",
        folder_type=FolderType.temp,
        cls=None,
        format="flac",
    )

    with av.open(tmp_path / result[0].filename) as container:
        stream = container.streams.audio[0]
        decoded_samples = sum(frame.samples for frame in container.decode(audio=0))

    assert len(stream.layout.channels) == channels
    assert decoded_samples / stream.codec_context.sample_rate == pytest.approx(duration)
