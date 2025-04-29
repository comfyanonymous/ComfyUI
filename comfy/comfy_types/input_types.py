from __future__ import annotations
from abc import ABC, abstractmethod
from av.container import InputContainer
from av.subtitles.stream import SubtitleStream
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Optional, TypedDict
import av
import io
import json
import numpy as np
import torch

ImageInput = torch.Tensor
"""
An image in format [B, H, W, C] where B is the batch size, C is the number of channels,
"""

class AudioInput(TypedDict):
    """
    TypedDict representing audio input.
    """

    waveform: torch.Tensor
    """
    Tensor in the format [B, C, T] where B is the batch size, C is the number of channels,
    """

    sample_rate: int

class VideoCodec(str, Enum):
    AUTO = "auto"
    H264 = "h264"

    @classmethod
    def as_input(cls) -> list[str]:
        """
        Returns a list of codec names that can be used as node input.
        """
        return [member.value for member in cls]

class VideoContainer(str, Enum):
    AUTO = "auto"
    MP4 = "mp4"

    @classmethod
    def as_input(cls) -> list[str]:
        """
        Returns a list of container names that can be used as node input.
        """
        return [member.value for member in cls]

    @classmethod
    def get_extension(cls, value) -> str:
        """
        Returns the file extension for the container.
        """
        if isinstance(value, str):
            value = cls(value)
        if value == VideoContainer.MP4 or value == VideoContainer.AUTO:
            return "mp4"
        return ""

@dataclass
class VideoComponents:
    """
    Dataclass representing the components of a video.
    """

    images: ImageInput
    frame_rate: Fraction
    audio: Optional[AudioInput] = None
    metadata: Optional[dict] = None

class VideoInput(ABC):
    """
    Abstract base class for video input types.
    """

    @abstractmethod
    def get_components(self) -> VideoComponents:
        """
        Abstract method to get the video components (images, audio, and frame rate).

        Returns:
            VideoComponents containing images, audio, and frame rate
        """
        pass

    @abstractmethod
    def save_to(
        self,
        path: str,
        format: VideoContainer = VideoContainer.AUTO,
        codec: VideoCodec = VideoCodec.AUTO,
        metadata: Optional[dict] = None
    ):
        """
        Abstract method to save the video input to a file.
        """
        pass

    # Provide a default implementation, but subclasses can provide optimized versions
    # if possible.
    def get_dimensions(self) -> tuple[int, int]:
        """
        Returns the dimensions of the video input.

        Returns:
            Tuple of (width, height)
        """
        components = self.get_components()
        return components.images.shape[2], components.images.shape[1]

class VideoFromFile(VideoInput):
    """
    Class representing video input from a file.
    """

    def __init__(self, file: str | io.BytesIO):
        """
        Initialize the VideoFromFile object based off of either a path on disk or a BytesIO object
        containing the file contents.
        """
        self.file = file

    def get_dimensions(self) -> tuple[int, int]:
        """
        Returns the dimensions of the video input.

        Returns:
            Tuple of (width, height)
        """
        if isinstance(self.file, io.BytesIO):
            self.file.seek(0)  # Reset the BytesIO object to the beginning
        with av.open(self.file, mode='r') as container:
            for stream in container.streams:
                if stream.type == 'video':
                    assert isinstance(stream, av.VideoStream)
                    return stream.width, stream.height
        raise ValueError(f"No video stream found in file '{self.file}'")

    def get_components_internal(self, container: InputContainer) -> VideoComponents:
        # Get video frames
        frames = []
        for frame in container.decode(video=0):
            img = frame.to_ndarray(format='rgb24')  # shape: (H, W, 3)
            img = torch.from_numpy(img) / 255.0  # shape: (H, W, 3)
            frames.append(img)

        images = torch.stack(frames) if len(frames) > 0 else torch.zeros(0, 3, 0, 0)

        # Get frame rate
        video_stream = next(s for s in container.streams if s.type == 'video')
        frame_rate = Fraction(video_stream.average_rate) if video_stream and video_stream.average_rate else Fraction(1)

        # Get audio if available
        audio = None
        try:
            container.seek(0)  # Reset the container to the beginning
            for stream in container.streams:
                if stream.type != 'audio':
                    continue
                assert isinstance(stream, av.AudioStream)
                audio_frames = []
                for packet in container.demux(stream):
                    for frame in packet.decode():
                        assert isinstance(frame, av.AudioFrame)
                        audio_frames.append(frame.to_ndarray())  # shape: (channels, samples)
                if len(audio_frames) > 0:
                    audio_data = np.concatenate(audio_frames, axis=1)  # shape: (channels, total_samples)
                    audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)  # shape: (1, channels, total_samples)
                    audio = AudioInput({
                        "waveform": audio_tensor,
                        "sample_rate": int(stream.sample_rate) if stream.sample_rate else 1,
                    })
        except StopIteration:
            pass  # No audio stream

        metadata = container.metadata
        return VideoComponents(images=images, audio=audio, frame_rate=frame_rate, metadata=metadata)

    def get_components(self) -> VideoComponents:
        if isinstance(self.file, io.BytesIO):
            self.file.seek(0)  # Reset the BytesIO object to the beginning
        with av.open(self.file, mode='r') as container:
            return self.get_components_internal(container)
        raise ValueError(f"No video stream found in file '{self.file}'")

    def save_to(
        self,
        path: str,
        format: VideoContainer = VideoContainer.AUTO,
        codec: VideoCodec = VideoCodec.AUTO,
        metadata: Optional[dict] = None
    ):
        if isinstance(self.file, io.BytesIO):
            self.file.seek(0)  # Reset the BytesIO object to the beginning
        with av.open(self.file, mode='r') as container:
            container_format = container.format.name
            video_encoding = container.streams.video[0].codec.name if len(container.streams.video) > 0 else None
            reuse_streams = True
            if format != VideoContainer.AUTO and format not in container_format.split(","):
                reuse_streams = False
            if codec != VideoCodec.AUTO and codec != video_encoding and video_encoding is not None:
                reuse_streams = False

            if not reuse_streams:
                components = self.get_components_internal(container)
                video = VideoFromComponents(components)
                return video.save_to(
                    path,
                    format=format,
                    codec=codec,
                    metadata=metadata
                )

            streams = container.streams
            with av.open(path, mode='w', options={"movflags": "use_metadata_tags"}) as output_container:
                # Copy over the original metadata
                for key, value in container.metadata.items():
                    if metadata is None or key not in metadata:
                        output_container.metadata[key] = value

                # Add our new metadata
                if metadata is not None:
                    for key, value in metadata.items():
                        if isinstance(value, str):
                            output_container.metadata[key] = value
                        else:
                            output_container.metadata[key] = json.dumps(value)

                # Add streams to the new container
                stream_map = {}
                for stream in streams:
                    if isinstance(stream, (av.VideoStream, av.AudioStream, SubtitleStream)):
                        out_stream = output_container.add_stream_from_template(template=stream, opaque=True)
                        stream_map[stream] = out_stream

                # Write packets to the new container
                for packet in container.demux():
                    if packet.stream in stream_map and packet.dts is not None:
                        packet.stream = stream_map[packet.stream]
                        output_container.mux(packet)

class VideoFromComponents(VideoInput):
    """
    Class representing video input from tensors.
    """

    def __init__(self, components: VideoComponents):
        self.components = components

    def get_components(self) -> VideoComponents:
        return VideoComponents(
            images=self.components.images,
            audio=self.components.audio,
            frame_rate=self.components.frame_rate
        )

    def save_to(
        self,
        path: str,
        format: VideoContainer = VideoContainer.AUTO,
        codec: VideoCodec = VideoCodec.AUTO,
        metadata: Optional[dict] = None
    ):
        if format != VideoContainer.AUTO and format != VideoContainer.MP4:
            raise ValueError("Only MP4 format is supported for now")
        if codec != VideoCodec.AUTO and codec != VideoCodec.H264:
            raise ValueError("Only H264 codec is supported for now")
        with av.open(path, mode='w', options={'movflags': 'use_metadata_tags'}) as output:
            # Add metadata before writing any streams
            if metadata is not None:
                for key, value in metadata.items():
                    output.metadata[key] = json.dumps(value)

            frame_rate = Fraction(round(self.components.frame_rate * 1000), 1000)
            # Create a video stream
            video_stream = output.add_stream('h264', rate=frame_rate)
            video_stream.width = self.components.images.shape[2]
            video_stream.height = self.components.images.shape[1]
            video_stream.pix_fmt = 'yuv420p'

            # Create an audio stream
            audio_sample_rate = 1
            audio_stream: Optional[av.AudioStream] = None
            if self.components.audio:
                audio_sample_rate = int(self.components.audio['sample_rate'])
                audio_stream = output.add_stream('aac', rate=audio_sample_rate)
                audio_stream.sample_rate = audio_sample_rate
                audio_stream.format = 'fltp'

            # Encode video
            for i, frame in enumerate(self.components.images):
                img = (frame * 255).clamp(0, 255).byte().cpu().numpy() # shape: (H, W, 3)
                frame = av.VideoFrame.from_ndarray(img, format='rgb24')
                frame = frame.reformat(format='yuv420p')  # Convert to YUV420P as required by h264
                packet = video_stream.encode(frame)
                output.mux(packet)

            # Flush video
            packet = video_stream.encode(None)
            output.mux(packet)

            if audio_stream and self.components.audio:
                # Encode audio
                samples_per_frame = int(audio_sample_rate / frame_rate)
                num_frames = self.components.audio['waveform'].shape[2] // samples_per_frame
                for i in range(num_frames):
                    start = i * samples_per_frame
                    end = start + samples_per_frame
                    # TODO(Feature) - Add support for stereo audio
                    chunk = self.components.audio['waveform'][0, 0, start:end].unsqueeze(0).numpy()
                    audio_frame = av.AudioFrame.from_ndarray(chunk, format='fltp', layout='mono')
                    audio_frame.sample_rate = audio_sample_rate
                    audio_frame.pts = i * samples_per_frame
                    for packet in audio_stream.encode(audio_frame):
                        output.mux(packet)

                # Flush audio
                for packet in audio_stream.encode(None):
                    output.mux(packet)

