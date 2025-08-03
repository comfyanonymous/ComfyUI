from __future__ import annotations
from av.container import InputContainer
from av.subtitles.stream import SubtitleStream
from fractions import Fraction
from typing import Optional
from comfy_api.latest._input import AudioInput, VideoInput
import av
import io
import json
import numpy as np
import torch
from comfy_api.latest._util import VideoContainer, VideoCodec, VideoComponents


def container_to_output_format(container_format: str | None) -> str | None:
    """
    A container's `format` may be a comma-separated list of formats.
    E.g., iso container's `format` may be `mov,mp4,m4a,3gp,3g2,mj2`.
    However, writing to a file/stream with `av.open` requires a single format,
    or `None` to auto-detect.
    """
    if not container_format:
        return None  # Auto-detect

    if "," not in container_format:
        return container_format

    formats = container_format.split(",")
    return formats[0]


def get_open_write_kwargs(
    dest: str | io.BytesIO, container_format: str, to_format: str | None
) -> dict:
    """Get kwargs for writing a `VideoFromFile` to a file/stream with `av.open`"""
    open_kwargs = {
        "mode": "w",
        # If isobmff, preserve custom metadata tags (workflow, prompt, extra_pnginfo)
        "options": {"movflags": "use_metadata_tags"},
    }

    is_write_to_buffer = isinstance(dest, io.BytesIO)
    if is_write_to_buffer:
        # Set output format explicitly, since it cannot be inferred from file extension
        if to_format == VideoContainer.AUTO:
            to_format = container_format.lower()
        elif isinstance(to_format, str):
            to_format = to_format.lower()
        open_kwargs["format"] = container_to_output_format(to_format)

    return open_kwargs


class VideoFromFile(VideoInput):
    """
    Class representing video input from a file.
    """

    def __init__(self, file: str | io.BytesIO):
        """
        Initialize the VideoFromFile object based off of either a path on disk or a BytesIO object
        containing the file contents.
        """
        self.__file = file

    def get_stream_source(self) -> str | io.BytesIO:
        """
        Return the underlying file source for efficient streaming.
        This avoids unnecessary memory copies when the source is already a file path.
        """
        if isinstance(self.__file, io.BytesIO):
            self.__file.seek(0)
        return self.__file

    def get_dimensions(self) -> tuple[int, int]:
        """
        Returns the dimensions of the video input.

        Returns:
            Tuple of (width, height)
        """
        if isinstance(self.__file, io.BytesIO):
            self.__file.seek(0)  # Reset the BytesIO object to the beginning
        with av.open(self.__file, mode='r') as container:
            for stream in container.streams:
                if stream.type == 'video':
                    assert isinstance(stream, av.VideoStream)
                    return stream.width, stream.height
        raise ValueError(f"No video stream found in file '{self.__file}'")

    def get_duration(self) -> float:
        """
        Returns the duration of the video in seconds.

        Returns:
            Duration in seconds
        """
        if isinstance(self.__file, io.BytesIO):
            self.__file.seek(0)
        with av.open(self.__file, mode="r") as container:
            if container.duration is not None:
                return float(container.duration / av.time_base)

            # Fallback: calculate from frame count and frame rate
            video_stream = next(
                (s for s in container.streams if s.type == "video"), None
            )
            if video_stream and video_stream.frames and video_stream.average_rate:
                return float(video_stream.frames / video_stream.average_rate)

            # Last resort: decode frames to count them
            if video_stream and video_stream.average_rate:
                frame_count = 0
                container.seek(0)
                for packet in container.demux(video_stream):
                    for _ in packet.decode():
                        frame_count += 1
                if frame_count > 0:
                    return float(frame_count / video_stream.average_rate)

        raise ValueError(f"Could not determine duration for file '{self.__file}'")

    def get_container_format(self) -> str:
        """
        Returns the container format of the video (e.g., 'mp4', 'mov', 'avi').

        Returns:
            Container format as string
        """
        if isinstance(self.__file, io.BytesIO):
            self.__file.seek(0)
        with av.open(self.__file, mode='r') as container:
            return container.format.name

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
        if isinstance(self.__file, io.BytesIO):
            self.__file.seek(0)  # Reset the BytesIO object to the beginning
        with av.open(self.__file, mode='r') as container:
            return self.get_components_internal(container)
        raise ValueError(f"No video stream found in file '{self.__file}'")

    def save_to(
        self,
        path: str | io.BytesIO,
        format: VideoContainer = VideoContainer.AUTO,
        codec: VideoCodec = VideoCodec.AUTO,
        metadata: Optional[dict] = None
    ):
        if isinstance(self.__file, io.BytesIO):
            self.__file.seek(0)  # Reset the BytesIO object to the beginning
        with av.open(self.__file, mode='r') as container:
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

            open_kwargs = get_open_write_kwargs(path, container_format, format)
            with av.open(path, **open_kwargs) as output_container:
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
        self.__components = components

    def get_components(self) -> VideoComponents:
        return VideoComponents(
            images=self.__components.images,
            audio=self.__components.audio,
            frame_rate=self.__components.frame_rate
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

            frame_rate = Fraction(round(self.__components.frame_rate * 1000), 1000)
            # Create a video stream
            video_stream = output.add_stream('h264', rate=frame_rate)
            video_stream.width = self.__components.images.shape[2]
            video_stream.height = self.__components.images.shape[1]
            video_stream.pix_fmt = 'yuv420p'

            # Create an audio stream
            audio_sample_rate = 1
            audio_stream: Optional[av.AudioStream] = None
            if self.__components.audio:
                audio_sample_rate = int(self.__components.audio['sample_rate'])
                audio_stream = output.add_stream('aac', rate=audio_sample_rate)
                audio_stream.sample_rate = audio_sample_rate
                audio_stream.format = 'fltp'

            # Encode video
            for i, frame in enumerate(self.__components.images):
                img = (frame * 255).clamp(0, 255).byte().cpu().numpy() # shape: (H, W, 3)
                frame = av.VideoFrame.from_ndarray(img, format='rgb24')
                frame = frame.reformat(format='yuv420p')  # Convert to YUV420P as required by h264
                packet = video_stream.encode(frame)
                output.mux(packet)

            # Flush video
            packet = video_stream.encode(None)
            output.mux(packet)

            if audio_stream and self.__components.audio:
                # Encode audio
                samples_per_frame = int(audio_sample_rate / frame_rate)
                num_frames = self.__components.audio['waveform'].shape[2] // samples_per_frame
                for i in range(num_frames):
                    start = i * samples_per_frame
                    end = start + samples_per_frame
                    # TODO(Feature) - Add support for stereo audio
                    chunk = (
                        self.__components.audio["waveform"][0, 0, start:end]
                        .unsqueeze(0)
                        .contiguous()
                        .numpy()
                    )
                    audio_frame = av.AudioFrame.from_ndarray(chunk, format='fltp', layout='mono')
                    audio_frame.sample_rate = audio_sample_rate
                    audio_frame.pts = i * samples_per_frame
                    for packet in audio_stream.encode(audio_frame):
                        output.mux(packet)

                # Flush audio
                for packet in audio_stream.encode(None):
                    output.mux(packet)


