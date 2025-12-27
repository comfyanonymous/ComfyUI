from __future__ import annotations
from av.container import InputContainer
from av.subtitles.stream import SubtitleStream
from fractions import Fraction
from typing import Optional
from .._input import AudioInput, VideoInput
import av
import io
import json
import numpy as np
import math
import torch
from .._util import VideoContainer, VideoCodec, VideoComponents


class _ReentrantBytesIO(io.BytesIO):
    """Read-only, seekable BytesIO-compatible view over shared immutable bytes."""

    def __init__(self, data: bytes):
        super().__init__(b"")  # Initialize base BytesIO with an empty buffer; we do not use its internal storage.
        if data is None:
            raise TypeError("data must be bytes, not None")
        self._data = data
        self._pos = 0
        self._len = len(data)

    def getvalue(self) -> bytes:
        if self.closed:
            raise ValueError("I/O operation on closed file.")
        return self._data

    def getbuffer(self) -> memoryview:
        if self.closed:
            raise ValueError("I/O operation on closed file.")
        return memoryview(self._data)

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if self.closed:
            raise ValueError("I/O operation on closed file.")
        if whence == io.SEEK_SET:
            new_pos = offset
        elif whence == io.SEEK_CUR:
            new_pos = self._pos + offset
        elif whence == io.SEEK_END:
            new_pos = self._len + offset
        else:
            raise ValueError(f"Invalid whence: {whence}")
        if new_pos < 0:
            raise ValueError("Negative seek position")
        self._pos = new_pos
        return self._pos

    def readinto(self, b) -> int:
        if self.closed:
            raise ValueError("I/O operation on closed file.")
        mv = memoryview(b)
        if mv.readonly:
            raise TypeError("readinto() argument must be writable")
        mv = mv.cast("B")
        if self._pos >= self._len:
            return 0
        n = min(len(mv), self._len - self._pos)
        mv[:n] = self._data[self._pos:self._pos + n]
        self._pos += n
        return n

    def readinto1(self, b) -> int:
        return self.readinto(b)

    def read(self, size: int = -1) -> bytes:
        if self.closed:
            raise ValueError("I/O operation on closed file.")
        if size is None or size < 0:
            size = self._len - self._pos
        if self._pos >= self._len:
            return b""
        end = min(self._pos + size, self._len)
        out = self._data[self._pos:end]
        self._pos = end
        return out

    def read1(self, size: int = -1) -> bytes:
        return self.read(size)

    def readline(self, size: int = -1) -> bytes:
        if self.closed:
            raise ValueError("I/O operation on closed file.")
        if self._pos >= self._len:
            return b""
        end_limit = self._len if size is None or size < 0 else min(self._len, self._pos + size)
        nl = self._data.find(b"\n", self._pos, end_limit)
        end = (nl + 1) if nl != -1 else end_limit
        out = self._data[self._pos:end]
        self._pos = end
        return out

    def readlines(self, hint: int = -1) -> list[bytes]:
        if self.closed:
            raise ValueError("I/O operation on closed file.")
        lines: list[bytes] = []
        total = 0
        while True:
            line = self.readline()
            if not line:
                break
            lines.append(line)
            total += len(line)
            if hint is not None and 0 <= hint <= total:
                break
        return lines

    def write(self, b) -> int:
        raise io.UnsupportedOperation("not writable")

    def writelines(self, lines) -> None:
        raise io.UnsupportedOperation("not writable")

    def truncate(self, size: int | None = None) -> int:
        raise io.UnsupportedOperation("not writable")


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

    __data: str | bytes

    def __init__(self, file: str | io.BytesIO | bytes | bytearray | memoryview):
        """
        Initialize the VideoFromFile object based off of either a path on disk or a BytesIO object
        containing the file contents.
        """
        if isinstance(file, str):
            self.__data = file
        elif isinstance(file, io.BytesIO):
            # Snapshot to immutable bytes once to ensure re-entrant, parallel-safe readers.
            self.__data = file.getbuffer().tobytes()
        elif isinstance(file, (bytes, bytearray, memoryview)):
            self.__data = bytes(file)
        else:
            raise TypeError(f"Unsupported video source type: {type(file)!r}")

    def get_stream_source(self) -> str | io.BytesIO:
        """
        Return the underlying file source for efficient streaming.
        This avoids unnecessary memory copies when the source is already a file path.
        """
        if isinstance(self.__data, str):
            return self.__data
        return _ReentrantBytesIO(self.__data)

    def get_dimensions(self) -> tuple[int, int]:
        """
        Returns the dimensions of the video input.

        Returns:
            Tuple of (width, height)
        """
        with av.open(self.get_stream_source(), mode="r") as container:
            for stream in container.streams:
                if stream.type == 'video':
                    assert isinstance(stream, av.VideoStream)
                    return stream.width, stream.height
        raise ValueError(f"No video stream found in {self._source_label()}")

    def get_duration(self) -> float:
        """
        Returns the duration of the video in seconds.

        Returns:
            Duration in seconds
        """
        with av.open(self.get_stream_source(), mode="r") as container:
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

        raise ValueError(f"Could not determine duration for file '{self._source_label()}'")

    def get_frame_count(self) -> int:
        """
        Returns the number of frames in the video without materializing them as
        torch tensors.
        """
        with av.open(self.get_stream_source(), mode="r") as container:
            video_stream = self._get_first_video_stream(container)
            # 1. Prefer the frames field if available
            if video_stream.frames and video_stream.frames > 0:
                return int(video_stream.frames)

            # 2. Try to estimate from duration and average_rate using only metadata
            if container.duration is not None and video_stream.average_rate:
                duration_seconds = float(container.duration / av.time_base)
                estimated_frames = int(round(duration_seconds * float(video_stream.average_rate)))
                if estimated_frames > 0:
                    return estimated_frames

            if (
                getattr(video_stream, "duration", None) is not None
                and getattr(video_stream, "time_base", None) is not None
                and video_stream.average_rate
            ):
                duration_seconds = float(video_stream.duration * video_stream.time_base)
                estimated_frames = int(round(duration_seconds * float(video_stream.average_rate)))
                if estimated_frames > 0:
                    return estimated_frames

            # 3. Last resort: decode frames and count them (streaming)
            frame_count = 0
            container.seek(0)
            for packet in container.demux(video_stream):
                for _ in packet.decode():
                    frame_count += 1

            if frame_count == 0:
                raise ValueError(f"Could not determine frame count for file '{self._source_label()}'")
            return frame_count

    def get_frame_rate(self) -> Fraction:
        """
        Returns the average frame rate of the video using container metadata
        without decoding all frames.
        """
        with av.open(self.get_stream_source(), mode="r") as container:
            video_stream = self._get_first_video_stream(container)
            # Preferred: use PyAV's average_rate (usually already a Fraction-like)
            if video_stream.average_rate:
                return Fraction(video_stream.average_rate)

            # Fallback: estimate from frames + duration if available
            if video_stream.frames and container.duration:
                duration_seconds = float(container.duration / av.time_base)
                if duration_seconds > 0:
                    return Fraction(video_stream.frames / duration_seconds).limit_denominator()

            # Last resort: match get_components_internal default
            return Fraction(1)

    def get_container_format(self) -> str:
        """
        Returns the container format of the video (e.g., 'mp4', 'mov', 'avi').

        Returns:
            Container format as string
        """
        with av.open(self.get_stream_source(), mode='r') as container:
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
        with av.open(self.get_stream_source(), mode='r') as container:
            return self.get_components_internal(container)

    def save_to(
        self,
        path: str | io.BytesIO,
        format: VideoContainer = VideoContainer.AUTO,
        codec: VideoCodec = VideoCodec.AUTO,
        metadata: Optional[dict] = None
    ):
        with av.open(self.get_stream_source(), mode='r') as container:
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

    def _get_first_video_stream(self, container: InputContainer):
        video_stream = next((s for s in container.streams if s.type == "video"), None)
        if video_stream is None:
            raise ValueError(f"No video stream found in file '{self._source_label()}'")
        return video_stream

    def _source_label(self) -> str:
        return self.__data if isinstance(self.__data, str) else f"<in-memory video: {len(self.__data)} bytes>"


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
        extra_kwargs = {}
        if isinstance(format, VideoContainer) and format != VideoContainer.AUTO:
            extra_kwargs["format"] = format.value
        with av.open(path, mode='w', options={'movflags': 'use_metadata_tags'}, **extra_kwargs) as output:
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
                waveform = self.__components.audio['waveform']
                waveform = waveform[:, :, :math.ceil((audio_sample_rate / frame_rate) * self.__components.images.shape[0])]
                frame = av.AudioFrame.from_ndarray(waveform.movedim(2, 1).reshape(1, -1).float().numpy(), format='flt', layout='mono' if waveform.shape[1] == 1 else 'stereo')
                frame.sample_rate = audio_sample_rate
                frame.pts = 0
                output.mux(audio_stream.encode(frame))

                # Flush encoder
                output.mux(audio_stream.encode(None))
