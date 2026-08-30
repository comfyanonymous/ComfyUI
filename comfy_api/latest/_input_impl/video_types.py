from av.bitstream import BitStreamFilterContext
from av.container import InputContainer
from av.subtitles.stream import SubtitleStream
from av.video.reformatter import ColorPrimaries, ColorRange, ColorTrc
from dataclasses import dataclass
from fractions import Fraction
from types import SimpleNamespace
from typing import Optional, Sequence
from .._input import AudioInput, VideoInput
import av
import io
import itertools
import json
import numpy as np
import math
import os
import torch
from .._util import VideoContainer, VideoCodec, VideoComponents
import logging


VIDEO_ENCODERS = {
    VideoCodec.H264: "h264",
    VideoCodec.AV1: "libsvtav1",
}
VIDEO_CONTAINER_FORMATS = {
    VideoContainer.MP4: "mp4",
    VideoContainer.MKV: "matroska",
    VideoContainer.WEBM: "webm",
}
WEBM_STREAM_CODECS = {
    "video": {"av1", "vp8", "vp9"},
    "audio": {"opus", "vorbis"},
    "subtitle": {"webvtt"},
}
BT2020_NCL = 9
BT709_NCL = 1
HDR_COLOR_TRANSFERS = {
    "HDR": ColorTrc.ARIB_STD_B67,
    "HDR PQ": ColorTrc.SMPTE2084,
}
VIDEO_COLOR_TRANSFERS = {
    "sRGB": ColorTrc.IEC61966_2_1,
    **HDR_COLOR_TRANSFERS,
}
VIDEO_TRANSFER_COLOR_SPACES = {
    ColorTrc.BT709: "sRGB",
    ColorTrc.IEC61966_2_1: "sRGB",
    ColorTrc.ARIB_STD_B67: "HDR",
    ColorTrc.SMPTE2084: "HDR PQ",
}


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
    is_write_to_buffer = isinstance(dest, io.BytesIO)
    open_kwargs = {"mode": "w"}

    if is_write_to_buffer:
        # Set output format explicitly, since it cannot be inferred from file extension
        if to_format == VideoContainer.AUTO:
            to_format = container_format.lower()
        elif isinstance(to_format, VideoContainer):
            to_format = VIDEO_CONTAINER_FORMATS[to_format]
        elif isinstance(to_format, str):
            to_format = to_format.lower()
        open_kwargs["format"] = container_to_output_format(to_format)

    output_format = open_kwargs["format"] if is_write_to_buffer else os.path.splitext(dest)[1].lower().lstrip(".")
    if output_format in ("mov", "mp4"):
        # Preserve custom metadata tags (workflow, prompt, extra_pnginfo) in isobmff.
        movflags = "use_metadata_tags" if is_write_to_buffer else "use_metadata_tags+faststart"
        open_kwargs["options"] = {"movflags": movflags}

    return open_kwargs


def video_stream_bit_depth(stream) -> int:
    if stream is None or stream.format is None or not stream.format.components:
        return 8
    return max(component.bits for component in stream.format.components)


def isobmff_hevc_filter(output_container, stream, out_stream):
    """Apple players need the 'hvc1' sample entry, not FFmpeg's default 'hev1'. Annex B input without
    extradata makes the muxer build hvcC from the first packet and strip in-band parameter sets;
    'hvc1' sources already have a complete hvcC and only need the tag PyAV reset."""
    if output_container.format.name not in ("mp4", "mov") or stream.codec.canonical_name != "hevc":
        return None
    try:
        codec_tag = stream.codec_context.codec_tag
    except UnicodeDecodeError:
        codec_tag = ""
    if codec_tag == "hvc1":
        out_stream.codec_context.codec_tag = "hvc1"
        return None
    hevc_filter = BitStreamFilterContext("hevc_mp4toannexb", stream, out_stream)
    out_stream.codec_context.codec_tag = "hvc1"
    out_stream.codec_context.extradata = None
    return hevc_filter


def filter_hevc_packet(hevc_filter, packet):
    if packet.has_sidedata("new_extradata"):
        raise ValueError("HEVC with multiple sample descriptions cannot be remuxed as hvc1; re-encode it instead")
    return hevc_filter.filter(packet)


def last_decodable_audio_stream(container: InputContainer):
    """Streams FFmpeg has no decoder for have no codec context, and decoding their
    packets crashes the process (e.g. APAC spatial-audio track in iPhone)."""
    stream = next(
        (s for s in reversed(container.streams.audio) if s.codec_context is not None),
        None,
    )
    if stream is None and len(container.streams.audio):
        logging.warning("No decodable audio stream found in video; ignoring audio.")
    return stream


def probe_audio_params(container: InputContainer, audio_stream, max_packets: int = 200):
    """Containers probed only up to a window (mpegts) leave audio codec parameters unset when
    audio starts beyond it; learn them by decoding ahead. The caller must seek back afterwards.
    Returns (sample_rate, channels), zeros when the stream never yields a decodable frame."""
    for i, packet in enumerate(container.demux(audio_stream)):
        try:
            frames = packet.decode()
        except av.error.FFmpegError:
            frames = ()
        if frames:
            return frames[0].sample_rate, frames[0].layout.nb_channels
        if i >= max_packets:
            break
    return 0, 0


def write_output_metadata(container: InputContainer, output, metadata: dict | None):
    """Copy the source container's metadata, then overlay the caller's tags."""
    for key, value in container.metadata.items():
        if metadata is None or key not in metadata:
            output.metadata[key] = value
    if metadata is not None:
        for key, value in metadata.items():
            output.metadata[key] = value if isinstance(value, str) else json.dumps(value)


def video_output_config(path: str | io.BytesIO, format: VideoContainer, codec: VideoCodec) -> tuple[dict, VideoContainer, VideoCodec]:
    if isinstance(format, str):
        format = VideoContainer(format)
    if isinstance(codec, str):
        codec = VideoCodec(codec)

    if format == VideoContainer.AUTO:
        extension = os.path.splitext(os.fspath(path))[1].lower() if isinstance(path, (str, os.PathLike)) else ""
        format = {
            ".mkv": VideoContainer.MKV,
            ".webm": VideoContainer.WEBM,
        }.get(extension, VideoContainer.MP4)
    if codec == VideoCodec.AUTO:
        codec = VideoCodec.AV1 if format == VideoContainer.WEBM else VideoCodec.H264
    if format == VideoContainer.WEBM and codec != VideoCodec.AV1:
        raise ValueError("WebM output requires the AV1 codec")

    # FFmpeg's faststart pass reopens the output by filename, so it cannot be used with file-like objects.
    open_kwargs = {"mode": "w", "format": VIDEO_CONTAINER_FORMATS[format]}
    if format == VideoContainer.MP4:
        movflags = "use_metadata_tags+faststart" if isinstance(path, (str, os.PathLike)) else "use_metadata_tags"
        open_kwargs["options"] = {"movflags": movflags}
    return open_kwargs, format, codec


def set_video_color_properties(target, color_space):
    is_hdr = color_space in HDR_COLOR_TRANSFERS
    target.color_primaries = ColorPrimaries.BT2020 if is_hdr else ColorPrimaries.BT709
    target.color_trc = VIDEO_COLOR_TRANSFERS[color_space]
    target.colorspace = BT2020_NCL if is_hdr else BT709_NCL
    target.color_range = ColorRange.MPEG


def copy_color_properties(source, target):
    target.color_primaries = source.color_primaries
    target.color_trc = source.color_trc
    target.colorspace = source.colorspace
    target.color_range = source.color_range


def video_stream_color_space(stream) -> str | None:
    if stream is None:
        return None
    return VIDEO_TRANSFER_COLOR_SPACES.get(stream.color_trc)


def video_encoder_options(codec: VideoCodec, crf: float | None) -> dict[str, str]:
    if crf is None:
        return {}
    if codec == VideoCodec.AV1 and crf == 0:
        return {"svtav1-params": "lossless=1"}
    return {"crf": str(crf)}


def webm_streams_compatible(streams) -> bool:
    for stream in streams:
        allowed_codecs = WEBM_STREAM_CODECS.get(stream.type)
        if allowed_codecs is not None and stream.codec_context is not None and stream.codec.canonical_name not in allowed_codecs:
            return False
    return True


class VideoFromFile(VideoInput):
    """
    Class representing video input from a file.
    """

    def __init__(self, file: str | io.BytesIO, *, start_time: float=0, duration: float=0):
        """
        Initialize the VideoFromFile object based off of either a path on disk or a BytesIO object
        containing the file contents.
        """
        self.__file = file
        self.__start_time = start_time
        self.__duration = duration

    def get_stream_source(self) -> str | io.BytesIO:
        """
        Return the underlying file source for efficient streaming.
        This avoids unnecessary memory copies when the source is already a file path.
        """
        if isinstance(self.__file, io.BytesIO):
            self.__file.seek(0)
        return self.__file

    def get_active_trim_window(self) -> tuple[float, float]:
        start_time = self.__start_time
        if start_time < 0:
            start_time = max(self._get_raw_duration() + start_time, 0.0)
        return float(start_time), float(self.__duration)

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

    def get_bit_depth(self) -> int:
        if isinstance(self.__file, io.BytesIO):
            self.__file.seek(0)  # Reset the BytesIO object to the beginning
        with av.open(self.__file, mode="r") as container:
            video_stream = container.streams.video[0] if len(container.streams.video) > 0 else None
            return video_stream_bit_depth(video_stream)

    def get_color_space(self) -> str:
        if isinstance(self.__file, io.BytesIO):
            self.__file.seek(0)
        with av.open(self.__file, mode="r") as container:
            video_stream = container.streams.video[0] if len(container.streams.video) > 0 else None
            return video_stream_color_space(video_stream) or "sRGB"

    def get_duration(self) -> float:
        """
        Returns the duration of the video in seconds.

        Returns:
            Duration in seconds
        """
        raw_duration = self._get_raw_duration()
        if self.__start_time < 0:
            duration_from_start = min(raw_duration, -self.__start_time)
        else:
            duration_from_start = raw_duration - self.__start_time
        if self.__duration:
            return min(self.__duration, duration_from_start)
        return duration_from_start

    def _get_raw_duration(self) -> float:
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
                frame_iterator = (
                    container.decode(video_stream)
                    if video_stream.codec.capabilities & 0x100
                    else container.demux(video_stream)
                )
                for packet in frame_iterator:
                    frame_count += 1
                if frame_count > 0:
                    return float(frame_count / video_stream.average_rate)

        raise ValueError(f"Could not determine duration for file '{self.__file}'")

    def get_frame_count(self) -> int:
        """
        Returns the number of frames in the video without materializing them as
        torch tensors.
        """
        if isinstance(self.__file, io.BytesIO):
            self.__file.seek(0)

        with av.open(self.__file, mode="r") as container:
            video_stream = self._get_first_video_stream(container)
            # 1. Prefer the frames field if available and usable
            if (
                video_stream.frames
                and video_stream.frames > 0
                and not self.__start_time
                and not self.__duration
            ):
                return int(video_stream.frames)

            # 2. Try to estimate from duration and average_rate using only metadata
            if (
                getattr(video_stream, "duration", None) is not None
                and getattr(video_stream, "time_base", None) is not None
                and video_stream.average_rate
            ):
                raw_duration = float(video_stream.duration * video_stream.time_base)
                if self.__start_time < 0:
                    duration_from_start = min(raw_duration, -self.__start_time)
                else:
                    duration_from_start = raw_duration - self.__start_time
                duration_seconds = min(self.__duration, duration_from_start) if self.__duration else duration_from_start
                estimated_frames = int(round(duration_seconds * float(video_stream.average_rate)))
                if estimated_frames > 0:
                    return estimated_frames

            # 3. Last resort: decode frames and count them (streaming)
            start_time, duration = self.get_active_trim_window()
            frame_count = 1
            start_pts = int(start_time / video_stream.time_base)
            end_pts = int((start_time + duration) / video_stream.time_base) if duration else None
            container.seek(start_pts, stream=video_stream)
            frame_iterator = (
                container.decode(video_stream)
                if video_stream.codec.capabilities & 0x100
                else container.demux(video_stream)
            )
            for frame in frame_iterator:
                if frame.pts >= start_pts:
                    break
            else:
                raise ValueError(f"Could not determine frame count for file '{self.__file}'\nNo frames exist for start_time {self.__start_time}")
            for frame in frame_iterator:
                if end_pts is not None and frame.pts >= end_pts:
                    break
                frame_count += 1
            return frame_count

    def get_frame_rate(self) -> Fraction:
        """
        Returns the average frame rate of the video using container metadata
        without decoding all frames.
        """
        if isinstance(self.__file, io.BytesIO):
            self.__file.seek(0)

        with av.open(self.__file, mode="r") as container:
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
        if isinstance(self.__file, io.BytesIO):
            self.__file.seek(0)
        with av.open(self.__file, mode='r') as container:
            return container.format.name

    def get_components_internal(self, container: InputContainer) -> VideoComponents:
        video_stream = self._get_first_video_stream(container)
        start_time, duration = self.get_active_trim_window()

        # Get video frames
        frames = []
        audio_frames = []
        alphas = None
        start_pts = int(start_time / video_stream.time_base)
        end_pts = int((start_time + duration) / video_stream.time_base)

        if start_pts != 0:
            container.seek(start_pts, stream=video_stream)

        image_format = 'gbrpf32le'
        process_image_format = lambda a: a
        align_graph = None
        audio = None

        streams = [video_stream]
        has_first_audio_frame = False
        checked_alpha = False

        # Default to False so we decode until EOF if duration is 0
        video_done = False
        audio_done = True

        audio_stream = last_decodable_audio_stream(container)
        if audio_stream is not None:
            streams += [audio_stream]
            resampler = av.audio.resampler.AudioResampler(format='fltp')
            audio_done = False

        for packet in container.demux(*streams):
            if video_done and audio_done:
                break

            if packet.stream.type == "video":
                if video_done:
                    continue
                try:
                    for frame in packet.decode():
                        if frame.pts < start_pts:
                            continue
                        if duration and frame.pts >= end_pts:
                            video_done = True
                            break

                        if not checked_alpha:
                            alpha_channel = False
                            for comp in frame.format.components:
                                if comp.is_alpha or frame.format.name == "pal8":
                                    alphas = []
                                    alpha_channel = True
                                    break
                            if frame.format.name in ("yuvj420p", "yuvj422p", "yuvj444p", "rgb24", "rgba", "pal8"):
                                process_image_format = lambda a: a.float() / 255.0
                                if alpha_channel:
                                    image_format = 'rgba'
                                else:
                                    image_format = 'rgb24'
                            else:
                                process_image_format = lambda a: a
                                if alpha_channel:
                                    image_format = 'gbrapf32le'
                                else:
                                    image_format = 'gbrpf32le'

                            checked_alpha = True

                        # Fix non-deterministic video decode when the video width is not a multiple of 32
                        # For non-yuvj pixel formats: most H.264/H.265 video and static images (e.g. lossy WebP via LoadImage)
                        # Pad both axes to a multiple of 32 and smear the border so the alignment padding never bleeds into the cropped edges
                        if image_format in ('gbrpf32le', 'gbrapf32le') and frame.width % 32 != 0:
                            if align_graph is None:
                                pad_w = ((frame.width + 31) // 32) * 32
                                pad_h = ((frame.height + 31) // 32) * 32
                                g = av.filter.Graph()
                                g_src = g.add_buffer(width=frame.width, height=frame.height,
                                                     format=frame.format.name, time_base=video_stream.time_base)
                                g_pad = g.add('pad', f'{pad_w}:{pad_h}:0:0')
                                g_fill = g.add('fillborders', f'left=0:right={pad_w - frame.width}:top=0:bottom={pad_h - frame.height}:mode=smear')
                                g_sink = g.add('buffersink')
                                g_src.link_to(g_pad)
                                g_pad.link_to(g_fill)
                                g_fill.link_to(g_sink)
                                g.configure()
                                align_graph = (g, g_src, g_sink)
                            align_graph[1].push(frame)
                            img = np.ascontiguousarray(align_graph[2].pull().to_ndarray(format=image_format)[:frame.height, :frame.width])
                        else:
                            img = frame.to_ndarray(format=image_format)
                        if frame.rotation != 0:
                            k = int(round(frame.rotation // 90))
                            img = np.rot90(img, k=k, axes=(0, 1)).copy()
                        if alphas is None:
                            frames.append(torch.from_numpy(img))
                        else:
                            frames.append(torch.from_numpy(img[..., :-1]))
                            alphas.append(torch.from_numpy(img[..., -1:]))
                except av.error.InvalidDataError:
                    logging.info("pyav decode error")

            elif packet.stream.type == "audio":
                if audio_done:
                    continue

                aframes = itertools.chain.from_iterable(
                    map(resampler.resample, packet.decode())
                )
                for frame in aframes:
                    if duration and frame.time > start_time + duration:
                        audio_done = True
                        break

                    if not has_first_audio_frame:
                        offset_seconds = start_time - frame.pts * audio_stream.time_base
                        to_skip = max(0, int(offset_seconds * audio_stream.sample_rate))
                        if to_skip < frame.samples:
                            has_first_audio_frame = True
                            audio_frames.append(frame.to_ndarray()[..., to_skip:])
                    else:
                        audio_frames.append(frame.to_ndarray())

        images = process_image_format(torch.stack(frames)) if len(frames) > 0 else torch.zeros(0, 0, 0, 3)
        if alphas is not None:
            alphas = process_image_format(torch.stack(alphas)) if len(alphas) > 0 else torch.zeros(0, 0, 0, 1)

        # Get frame rate
        frame_rate = Fraction(video_stream.average_rate) if video_stream.average_rate else Fraction(1)

        if len(audio_frames) > 0:
            audio_data = np.concatenate(audio_frames, axis=1)  # shape: (channels, total_samples)
            if duration:
                audio_data = audio_data[..., :int(duration * audio_stream.sample_rate)]

            audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)  # shape: (1, channels, total_samples)
            audio = AudioInput({
                "waveform": audio_tensor,
                "sample_rate": int(audio_stream.sample_rate) if audio_stream.sample_rate else 1,
            })

        metadata = container.metadata
        return VideoComponents(images=images, alpha=alphas, audio=audio, frame_rate=frame_rate, metadata=metadata)

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
        metadata: Optional[dict] = None,
        bit_depth: int | None = None,
        crf: float | None = None,
        color_space: str | None = None,
    ):
        if color_space is not None and color_space not in VIDEO_COLOR_TRANSFERS:
            raise ValueError(f"Unsupported video color space: {color_space}")
        _, output_format, _ = video_output_config(path, format, codec)
        if isinstance(self.__file, io.BytesIO):
            self.__file.seek(0)  # Reset the BytesIO object to the beginning
        with av.open(self.__file, mode='r') as container:
            container_format = container.format.name
            video_stream = container.streams.video[0] if len(container.streams.video) > 0 else None
            video_encoding = video_stream.codec.canonical_name if video_stream is not None else None
            source_bit_depth = video_stream_bit_depth(video_stream)
            source_color_space = video_stream_color_space(video_stream)
            if source_color_space is not None and color_space is not None and source_color_space != color_space:
                raise ValueError(
                    f"Cannot save {source_color_space} video as {color_space} without color conversion; "
                    f"use auto or {source_color_space}"
                )
            reuse_streams = True
            if format != VideoContainer.AUTO and VIDEO_CONTAINER_FORMATS[VideoContainer(format)] not in container_format.split(","):
                reuse_streams = False
            if output_format == VideoContainer.WEBM and not webm_streams_compatible(container.streams):
                reuse_streams = False
            if codec != VideoCodec.AUTO and codec != video_encoding and video_encoding is not None:
                reuse_streams = False
            if bit_depth is not None and video_encoding is not None and bit_depth != source_bit_depth:
                reuse_streams = False
            if crf is not None:
                reuse_streams = False
            if color_space is not None:
                reuse_streams = False
            if self.__start_time or self.__duration:
                reuse_streams = False

            if not reuse_streams:
                if bit_depth is None:
                    bit_depth = source_bit_depth
                return self._save_transcoded(container, path, format=format, codec=codec, metadata=metadata, bit_depth=bit_depth, crf=crf, color_space=color_space)

            streams = container.streams

            open_kwargs = get_open_write_kwargs(path, container_format, format)
            with av.open(path, **open_kwargs) as output_container:
                # Add metadata before writing any streams
                write_output_metadata(container, output_container, metadata)

                # Add streams to the new container. Streams with no codec context cannot be used as an output template.
                stream_map = {}
                hevc_filters = {}
                for stream in streams:
                    if isinstance(stream, (av.VideoStream, av.AudioStream, SubtitleStream)):
                        if stream.codec_context is None:
                            logging.warning("Skipping %s stream %d with unsupported codec", stream.type, stream.index)
                            continue
                        out_stream = output_container.add_stream_from_template(template=stream, opaque=True)
                        hevc_filter = isobmff_hevc_filter(output_container, stream, out_stream)
                        if hevc_filter is not None:
                            hevc_filters[stream] = hevc_filter
                        stream_map[stream] = out_stream

                # Write packets to the new container
                for packet in container.demux():
                    if packet.stream in stream_map and packet.dts is not None:
                        out_stream = stream_map[packet.stream]
                        hevc_filter = hevc_filters.get(packet.stream)
                        for out_packet in filter_hevc_packet(hevc_filter, packet) if hevc_filter else (packet,):
                            out_packet.stream = out_stream
                            output_container.mux(out_packet)

    def _save_transcoded(
        self,
        container: InputContainer,
        path: str | io.BytesIO,
        format: VideoContainer,
        codec: VideoCodec,
        metadata: dict | None,
        bit_depth: int,
        crf: float | None = None,
        color_space: str | None = None,
    ):
        """Re-encode one frame at a time; peak memory does not scale with video length."""
        open_kwargs, output_format, output_codec = video_output_config(path, format, codec)
        video_stream = self._get_first_video_stream(container)
        start_time, duration = self.get_active_trim_window()
        start_pts = int(start_time / video_stream.time_base)
        end_pts = int((start_time + duration) / video_stream.time_base) if duration else None
        stream_end_pts = None
        if video_stream.duration is not None:
            stream_end_pts = (video_stream.start_time or 0) + video_stream.duration
        output_end_pts = end_pts
        if stream_end_pts is not None and (output_end_pts is None or stream_end_pts < output_end_pts):
            output_end_pts = stream_end_pts
        if start_pts != 0:
            container.seek(start_pts, stream=video_stream)

        audio_stream = last_decodable_audio_stream(container)
        source_color_space = video_stream_color_space(video_stream)
        preserve_source_color = source_color_space is not None
        pix_fmt = "yuv420p10le" if bit_depth >= 10 else "yuv420p"
        rate = Fraction(video_stream.average_rate) if video_stream.average_rate else Fraction(1)

        resampler = None
        sample_rate = 0
        audio_time_base = None
        duration_cap = None
        if audio_stream is not None:
            sample_rate = audio_stream.codec_context.sample_rate
            channels = audio_stream.codec_context.channels
            if not sample_rate:
                sample_rate, channels = probe_audio_params(container, audio_stream)
                container.seek(start_pts, stream=video_stream)
                if sample_rate:
                    audio_stream.codec_context.flush_buffers()
                else:
                    logging.warning("Audio stream parameters could not be determined; ignoring audio.")
                    audio_stream = None
        if audio_stream is not None:
            if output_format == VideoContainer.WEBM:
                sample_rate = 48000
            audio_time_base = Fraction(1, sample_rate)
            layout = {1: "mono", 2: "stereo", 6: "5.1"}.get(channels, "stereo")
            resampler = av.audio.resampler.AudioResampler(format="fltp", layout=layout, rate=sample_rate)
            if duration:
                duration_cap = math.ceil(duration * sample_rate)

        streams = [video_stream] if audio_stream is None else [video_stream, audio_stream]
        pts_step = max(1, int(round((1 / rate) / video_stream.time_base)))
        video_done = False
        audio_done = audio_stream is None
        video_pts_offset = None
        last_video_pts = None
        last_video_end = None
        # rebased pts -> true display duration: the mp4 muxer pads the last sample with 1/rate otherwise
        video_frame_durations = {}
        source_size = None
        rotation_k = 0
        rotation_filter = None
        audio_started = False
        samples_written = 0
        pending_audio = []
        # The output opens lazily on the first kept frame: it decides the geometry (90/270 rotation swaps dims),
        # and never seeking back keeps webm/mkv leading audio intact.
        output = None
        out_video = None
        out_audio = None

        def audio_frame_from_ndarray(nd_planar):
            frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(nd_planar), format="fltp", layout=layout)
            frame.sample_rate = sample_rate
            return frame

        def drain_audio(final=False):
            # Audio may cover the pts span of the video written so far, capped by the requested duration
            nonlocal samples_written, audio_done
            if last_video_end is None:
                cap = 0
            else:
                cap = math.ceil(last_video_end * video_stream.time_base * sample_rate)
            if duration_cap is not None:
                cap = min(cap, duration_cap)
            while pending_audio and not audio_done:
                frame = pending_audio[0]
                if samples_written + frame.samples <= cap:
                    frame.pts = samples_written
                    frame.time_base = audio_time_base
                    output.mux(out_audio.encode(frame))
                    samples_written += frame.samples
                    pending_audio.pop(0)
                    continue
                if final:
                    keep = frame.to_ndarray()[..., :cap - samples_written]
                    if keep.shape[-1] > 0:
                        tail = audio_frame_from_ndarray(keep)
                        tail.pts = samples_written
                        tail.time_base = audio_time_base
                        output.mux(out_audio.encode(tail))
                        samples_written += keep.shape[-1]
                    pending_audio.clear()
                break
            if duration_cap is not None and samples_written >= duration_cap:
                audio_done = True
            return cap

        try:
            for packet in container.demux(*streams):
                if video_done and audio_done:
                    break

                if packet.stream == video_stream and not video_done:
                    try:
                        frames = packet.decode()
                    except av.error.InvalidDataError:
                        logging.info("pyav decode error")
                        continue
                    for frame in frames:
                        if frame.pts is not None and frame.pts < start_pts:
                            continue
                        if end_pts is not None and frame.pts is not None and frame.pts >= end_pts:
                            video_done = True
                            if last_video_pts is not None:
                                # the source continues past the window: hold the last kept frame to the window end
                                end_offset = video_pts_offset if video_pts_offset is not None else start_pts
                                last_video_end = max(last_video_end, end_pts - end_offset)
                            break
                        # the source's true display duration of this frame; average_rate is not a
                        # frame duration (sparse/VFR sources), so it is only the fallback
                        frame_duration = frame.duration if frame.duration else pts_step
                        if end_pts is not None and frame.pts is not None:
                            frame_duration = min(frame_duration, end_pts - frame.pts)
                        if output is None:
                            rotation_k = int(round(frame.rotation // 90)) % 4 if frame.rotation else 0
                            if rotation_k % 2:
                                out_width, out_height = frame.height, frame.width
                            else:
                                out_width, out_height = frame.width, frame.height
                            if out_width % 2 or out_height % 2:
                                raise ValueError(f"{output_codec.value.upper()} output requires even dimensions, got {out_width}x{out_height}")
                            source_size = (frame.width, frame.height)
                            output = av.open(path, **open_kwargs)
                            # Add metadata before writing any streams
                            write_output_metadata(container, output, metadata)
                            out_video = output.add_stream(VIDEO_ENCODERS[output_codec], rate=rate)
                            # no B-frames: reordering makes mp4 sample durations follow decode order,
                            # so irregular-VFR spans and trim windows land wrong
                            out_video.codec_context.max_b_frames = 0
                            out_video.width = out_width
                            out_video.height = out_height
                            out_video.pix_fmt = pix_fmt
                            out_video.options = video_encoder_options(output_codec, crf)
                            if preserve_source_color:
                                copy_color_properties(video_stream, out_video.codec_context)
                            elif color_space is not None:
                                set_video_color_properties(out_video.codec_context, color_space)
                            # source pts pass through (rebased to 0), so variable frame rate survives
                            out_video.codec_context.time_base = video_stream.time_base
                            if audio_stream is not None:
                                audio_codec = "libopus" if output_format == VideoContainer.WEBM else "aac"
                                out_audio = output.add_stream(audio_codec, rate=sample_rate, layout=layout)
                        if (frame.width, frame.height) != source_size:
                            # encoding would silently rescale the new geometry into the old one
                            raise ValueError(
                                f"Video resolution changes mid-stream "
                                f"({source_size[0]}x{source_size[1]} -> {frame.width}x{frame.height}); cannot transcode"
                            )
                        if rotation_k:
                            if rotation_filter is None:
                                g = av.filter.Graph()
                                g_src = g.add_buffer(width=frame.width, height=frame.height,
                                                     format=frame.format.name, time_base=video_stream.time_base)
                                tail = g_src
                                for filter_name, filter_args in {1: [("transpose", "cclock")],
                                                                 2: [("hflip", None), ("vflip", None)],
                                                                 3: [("transpose", "clock")]}[rotation_k]:
                                    step = g.add(filter_name, filter_args)
                                    tail.link_to(step)
                                    tail = step
                                g_sink = g.add("buffersink")
                                tail.link_to(g_sink)
                                g.configure()
                                rotation_filter = (g_src, g_sink)
                            rotation_filter[0].push(frame)
                            frame = rotation_filter[1].pull()
                        if frame.color_range == ColorRange.JPEG and not preserve_source_color:
                            # compress full-range sources (yuvj/MJPEG) to limited range
                            frame = frame.reformat(format=pix_fmt, src_color_range="JPEG", dst_color_range="MPEG")
                        else:
                            frame = frame.reformat(format=pix_fmt)
                        if preserve_source_color:
                            copy_color_properties(video_stream, frame)
                        elif color_space is not None:
                            set_video_color_properties(frame, color_space)
                        frame_output_end = None
                        if frame.pts is not None:
                            if video_pts_offset is None:
                                video_pts_offset = frame.pts
                            frame.pts -= video_pts_offset
                            if output_end_pts is not None:
                                frame_output_end = output_end_pts - video_pts_offset
                                if frame.pts + frame_duration > frame_output_end:
                                    clamped_pts = frame_output_end - frame_duration
                                    if clamped_pts >= 0 and (last_video_pts is None or clamped_pts > last_video_pts):
                                        frame.pts = min(frame.pts, clamped_pts)
                                    elif frame.pts < frame_output_end:
                                        frame_duration = frame_output_end - frame.pts
                                    else:
                                        continue
                        if frame.pts is None or (last_video_pts is not None and frame.pts <= last_video_pts):
                            # broken sources emit missing/backward timestamps mid-stream, which the
                            # muxer rejects; nudge them forward by one nominal frame interval
                            frame.pts = 0 if last_video_pts is None else last_video_pts + pts_step
                            if frame_output_end is not None and frame.pts + frame_duration > frame_output_end:
                                if frame.pts >= frame_output_end:
                                    continue
                                frame_duration = frame_output_end - frame.pts
                        last_video_pts = frame.pts
                        last_video_end = frame.pts + frame_duration
                        video_frame_durations[frame.pts] = frame_duration
                        # the decoded pict_type would force x264's frame types (intra-only
                        # sources like MJPEG/ProRes would come out all-keyframe)
                        frame.pict_type = 0
                        for out_packet in out_video.encode(frame):
                            out_packet.duration = video_frame_durations.pop(out_packet.pts, 0)
                            output.mux(out_packet)
                        drain_audio()

                elif packet.stream == audio_stream and not audio_done:
                    for resampled in itertools.chain.from_iterable(map(resampler.resample, packet.decode())):
                        frame_start = None
                        if resampled.pts is not None:
                            # passthrough frames keep the source stream's time base
                            tb = resampled.time_base if resampled.time_base else audio_time_base
                            frame_start = float(resampled.pts * tb)
                            if duration and not audio_started and frame_start >= start_time + duration:
                                audio_done = True
                                break
                        if not audio_started:
                            if frame_start is None:
                                frame_start = 0.0
                            to_skip = max(0, int((start_time - frame_start) * sample_rate))
                            if to_skip >= resampled.samples:
                                continue
                            audio_started = True
                            if duration and frame_start > start_time:
                                duration_cap = min(duration_cap, math.ceil((start_time + duration - frame_start) * sample_rate))
                            if to_skip:
                                pending_audio.append(audio_frame_from_ndarray(resampled.to_ndarray()[..., to_skip:]))
                                continue
                        pending_audio.append(resampled)
                        if video_done:
                            # the video window is complete so the cap is final, but containers
                            # that interleave audio behind video (fragmented mp4) still owe most
                            # of it: stop only once the demuxed audio covers the cap
                            cap = drain_audio()
                            if pending_audio or samples_written >= cap:
                                drain_audio(final=True)
                                audio_done = True
                                break

            if output is None:
                raise ValueError(f"No decodable video frames found in file '{self.__file}'")
            if out_audio is not None and not audio_done:
                drain_audio(final=True)
            window_fill = last_video_end - last_video_pts if video_done and last_video_pts is not None else 0
            for out_packet in out_video.encode(None):
                duration = video_frame_durations.pop(out_packet.pts, 0)
                if out_packet.pts == last_video_pts:
                    duration = max(duration, window_fill)
                out_packet.duration = duration
                output.mux(out_packet)
            if out_audio is not None:
                output.mux(out_audio.encode(None))
        except BaseException:
            if output is not None:
                output.close()
                if isinstance(path, (str, os.PathLike)) and os.path.exists(path):
                    os.remove(path)
            raise
        else:
            if output is not None:
                output.close()

    def _get_first_video_stream(self, container: InputContainer):
        if len(container.streams.video):
            return container.streams.video[0]
        raise ValueError(f"No video stream found in file '{self.__file}'")

    def as_trimmed(
        self, start_time: float = 0, duration: float = 0, strict_duration: bool = True
    ) -> VideoInput | None:
        trimmed = VideoFromFile(
            self.get_stream_source(),
            start_time=start_time + self.__start_time,
            duration=duration,
        )
        if trimmed.get_duration() < duration and strict_duration:
            return None
        return trimmed


class VideoFromComponents(VideoInput):
    """
    Class representing video input from tensors.
    """

    def __init__(self, components: VideoComponents, bit_depth: int = 8, color_space: str = "sRGB"):
        if color_space not in VIDEO_COLOR_TRANSFERS:
            raise ValueError(f"Unsupported video color space: {color_space}")
        self.__components = components
        # Tensor components have no inherent bit depth; this is the depth used when encoding.
        self.__bit_depth = bit_depth
        self.__color_space = color_space

    def get_components(self) -> VideoComponents:
        return VideoComponents(
            images=self.__components.images,
            audio=self.__components.audio,
            frame_rate=self.__components.frame_rate,
        )

    def get_bit_depth(self) -> int:
        return self.__bit_depth

    def get_color_space(self) -> str:
        return self.__color_space

    def save_to(
        self,
        path: str,
        format: VideoContainer = VideoContainer.AUTO,
        codec: VideoCodec = VideoCodec.AUTO,
        metadata: Optional[dict] = None,
        bit_depth: int | None = None,
        crf: float | None = None,
        color_space: str | None = None,
    ):
        """Save the video to a file path or BytesIO buffer."""
        if color_space is None:
            color_space = self.__color_space
        if color_space is not None and color_space not in VIDEO_COLOR_TRANSFERS:
            raise ValueError(f"Unsupported video color space: {color_space}")
        open_kwargs, output_format, output_codec = video_output_config(path, format, codec)
        # None means "use the depth this video was created with" (CreateVideo's choice).
        if bit_depth is None:
            bit_depth = self.__bit_depth
        is_10bit = bit_depth >= 10
        with av.open(path, **open_kwargs) as output:
            # Add metadata before writing any streams
            if metadata is not None:
                for key, value in metadata.items():
                    output.metadata[key] = json.dumps(value)

            frame_rate = Fraction(round(self.__components.frame_rate * 1000), 1000)
            # Create a video stream
            pix_fmt = "yuv420p10le" if is_10bit else "yuv420p"
            video_stream = output.add_stream(VIDEO_ENCODERS[output_codec], rate=frame_rate)
            video_stream.width = self.__components.images.shape[2]
            video_stream.height = self.__components.images.shape[1]
            video_stream.pix_fmt = pix_fmt
            video_stream.options = video_encoder_options(output_codec, crf)
            if color_space is not None:
                set_video_color_properties(video_stream.codec_context, color_space)

            # Create an audio stream
            audio_sample_rate = 1
            audio_resampler = None
            audio_stream: Optional[av.AudioStream] = None
            if self.__components.audio:
                source_audio_sample_rate = int(self.__components.audio['sample_rate'])
                audio_sample_rate = 48000 if output_format == VideoContainer.WEBM else source_audio_sample_rate
                waveform = self.__components.audio['waveform']
                waveform = waveform[0, :, :math.ceil((source_audio_sample_rate / frame_rate) * self.__components.images.shape[0])]
                layout = {1: 'mono', 2: 'stereo', 6: '5.1'}.get(waveform.shape[0], 'stereo')
                audio_codec = "libopus" if output_format == VideoContainer.WEBM else "aac"
                audio_stream = output.add_stream(audio_codec, rate=audio_sample_rate, layout=layout)
                if audio_sample_rate != source_audio_sample_rate:
                    audio_resampler = av.audio.resampler.AudioResampler(format="fltp", layout=layout, rate=audio_sample_rate)

            # Encode video
            for i, frame in enumerate(self.__components.images):
                if is_10bit:
                    # 16-bit RGB keeps float precision through the conversion to 10-bit YUV.
                    img = (frame.float() * 65535).clamp(0, 65535).cpu().numpy().astype(np.uint16)  # shape: (H, W, 3)
                    frame = av.VideoFrame.from_ndarray(img, format="rgb48le")
                else:
                    img = (frame * 255).clamp(0, 255).byte().cpu().numpy() # shape: (H, W, 3)
                    frame = av.VideoFrame.from_ndarray(img, format='rgb24')
                dst_colorspace = None
                if color_space == "sRGB":
                    dst_colorspace = BT709_NCL
                elif color_space in HDR_COLOR_TRANSFERS:
                    dst_colorspace = BT2020_NCL
                frame = frame.reformat(format=pix_fmt, dst_colorspace=dst_colorspace)
                if color_space is not None:
                    set_video_color_properties(frame, color_space)
                packet = video_stream.encode(frame)
                output.mux(packet)

            # Flush video
            packet = video_stream.encode(None)
            output.mux(packet)

            if audio_stream and self.__components.audio:
                frame = av.AudioFrame.from_ndarray(waveform.float().cpu().contiguous().numpy(), format='fltp', layout=layout)
                frame.sample_rate = source_audio_sample_rate
                frame.pts = 0
                frames = [frame] if audio_resampler is None else audio_resampler.resample(frame)
                for frame in frames:
                    output.mux(audio_stream.encode(frame))
                if audio_resampler is not None:
                    for frame in audio_resampler.resample(None):
                        output.mux(audio_stream.encode(frame))

                # Flush encoder
                output.mux(audio_stream.encode(None))

    def as_trimmed(
        self,
        start_time: float | None = None,
        duration: float | None = None,
        strict_duration: bool = True,
    ) -> VideoInput | None:
        if self.get_duration() < start_time + duration:
            return None
        #TODO Consider tracking duration and trimming at time of save?
        return VideoFromFile(self.get_stream_source(), start_time=start_time, duration=duration)


def _fit_images(images: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Scale a (N, H, W, C) batch to fit inside width x height, keeping the aspect ratio, centered on black."""
    src_h, src_w = images.shape[1], images.shape[2]
    scale = min(width / src_w, height / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    samples = images.movedim(-1, 1)
    if (new_w, new_h) != (src_w, src_h):
        samples = torch.nn.functional.interpolate(samples, size=(new_h, new_w), mode="bilinear", align_corners=False)
    left = (width - new_w) // 2
    top = (height - new_h) // 2
    samples = torch.nn.functional.pad(samples, (left, width - new_w - left, top, height - new_h - top))
    return samples.movedim(1, -1)


def _match_audio_channels(waveform: torch.Tensor, channels: int) -> torch.Tensor:
    have = waveform.shape[1]
    if have == channels:
        return waveform
    if channels == 1:
        return waveform.mean(dim=1, keepdim=True)
    if have == 1:
        return waveform.repeat(1, channels, 1)
    if have > channels:
        return waveform[:, :channels]
    return torch.nn.functional.pad(waveform, (0, 0, 0, channels - have))


def _display_dimensions(video: VideoInput) -> tuple[int, int]:
    """(width, height) as displayed. VideoFromFile.get_dimensions() reports the stored frame size,
    which ignores a 90/270-degree display rotation that transcoding bakes in."""
    if not isinstance(video, VideoFromFile):
        return video.get_dimensions()
    with av.open(video.get_stream_source(), mode="r") as container:
        if not len(container.streams.video):
            raise ValueError("No video stream found in video")
        stream = container.streams.video[0]
        width, height = stream.width, stream.height
        try:
            for frame in container.decode(stream):
                if frame.rotation and (int(round(frame.rotation // 90)) % 4) % 2:
                    width, height = height, width
                break
        except av.error.FFmpegError:
            pass
    return width, height


_TRIM_EPSILON = 1e-6


def _segment_filter_graph(frame, time_base: Fraction, rotation_k: int, target_size: tuple[int, int] | None):
    """Bake the display rotation and, when the rotated frame differs from target_size, scale it to fit
    inside the target (aspect preserved) and pad the remainder with black. Returns (graph, source, sink) or None;
    the graph must stay referenced while its contexts are in use."""
    steps = {
        0: [],
        1: [("transpose", "cclock")],
        2: [("hflip", None), ("vflip", None)],
        3: [("transpose", "clock")],
    }[rotation_k]
    rotated = (frame.height, frame.width) if rotation_k % 2 else (frame.width, frame.height)
    if target_size is not None and rotated != target_size:
        width, height = target_size
        steps = steps + [
            ("scale", f"{width}:{height}:force_original_aspect_ratio=decrease:force_divisible_by=2"),
            ("pad", f"{width}:{height}:(ow-iw)/2:(oh-ih)/2"),
        ]
    if not steps:
        return None
    graph = av.filter.Graph()
    source = graph.add_buffer(width=frame.width, height=frame.height, format=frame.format.name, time_base=time_base)
    tail = source
    for filter_name, filter_args in steps:
        step = graph.add(filter_name, filter_args)
        tail.link_to(step)
        tail = step
    sink = graph.add("buffersink")
    tail.link_to(sink)
    graph.configure()
    return graph, source, sink


@dataclass
class _ConcatSegment:
    """Header-level facts about one concatenation source, read once before encoding."""
    time_base: Fraction
    rate: Fraction
    bit_depth: int
    color_space: str | None
    color_props: SimpleNamespace
    audio_index: int | None
    sample_rate: int
    channels: int


class _ConcatEncoder:
    """Output side of ``VideoConcatenated.save_to``: one encoder that every source segment feeds in turn.

    This mirrors the writer half of ``VideoFromFile._save_transcoded`` (lazy output open, pts
    rebasing/clamping, audio drain) plus a per-segment base offset and silence padding so the audio
    track stays gapless across segment boundaries. Keep the two in sync.

    Every frame pushed here must already be rebased to its segment start, expressed in ``time_base``
    units *and* stamped ``frame.time_base = time_base``: PyAV rescales ``frame.pts`` from
    ``frame.time_base`` to the codec time base on encode.

    The audio track must be gapless: PyAV does not reject pts gaps, it silently lets every later
    segment's audio run ahead of its video.
    """

    def __init__(
        self,
        path: str | io.BytesIO,
        open_kwargs: dict,
        output_format: VideoContainer,
        output_codec: VideoCodec,
        metadata: dict | None,
        *,
        time_base: Fraction,
        rate: Fraction,
        pix_fmt: str,
        crf: float | None,
        color_space: str | None,
        color_props: SimpleNamespace | None,
        audio: tuple[int, str, int] | None,
    ):
        self.path = path
        self.open_kwargs = open_kwargs
        self.output_format = output_format
        self.output_codec = output_codec
        self.metadata = metadata
        self.time_base = time_base
        self.rate = rate
        self.pix_fmt = pix_fmt
        self.crf = crf
        self.color_space = color_space
        self.color_props = color_props
        # (sample_rate, layout, channels) of the output audio stream, or None for no audio
        self.audio = audio
        self.audio_time_base = Fraction(1, audio[0]) if audio is not None else None

        self.output = None
        self.out_video = None
        self.out_audio = None
        self.output_size: tuple[int, int] | None = None
        self.last_video_pts = None
        self.last_video_end = None
        # rebased pts -> true display duration: the mp4 muxer pads the last sample with 1/rate otherwise
        self.video_frame_durations = {}
        # one packet is held back so its duration can still be fixed up at a segment boundary
        self.held_packet = None
        self.samples_written = 0
        self.pending_audio = []

        self.seg_base = 0
        self.seg_end = None
        self.seg_video_done = False
        self.seg_audio_done = True
        self.duration_cap = None
        self.pts_step = 1

    @property
    def segment_done(self) -> bool:
        return self.seg_video_done and self.seg_audio_done

    def _samples_at(self, pts: int) -> int:
        return math.ceil(pts * self.time_base * self.audio[0])

    def audio_frame_from_ndarray(self, nd_planar):
        frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(nd_planar), format="fltp", layout=self.audio[1])
        frame.sample_rate = self.audio[0]
        return frame

    def begin_segment(self, *, pts_step: int, audio_cap_samples: int | None, has_audio: bool):
        if self.out_audio is not None and not self.seg_audio_done:
            self._drain_audio(final=True)
        # a finished segment never leaves audio behind for the next one
        self.pending_audio.clear()
        self.seg_base = self.last_video_end or 0
        self.seg_end = None
        self.seg_video_done = False
        self.pts_step = pts_step
        self.duration_cap = None
        if self.audio is None:
            self.seg_audio_done = True
            return
        target = self._samples_at(self.seg_base)
        if self.out_audio is not None:
            self._pad_silence(target - self.samples_written)
        if audio_cap_samples is not None:
            self.duration_cap = target + audio_cap_samples
        self.seg_audio_done = not has_audio

    def _open(self, frame):
        if frame.width % 2 or frame.height % 2:
            raise ValueError(f"{self.output_codec.value.upper()} output requires even dimensions, got {frame.width}x{frame.height}")
        self.output_size = (frame.width, frame.height)
        self.output = av.open(self.path, **self.open_kwargs)
        # Add metadata before writing any streams
        if self.metadata is not None:
            for key, value in self.metadata.items():
                self.output.metadata[key] = value if isinstance(value, str) else json.dumps(value)
        self.out_video = self.output.add_stream(VIDEO_ENCODERS[self.output_codec], rate=self.rate)
        # no B-frames: reordering makes mp4 sample durations follow decode order,
        # so irregular-VFR spans and segment boundaries land wrong
        self.out_video.codec_context.max_b_frames = 0
        self.out_video.width = frame.width
        self.out_video.height = frame.height
        self.out_video.pix_fmt = self.pix_fmt
        self.out_video.options = video_encoder_options(self.output_codec, self.crf)
        if self.color_props is not None:
            copy_color_properties(self.color_props, self.out_video.codec_context)
        elif self.color_space is not None:
            set_video_color_properties(self.out_video.codec_context, self.color_space)
        # all segments are rescaled into this time base, so variable frame rate survives
        self.out_video.codec_context.time_base = self.time_base
        if self.audio is not None:
            sample_rate, layout, _ = self.audio
            audio_codec = "libopus" if self.output_format == VideoContainer.WEBM else "aac"
            self.out_audio = self.output.add_stream(audio_codec, rate=sample_rate, layout=layout)

    def push_video(self, frame, frame_duration: int, segment_end: int | None):
        if self.seg_end is None and segment_end is not None:
            self.seg_end = self.seg_base + segment_end
        if self.output is None:
            self._open(frame)
        end = self.seg_end
        if frame.pts is not None:
            frame.pts += self.seg_base
            if end is not None and frame.pts + frame_duration > end:
                clamped_pts = end - frame_duration
                if clamped_pts >= self.seg_base and (self.last_video_pts is None or clamped_pts > self.last_video_pts):
                    frame.pts = min(frame.pts, clamped_pts)
                elif frame.pts < end:
                    frame_duration = end - frame.pts
                else:
                    return
        if frame.pts is None or (self.last_video_pts is not None and frame.pts <= self.last_video_pts):
            # broken sources emit missing/backward timestamps mid-stream, which the
            # muxer rejects; nudge them forward by one nominal frame interval
            frame.pts = self.seg_base if self.last_video_pts is None else max(self.seg_base, self.last_video_pts + self.pts_step)
            if end is not None and frame.pts + frame_duration > end:
                if frame.pts >= end:
                    return
                frame_duration = end - frame.pts
        self.last_video_pts = frame.pts
        self.last_video_end = frame.pts + frame_duration
        self.video_frame_durations[frame.pts] = frame_duration
        # the decoded pict_type would force x264's frame types (intra-only
        # sources like MJPEG/ProRes would come out all-keyframe)
        frame.pict_type = 0
        for packet in self.out_video.encode(frame):
            self._emit(packet)
        self._drain_audio()

    def _emit(self, packet):
        packet.duration = self.video_frame_durations.pop(packet.pts, 0)
        held = self.held_packet
        if held is not None:
            if not held.duration and held.pts is not None and packet.pts is not None and packet.pts > held.pts:
                held.duration = packet.pts - held.pts
            self.output.mux(held)
        self.held_packet = packet

    def _mux_held(self):
        held = self.held_packet
        if held is None:
            return
        if held.pts == self.last_video_pts and self.last_video_end is not None:
            held.duration = max(held.duration or 0, self.last_video_end - held.pts)
        self.output.mux(held)
        self.held_packet = None

    def end_video(self, segment_end: int | None):
        """The segment's video is complete; ``segment_end`` (segment-relative) is set when the source
        continues past its trim window, so the last kept frame is held to the window end."""
        self.seg_video_done = True
        if segment_end is not None and self.last_video_pts is not None:
            self.last_video_end = max(self.last_video_end, self.seg_base + segment_end)

    def end_audio(self):
        if not self.seg_audio_done:
            self._drain_audio(final=True)
            self.seg_audio_done = True

    def push_audio(self, frame) -> bool:
        """Queue a resampled audio frame; returns True once the segment's audio is complete."""
        if self.seg_audio_done:
            return True
        self.pending_audio.append(frame)
        if self.seg_video_done:
            # the video window is complete so the cap is final, but containers
            # that interleave audio behind video (fragmented mp4) still owe most
            # of it: stop only once the demuxed audio covers the cap
            cap = self._drain_audio()
            if self.pending_audio or self.samples_written >= cap:
                self._drain_audio(final=True)
                self.seg_audio_done = True
        return self.seg_audio_done

    def push_audio_cap(self, samples: int):
        cap = self._samples_at(self.seg_base) + samples
        self.duration_cap = cap if self.duration_cap is None else min(self.duration_cap, cap)

    def _drain_audio(self, final=False):
        # Audio may cover the pts span of the video written so far, capped by the segment's window
        if self.audio is None:
            return 0
        if self.last_video_end is None:
            cap = 0
        else:
            cap = self._samples_at(self.last_video_end)
        if self.duration_cap is not None:
            cap = min(cap, self.duration_cap)
        while self.pending_audio and not self.seg_audio_done:
            frame = self.pending_audio[0]
            if self.samples_written + frame.samples <= cap:
                frame.pts = self.samples_written
                frame.time_base = self.audio_time_base
                self.output.mux(self.out_audio.encode(frame))
                self.samples_written += frame.samples
                self.pending_audio.pop(0)
                continue
            if final:
                keep = frame.to_ndarray()[..., :cap - self.samples_written]
                if keep.shape[-1] > 0:
                    tail = self.audio_frame_from_ndarray(keep)
                    tail.pts = self.samples_written
                    tail.time_base = self.audio_time_base
                    self.output.mux(self.out_audio.encode(tail))
                    self.samples_written += keep.shape[-1]
                self.pending_audio.clear()
            break
        if self.duration_cap is not None and self.samples_written >= self.duration_cap:
            # the window is full: whatever is still queued belongs past this segment's end
            self.seg_audio_done = True
            self.pending_audio.clear()
        return cap

    def _pad_silence(self, samples: int):
        channels = self.audio[2]
        while samples > 0:
            chunk = min(samples, 1024)
            frame = self.audio_frame_from_ndarray(np.zeros((channels, chunk), dtype=np.float32))
            frame.pts = self.samples_written
            frame.time_base = self.audio_time_base
            self.output.mux(self.out_audio.encode(frame))
            self.samples_written += chunk
            samples -= chunk

    def end_segment(self):
        """Hold the segment's last frame to the segment end (its window end for trimmed sources)."""
        if self.last_video_pts is None:
            return
        hold = self.last_video_end - self.last_video_pts
        held = self.held_packet
        if held is not None and held.pts == self.last_video_pts:
            held.duration = max(held.duration or 0, hold)
        elif self.last_video_pts in self.video_frame_durations:
            self.video_frame_durations[self.last_video_pts] = max(self.video_frame_durations[self.last_video_pts], hold)

    def finish(self):
        if self.output is None:
            raise ValueError("No decodable video frames found in concatenated video")
        if self.out_audio is not None:
            if not self.seg_audio_done:
                self._drain_audio(final=True)
            # a final clip without (or with short) audio still needs the track to reach the video end
            self._pad_silence(self._samples_at(self.last_video_end) - self.samples_written)
        for packet in self.out_video.encode(None):
            self._emit(packet)
        self._mux_held()
        if self.out_audio is not None:
            self.output.mux(self.out_audio.encode(None))
        self.output.close()
        self.output = None

    def abort(self):
        if self.output is None:
            return
        self.output.close()
        self.output = None
        if isinstance(self.path, (str, os.PathLike)) and os.path.exists(self.path):
            os.remove(self.path)


class VideoConcatenated(VideoInput):
    """
    Lazy back-to-back concatenation of VideoInputs, in order. The sources are decoded and encoded
    in a single streaming pass when the video is saved, so peak memory does not scale with length.
    Saving keeps each clip's own frame timing (variable frame rate); get_components() returns a
    constant-frame-rate tensor at the first clip's rate, retiming the others by repeating/dropping frames.
    """

    def __init__(self, sources: Sequence[VideoInput], *, resize: str = "fit"):
        """
        resize: "fit" scales later clips to fit inside the first clip's frame (aspect preserved,
        black bars); "error" raises when the resolutions differ.
        """
        flat = []
        for source in sources:
            # a nested concatenation with the same resize mode is just more sources; a different
            # mode stays opaque so its own canvas rule is honored
            if isinstance(source, VideoConcatenated) and source._resize == resize:
                flat.extend(source._sources)
            else:
                flat.append(source)
        if not flat:
            raise ValueError("VideoConcatenated needs at least one source video")
        if resize not in ("fit", "error"):
            raise ValueError(f"Unsupported resize mode: {resize}")
        # single underscore on purpose: nested concatenations read each other's sources
        self._sources = flat
        self._resize = resize
        self._stream_cache: io.BytesIO | None = None
        self._dimensions: tuple[int, int] | None = None

        color_spaces = {source.get_color_space() for source in flat} - {"auto"}
        if len(color_spaces) > 1:
            raise ValueError(
                f"Cannot concatenate videos with different color spaces ({', '.join(sorted(color_spaces))}) without color conversion"
            )
        if resize == "error":
            dimensions = [_display_dimensions(source) for source in flat]
            self._dimensions = dimensions[0]
            for index, dims in enumerate(dimensions):
                if dims != dimensions[0]:
                    raise ValueError(
                        f"video {index} is {dims[0]}x{dims[1]}; expected {dimensions[0][0]}x{dimensions[0][1]} (resize is set to error)"
                    )

    def get_dimensions(self) -> tuple[int, int]:
        # the first clip defines the canvas, as displayed (rotation baked in)
        if self._dimensions is None:
            self._dimensions = _display_dimensions(self._sources[0])
        return self._dimensions

    def get_frame_rate(self) -> Fraction:
        return self._sources[0].get_frame_rate()

    def get_duration(self) -> float:
        return float(sum(source.get_duration() for source in self._sources))

    def get_frame_count(self) -> int:
        # the constant-frame-rate view at the first clip's rate, so that
        # frame_count / get_frame_rate() == get_duration() and it matches get_components()
        rate = self.get_frame_rate()
        total = 0
        for source in self._sources:
            count = source.get_frame_count()
            if count <= 0:
                continue
            source_rate = source.get_frame_rate()
            total += count if source_rate == rate else max(1, int(round(Fraction(count) * rate / source_rate)))
        return int(total)

    def get_bit_depth(self) -> int:
        return max(source.get_bit_depth() for source in self._sources)

    def get_color_space(self) -> str:
        for source in self._sources:
            color_space = source.get_color_space()
            if color_space != "auto":
                return color_space
        return "sRGB"

    def get_container_format(self) -> str:
        # what get_stream_source() materializes
        return "mp4"

    def get_stream_source(self) -> io.BytesIO:
        if self._stream_cache is None:
            buffer = io.BytesIO()
            self.save_to(buffer, format=VideoContainer.MP4, codec=VideoCodec.H264)
            self._stream_cache = buffer
        self._stream_cache.seek(0)
        return self._stream_cache

    def get_components(self) -> VideoComponents:
        parts = [source.get_components() for source in self._sources]
        frame_rate = parts[0].frame_rate
        height, width = parts[0].images.shape[1], parts[0].images.shape[2]
        images = []
        for index, part in enumerate(parts):
            part_images = part.images
            if part.frame_rate != frame_rate and part_images.shape[0] > 0:
                # the tensor form is constant frame rate: repeat/drop frames (nearest) so the clip
                # keeps its duration at the first clip's rate
                count = max(1, int(round(Fraction(part_images.shape[0]) * frame_rate / part.frame_rate)))
                positions = torch.arange(count, dtype=torch.float64) * float(part.frame_rate / frame_rate)
                part_images = part_images[positions.floor().long().clamp(max=part_images.shape[0] - 1)]
            if (part_images.shape[1], part_images.shape[2]) != (height, width):
                if self._resize == "error":
                    raise ValueError(f"video {index} is {part_images.shape[2]}x{part_images.shape[1]}; expected {width}x{height}")
                part_images = _fit_images(part_images, width, height)
            images.append(part_images)
        images_per_part = images
        images = torch.cat(images, dim=0)

        audio = None
        audio_parts = [part.audio for part in parts if part.audio is not None]
        if audio_parts:
            # never degrade a source: the output carries the best rate and channel count present
            sample_rate = max(int(part_audio["sample_rate"]) for part_audio in audio_parts)
            # same layout clamp as save_to (mono / stereo / 5.1)
            channels = {1: 1, 2: 2, 6: 6}.get(max(part_audio["waveform"].shape[1] for part_audio in audio_parts), 2)
            blocks = []
            for part, part_images in zip(parts, images_per_part):
                samples = int(round(part_images.shape[0] / frame_rate * sample_rate))
                if part.audio is None:
                    blocks.append(torch.zeros(1, channels, samples))
                    continue
                waveform = part.audio["waveform"]
                part_rate = int(part.audio["sample_rate"])
                if part_rate != sample_rate:
                    import torchaudio
                    waveform = torchaudio.functional.resample(waveform, part_rate, sample_rate)
                waveform = _match_audio_channels(waveform, channels)[..., :samples]
                if waveform.shape[-1] < samples:
                    waveform = torch.nn.functional.pad(waveform, (0, samples - waveform.shape[-1]))
                blocks.append(waveform)
            audio = AudioInput({"waveform": torch.cat(blocks, dim=2), "sample_rate": sample_rate})
        return VideoComponents(images=images, audio=audio, frame_rate=frame_rate)

    def as_trimmed(
        self, start_time: float = 0, duration: float = 0, strict_duration: bool = True
    ) -> VideoInput | None:
        start_time = start_time or 0.0
        duration = duration or 0.0
        total = self.get_duration()
        if start_time < 0:
            start_time = max(total + start_time, 0.0)
        remaining = duration if duration else math.inf
        pieces = []
        offset = 0.0
        for source in self._sources:
            if remaining <= _TRIM_EPSILON:
                break
            source_duration = source.get_duration()
            piece_start = start_time - offset
            offset += source_duration
            if piece_start >= source_duration - _TRIM_EPSILON:
                continue
            piece_start = max(piece_start, 0.0)
            take = min(source_duration - piece_start, remaining)
            if take <= _TRIM_EPSILON:
                break
            to_end = take >= source_duration - piece_start - _TRIM_EPSILON
            if piece_start == 0 and to_end:
                pieces.append(source)
            else:
                piece = source.as_trimmed(piece_start, take, strict_duration=False)
                if piece is None and to_end:
                    # float rounding can push an exact remainder one ulp past the source's own
                    # duration; "to the end" (duration 0) is what was meant
                    piece = source.as_trimmed(piece_start, 0, strict_duration=False)
                if piece is None:
                    return None
                pieces.append(piece)
            remaining -= take
        if not pieces:
            return None
        if strict_duration and duration and sum(piece.get_duration() for piece in pieces) < duration - 1e-6:
            return None
        if len(pieces) == 1:
            return pieces[0]
        return VideoConcatenated(pieces, resize=self._resize)

    @staticmethod
    def _as_file_source(video: VideoInput) -> "VideoFromFile":
        if isinstance(video, VideoFromFile):
            return video
        start_time, duration = 0.0, 0.0
        # a subclass that streams its own untrimmed source still owes us its trim window;
        # the base get_stream_source() encodes via save_to, which already applies it
        if type(video).get_stream_source is not VideoInput.get_stream_source:
            start_time, duration = video.get_active_trim_window()
        return VideoFromFile(video.get_stream_source(), start_time=start_time, duration=duration)

    @staticmethod
    def _probe_segment(segment: "VideoFromFile", index: int) -> _ConcatSegment:
        with av.open(segment.get_stream_source(), mode="r") as container:
            if not len(container.streams.video):
                raise ValueError(f"No video stream found in video {index}")
            video_stream = container.streams.video[0]
            audio_stream = last_decodable_audio_stream(container)
            sample_rate = channels = 0
            if audio_stream is not None:
                sample_rate = audio_stream.codec_context.sample_rate
                channels = audio_stream.codec_context.channels
                if not sample_rate:
                    sample_rate, channels = probe_audio_params(container, audio_stream)
                    if not sample_rate:
                        logging.warning("Audio stream parameters of video %d could not be determined; ignoring its audio.", index)
                        audio_stream = None
            return _ConcatSegment(
                time_base=video_stream.time_base,
                rate=Fraction(video_stream.average_rate) if video_stream.average_rate else Fraction(1),
                bit_depth=video_stream_bit_depth(video_stream),
                color_space=video_stream_color_space(video_stream),
                color_props=SimpleNamespace(
                    color_primaries=video_stream.color_primaries,
                    color_trc=video_stream.color_trc,
                    colorspace=video_stream.colorspace,
                    color_range=video_stream.color_range,
                ),
                audio_index=audio_stream.index if audio_stream is not None else None,
                sample_rate=int(sample_rate),
                channels=int(channels),
            )

    def save_to(
        self,
        path: str | io.BytesIO,
        format: VideoContainer = VideoContainer.AUTO,
        codec: VideoCodec = VideoCodec.AUTO,
        metadata: Optional[dict] = None,
        bit_depth: int | None = None,
        crf: float | None = None,
        color_space: str | None = None,
    ):
        """Re-encode every source, one frame at a time, into a single output; peak memory does not scale with length."""
        if color_space is not None and color_space not in VIDEO_COLOR_TRANSFERS:
            raise ValueError(f"Unsupported video color space: {color_space}")
        open_kwargs, output_format, output_codec = video_output_config(path, format, codec)
        segments = [self._as_file_source(source) for source in self._sources]
        infos = [self._probe_segment(segment, index) for index, segment in enumerate(segments)]

        time_bases = {info.time_base for info in infos}
        # exact for 24/25/30/50/60 and the NTSC 23.976/29.97/59.94 rates; every frame's pts is
        # rescaled independently from its source, so rounding never accumulates
        time_base = infos[0].time_base if len(time_bases) == 1 else Fraction(1, 360000)
        rate = infos[0].rate
        if bit_depth is None:
            bit_depth = max(info.bit_depth for info in infos)
        pix_fmt = "yuv420p10le" if bit_depth >= 10 else "yuv420p"

        color_props = None
        source_color_space = None
        for info in infos:
            if info.color_space is None:
                continue
            if color_space is not None and info.color_space != color_space:
                raise ValueError(
                    f"Cannot save {info.color_space} video as {color_space} without color conversion; "
                    f"use auto or {info.color_space}"
                )
            if source_color_space is None:
                source_color_space = info.color_space
                color_props = info.color_props
            elif info.color_space != source_color_space:
                raise ValueError(
                    f"Cannot concatenate {source_color_space} and {info.color_space} videos without color conversion"
                )

        audio = None
        audio_infos = [info for info in infos if info.audio_index is not None]
        if audio_infos:
            # never degrade a source: the output carries the best rate and channel count present
            sample_rate = 48000 if output_format == VideoContainer.WEBM else max(info.sample_rate for info in audio_infos)
            layout = {1: "mono", 2: "stereo", 6: "5.1"}.get(max(info.channels for info in audio_infos), "stereo")
            audio = (sample_rate, layout, {"mono": 1, "stereo": 2, "5.1": 6}[layout])

        encoder = _ConcatEncoder(
            path, open_kwargs, output_format, output_codec, metadata,
            time_base=time_base, rate=rate, pix_fmt=pix_fmt, crf=crf,
            color_space=color_space, color_props=color_props, audio=audio,
        )
        try:
            for index, (segment, info) in enumerate(zip(segments, infos)):
                # every open goes through get_stream_source() (it rewinds BytesIO sources), and
                # segments run strictly one after another, so the same source may appear twice
                with av.open(segment.get_stream_source(), mode="r") as container:
                    self._encode_segment(encoder, container, segment, info, index, scale_to=encoder.output_size)
            encoder.finish()
        except BaseException:
            encoder.abort()
            raise

    def _encode_segment(self, encoder: _ConcatEncoder, container: InputContainer, segment: "VideoFromFile", info: _ConcatSegment, index: int, *, scale_to: tuple[int, int] | None):
        """Input side; mirrors the demux/decode half of VideoFromFile._save_transcoded for one source."""
        video_stream = container.streams.video[0]
        source_time_base = video_stream.time_base
        start_time, duration = segment.get_active_trim_window()
        start_pts = int(start_time / source_time_base)
        end_pts = int((start_time + duration) / source_time_base) if duration else None
        stream_end_pts = None
        if video_stream.duration is not None:
            stream_end_pts = (video_stream.start_time or 0) + video_stream.duration
        output_end_pts = end_pts
        if stream_end_pts is not None and (output_end_pts is None or stream_end_pts < output_end_pts):
            output_end_pts = stream_end_pts
        if start_pts != 0:
            container.seek(start_pts, stream=video_stream)

        audio_stream = None
        resampler = None
        audio_cap_samples = None
        if encoder.audio is not None and info.audio_index is not None:
            audio_stream = container.streams[info.audio_index]
            target_rate, target_layout, _ = encoder.audio
            resampler = av.audio.resampler.AudioResampler(format="fltp", layout=target_layout, rate=target_rate)
            if duration:
                audio_cap_samples = math.ceil(duration * target_rate)

        def rescale(value: int) -> int:
            return int(round(value * source_time_base / encoder.time_base))

        source_pts_step = max(1, int(round((1 / info.rate) / source_time_base)))
        encoder.begin_segment(
            pts_step=max(1, int(round((1 / info.rate) / encoder.time_base))),
            audio_cap_samples=audio_cap_samples,
            has_audio=audio_stream is not None,
        )

        # per-source: a recognized tag means the frames already carry the output color space;
        # an untagged full-range (yuvj/MJPEG) source is compressed to limited range instead
        preserve_source_color = video_stream_color_space(video_stream) is not None
        streams = [video_stream] if audio_stream is None else [video_stream, audio_stream]
        video_done = False
        video_pts_offset = None
        source_size = None
        filter_graph = None
        audio_started = False
        label = f"video {index}"

        for packet in container.demux(*streams):
            if encoder.segment_done:
                break

            if packet.stream == video_stream and not video_done:
                try:
                    frames = packet.decode()
                except av.error.InvalidDataError:
                    logging.info("pyav decode error")
                    continue
                for frame in frames:
                    if frame.pts is not None and frame.pts < start_pts:
                        continue
                    if end_pts is not None and frame.pts is not None and frame.pts >= end_pts:
                        video_done = True
                        # the source continues past the window: hold the last kept frame to the window end
                        end_offset = video_pts_offset if video_pts_offset is not None else start_pts
                        encoder.end_video(rescale(end_pts - end_offset))
                        break
                    # the source's true display duration of this frame; average_rate is not a
                    # frame duration (sparse/VFR sources), so it is only the fallback
                    frame_duration = frame.duration if frame.duration else source_pts_step
                    if end_pts is not None and frame.pts is not None:
                        frame_duration = min(frame_duration, end_pts - frame.pts)
                    if source_size is None:
                        rotation_k = int(round(frame.rotation // 90)) % 4 if frame.rotation else 0
                        rotated = (frame.height, frame.width) if rotation_k % 2 else (frame.width, frame.height)
                        source_size = (frame.width, frame.height)
                        target_size = scale_to
                        if target_size is not None and rotated != target_size:
                            if self._resize == "error":
                                raise ValueError(f"{label} is {rotated[0]}x{rotated[1]}; expected {target_size[0]}x{target_size[1]}")
                            if rotated[0] > target_size[0] or rotated[1] > target_size[1]:
                                logging.warning(
                                    "Concatenate: %s (%dx%d) is downscaled to fit the first video's %dx%d frame",
                                    label, rotated[0], rotated[1], target_size[0], target_size[1],
                                )
                        filter_graph = _segment_filter_graph(frame, source_time_base, rotation_k, target_size)
                    if (frame.width, frame.height) != source_size:
                        # encoding would silently rescale the new geometry into the old one
                        raise ValueError(
                            f"Video resolution changes mid-stream in {label} "
                            f"({source_size[0]}x{source_size[1]} -> {frame.width}x{frame.height}); cannot transcode"
                        )
                    if filter_graph is not None:
                        filter_graph[1].push(frame)
                        frame = filter_graph[2].pull()
                    if frame.color_range == ColorRange.JPEG and not preserve_source_color:
                        # compress full-range sources (yuvj/MJPEG) to limited range
                        frame = frame.reformat(format=encoder.pix_fmt, src_color_range="JPEG", dst_color_range="MPEG")
                    else:
                        frame = frame.reformat(format=encoder.pix_fmt)
                    if preserve_source_color:
                        copy_color_properties(video_stream, frame)
                    elif encoder.color_space is not None:
                        set_video_color_properties(frame, encoder.color_space)
                    segment_end = None
                    if frame.pts is not None:
                        # source pts pass through, rebased to the segment start and rescaled to the output time base
                        if video_pts_offset is None:
                            video_pts_offset = frame.pts
                        frame.pts = rescale(frame.pts - video_pts_offset)
                        if output_end_pts is not None:
                            segment_end = rescale(output_end_pts - video_pts_offset)
                    frame.time_base = encoder.time_base
                    encoder.push_video(frame, max(1, rescale(frame_duration)), segment_end)

            elif packet.stream == audio_stream and not encoder.seg_audio_done:
                target_rate = encoder.audio[0]
                for resampled in itertools.chain.from_iterable(map(resampler.resample, packet.decode())):
                    frame_start = None
                    if resampled.pts is not None:
                        # passthrough frames keep the source stream's time base
                        tb = resampled.time_base if resampled.time_base else encoder.audio_time_base
                        frame_start = float(resampled.pts * tb)
                        if duration and not audio_started and frame_start >= start_time + duration:
                            encoder.end_audio()
                            break
                    if not audio_started:
                        if frame_start is None:
                            frame_start = 0.0
                        to_skip = max(0, int((start_time - frame_start) * target_rate))
                        if to_skip >= resampled.samples:
                            continue
                        audio_started = True
                        if duration and frame_start > start_time:
                            encoder.push_audio_cap(math.ceil((start_time + duration - frame_start) * target_rate))
                        if to_skip:
                            if encoder.push_audio(encoder.audio_frame_from_ndarray(resampled.to_ndarray()[..., to_skip:])):
                                break
                            continue
                    if encoder.push_audio(resampled):
                        break

        if not video_done:
            encoder.end_video(None)
        if resampler is not None and audio_started and not encoder.seg_audio_done:
            # the resampler holds a few samples of delay; release them so the segment's audio is complete
            for resampled in resampler.resample(None):
                if encoder.push_audio(resampled):
                    break
        encoder.end_segment()
