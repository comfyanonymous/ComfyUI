from __future__ import annotations

import os
import av
import torch
import folder_paths
import json
from typing import Callable, Optional
from typing_extensions import override
from fractions import Fraction
from comfy_api.latest import ComfyExtension, io, ui, Input, InputImpl, Types
from comfy.cli_args import args
from comfy_execution.utils import get_executing_context
from comfy_extras.nodes_image_stream import FrameProgressTracker, drain_image_stream
from server import PromptServer


class SavedVideoStream(Input.ImageStream):
    def __init__(
        self,
        stream: Input.ImageStream,
        saver,
        output_factory: Callable[[], tuple[str, ui.PreviewVideo | None]],
        format: str,
        codec,
        metadata: Optional[dict],
        emit_preview_on_finalize: bool = True,
        preview_node_id: Optional[str] = None,
        preview_display_node_id: Optional[str] = None,
    ):
        super().__init__()
        self._stream = stream
        self._saver = saver
        self._output_factory = output_factory
        self._path: str | None = None
        self._format = Types.VideoContainer(format)
        self._codec = Types.VideoCodec(codec)
        self._metadata = metadata
        self._preview_ui: ui.PreviewVideo | None = None
        self._emit_preview_on_finalize = emit_preview_on_finalize
        self._preview_node_id = preview_node_id
        self._preview_display_node_id = preview_display_node_id
        self._save_state = None

    def _emit_preview(self) -> None:
        if not self._emit_preview_on_finalize or self._preview_ui is None:
            return

        current = get_executing_context()
        if current is None:
            return

        server = getattr(PromptServer, "instance", None)
        if server is None or server.client_id is None:
            return

        server.send_sync(
            "executed",
            {
                "node": self._preview_node_id or current.node_id,
                "display_node": self._preview_display_node_id or self._preview_node_id or current.node_id,
                "output": self._preview_ui.as_dict(),
                "prompt_id": current.prompt_id,
            },
            server.client_id,
        )

    def _discard_partial_output(self) -> None:
        if self._save_state is not None:
            self._save_state[0].close()
            self._save_state = None
        if self._path is not None and os.path.exists(self._path):
            os.remove(self._path)
        self._path = None
        self._preview_ui = None

    def get_preview_ui(self) -> ui.PreviewVideo | None:
        return self._preview_ui

    def get_dimensions(self) -> tuple[int, int]:
        return self._stream.get_dimensions()

    def do_reset(self) -> None:
        self._discard_partial_output()
        self._stream.reset()
        self._path, self._preview_ui = self._output_factory()
        assert self._path is not None
        open(self._path, "ab").close()
        self._save_state = self._saver.save_start(
            self._path,
            format=self._format,
            codec=self._codec,
            metadata=self._metadata,
        )

    def do_pull(self, max_frames: int) -> Input.Image:
        assert self._save_state is not None
        chunk = self._stream.pull(max_frames)
        self._saver.save_add(self._save_state[0], self._save_state[1], chunk)
        if chunk.shape[0] < max_frames:
            self._saver.save_finalize(*self._save_state)
            self._save_state = None
            self._emit_preview()
            self._path = None
        return chunk


def _build_saved_stream(
    hidden,
    stream: Input.ImageStream,
    audio: Optional[Input.Audio],
    fps: float,
    filename_prefix,
    format: str,
    codec,
    emit_preview: bool = True,
) -> SavedVideoStream:
    width, height = stream.get_dimensions()
    saved_metadata = None
    if not args.disable_metadata:
        metadata = {}
        if hidden.extra_pnginfo is not None:
            metadata.update(hidden.extra_pnginfo)
        if hidden.prompt is not None:
            metadata["prompt"] = hidden.prompt
        if len(metadata) > 0:
            saved_metadata = metadata

    preview_node_id = hidden.unique_id
    preview_display_node_id = preview_node_id
    if hidden.dynprompt is not None and preview_node_id is not None:
        preview_display_node_id = hidden.dynprompt.get_display_node_id(preview_node_id)

    def output_factory() -> tuple[str, ui.PreviewVideo]:
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix,
            folder_paths.get_output_directory(),
            width,
            height,
        )
        file = f"{filename}_{counter:05}_.{Types.VideoContainer.get_extension(format)}"
        return (
            os.path.join(full_output_folder, file),
            ui.PreviewVideo([ui.SavedResult(file, subfolder, io.FolderType.output)]),
        )

    return SavedVideoStream(
        stream,
        InputImpl.VideoFromComponents(
            Types.VideoComponents(
                images=torch.zeros((0, height, width, 3)),
                audio=audio,
                frame_rate=Fraction(fps),
            )
        ),
        output_factory,
        format,
        codec,
        saved_metadata,
        emit_preview,
        preview_node_id,
        preview_display_node_id,
    )

class SaveWEBM(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveWEBM",
            search_aliases=["export webm"],
            category="image/video",
            is_experimental=True,
            inputs=[
                io.Image.Input("images"),
                io.String.Input("filename_prefix", default="ComfyUI"),
                io.Combo.Input("codec", options=["vp9", "av1"]),
                io.Float.Input("fps", default=24.0, min=0.01, max=1000.0, step=0.01),
                io.Float.Input("crf", default=32.0, min=0, max=63.0, step=1, tooltip="Higher crf means lower quality with a smaller file size, lower crf means higher quality higher filesize."),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, images, codec, fps, filename_prefix, crf) -> io.NodeOutput:
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), images[0].shape[1], images[0].shape[0]
        )

        file = f"{filename}_{counter:05}_.webm"
        container = av.open(os.path.join(full_output_folder, file), mode="w")

        if cls.hidden.prompt is not None:
            container.metadata["prompt"] = json.dumps(cls.hidden.prompt)

        if cls.hidden.extra_pnginfo is not None:
            for x in cls.hidden.extra_pnginfo:
                container.metadata[x] = json.dumps(cls.hidden.extra_pnginfo[x])

        codec_map = {"vp9": "libvpx-vp9", "av1": "libsvtav1"}
        stream = container.add_stream(codec_map[codec], rate=Fraction(round(fps * 1000), 1000))
        stream.width = images.shape[-2]
        stream.height = images.shape[-3]
        stream.pix_fmt = "yuv420p10le" if codec == "av1" else "yuv420p"
        stream.bit_rate = 0
        stream.options = {'crf': str(crf)}
        if codec == "av1":
            stream.options["preset"] = "6"

        for frame in images:
            frame = av.VideoFrame.from_ndarray(torch.clamp(frame[..., :3] * 255, min=0, max=255).to(device=torch.device("cpu"), dtype=torch.uint8).numpy(), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        container.mux(stream.encode())
        container.close()

        return io.NodeOutput(ui=ui.PreviewVideo([ui.SavedResult(file, subfolder, io.FolderType.output)]))

class SaveVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveVideo",
            search_aliases=["export video"],
            display_name="Save Video",
            category="image/video",
            essentials_category="Basics",
            description="Saves the input images to your ComfyUI output directory.",
            inputs=[
                io.Video.Input("video", tooltip="The video to save."),
                io.String.Input("filename_prefix", default="video/ComfyUI", tooltip="The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."),
                io.Combo.Input("format", options=Types.VideoContainer.as_input(), default="auto", tooltip="The format to save the video as."),
                io.Combo.Input("codec", options=Types.VideoCodec.as_input(), default="auto", tooltip="The codec to use for the video."),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, video: Input.Video, filename_prefix, format: str, codec) -> io.NodeOutput:
        width, height = video.get_dimensions()
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix,
            folder_paths.get_output_directory(),
            width,
            height
        )
        saved_metadata = None
        if not args.disable_metadata:
            metadata = {}
            if cls.hidden.extra_pnginfo is not None:
                metadata.update(cls.hidden.extra_pnginfo)
            if cls.hidden.prompt is not None:
                metadata["prompt"] = cls.hidden.prompt
            if len(metadata) > 0:
                saved_metadata = metadata
        file = f"{filename}_{counter:05}_.{Types.VideoContainer.get_extension(format)}"
        video.save_to(
            os.path.join(full_output_folder, file),
            format=Types.VideoContainer(format),
            codec=codec,
            metadata=saved_metadata
        )

        return io.NodeOutput(ui=ui.PreviewVideo([ui.SavedResult(file, subfolder, io.FolderType.output)]))


class SavePassthroughVideoStream(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SavePassthroughVideoStream",
            search_aliases=["stream to video", "save image stream", "export video stream", "passthrough video stream"],
            display_name="Save+Passthrough Video Stream",
            category="image/video",
            essentials_category="Basics",
            description="Saves frames as they pass through the input image stream.",
            has_intermediate_output=True,
            inputs=[
                io.ImageStream.Input("stream", tooltip="The image stream to save."),
                io.Float.Input("fps", default=30.0, min=1.0, max=120.0, step=1.0),
                io.String.Input("filename_prefix", default="video/ComfyUI", tooltip="The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."),
                io.Combo.Input("format", options=Types.VideoContainer.as_input(), default="auto", tooltip="The format to save the video as."),
                io.Combo.Input("codec", options=Types.VideoCodec.as_input(), default="auto", tooltip="The codec to use for the video."),
                io.Audio.Input("audio", optional=True, tooltip="The audio to add to the video."),
            ],
            outputs=[
                io.ImageStream.Output(display_name="passthrough"),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo, io.Hidden.unique_id, io.Hidden.dynprompt],
        )

    @classmethod
    def execute(
        cls,
        stream: Input.ImageStream,
        fps: float,
        filename_prefix,
        format: str,
        codec,
        audio: Optional[Input.Audio] = None,
    ) -> io.NodeOutput:
        return io.NodeOutput(_build_saved_stream(cls.hidden, stream, audio, fps, filename_prefix, format, codec))


class SaveVideoStream(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveVideoStream",
            search_aliases=["save image stream", "export video stream", "stream to video"],
            display_name="Save Video Stream",
            category="image/video",
            essentials_category="Basics",
            description="Saves an image stream by draining it directly to EOF.",
            is_output_node=True,
            inputs=[
                io.ImageStream.Input("stream", tooltip="The image stream to save."),
                io.Float.Input("fps", default=30.0, min=1.0, max=120.0, step=1.0),
                io.Int.Input("chunk_size", default=8, min=1, max=4096),
                io.String.Input("filename_prefix", default="video/ComfyUI", tooltip="The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."),
                io.Combo.Input("format", options=Types.VideoContainer.as_input(), default="auto", tooltip="The format to save the video as."),
                io.Combo.Input("codec", options=Types.VideoCodec.as_input(), default="auto", tooltip="The codec to use for the video."),
                io.Audio.Input("audio", optional=True, tooltip="The audio to add to the video."),
            ],
            outputs=[],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo, io.Hidden.unique_id, io.Hidden.dynprompt],
        )

    @classmethod
    def execute(
        cls,
        stream: Input.ImageStream,
        fps: float,
        chunk_size: int,
        filename_prefix,
        format: str,
        codec,
        audio: Optional[Input.Audio] = None,
    ) -> io.NodeOutput:
        saved_stream = _build_saved_stream(
            cls.hidden,
            stream,
            audio,
            fps,
            filename_prefix,
            format,
            codec,
            emit_preview=False,
        )
        drain_image_stream(saved_stream, chunk_size, progress=FrameProgressTracker())
        return io.NodeOutput(ui=saved_stream.get_preview_ui())


class CreateVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CreateVideo",
            search_aliases=["images to video"],
            display_name="Create Video",
            category="image/video",
            description="Create a video from images.",
            inputs=[
                io.Image.Input("images", tooltip="The images to create a video from."),
                io.Float.Input("fps", default=30.0, min=1.0, max=120.0, step=1.0),
                io.Audio.Input("audio", optional=True, tooltip="The audio to add to the video."),
            ],
            outputs=[
                io.Video.Output(),
            ],
        )

    @classmethod
    def execute(cls, images: Input.Image, fps: float, audio: Optional[Input.Audio] = None) -> io.NodeOutput:
        return io.NodeOutput(
            InputImpl.VideoFromComponents(Types.VideoComponents(images=images, audio=audio, frame_rate=Fraction(fps)))
        )

class GetVideoComponents(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GetVideoComponents",
            search_aliases=["extract frames", "split video", "video to images", "demux"],
            display_name="Get Video Components",
            category="image/video",
            description="Extracts all components from a video: frames, audio, and framerate.",
            inputs=[
                io.Video.Input("video", tooltip="The video to extract components from."),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.Audio.Output(display_name="audio"),
                io.Float.Output(display_name="fps"),
            ],
        )

    @classmethod
    def execute(cls, video: Input.Video) -> io.NodeOutput:
        components = video.get_components()
        return io.NodeOutput(components.images, components.audio, float(components.frame_rate))


class LoadVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["video"])
        return io.Schema(
            node_id="LoadVideo",
            search_aliases=["import video", "open video", "video file"],
            display_name="Load Video",
            category="image/video",
            essentials_category="Basics",
            inputs=[
                io.Combo.Input("file", options=sorted(files), upload=io.UploadType.video),
            ],
            outputs=[
                io.Video.Output(),
            ],
        )

    @classmethod
    def execute(cls, file) -> io.NodeOutput:
        video_path = folder_paths.get_annotated_filepath(file)
        return io.NodeOutput(InputImpl.VideoFromFile(video_path))

    @classmethod
    def fingerprint_inputs(s, file):
        video_path = folder_paths.get_annotated_filepath(file)
        mod_time = os.path.getmtime(video_path)
        # Instead of hashing the file, we can just use the modification time to avoid
        # rehashing large files.
        return mod_time

    @classmethod
    def validate_inputs(s, file):
        if not folder_paths.exists_annotated_filepath(file):
            return "Invalid video file: {}".format(file)

        return True

class VideoSlice(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Video Slice",
            display_name="Video Slice",
            search_aliases=[
                "trim video duration",
                "skip first frames",
                "frame load cap",
                "start time",
            ],
            category="image/video",
            essentials_category="Video Tools",
            inputs=[
                io.Video.Input("video"),
                io.Float.Input(
                    "start_time",
                    default=0.0,
                    max=1e5,
                    min=-1e5,
                    step=0.001,
                    tooltip="Start time in seconds",
                ),
                io.Float.Input(
                    "duration",
                    default=0.0,
                    min=0.0,
                    step=0.001,
                    tooltip="Duration in seconds, or 0 for unlimited duration",
                ),
                io.Boolean.Input(
                    "strict_duration",
                    default=False,
                    tooltip="If True, when the specified duration is not possible, an error will be raised.",
                ),
            ],
            outputs=[
                io.Video.Output(),
            ],
        )

    @classmethod
    def execute(cls, video: io.Video.Type, start_time: float, duration: float, strict_duration: bool) -> io.NodeOutput:
        trimmed = video.as_trimmed(start_time, duration, strict_duration=strict_duration)
        if trimmed is not None:
            return io.NodeOutput(trimmed)
        raise ValueError(
            f"Failed to slice video:\nSource duration: {video.get_duration()}\nStart time: {start_time}\nTarget duration: {duration}"
        )


class VideoExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SaveWEBM,
            SaveVideo,
            SavePassthroughVideoStream,
            SaveVideoStream,
            CreateVideo,
            GetVideoComponents,
            LoadVideo,
            VideoSlice,
        ]

async def comfy_entrypoint() -> VideoExtension:
    return VideoExtension()
