from __future__ import annotations

import json
import os
from fractions import Fraction

import av
import torch

import folder_paths
from comfy.cli_args import args
from comfy_api.input import AudioInput, ImageInput, VideoInput
from comfy_api.input_impl import VideoFromComponents, VideoFromFile
from comfy_api.util import VideoCodec, VideoComponents, VideoContainer
from comfy_api.v3 import io, ui


class CreateVideo(io.ComfyNodeV3):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CreateVideo_V3",
            display_name="Create Video _V3",
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
    def execute(cls, images: ImageInput, fps: float, audio: AudioInput = None):
        return io.NodeOutput(VideoFromComponents(
            VideoComponents(
            images=images,
            audio=audio,
            frame_rate=Fraction(fps),
            )
        ))


class GetVideoComponents(io.ComfyNodeV3):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GetVideoComponents_V3",
            display_name="Get Video Components _V3",
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
    def execute(cls, video: VideoInput):
        components = video.get_components()
        return io.NodeOutput(components.images, components.audio, float(components.frame_rate))


class LoadVideo(io.ComfyNodeV3):
    @classmethod
    def define_schema(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["video"])
        return io.Schema(
            node_id="LoadVideo_V3",
            display_name="Load Video _V3",
            category="image/video",
            inputs=[
                io.Combo.Input("file", options=sorted(files), upload=io.UploadType.video),
            ],
            outputs=[
                io.Video.Output(),
            ],
        )

    @classmethod
    def execute(cls, file):
        video_path = folder_paths.get_annotated_filepath(file)
        return io.NodeOutput(VideoFromFile(video_path))

    @classmethod
    def fingerprint_inputs(s, file):
        video_path = folder_paths.get_annotated_filepath(file)
        mod_time = os.path.getmtime(video_path)
        # Instead of hashing the file, we can just use the modification time to avoid rehashing large files.
        return mod_time

    @classmethod
    def validate_inputs(s, file):
        if not folder_paths.exists_annotated_filepath(file):
            return "Invalid video file: {}".format(file)
        return True


class SaveVideo(io.ComfyNodeV3):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveVideo_V3",
            display_name="Save Video _V3",
            category="image/video",
            description="Saves the input images to your ComfyUI output directory.",
            inputs=[
                io.Video.Input("video", tooltip="The video to save."),
                io.String.Input("filename_prefix", default="video/ComfyUI", tooltip="The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."),
                io.Combo.Input("format", options=VideoContainer.as_input(), default="auto", tooltip="The format to save the video as."),
                io.Combo.Input("codec", options=VideoCodec.as_input(), default="auto", tooltip="The codec to use for the video."),
            ],
            outputs=[],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, video: VideoInput, filename_prefix, format, codec):
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
        file = f"{filename}_{counter:05}_.{VideoContainer.get_extension(format)}"
        video.save_to(
            os.path.join(full_output_folder, file),
            format=format,
            codec=codec,
            metadata=saved_metadata
        )
        return io.NodeOutput(ui=ui.PreviewVideo([ui.SavedResult(file, subfolder, io.FolderType.output)]))


class SaveWEBM(io.ComfyNodeV3):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveWEBM_V3",
            category="image/video",
            is_experimental=True,
            inputs=[
                io.Image.Input("images"),
                io.String.Input("filename_prefix", default="ComfyUI"),
                io.Combo.Input("codec", options=["vp9", "av1"]),
                io.Float.Input("fps", default=24.0, min=0.01, max=1000.0, step=0.01),
                io.Float.Input("crf", default=32.0, min=0, max=63.0, step=1, tooltip="Higher crf means lower quality with a smaller file size, lower crf means higher quality higher filesize."),
            ],
            outputs=[],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, images, codec, fps, filename_prefix, crf):
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


NODES_LIST = [CreateVideo, GetVideoComponents, LoadVideo, SaveVideo, SaveWEBM]
