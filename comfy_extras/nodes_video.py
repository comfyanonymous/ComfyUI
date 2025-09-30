from __future__ import annotations

import os
import io
import av
import torch
import folder_paths
import json
from typing import Optional
from typing_extensions import override
from fractions import Fraction
from comfy_api.input import AudioInput, ImageInput, VideoInput
from comfy_api.input_impl import VideoFromComponents, VideoFromFile
from comfy_api.util import VideoCodec, VideoComponents, VideoContainer
from comfy_api.latest import ComfyExtension, io, ui
from comfy.cli_args import args
import comfy.utils

class EncodeVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EncodeVideo",
            display_name="Encode Video",
            category="image/video",
            description="Encode a video using an image encoder.",
            inputs=[
                io.Video.Input("video", tooltip="The video to be encoded."),
                io.Int.Input(
                    "processing_batch_size", default=-1, min=-1,
                    tooltip=(
                        "Number of frames/segments to process at a time during encoding.\n"
                        "-1 means process all at once. Smaller values reduce GPU memory usage."
                    ),
                ),
                io.Int.Input("step_size", default=8, min=1, max=32,
                    tooltip=(
                        "Stride (in frames) between the start of consecutive segments.\n"
                        "Smaller step = more overlap and smoother temporal coverage "
                        "but higher compute cost. Larger step = faster but may miss detail."
                    ),
                ),
                io.Vae.Input("vae", optional=True),
                io.ClipVision.Input("clip_vision", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="encoded_video"),
            ],
        )
    
    @classmethod
    def execute(cls, video, processing_batch_size, step_size, vae = None, clip_vision = None):

        t, c, h, w = video.shape
        b = 1
        batch_size = b * t

        if vae is not None and clip_vision is not None:
            raise ValueError("Must either have vae or clip_vision.")
        elif vae is None and clip_vision is None:
            raise ValueError("Can't have VAE and Clip Vision passed at the same time!")
        vae = vae if vae is not None else clip_vision

        if hasattr(vae.first_stage_model, "video_encoding"):
            data, num_segments, output_fn = vae.first_stage_model.video_encoding(video, step_size)
            batch_size = b * num_segments
        else:
            data = video.view(batch_size, c, h, w)
            output_fn = lambda x: x.view(b, t, -1)

        if processing_batch_size != -1:
            batch_size = processing_batch_size

        outputs = []
        total = data.shape[0]
        pbar = comfy.utils.ProgressBar(total/batch_size)
        with torch.inference_mode(): 
            for i in range(0, total, batch_size):
                chunk = data[i : i + batch_size]
                out = vae.encode(chunk)
                outputs.append(out)
                del out, chunk
                torch.cuda.empty_cache()
                pbar.update(1)

        output = torch.cat(outputs)

        return io.NodeOutput(output_fn(output))

class ResampleVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ResampleVideo",
            display_name="Resample Video",
            category="image/video",
            inputs = [
                io.Video.Input("video"),
                io.Int.Input("target_fps", min=1, default=25)
            ],
            outputs=[io.Video.Output(display_name="video")]
        )
    @classmethod
    def execute(cls, video, target_fps: int):
        # doesn't support upsampling 
        with av.open(video.get_stream_source(), mode="r") as container:
            stream = container.streams.video[0]
            frames = []

            src_rate = stream.average_rate or stream.guessed_rate
            src_fps = float(src_rate) if src_rate else None

            # yield original frames if asked for upsampling or src is unknown
            if src_fps is None or target_fps > src_fps:
                for packet in container.demux(stream):
                    for frame in packet.decode():
                        arr = torch.from_numpy(frame.to_ndarray(format="rgb24")).float() / 255.0
                        frames.append(arr)
                return io.NodeOutput(torch.stack(frames))

            stream.thread_type = "AUTO"

            next_time = 0.0
            step = 1.0 / target_fps

            for packet in container.demux(stream):
                for frame in packet.decode():
                    if frame.time is None:
                        continue
                    t = frame.time
                    while t >= next_time:
                        arr = torch.from_numpy(frame.to_ndarray(format="rgb24")).float() / 255.0
                        frames.append(arr)
                        next_time += step

            return io.NodeOutput(torch.stack(frames))

class VideoToImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VideoToImage",
            category="image/video",
            display_name = "Video To Images",
            inputs=[io.Video.Input("video")],
            outputs=[io.Image.Output("images")]
        )
    @classmethod
    def execute(cls, video):
        with av.open(video.get_stream_source(), mode="r") as container:
            components = video.get_components_internal(container)

        images = components.images
        return io.NodeOutput(images)

class SaveWEBM(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveWEBM",
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
            display_name="Save Video",
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
    def execute(cls, video: VideoInput, filename_prefix, format, codec) -> io.NodeOutput:
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


class CreateVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CreateVideo",
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
    def execute(cls, images: ImageInput, fps: float, audio: Optional[AudioInput] = None) -> io.NodeOutput:
        return io.NodeOutput(
            VideoFromComponents(VideoComponents(images=images, audio=audio, frame_rate=Fraction(fps)))
        )

class GetVideoComponents(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GetVideoComponents",
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
    def execute(cls, video: VideoInput) -> io.NodeOutput:
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
            display_name="Load Video",
            category="image/video",
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
        return io.NodeOutput(VideoFromFile(video_path))

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


class VideoExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SaveWEBM,
            SaveVideo,
            CreateVideo,
            GetVideoComponents,
            LoadVideo,
            EncodeVideo,
            ResampleVideo,
            VideoToImage
        ]

async def comfy_entrypoint() -> VideoExtension:
    return VideoExtension()
