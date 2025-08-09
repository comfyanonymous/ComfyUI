import os
import numpy as np
import folder_paths
import av
from fractions import Fraction
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from comfy.cli_args import args
import json
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

class GetWorkflowMetadata(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GetWorkflowMetadata",
            category="image",
            display_name="Get Workflow Metadata",
            description="Gets the workflow metadata from the current workflow.",
            inputs=[],
            hidden=[
                io.Hidden.extra_pnginfo
            ],
            outputs=[
                io.Metadata.Output("metadata", display_name="workflow")
            ],
        )
    @classmethod
    def execute(self, extra_pnginfo):
        metadata = PngInfo()
        if not args.disable_metadata:
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata.add_text(x, json.dumps(extra_pnginfo[x]))
        return io.NodeOutput(metadata)

class EmptyMetadata(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EmptyMetadata",
            category="image",
            display_name="Empty Metadata",
            description="Create a blank / empty metadata to add upon.",
            inputs=[],
            outputs=[
                io.Metadata.Output("metadata", display_name="metadata")
            ],
        )
    @classmethod
    def execute(self):
        metadata = PngInfo()
        return io.NodeOutput(metadata)

class AddMetadataValue(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AddMetadataValue",
            category="image",
            display_name="Add Metadata Value",
            description="Add an arbitrary value to the metadata.",
            inputs=[
                io.String.Input("key", tooltip="Key to save the value at."),
                io.String.Input("value", tooltip="What to add to the metadata."),
                io.Metadata.Input("metadata", display_name="metadata")
            ],
            outputs=[
                io.Metadata.Output("modified_metadata", display_name="metadata")
            ],
        )
    @classmethod
    def execute(self, key, value, metadata):
        metadata.add_text(key, json.dumps(value))
        return io.NodeOutput(metadata)



class SaveImageCustomMetadata(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveImageCustomMetadata",
            category="image",
            display_name="Save Image With Custom Metadata",
            description="Saves the input images to your ComfyUI output directory with custom metadata.",
            inputs=[
                io.Image.Input("images", tooltip="The images to save."),
                io.String.Input("filename_prefix", default="ComfyUI", tooltip="The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."),
                io.Metadata.Input("metadata", display_name="metadata", tooltip="Metadata to save with the image.", optional=True)
            ],
            outputs=[],
            is_output_node=True,
        )
    @classmethod
    def execute(self, images, filename_prefix, metadata=None):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory(), images[0].shape[1], images[0].shape[0])
        results = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": "output"
            })
            counter += 1

        return io.NodeOutput(ui = {"images": results})

class SaveWEBMCustomMetadata(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveWebmCustomMetadata",
            category="image/video",
            display_name="Save Webm With Custom Metadata",
            inputs=[
                io.Image.Input("images", tooltip="The images to save."),
                io.String.Input("filename_prefix", default="ComfyUI"),
                io.Combo.Input("codec", ["vp9", "av1"]),
                io.Float.Input("fps", default=24.0, min=0.01, max=1000.0, step=0.01),
                io.Float.Input("crf", default=32.0, min=0.0, max=63.0, step=1.0, tooltip="Higher crf means lower quality with a smaller file size, lower crf means higher quality higher filesize."),
                io.Metadata.Input("metadata", display_name="metadata", tooltip="Metadata to save with the image.", optional=True)
            ],
            outputs=[],
            is_experimental=True,
            is_output_node=True
        )

    @classmethod
    def execute(self, images, codec, fps, filename_prefix, crf, metadata=None):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory(), images[0].shape[1], images[0].shape[0])

        file = f"{filename}_{counter:05}_.webm"
        container = av.open(os.path.join(full_output_folder, file), mode="w")

        if metadata is not None:
            for x in metadata:
                container.metadata[x] = json.dumps(metadata[x])

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

        results = [{
            "filename": file,
            "subfolder": subfolder,
            "type": "output"
        }]

        return io.NodeOutput(ui = {"images": results, "animated": (True,)})


class MetadataExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            GetWorkflowMetadata,
            EmptyMetadata,
            AddMetadataValue,
            SaveImageCustomMetadata,
            SaveWEBMCustomMetadata
        ]

async def comfy_entrypoint() -> MetadataExtension:
    return MetadataExtension()
