from __future__ import annotations

import os
from pathlib import Path

import folder_paths
import nodes
from comfy_api.input_impl import VideoFromFile
from comfy_api.v3 import io, ui


def normalize_path(path):
    return path.replace("\\", "/")


class Load3D(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        input_dir = os.path.join(folder_paths.get_input_directory(), "3d")

        os.makedirs(input_dir, exist_ok=True)

        input_path = Path(input_dir)
        base_path = Path(folder_paths.get_input_directory())

        files = [
            normalize_path(str(file_path.relative_to(base_path)))
            for file_path in input_path.rglob("*")
            if file_path.suffix.lower() in {".gltf", ".glb", ".obj", ".fbx", ".stl"}
        ]

        return io.Schema(
            node_id="Load3D_V3",
            display_name="Load 3D _V3",
            category="3d",
            is_experimental=True,
            inputs=[
                io.Combo.Input("model_file", options=sorted(files), upload=io.UploadType.model),
                io.Load3D.Input("image"),
                io.Int.Input("width", default=1024, min=1, max=4096, step=1),
                io.Int.Input("height", default=1024, min=1, max=4096, step=1),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Mask.Output(display_name="mask"),
                io.String.Output(display_name="mesh_path"),
                io.Image.Output(display_name="normal"),
                io.Image.Output(display_name="lineart"),
                io.Load3DCamera.Output(display_name="camera_info"),
                io.Video.Output(display_name="recording_video"),
            ],
        )

    @classmethod
    def execute(cls, model_file, image, **kwargs):
        image_path = folder_paths.get_annotated_filepath(image["image"])
        mask_path = folder_paths.get_annotated_filepath(image["mask"])
        normal_path = folder_paths.get_annotated_filepath(image["normal"])
        lineart_path = folder_paths.get_annotated_filepath(image["lineart"])

        load_image_node = nodes.LoadImage()
        output_image, ignore_mask = load_image_node.load_image(image=image_path)
        ignore_image, output_mask = load_image_node.load_image(image=mask_path)
        normal_image, ignore_mask2 = load_image_node.load_image(image=normal_path)
        lineart_image, ignore_mask3 = load_image_node.load_image(image=lineart_path)

        video = None
        if image["recording"] != "":
            recording_video_path = folder_paths.get_annotated_filepath(image["recording"])
            video = VideoFromFile(recording_video_path)

        return io.NodeOutput(
            output_image, output_mask, model_file, normal_image, lineart_image, image["camera_info"], video
        )


class Load3DAnimation(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        input_dir = os.path.join(folder_paths.get_input_directory(), "3d")

        os.makedirs(input_dir, exist_ok=True)

        input_path = Path(input_dir)
        base_path = Path(folder_paths.get_input_directory())

        files = [
            normalize_path(str(file_path.relative_to(base_path)))
            for file_path in input_path.rglob("*")
            if file_path.suffix.lower() in {".gltf", ".glb", ".fbx"}
        ]

        return io.Schema(
            node_id="Load3DAnimation_V3",
            display_name="Load 3D - Animation _V3",
            category="3d",
            is_experimental=True,
            inputs=[
                io.Combo.Input("model_file", options=sorted(files), upload=io.UploadType.model),
                io.Load3DAnimation.Input("image"),
                io.Int.Input("width", default=1024, min=1, max=4096, step=1),
                io.Int.Input("height", default=1024, min=1, max=4096, step=1),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Mask.Output(display_name="mask"),
                io.String.Output(display_name="mesh_path"),
                io.Image.Output(display_name="normal"),
                io.Load3DCamera.Output(display_name="camera_info"),
                io.Video.Output(display_name="recording_video"),
            ],
        )

    @classmethod
    def execute(cls, model_file, image, **kwargs):
        image_path = folder_paths.get_annotated_filepath(image["image"])
        mask_path = folder_paths.get_annotated_filepath(image["mask"])
        normal_path = folder_paths.get_annotated_filepath(image["normal"])

        load_image_node = nodes.LoadImage()
        output_image, ignore_mask = load_image_node.load_image(image=image_path)
        ignore_image, output_mask = load_image_node.load_image(image=mask_path)
        normal_image, ignore_mask2 = load_image_node.load_image(image=normal_path)

        video = None
        if image['recording'] != "":
            recording_video_path = folder_paths.get_annotated_filepath(image["recording"])
            video = VideoFromFile(recording_video_path)

        return io.NodeOutput(output_image, output_mask, model_file, normal_image, image["camera_info"], video)


class Preview3D(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Preview3D_V3",  # frontend expects "Preview3D" to work
            display_name="Preview 3D _V3",
            category="3d",
            is_experimental=True,
            is_output_node=True,
            inputs=[
                io.String.Input("model_file", default="", multiline=False),
                io.Load3DCamera.Input("camera_info", optional=True),
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, model_file, camera_info=None):
        return io.NodeOutput(ui=ui.PreviewUI3D([model_file, camera_info], cls=cls))


class Preview3DAnimation(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Preview3DAnimation_V3",  # frontend expects "Preview3DAnimation" to work
            display_name="Preview 3D - Animation _V3",
            category="3d",
            is_experimental=True,
            is_output_node=True,
            inputs=[
                io.String.Input("model_file", default="", multiline=False),
                io.Load3DCamera.Input("camera_info", optional=True),
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, model_file, camera_info=None):
        return io.NodeOutput(ui=ui.PreviewUI3D([model_file, camera_info], cls=cls))


NODES_LIST = [
    Load3D,
    Load3DAnimation,
    Preview3D,
    Preview3DAnimation,
]
