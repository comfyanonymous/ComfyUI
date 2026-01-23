import nodes
import folder_paths
import os

from typing_extensions import override
from comfy_api.latest import IO, ComfyExtension, InputImpl, UI

from pathlib import Path


def normalize_path(path):
    return path.replace('\\', '/')

class Load3D(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        input_dir = os.path.join(folder_paths.get_input_directory(), "3d")

        os.makedirs(input_dir, exist_ok=True)

        input_path = Path(input_dir)
        base_path = Path(folder_paths.get_input_directory())

        files = [
            normalize_path(str(file_path.relative_to(base_path)))
            for file_path in input_path.rglob("*")
            if file_path.suffix.lower() in {'.gltf', '.glb', '.obj', '.fbx', '.stl', '.spz', '.splat', '.ply', '.ksplat'}
        ]
        return IO.Schema(
            node_id="Load3D",
            display_name="Load 3D & Animation",
            category="3d",
            is_experimental=True,
            inputs=[
                IO.Combo.Input("model_file", options=sorted(files), upload=IO.UploadType.model),
                IO.Load3D.Input("image"),
                IO.Int.Input("width", default=1024, min=1, max=4096, step=1),
                IO.Int.Input("height", default=1024, min=1, max=4096, step=1),
            ],
            outputs=[
                IO.Image.Output(display_name="image"),
                IO.Mask.Output(display_name="mask"),
                IO.String.Output(display_name="mesh_path"),
                IO.Image.Output(display_name="normal"),
                IO.Load3DCamera.Output(display_name="camera_info"),
                IO.Video.Output(display_name="recording_video"),
            ],
        )

    @classmethod
    def execute(cls, model_file, image, **kwargs) -> IO.NodeOutput:
        image_path = folder_paths.get_annotated_filepath(image['image'])
        mask_path = folder_paths.get_annotated_filepath(image['mask'])
        normal_path = folder_paths.get_annotated_filepath(image['normal'])

        load_image_node = nodes.LoadImage()
        output_image, ignore_mask = load_image_node.load_image(image=image_path)
        ignore_image, output_mask = load_image_node.load_image(image=mask_path)
        normal_image, ignore_mask2 = load_image_node.load_image(image=normal_path)

        video = None

        if image['recording'] != "":
            recording_video_path = folder_paths.get_annotated_filepath(image['recording'])

            video = InputImpl.VideoFromFile(recording_video_path)

        return IO.NodeOutput(output_image, output_mask, model_file, normal_image, image['camera_info'], video)

    process = execute  # TODO: remove


class Preview3D(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Preview3D",
            search_aliases=["view mesh", "3d viewer"],
            display_name="Preview 3D & Animation",
            category="3d",
            is_experimental=True,
            is_output_node=True,
            inputs=[
                IO.String.Input("model_file", default="", multiline=False),
                IO.Load3DCamera.Input("camera_info", optional=True),
                IO.Image.Input("bg_image", optional=True),
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, model_file, **kwargs) -> IO.NodeOutput:
        camera_info = kwargs.get("camera_info", None)
        bg_image = kwargs.get("bg_image", None)
        return IO.NodeOutput(ui=UI.PreviewUI3D(model_file, camera_info, bg_image=bg_image))

    process = execute  # TODO: remove


class Load3DExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            Load3D,
            Preview3D,
        ]


async def comfy_entrypoint() -> Load3DExtension:
    return Load3DExtension()
