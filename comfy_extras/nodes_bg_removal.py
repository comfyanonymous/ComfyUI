import folder_paths
from typing_extensions import override
from comfy_api.latest import ComfyExtension, IO
from comfy.bg_removal_model import load


class LoadBackGroundRemovalModel(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        files = folder_paths.get_filename_list("background_removal")
        return IO.Schema(
            node_id="LoadBackGroundRemovalModel",
            category="loaders",
            inputs=[
                IO.Combo.Input("background_removal_name", options=sorted(files)),
            ],
            outputs=[
                IO.BackgroundRemoval.Output("bg_model")
            ]
        )
    @classmethod
    def execute(cls, background_removal_name):
        path = folder_paths.get_full_path_or_raise("background_removal", background_removal_name)
        bg = load(path)
        if bg is None:
            raise RuntimeError("ERROR: clip vision file is invalid and does not contain a valid vision model.")
        return IO.NodeOutput(bg)

class RemoveBackGround(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="RemoveBackGround",
            category="image/background removal",
            inputs=[
                IO.Image.Input("image"),
                IO.BackgroundRemoval.Input("bg_removal_model")
            ],
            outputs=[
                IO.Mask.Output("mask")
            ]
        )
    @classmethod
    def execute(cls, image, bg_removal_model):
        mask = bg_removal_model.encode_image(image)
        return IO.NodeOutput(mask)

class BackGroundRemovalExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            LoadBackGroundRemovalModel,
            RemoveBackground
        ]


async def comfy_entrypoint() -> BackGroundRemovalExtension:
    return BackgroundRemovalExtension()
