import os

import folder_paths
from comfy_api.latest import io
from comfy_api_sealed_worker.ply_types import PLY


class SavePLY(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SavePLY",
            display_name="Save PLY",
            category="3d",
            is_output_node=True,
            inputs=[
                io.Ply.Input("ply"),
                io.String.Input("filename_prefix", default="pointcloud/ComfyUI"),
            ],
        )

    @classmethod
    def execute(cls, ply: PLY, filename_prefix: str) -> io.NodeOutput:
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory()
        )
        f = f"{filename}_{counter:05}_.ply"
        ply.save_to(os.path.join(full_output_folder, f))
        return io.NodeOutput(ui={"pointclouds": [{"filename": f, "subfolder": subfolder, "type": "output"}]})


NODE_CLASS_MAPPINGS = {
    "SavePLY": SavePLY,
}
