import os

import folder_paths
from comfy_api.latest import io
from comfy_api_sealed_worker.npz_types import NPZ


class SaveNPZ(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveNPZ",
            display_name="Save NPZ",
            category="3d",
            is_output_node=True,
            inputs=[
                io.Npz.Input("npz"),
                io.String.Input("filename_prefix", default="da3_streaming/ComfyUI"),
            ],
        )

    @classmethod
    def execute(cls, npz: NPZ, filename_prefix: str) -> io.NodeOutput:
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory()
        )
        batch_dir = os.path.join(full_output_folder, f"{filename}_{counter:05}")
        os.makedirs(batch_dir, exist_ok=True)
        filenames = []
        for i, frame_bytes in enumerate(npz.frames):
            f = f"frame_{i:06d}.npz"
            with open(os.path.join(batch_dir, f), "wb") as fh:
                fh.write(frame_bytes)
            filenames.append(f)
        return io.NodeOutput(ui={"npz_files": [{"folder": os.path.join(subfolder, f"{filename}_{counter:05}"), "count": len(filenames), "type": "output"}]})


NODE_CLASS_MAPPINGS = {
    "SaveNPZ": SaveNPZ,
}
