from comfy_api.latest import io, ComfyExtension
import comfy.asset_management
import comfy.sd
import folder_paths
import logging
import os


class AssetTestNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AssetTestNode",
            is_experimental=True,
            inputs=[
                io.Combo.Input("ckpt_name", folder_paths.get_filename_list("checkpoints")),
            ],
            outputs=[
                io.Model.Output(),
                io.Clip.Output(),
                io.Vae.Output(),
            ],
        )

    @classmethod
    def execute(cls, ckpt_name: str):
        hash = None
        # lets get the full path just so we can retrieve the hash from db, if exists
        try:
            full_path = folder_paths.get_full_path("checkpoints", ckpt_name)
            if full_path is None:
                raise Exception(f"Model {ckpt_name} not found")
            from app.model_processor import model_processor
            hash = model_processor.retrieve_hash(full_path)
        except Exception as e:
            logging.error(f"Could not get model by hash with error: {e}")
        subdir, name = os.path.split(ckpt_name)
        asset_info = comfy.asset_management.AssetInfo(hash=hash, name=name, tags=["models", "checkpoints"], metadata={"subdir": subdir})
        asset = comfy.asset_management.resolve(asset_info)
        # /\ the stuff above should happen in execution code instead of inside the node
        # \/ the stuff below should happen in the node - confirm is a model asset, do stuff to it (already loaded? or should be called to 'load'?)
        if asset is None:
            raise Exception(f"Model {asset_info.name} not found")
        assert isinstance(asset, comfy.asset_management.ModelReturnedAsset)
        out = comfy.sd.load_state_dict_guess_config(asset.state_dict, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"), metadata=asset.metadata)
        return io.NodeOutput(out[0], out[1], out[2])


class AssetTestExtension(ComfyExtension):
    @classmethod
    async def get_node_list(cls):
        return [AssetTestNode]


def comfy_entrypoint():
    return AssetTestExtension()
