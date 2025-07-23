from __future__ import annotations

import comfy.model_management
import comfy.sd
import folder_paths
from comfy_api.v3 import io


class CLIPTextEncodeHiDream(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CLIPTextEncodeHiDream_V3",
            category="advanced/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("clip_l", multiline=True, dynamic_prompts=True),
                io.String.Input("clip_g", multiline=True, dynamic_prompts=True),
                io.String.Input("t5xxl", multiline=True, dynamic_prompts=True),
                io.String.Input("llama", multiline=True, dynamic_prompts=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ]
        )

    @classmethod
    def execute(cls, clip, clip_l, clip_g, t5xxl, llama):
        tokens = clip.tokenize(clip_g)
        tokens["l"] = clip.tokenize(clip_l)["l"]
        tokens["t5xxl"] = clip.tokenize(t5xxl)["t5xxl"]
        tokens["llama"] = clip.tokenize(llama)["llama"]
        return io.NodeOutput(clip.encode_from_tokens_scheduled(tokens))


class QuadrupleCLIPLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="QuadrupleCLIPLoader_V3",
            category="advanced/loaders",
            description="[Recipes]\n\nhidream: long clip-l, long clip-g, t5xxl, llama_8b_3.1_instruct",
            inputs=[
                io.Combo.Input("clip_name1", options=folder_paths.get_filename_list("text_encoders")),
                io.Combo.Input("clip_name2", options=folder_paths.get_filename_list("text_encoders")),
                io.Combo.Input("clip_name3", options=folder_paths.get_filename_list("text_encoders")),
                io.Combo.Input("clip_name4", options=folder_paths.get_filename_list("text_encoders")),
            ],
            outputs=[
                io.Clip.Output(),
            ]
        )

    @classmethod
    def execute(cls, clip_name1, clip_name2, clip_name3, clip_name4):
        clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", clip_name1)
        clip_path2 = folder_paths.get_full_path_or_raise("text_encoders", clip_name2)
        clip_path3 = folder_paths.get_full_path_or_raise("text_encoders", clip_name3)
        clip_path4 = folder_paths.get_full_path_or_raise("text_encoders", clip_name4)
        return io.NodeOutput(
            comfy.sd.load_clip(
                ckpt_paths=[clip_path1, clip_path2, clip_path3, clip_path4],
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
            )
        )


NODES_LIST = [
    CLIPTextEncodeHiDream,
    QuadrupleCLIPLoader,
]
