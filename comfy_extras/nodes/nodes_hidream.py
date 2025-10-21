from typing_extensions import override

import comfy.sd
import comfy.model_management
from comfy_api.latest import ComfyExtension, io
from comfy.cmd import folder_paths
from comfy import model_downloader


class QuadrupleCLIPLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="QuadrupleCLIPLoader",
            category="advanced/loaders",
            description="[Recipes]\n\nhidream: long clip-l, long clip-g, t5xxl, llama_8b_3.1_instruct",
            inputs=[
                io.Combo.Input("clip_name1", options=model_downloader.get_filename_list("text_encoders")),
                io.Combo.Input("clip_name2", options=model_downloader.get_filename_list("text_encoders")),
                io.Combo.Input("clip_name3", options=model_downloader.get_filename_list("text_encoders")),
                io.Combo.Input("clip_name4", options=model_downloader.get_filename_list("text_encoders")),
            ],
            outputs=[
                io.Clip.Output(),
            ]
        )

    @classmethod
    def execute(cls, clip_name1, clip_name2, clip_name3, clip_name4):
        clip_path1 = model_downloader.get_full_path_or_raise("text_encoders", clip_name1)
        clip_path2 = model_downloader.get_full_path_or_raise("text_encoders", clip_name2)
        clip_path3 = model_downloader.get_full_path_or_raise("text_encoders", clip_name3)
        clip_path4 = model_downloader.get_full_path_or_raise("text_encoders", clip_name4)
        clip = comfy.sd.load_clip(ckpt_paths=[clip_path1, clip_path2, clip_path3, clip_path4], embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return io.NodeOutput(clip)


class CLIPTextEncodeHiDream(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CLIPTextEncodeHiDream",
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


class HiDreamExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            QuadrupleCLIPLoader,
            CLIPTextEncodeHiDream,
        ]


async def comfy_entrypoint() -> HiDreamExtension:
    return HiDreamExtension()
