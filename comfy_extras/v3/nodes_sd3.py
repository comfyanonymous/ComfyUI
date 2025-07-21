from __future__ import annotations

import torch

import comfy.model_management
import comfy.sd
import folder_paths
import nodes
from comfy_api.v3 import io, resources
from comfy_extras.v3.nodes_slg import SkipLayerGuidanceDiT


class CLIPTextEncodeSD3(io.ComfyNodeV3):
    @classmethod
    def define_schema(cls):
        return io.SchemaV3(
            node_id="CLIPTextEncodeSD3_V3",
            category="advanced/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("clip_l", multiline=True, dynamic_prompts=True),
                io.String.Input("clip_g", multiline=True, dynamic_prompts=True),
                io.String.Input("t5xxl", multiline=True, dynamic_prompts=True),
                io.Combo.Input("empty_padding", options=["none", "empty_prompt"]),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, clip_l, clip_g, t5xxl, empty_padding: str):
        no_padding = empty_padding == "none"

        tokens = clip.tokenize(clip_g)
        if len(clip_g) == 0 and no_padding:
            tokens["g"] = []

        if len(clip_l) == 0 and no_padding:
            tokens["l"] = []
        else:
            tokens["l"] = clip.tokenize(clip_l)["l"]

        if len(t5xxl) == 0 and no_padding:
            tokens["t5xxl"] =  []
        else:
            tokens["t5xxl"] = clip.tokenize(t5xxl)["t5xxl"]
        if len(tokens["l"]) != len(tokens["g"]):
            empty = clip.tokenize("")
            while len(tokens["l"]) < len(tokens["g"]):
                tokens["l"] += empty["l"]
            while len(tokens["l"]) > len(tokens["g"]):
                tokens["g"] += empty["g"]
        return io.NodeOutput(clip.encode_from_tokens_scheduled(tokens))


class EmptySD3LatentImage(io.ComfyNodeV3):
    @classmethod
    def define_schema(cls):
        return io.SchemaV3(
            node_id="EmptySD3LatentImage_V3",
            category="latent/sd3",
            inputs=[
                io.Int.Input("width", default=1024, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=1024, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
            ],
            outputs=[
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(cls, width: int, height: int, batch_size=1):
        latent = torch.zeros(
            [batch_size, 16, height // 8, width // 8], device=comfy.model_management.intermediate_device()
        )
        return io.NodeOutput({"samples":latent})


class SkipLayerGuidanceSD3(SkipLayerGuidanceDiT):
    """
    Enhance guidance towards detailed dtructure by having another set of CFG negative with skipped layers.
    Inspired by Perturbed Attention Guidance (https://arxiv.org/abs/2403.17377)
    Experimental implementation by Dango233@StabilityAI.
    """
    @classmethod
    def define_schema(cls):
        return io.SchemaV3(
            node_id="SkipLayerGuidanceSD3_V3",
            category="advanced/guidance",
            inputs=[
                io.Model.Input("model"),
                io.String.Input("layers", default="7, 8, 9", multiline=False),
                io.Float.Input("scale", default=3.0, min=0.0, max=10.0, step=0.1),
                io.Float.Input("start_percent", default=0.01, min=0.0, max=1.0, step=0.001),
                io.Float.Input("end_percent", default=0.15, min=0.0, max=1.0, step=0.001),
            ],
            outputs=[
                io.Model.Output(),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, layers: str, scale: float, start_percent: float, end_percent: float):
        return SkipLayerGuidanceDiT.execute(
            model=model, scale=scale, start_percent=start_percent, end_percent=end_percent, double_layers=layers
        )


class TripleCLIPLoader(io.ComfyNodeV3):
    @classmethod
    def define_schema(cls):
        return io.SchemaV3(
            node_id="TripleCLIPLoader_V3",
            category="advanced/loaders",
            description="[Recipes]\n\nsd3: clip-l, clip-g, t5",
            inputs=[
                io.Combo.Input("clip_name1", options=folder_paths.get_filename_list("text_encoders")),
                io.Combo.Input("clip_name2", options=folder_paths.get_filename_list("text_encoders")),
                io.Combo.Input("clip_name3", options=folder_paths.get_filename_list("text_encoders")),
            ],
            outputs=[
                io.Clip.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip_name1: str, clip_name2: str, clip_name3: str):
        clip_data =[
            cls.resources.get(resources.TorchDictFolderFilename("text_encoders", clip_name1)),
            cls.resources.get(resources.TorchDictFolderFilename("text_encoders", clip_name2)),
            cls.resources.get(resources.TorchDictFolderFilename("text_encoders", clip_name3)),
        ]
        return io.NodeOutput(
            comfy.sd.load_text_encoder_state_dicts(
                clip_data, embedding_directory=folder_paths.get_folder_paths("embeddings")
            )
        )

NODES_LIST = [
    CLIPTextEncodeSD3,
    EmptySD3LatentImage,
    SkipLayerGuidanceSD3,
    TripleCLIPLoader,
]
