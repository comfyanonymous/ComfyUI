from transformers import CLIPVisionModelWithProjection, CLIPVisionConfig, CLIPImageProcessor, modeling_utils
from .utils import load_torch_file, transformers_convert
import os
import torch
import comfy.ops

class ClipVisionModel():
    def __init__(self, json_config):
        config = CLIPVisionConfig.from_json_file(json_config)
        with comfy.ops.use_comfy_ops():
            with modeling_utils.no_init_weights():
                self.model = CLIPVisionModelWithProjection(config)
        self.processor = CLIPImageProcessor(crop_size=224,
                                            do_center_crop=True,
                                            do_convert_rgb=True,
                                            do_normalize=True,
                                            do_resize=True,
                                            image_mean=[ 0.48145466,0.4578275,0.40821073],
                                            image_std=[0.26862954,0.26130258,0.27577711],
                                            resample=3, #bicubic
                                            size=224)

    def load_sd(self, sd):
        return self.model.load_state_dict(sd, strict=False)

    def encode_image(self, image):
        img = torch.clip((255. * image[0]), 0, 255).round().int()
        inputs = self.processor(images=[img], return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs

def convert_to_transformers(sd, prefix):
    sd_k = sd.keys()
    if "{}transformer.resblocks.0.attn.in_proj_weight".format(prefix) in sd_k:
        keys_to_replace = {
            "{}class_embedding".format(prefix): "vision_model.embeddings.class_embedding",
            "{}conv1.weight".format(prefix): "vision_model.embeddings.patch_embedding.weight",
            "{}positional_embedding".format(prefix): "vision_model.embeddings.position_embedding.weight",
            "{}ln_post.bias".format(prefix): "vision_model.post_layernorm.bias",
            "{}ln_post.weight".format(prefix): "vision_model.post_layernorm.weight",
            "{}ln_pre.bias".format(prefix): "vision_model.pre_layrnorm.bias",
            "{}ln_pre.weight".format(prefix): "vision_model.pre_layrnorm.weight",
        }

        for x in keys_to_replace:
            if x in sd_k:
                sd[keys_to_replace[x]] = sd.pop(x)

        if "{}proj".format(prefix) in sd_k:
            sd['visual_projection.weight'] = sd.pop("{}proj".format(prefix)).transpose(0, 1)

        sd = transformers_convert(sd, prefix, "vision_model.", 32)
    return sd

def load_clipvision_from_sd(sd, prefix="", convert_keys=False):
    if convert_keys:
        sd = convert_to_transformers(sd, prefix)
    if "vision_model.encoder.layers.30.layer_norm1.weight" in sd:
        json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "clip_vision_config_h.json")
    else:
        json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "clip_vision_config_vitl.json")
    clip = ClipVisionModel(json_config)
    m, u = clip.load_sd(sd)
    u = set(u)
    keys = list(sd.keys())
    for k in keys:
        if k not in u:
            t = sd.pop(k)
            del t
    return clip

def load(ckpt_path):
    sd = load_torch_file(ckpt_path)
    return load_clipvision_from_sd(sd)
