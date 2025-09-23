import json
import logging
from typing import Optional

import torch

from . import clip_model
from . import model_management
from . import model_patcher
from . import ops
from .image_encoders import dino2
from .component_model import files
from .model_management import load_models_gpu
from .utils import load_torch_file, transformers_convert, state_dict_prefix_replace

logger = logging.getLogger(__name__)


class Output:
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, item):
        setattr(self, key, item)


def clip_preprocess(image, size=224, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711], crop=True):
    image = image[:, :, :, :3] if image.shape[3] > 3 else image
    mean = torch.tensor(mean, device=image.device, dtype=image.dtype)
    std = torch.tensor(std, device=image.device, dtype=image.dtype)
    image = image.movedim(-1, 1)
    if not (image.shape[2] == size and image.shape[3] == size):
        if crop:
            scale = (size / min(image.shape[2], image.shape[3]))
            scale_size = (round(scale * image.shape[2]), round(scale * image.shape[3]))
        else:
            scale_size = (size, size)

        image = torch.nn.functional.interpolate(image, size=scale_size, mode="bicubic", antialias=True)
        h = (image.shape[2] - size) // 2
        w = (image.shape[3] - size) // 2
        image = image[:, :, h:h + size, w:w + size]
    image = torch.clip((255. * image), 0, 255).round() / 255.0
    return (image - mean.view([3, 1, 1])) / std.view([3, 1, 1])


IMAGE_ENCODERS = {
    "clip_vision_model": clip_model.CLIPVisionModelProjection,
    "siglip_vision_model": clip_model.CLIPVisionModelProjection,
    "dinov2": dino2.Dinov2Model,
}


class ClipVisionModel():
    def __init__(self, json_config: dict | str):
        if isinstance(json_config, dict):
            config = json_config
        elif json_config is not None and isinstance(json_config, str):
            if json_config.startswith("{"):
                config = json.loads(json_config)
            else:
                with open(json_config) as f:
                    config = json.load(f)
        else:
            raise ValueError(f"json_config had invalid value={json_config}")

        self.image_size = config.get("image_size", 224)
        self.image_mean = config.get("image_mean", [0.48145466, 0.4578275, 0.40821073])
        self.image_std = config.get("image_std", [0.26862954, 0.26130258, 0.27577711])
        model_type = config.get("model_type", "clip_vision_model")
        model_class = IMAGE_ENCODERS.get(model_type)
        if model_type == "siglip_vision_model":
            self.return_all_hidden_states = True
        else:
            self.return_all_hidden_states = False

        self.load_device = model_management.text_encoder_device()
        offload_device = model_management.text_encoder_offload_device()
        self.dtype = model_management.text_encoder_dtype(self.load_device)
        self.model = model_class(config, self.dtype, offload_device, ops.manual_cast)
        self.model.eval()

        self.patcher = model_patcher.ModelPatcher(self.model, load_device=self.load_device, offload_device=offload_device)

    def load_sd(self, sd):
        return self.model.load_state_dict(sd, strict=False)

    def get_sd(self):
        return self.model.state_dict()

    def encode_image(self, image, crop=True):
        load_models_gpu([self.patcher])
        pixel_values = clip_preprocess(image.to(self.load_device), size=self.image_size, mean=self.image_mean, std=self.image_std, crop=crop).float()
        out = self.model(pixel_values=pixel_values, intermediate_output='all' if self.return_all_hidden_states else -2)

        outputs = Output()
        outputs["last_hidden_state"] = out[0].to(model_management.intermediate_device())
        outputs["image_embeds"] = out[2].to(model_management.intermediate_device())
        if self.return_all_hidden_states:
            all_hs = out[1].to(model_management.intermediate_device())
            outputs["penultimate_hidden_states"] = all_hs[:, -2]
            outputs["all_hidden_states"] = all_hs
        else:
            outputs["penultimate_hidden_states"] = out[1].to(model_management.intermediate_device())
        outputs["mm_projected"] = out[3]
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

        sd = transformers_convert(sd, prefix, "vision_model.", 48)
    else:
        replace_prefix = {prefix: ""}
        sd = state_dict_prefix_replace(sd, replace_prefix)
    return sd


def load_clipvision_from_sd(sd, prefix="", convert_keys=False) -> Optional[ClipVisionModel]:
    json_config: dict = {}
    if convert_keys:
        sd = convert_to_transformers(sd, prefix)
    if "vision_model.encoder.layers.47.layer_norm1.weight" in sd:
        json_config = files.get_path_as_dict(None, "clip_vision_config_g.json")
    elif "vision_model.encoder.layers.30.layer_norm1.weight" in sd:
        json_config = files.get_path_as_dict(None, "clip_vision_config_h.json")
    elif "vision_model.encoder.layers.22.layer_norm1.weight" in sd:
        embed_shape = sd["vision_model.embeddings.position_embedding.weight"].shape[0]
        if sd["vision_model.encoder.layers.0.layer_norm1.weight"].shape[0] == 1152:
            if embed_shape == 729:
                json_config = files.get_path_as_dict(None, "clip_vision_siglip_384.json")
            elif embed_shape == 1024:
                json_config = files.get_path_as_dict(None, "clip_vision_siglip_512.json")
        elif embed_shape == 577:
            if "multi_modal_projector.linear_1.bias" in sd:
                json_config = files.get_path_as_dict(None, "clip_vision_config_vitl_336_llava.json")
            else:
                json_config = files.get_path_as_dict(None, "clip_vision_config_vitl_336.json")
        else:
            json_config = files.get_path_as_dict(None, "clip_vision_config_vitl.json")

    # Dinov2
    elif 'encoder.layer.39.layer_scale2.lambda1' in sd:
        json_config = files.get_path_as_dict(None, "dino2_giant.json", package="comfy.image_encoders")
    elif 'encoder.layer.23.layer_scale2.lambda1' in sd:
        json_config = files.get_path_as_dict(None, "dino2_large.json", package="comfy.image_encoders")
    else:
        return None

    clip = ClipVisionModel(json_config)
    m, u = clip.load_sd(sd)
    if len(m) > 0:
        logger.warning("missing clip vision: {}".format(m))
    u = set(u)
    keys = list(sd.keys())
    for k in keys:
        if k not in u:
            sd.pop(k)
    return clip


def load(ckpt_path):
    sd = load_torch_file(ckpt_path)
    if "visual.transformer.resblocks.0.attn.in_proj_weight" in sd:
        return load_clipvision_from_sd(sd, prefix="visual.", convert_keys=True)
    else:
        return load_clipvision_from_sd(sd)
