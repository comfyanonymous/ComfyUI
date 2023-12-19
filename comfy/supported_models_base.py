import torch
from . import model_base
from . import utils
from . import latent_formats

class ClipTarget:
    def __init__(self, tokenizer, clip):
        self.clip = clip
        self.tokenizer = tokenizer
        self.params = {}

class BASE:
    unet_config = {}
    unet_extra_config = {
        "num_heads": -1,
        "num_head_channels": 64,
    }

    clip_prefix = []
    clip_vision_prefix = None
    noise_aug_config = None
    sampling_settings = {}
    latent_format = latent_formats.LatentFormat

    manual_cast_dtype = None

    @classmethod
    def matches(s, unet_config):
        for k in s.unet_config:
            if s.unet_config[k] != unet_config[k]:
                return False
        return True

    def model_type(self, state_dict, prefix=""):
        return model_base.ModelType.EPS

    def inpaint_model(self):
        return self.unet_config["in_channels"] > 4

    def __init__(self, unet_config):
        self.unet_config = unet_config
        self.latent_format = self.latent_format()
        for x in self.unet_extra_config:
            self.unet_config[x] = self.unet_extra_config[x]

    def get_model(self, state_dict, prefix="", device=None):
        if self.noise_aug_config is not None:
            out = model_base.SD21UNCLIP(self, self.noise_aug_config, model_type=self.model_type(state_dict, prefix), device=device)
        else:
            out = model_base.BaseModel(self, model_type=self.model_type(state_dict, prefix), device=device)
        if self.inpaint_model():
            out.set_inpaint()
        return out

    def process_clip_state_dict(self, state_dict):
        return state_dict

    def process_unet_state_dict(self, state_dict):
        return state_dict

    def process_vae_state_dict(self, state_dict):
        return state_dict

    def process_clip_state_dict_for_saving(self, state_dict):
        replace_prefix = {"": "cond_stage_model."}
        return utils.state_dict_prefix_replace(state_dict, replace_prefix)

    def process_unet_state_dict_for_saving(self, state_dict):
        replace_prefix = {"": "model.diffusion_model."}
        return utils.state_dict_prefix_replace(state_dict, replace_prefix)

    def process_vae_state_dict_for_saving(self, state_dict):
        replace_prefix = {"": "first_stage_model."}
        return utils.state_dict_prefix_replace(state_dict, replace_prefix)

    def set_manual_cast(self, manual_cast_dtype):
        self.manual_cast_dtype = manual_cast_dtype
