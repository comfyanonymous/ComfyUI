import torch
from . import model_base
from . import utils

from . import sd1_clip
from . import sd2_clip
from . import sdxl_clip

from . import supported_models_base

class SD15(supported_models_base.BASE):
    unet_config = {
        "context_dim": 768,
        "model_channels": 320,
        "use_linear_in_transformer": False,
        "adm_in_channels": None,
    }

    unet_extra_config = {
        "num_heads": 8,
        "num_head_channels": -1,
    }

    vae_scale_factor = 0.18215

    def process_clip_state_dict(self, state_dict):
        k = list(state_dict.keys())
        for x in k:
            if x.startswith("cond_stage_model.transformer.") and not x.startswith("cond_stage_model.transformer.text_model."):
                y = x.replace("cond_stage_model.transformer.", "cond_stage_model.transformer.text_model.")
                state_dict[y] = state_dict.pop(x)

        if 'cond_stage_model.transformer.text_model.embeddings.position_ids' in state_dict:
            ids = state_dict['cond_stage_model.transformer.text_model.embeddings.position_ids']
            if ids.dtype == torch.float32:
                state_dict['cond_stage_model.transformer.text_model.embeddings.position_ids'] = ids.round()

        return state_dict

    def clip_target(self):
        return supported_models_base.ClipTarget(sd1_clip.SD1Tokenizer, sd1_clip.SD1ClipModel)

class SD20(supported_models_base.BASE):
    unet_config = {
        "context_dim": 1024,
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "adm_in_channels": None,
    }

    vae_scale_factor = 0.18215

    def v_prediction(self, state_dict):
        if self.unet_config["in_channels"] == 4: #SD2.0 inpainting models are not v prediction
            k = "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.norm1.bias"
            out = state_dict[k]
            if torch.std(out, unbiased=False) > 0.09: # not sure how well this will actually work. I guess we will find out.
                return True
        return False

    def process_clip_state_dict(self, state_dict):
        state_dict = utils.transformers_convert(state_dict, "cond_stage_model.model.", "cond_stage_model.transformer.text_model.", 24)
        return state_dict

    def clip_target(self):
        return supported_models_base.ClipTarget(sd2_clip.SD2Tokenizer, sd2_clip.SD2ClipModel)

class SD21UnclipL(SD20):
    unet_config = {
        "context_dim": 1024,
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "adm_in_channels": 1536,
    }

    clip_vision_prefix = "embedder.model.visual."
    noise_aug_config = {"noise_schedule_config": {"timesteps": 1000, "beta_schedule": "squaredcos_cap_v2"}, "timestep_dim": 768}


class SD21UnclipH(SD20):
    unet_config = {
        "context_dim": 1024,
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "adm_in_channels": 2048,
    }

    clip_vision_prefix = "embedder.model.visual."
    noise_aug_config = {"noise_schedule_config": {"timesteps": 1000, "beta_schedule": "squaredcos_cap_v2"}, "timestep_dim": 1024}

class SDXLRefiner(supported_models_base.BASE):
    unet_config = {
        "model_channels": 384,
        "use_linear_in_transformer": True,
        "context_dim": 1280,
        "adm_in_channels": 2560,
        "transformer_depth": [0, 4, 4, 0],
    }

    vae_scale_factor = 0.13025

    def get_model(self, state_dict):
        return model_base.SDXLRefiner(self.unet_config)

    def process_clip_state_dict(self, state_dict):
        keys_to_replace = {}
        replace_prefix = {}

        state_dict = utils.transformers_convert(state_dict, "conditioner.embedders.0.model.", "cond_stage_model.clip_g.transformer.text_model.", 32)
        keys_to_replace["conditioner.embedders.0.model.text_projection"] = "cond_stage_model.clip_g.text_projection"

        state_dict = supported_models_base.state_dict_key_replace(state_dict, keys_to_replace)
        return state_dict

    def clip_target(self):
        return supported_models_base.ClipTarget(sdxl_clip.SDXLTokenizer, sdxl_clip.SDXLRefinerClipModel)

class SDXL(supported_models_base.BASE):
    unet_config = {
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "transformer_depth": [0, 2, 10],
        "context_dim": 2048,
        "adm_in_channels": 2816
    }

    vae_scale_factor = 0.13025

    def get_model(self, state_dict):
        return model_base.SDXL(self.unet_config)

    def process_clip_state_dict(self, state_dict):
        keys_to_replace = {}
        replace_prefix = {}

        replace_prefix["conditioner.embedders.0.transformer.text_model"] = "cond_stage_model.clip_l.transformer.text_model"
        state_dict = utils.transformers_convert(state_dict, "conditioner.embedders.1.model.", "cond_stage_model.clip_g.transformer.text_model.", 32)
        keys_to_replace["conditioner.embedders.1.model.text_projection"] = "cond_stage_model.clip_g.text_projection"

        state_dict = supported_models_base.state_dict_prefix_replace(state_dict, replace_prefix)
        state_dict = supported_models_base.state_dict_key_replace(state_dict, keys_to_replace)
        return state_dict

    def clip_target(self):
        return supported_models_base.ClipTarget(sdxl_clip.SDXLTokenizer, sdxl_clip.SDXLClipModel)


models = [SD15, SD20, SD21UnclipL, SD21UnclipH, SDXLRefiner, SDXL]
