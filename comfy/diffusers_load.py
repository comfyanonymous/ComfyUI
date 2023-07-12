import json
import os
import yaml

import folder_paths
from comfy.sd import load_checkpoint
import os.path as osp
import re
import torch
from safetensors.torch import load_file, save_file
from . import diffusers_convert


def load_diffusers(model_path, fp16=True, output_vae=True, output_clip=True, embedding_directory=None):
    diffusers_unet_conf = json.load(open(osp.join(model_path, "unet/config.json")))
    diffusers_scheduler_conf = json.load(open(osp.join(model_path, "scheduler/scheduler_config.json")))

    # magic
    v2 = diffusers_unet_conf["sample_size"] == 96
    if 'prediction_type' in diffusers_scheduler_conf:
        v_pred = diffusers_scheduler_conf['prediction_type'] == 'v_prediction'

    if v2:
        if v_pred:
            config_path = folder_paths.get_full_path("configs", 'v2-inference-v.yaml')
        else:
            config_path = folder_paths.get_full_path("configs", 'v2-inference.yaml')
    else:
        config_path = folder_paths.get_full_path("configs", 'v1-inference.yaml')

    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    model_config_params = config['model']['params']
    clip_config = model_config_params['cond_stage_config']
    scale_factor = model_config_params['scale_factor']
    vae_config = model_config_params['first_stage_config']
    vae_config['scale_factor'] = scale_factor
    model_config_params["unet_config"]["params"]["use_fp16"] = fp16

    unet_path = osp.join(model_path, "unet", "diffusion_pytorch_model.safetensors")
    vae_path = osp.join(model_path, "vae", "diffusion_pytorch_model.safetensors")
    text_enc_path = osp.join(model_path, "text_encoder", "model.safetensors")

    # Load models from safetensors if it exists, if it doesn't pytorch
    if osp.exists(unet_path):
        unet_state_dict = load_file(unet_path, device="cpu")
    else:
        unet_path = osp.join(model_path, "unet", "diffusion_pytorch_model.bin")
        unet_state_dict = torch.load(unet_path, map_location="cpu")

    if osp.exists(vae_path):
        vae_state_dict = load_file(vae_path, device="cpu")
    else:
        vae_path = osp.join(model_path, "vae", "diffusion_pytorch_model.bin")
        vae_state_dict = torch.load(vae_path, map_location="cpu")

    if osp.exists(text_enc_path):
        text_enc_dict = load_file(text_enc_path, device="cpu")
    else:
        text_enc_path = osp.join(model_path, "text_encoder", "pytorch_model.bin")
        text_enc_dict = torch.load(text_enc_path, map_location="cpu")

    # Convert the UNet model
    unet_state_dict = diffusers_convert.convert_unet_state_dict(unet_state_dict)
    unet_state_dict = {"model.diffusion_model." + k: v for k, v in unet_state_dict.items()}

    # Convert the VAE model
    vae_state_dict = diffusers_convert.convert_vae_state_dict(vae_state_dict)
    vae_state_dict = {"first_stage_model." + k: v for k, v in vae_state_dict.items()}

    # Easiest way to identify v2.0 model seems to be that the text encoder (OpenCLIP) is deeper
    is_v20_model = "text_model.encoder.layers.22.layer_norm2.bias" in text_enc_dict

    if is_v20_model:
        # Need to add the tag 'transformer' in advance so we can knock it out from the final layer-norm
        text_enc_dict = {"transformer." + k: v for k, v in text_enc_dict.items()}
        text_enc_dict = diffusers_convert.convert_text_enc_state_dict_v20(text_enc_dict)
        text_enc_dict = {"cond_stage_model.model." + k: v for k, v in text_enc_dict.items()}
    else:
        text_enc_dict = diffusers_convert.convert_text_enc_state_dict(text_enc_dict)
        text_enc_dict = {"cond_stage_model.transformer." + k: v for k, v in text_enc_dict.items()}

    # Put together new checkpoint
    sd = {**unet_state_dict, **vae_state_dict, **text_enc_dict}

    return load_checkpoint(embedding_directory=embedding_directory, state_dict=sd, config=config)
