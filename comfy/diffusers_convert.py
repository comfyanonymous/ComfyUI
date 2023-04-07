import json
import os
import yaml

# because of local import nonsense
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import folder_paths
from comfy.ldm.util import instantiate_from_config
from comfy.sd import ModelPatcher, load_model_weights, CLIP, VAE
import os.path as osp
import re
import torch
from safetensors.torch import load_file, save_file

# conversion code from https://github.com/huggingface/diffusers/blob/main/scripts/convert_diffusers_to_original_stable_diffusion.py

# =================#
# UNet Conversion #
# =================#

unet_conversion_map = [
    # (stable-diffusion, HF Diffusers)
    ("time_embed.0.weight", "time_embedding.linear_1.weight"),
    ("time_embed.0.bias", "time_embedding.linear_1.bias"),
    ("time_embed.2.weight", "time_embedding.linear_2.weight"),
    ("time_embed.2.bias", "time_embedding.linear_2.bias"),
    ("input_blocks.0.0.weight", "conv_in.weight"),
    ("input_blocks.0.0.bias", "conv_in.bias"),
    ("out.0.weight", "conv_norm_out.weight"),
    ("out.0.bias", "conv_norm_out.bias"),
    ("out.2.weight", "conv_out.weight"),
    ("out.2.bias", "conv_out.bias"),
]

unet_conversion_map_resnet = [
    # (stable-diffusion, HF Diffusers)
    ("in_layers.0", "norm1"),
    ("in_layers.2", "conv1"),
    ("out_layers.0", "norm2"),
    ("out_layers.3", "conv2"),
    ("emb_layers.1", "time_emb_proj"),
    ("skip_connection", "conv_shortcut"),
]

unet_conversion_map_layer = []
# hardcoded number of downblocks and resnets/attentions...
# would need smarter logic for other networks.
for i in range(4):
    # loop over downblocks/upblocks

    for j in range(2):
        # loop over resnets/attentions for downblocks
        hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
        sd_down_res_prefix = f"input_blocks.{3 * i + j + 1}.0."
        unet_conversion_map_layer.append((sd_down_res_prefix, hf_down_res_prefix))

        if i < 3:
            # no attention layers in down_blocks.3
            hf_down_atn_prefix = f"down_blocks.{i}.attentions.{j}."
            sd_down_atn_prefix = f"input_blocks.{3 * i + j + 1}.1."
            unet_conversion_map_layer.append((sd_down_atn_prefix, hf_down_atn_prefix))

    for j in range(3):
        # loop over resnets/attentions for upblocks
        hf_up_res_prefix = f"up_blocks.{i}.resnets.{j}."
        sd_up_res_prefix = f"output_blocks.{3 * i + j}.0."
        unet_conversion_map_layer.append((sd_up_res_prefix, hf_up_res_prefix))

        if i > 0:
            # no attention layers in up_blocks.0
            hf_up_atn_prefix = f"up_blocks.{i}.attentions.{j}."
            sd_up_atn_prefix = f"output_blocks.{3 * i + j}.1."
            unet_conversion_map_layer.append((sd_up_atn_prefix, hf_up_atn_prefix))

    if i < 3:
        # no downsample in down_blocks.3
        hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0.conv."
        sd_downsample_prefix = f"input_blocks.{3 * (i + 1)}.0.op."
        unet_conversion_map_layer.append((sd_downsample_prefix, hf_downsample_prefix))

        # no upsample in up_blocks.3
        hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
        sd_upsample_prefix = f"output_blocks.{3 * i + 2}.{1 if i == 0 else 2}."
        unet_conversion_map_layer.append((sd_upsample_prefix, hf_upsample_prefix))

hf_mid_atn_prefix = "mid_block.attentions.0."
sd_mid_atn_prefix = "middle_block.1."
unet_conversion_map_layer.append((sd_mid_atn_prefix, hf_mid_atn_prefix))

for j in range(2):
    hf_mid_res_prefix = f"mid_block.resnets.{j}."
    sd_mid_res_prefix = f"middle_block.{2 * j}."
    unet_conversion_map_layer.append((sd_mid_res_prefix, hf_mid_res_prefix))


def convert_unet_state_dict(unet_state_dict):
    # buyer beware: this is a *brittle* function,
    # and correct output requires that all of these pieces interact in
    # the exact order in which I have arranged them.
    mapping = {k: k for k in unet_state_dict.keys()}
    for sd_name, hf_name in unet_conversion_map:
        mapping[hf_name] = sd_name
    for k, v in mapping.items():
        if "resnets" in k:
            for sd_part, hf_part in unet_conversion_map_resnet:
                v = v.replace(hf_part, sd_part)
            mapping[k] = v
    for k, v in mapping.items():
        for sd_part, hf_part in unet_conversion_map_layer:
            v = v.replace(hf_part, sd_part)
        mapping[k] = v
    new_state_dict = {v: unet_state_dict[k] for k, v in mapping.items()}
    return new_state_dict


# ================#
# VAE Conversion #
# ================#

vae_conversion_map = [
    # (stable-diffusion, HF Diffusers)
    ("nin_shortcut", "conv_shortcut"),
    ("norm_out", "conv_norm_out"),
    ("mid.attn_1.", "mid_block.attentions.0."),
]

for i in range(4):
    # down_blocks have two resnets
    for j in range(2):
        hf_down_prefix = f"encoder.down_blocks.{i}.resnets.{j}."
        sd_down_prefix = f"encoder.down.{i}.block.{j}."
        vae_conversion_map.append((sd_down_prefix, hf_down_prefix))

    if i < 3:
        hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0."
        sd_downsample_prefix = f"down.{i}.downsample."
        vae_conversion_map.append((sd_downsample_prefix, hf_downsample_prefix))

        hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
        sd_upsample_prefix = f"up.{3 - i}.upsample."
        vae_conversion_map.append((sd_upsample_prefix, hf_upsample_prefix))

    # up_blocks have three resnets
    # also, up blocks in hf are numbered in reverse from sd
    for j in range(3):
        hf_up_prefix = f"decoder.up_blocks.{i}.resnets.{j}."
        sd_up_prefix = f"decoder.up.{3 - i}.block.{j}."
        vae_conversion_map.append((sd_up_prefix, hf_up_prefix))

# this part accounts for mid blocks in both the encoder and the decoder
for i in range(2):
    hf_mid_res_prefix = f"mid_block.resnets.{i}."
    sd_mid_res_prefix = f"mid.block_{i + 1}."
    vae_conversion_map.append((sd_mid_res_prefix, hf_mid_res_prefix))

vae_conversion_map_attn = [
    # (stable-diffusion, HF Diffusers)
    ("norm.", "group_norm."),
    ("q.", "query."),
    ("k.", "key."),
    ("v.", "value."),
    ("proj_out.", "proj_attn."),
]


def reshape_weight_for_sd(w):
    # convert HF linear weights to SD conv2d weights
    return w.reshape(*w.shape, 1, 1)


def convert_vae_state_dict(vae_state_dict):
    mapping = {k: k for k in vae_state_dict.keys()}
    for k, v in mapping.items():
        for sd_part, hf_part in vae_conversion_map:
            v = v.replace(hf_part, sd_part)
        mapping[k] = v
    for k, v in mapping.items():
        if "attentions" in k:
            for sd_part, hf_part in vae_conversion_map_attn:
                v = v.replace(hf_part, sd_part)
            mapping[k] = v
    new_state_dict = {v: vae_state_dict[k] for k, v in mapping.items()}
    weights_to_convert = ["q", "k", "v", "proj_out"]
    for k, v in new_state_dict.items():
        for weight_name in weights_to_convert:
            if f"mid.attn_1.{weight_name}.weight" in k:
                print(f"Reshaping {k} for SD format")
                new_state_dict[k] = reshape_weight_for_sd(v)
    return new_state_dict


# =========================#
# Text Encoder Conversion #
# =========================#


textenc_conversion_lst = [
    # (stable-diffusion, HF Diffusers)
    ("resblocks.", "text_model.encoder.layers."),
    ("ln_1", "layer_norm1"),
    ("ln_2", "layer_norm2"),
    (".c_fc.", ".fc1."),
    (".c_proj.", ".fc2."),
    (".attn", ".self_attn"),
    ("ln_final.", "transformer.text_model.final_layer_norm."),
    ("token_embedding.weight", "transformer.text_model.embeddings.token_embedding.weight"),
    ("positional_embedding", "transformer.text_model.embeddings.position_embedding.weight"),
]
protected = {re.escape(x[1]): x[0] for x in textenc_conversion_lst}
textenc_pattern = re.compile("|".join(protected.keys()))

# Ordering is from https://github.com/pytorch/pytorch/blob/master/test/cpp/api/modules.cpp
code2idx = {"q": 0, "k": 1, "v": 2}


def convert_text_enc_state_dict_v20(text_enc_dict):
    new_state_dict = {}
    capture_qkv_weight = {}
    capture_qkv_bias = {}
    for k, v in text_enc_dict.items():
        if (
                k.endswith(".self_attn.q_proj.weight")
                or k.endswith(".self_attn.k_proj.weight")
                or k.endswith(".self_attn.v_proj.weight")
        ):
            k_pre = k[: -len(".q_proj.weight")]
            k_code = k[-len("q_proj.weight")]
            if k_pre not in capture_qkv_weight:
                capture_qkv_weight[k_pre] = [None, None, None]
            capture_qkv_weight[k_pre][code2idx[k_code]] = v
            continue

        if (
                k.endswith(".self_attn.q_proj.bias")
                or k.endswith(".self_attn.k_proj.bias")
                or k.endswith(".self_attn.v_proj.bias")
        ):
            k_pre = k[: -len(".q_proj.bias")]
            k_code = k[-len("q_proj.bias")]
            if k_pre not in capture_qkv_bias:
                capture_qkv_bias[k_pre] = [None, None, None]
            capture_qkv_bias[k_pre][code2idx[k_code]] = v
            continue

        relabelled_key = textenc_pattern.sub(lambda m: protected[re.escape(m.group(0))], k)
        new_state_dict[relabelled_key] = v

    for k_pre, tensors in capture_qkv_weight.items():
        if None in tensors:
            raise Exception("CORRUPTED MODEL: one of the q-k-v values for the text encoder was missing")
        relabelled_key = textenc_pattern.sub(lambda m: protected[re.escape(m.group(0))], k_pre)
        new_state_dict[relabelled_key + ".in_proj_weight"] = torch.cat(tensors)

    for k_pre, tensors in capture_qkv_bias.items():
        if None in tensors:
            raise Exception("CORRUPTED MODEL: one of the q-k-v values for the text encoder was missing")
        relabelled_key = textenc_pattern.sub(lambda m: protected[re.escape(m.group(0))], k_pre)
        new_state_dict[relabelled_key + ".in_proj_bias"] = torch.cat(tensors)

    return new_state_dict


def convert_text_enc_state_dict(text_enc_dict):
    return text_enc_dict


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
    unet_state_dict = convert_unet_state_dict(unet_state_dict)
    unet_state_dict = {"model.diffusion_model." + k: v for k, v in unet_state_dict.items()}

    # Convert the VAE model
    vae_state_dict = convert_vae_state_dict(vae_state_dict)
    vae_state_dict = {"first_stage_model." + k: v for k, v in vae_state_dict.items()}

    # Easiest way to identify v2.0 model seems to be that the text encoder (OpenCLIP) is deeper
    is_v20_model = "text_model.encoder.layers.22.layer_norm2.bias" in text_enc_dict

    if is_v20_model:
        # Need to add the tag 'transformer' in advance so we can knock it out from the final layer-norm
        text_enc_dict = {"transformer." + k: v for k, v in text_enc_dict.items()}
        text_enc_dict = convert_text_enc_state_dict_v20(text_enc_dict)
        text_enc_dict = {"cond_stage_model.model." + k: v for k, v in text_enc_dict.items()}
    else:
        text_enc_dict = convert_text_enc_state_dict(text_enc_dict)
        text_enc_dict = {"cond_stage_model.transformer." + k: v for k, v in text_enc_dict.items()}

    # Put together new checkpoint
    sd = {**unet_state_dict, **vae_state_dict, **text_enc_dict}

    clip = None
    vae = None

    class WeightsLoader(torch.nn.Module):
        pass

    w = WeightsLoader()
    load_state_dict_to = []
    if output_vae:
        vae = VAE(scale_factor=scale_factor, config=vae_config)
        w.first_stage_model = vae.first_stage_model
        load_state_dict_to = [w]

    if output_clip:
        clip = CLIP(config=clip_config, embedding_directory=embedding_directory)
        w.cond_stage_model = clip.cond_stage_model
        load_state_dict_to = [w]

    model = instantiate_from_config(config["model"])
    model = load_model_weights(model, sd, verbose=False, load_state_dict_to=load_state_dict_to)

    if fp16:
        model = model.half()

    return ModelPatcher(model), clip, vae
