import io
import torch
import requests
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from torchvision.transforms import ToTensor
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    assign_to_checkpoint,
    conv_attn_to_linear,
    create_vae_diffusers_config,
    renew_vae_attention_paths,
    renew_vae_resnet_paths,
)
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    UniPCMultistepScheduler,
)

SCHEDULERS = {
    'DDIM' : DDIMScheduler,
    'DDPM' : DDPMScheduler,
    'DEISMultistep' : DEISMultistepScheduler,
    'DPMSolverMultistep' : DPMSolverMultistepScheduler,
    'DPMSolverSinglestep' : DPMSolverSinglestepScheduler,
    'EulerAncestralDiscrete' : EulerAncestralDiscreteScheduler,
    'EulerDiscrete' : EulerDiscreteScheduler,
    'HeunDiscrete' : HeunDiscreteScheduler,
    'KDPM2AncestralDiscrete' : KDPM2AncestralDiscreteScheduler,
    'KDPM2Discrete' : KDPM2DiscreteScheduler,
    'UniPCMultistep' : UniPCMultistepScheduler
}

SCHEDULERS_hunyuan = ["ddpm", "ddim", "dpmms"]

def token_auto_concat_embeds(pipe, positive, negative):
    max_length = pipe.tokenizer.model_max_length
    positive_length = pipe.tokenizer(positive, return_tensors="pt").input_ids.shape[-1]
    negative_length = pipe.tokenizer(negative, return_tensors="pt").input_ids.shape[-1]
    
    print(f'Token length is model maximum: {max_length}, positive length: {positive_length}, negative length: {negative_length}.')
    if max_length < positive_length or max_length < negative_length:
        print('Concatenated embedding.')
        if positive_length > negative_length:
            positive_ids = pipe.tokenizer(positive, return_tensors="pt").input_ids.to("cuda")
            negative_ids = pipe.tokenizer(negative, truncation=False, padding="max_length", max_length=positive_ids.shape[-1], return_tensors="pt").input_ids.to("cuda")
        else:
            negative_ids = pipe.tokenizer(negative, return_tensors="pt").input_ids.to("cuda")  
            positive_ids = pipe.tokenizer(positive, truncation=False, padding="max_length", max_length=negative_ids.shape[-1],  return_tensors="pt").input_ids.to("cuda")
    else:
        positive_ids = pipe.tokenizer(positive, truncation=False, padding="max_length", max_length=max_length,  return_tensors="pt").input_ids.to("cuda")
        negative_ids = pipe.tokenizer(negative, truncation=False, padding="max_length", max_length=max_length, return_tensors="pt").input_ids.to("cuda")
    
    positive_concat_embeds = []
    negative_concat_embeds = []
    for i in range(0, positive_ids.shape[-1], max_length):
        positive_concat_embeds.append(pipe.text_encoder(positive_ids[:, i: i + max_length])[0])
        negative_concat_embeds.append(pipe.text_encoder(negative_ids[:, i: i + max_length])[0])
    
    positive_prompt_embeds = torch.cat(positive_concat_embeds, dim=1)
    negative_prompt_embeds = torch.cat(negative_concat_embeds, dim=1)
    return positive_prompt_embeds, negative_prompt_embeds

# Reference from : https://github.com/huggingface/diffusers/blob/main/scripts/convert_vae_pt_to_diffusers.py
def custom_convert_ldm_vae_checkpoint(checkpoint, config):
    vae_state_dict = checkpoint

    new_checkpoint = {}

    new_checkpoint["encoder.conv_in.weight"] = vae_state_dict["encoder.conv_in.weight"]
    new_checkpoint["encoder.conv_in.bias"] = vae_state_dict["encoder.conv_in.bias"]
    new_checkpoint["encoder.conv_out.weight"] = vae_state_dict["encoder.conv_out.weight"]
    new_checkpoint["encoder.conv_out.bias"] = vae_state_dict["encoder.conv_out.bias"]
    new_checkpoint["encoder.conv_norm_out.weight"] = vae_state_dict["encoder.norm_out.weight"]
    new_checkpoint["encoder.conv_norm_out.bias"] = vae_state_dict["encoder.norm_out.bias"]

    new_checkpoint["decoder.conv_in.weight"] = vae_state_dict["decoder.conv_in.weight"]
    new_checkpoint["decoder.conv_in.bias"] = vae_state_dict["decoder.conv_in.bias"]
    new_checkpoint["decoder.conv_out.weight"] = vae_state_dict["decoder.conv_out.weight"]
    new_checkpoint["decoder.conv_out.bias"] = vae_state_dict["decoder.conv_out.bias"]
    new_checkpoint["decoder.conv_norm_out.weight"] = vae_state_dict["decoder.norm_out.weight"]
    new_checkpoint["decoder.conv_norm_out.bias"] = vae_state_dict["decoder.norm_out.bias"]

    new_checkpoint["quant_conv.weight"] = vae_state_dict["quant_conv.weight"]
    new_checkpoint["quant_conv.bias"] = vae_state_dict["quant_conv.bias"]
    new_checkpoint["post_quant_conv.weight"] = vae_state_dict["post_quant_conv.weight"]
    new_checkpoint["post_quant_conv.bias"] = vae_state_dict["post_quant_conv.bias"]

    # Retrieves the keys for the encoder down blocks only
    num_down_blocks = len({".".join(layer.split(".")[:3]) for layer in vae_state_dict if "encoder.down" in layer})
    down_blocks = {
        layer_id: [key for key in vae_state_dict if f"down.{layer_id}" in key] for layer_id in range(num_down_blocks)
    }

    # Retrieves the keys for the decoder up blocks only
    num_up_blocks = len({".".join(layer.split(".")[:3]) for layer in vae_state_dict if "decoder.up" in layer})
    up_blocks = {
        layer_id: [key for key in vae_state_dict if f"up.{layer_id}" in key] for layer_id in range(num_up_blocks)
    }

    for i in range(num_down_blocks):
        resnets = [key for key in down_blocks[i] if f"down.{i}" in key and f"down.{i}.downsample" not in key]

        if f"encoder.down.{i}.downsample.conv.weight" in vae_state_dict:
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.weight"] = vae_state_dict.pop(
                f"encoder.down.{i}.downsample.conv.weight"
            )
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.bias"] = vae_state_dict.pop(
                f"encoder.down.{i}.downsample.conv.bias"
            )

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"down.{i}.block", "new": f"down_blocks.{i}.resnets"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_resnets = [key for key in vae_state_dict if "encoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"encoder.mid.block_{i}" in key]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_attentions = [key for key in vae_state_dict if "encoder.mid.attn" in key]
    paths = renew_vae_attention_paths(mid_attentions)
    meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
    conv_attn_to_linear(new_checkpoint)

    for i in range(num_up_blocks):
        block_id = num_up_blocks - 1 - i
        resnets = [
            key for key in up_blocks[block_id] if f"up.{block_id}" in key and f"up.{block_id}.upsample" not in key
        ]

        if f"decoder.up.{block_id}.upsample.conv.weight" in vae_state_dict:
            new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.weight"] = vae_state_dict[
                f"decoder.up.{block_id}.upsample.conv.weight"
            ]
            new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.bias"] = vae_state_dict[
                f"decoder.up.{block_id}.upsample.conv.bias"
            ]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"up.{block_id}.block", "new": f"up_blocks.{i}.resnets"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_resnets = [key for key in vae_state_dict if "decoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"decoder.mid.block_{i}" in key]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_attentions = [key for key in vae_state_dict if "decoder.mid.attn" in key]
    paths = renew_vae_attention_paths(mid_attentions)
    meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
    conv_attn_to_linear(new_checkpoint)
    return new_checkpoint

# Reference from : https://github.com/huggingface/diffusers/blob/main/scripts/convert_vae_pt_to_diffusers.py
def vae_pt_to_vae_diffuser(
    checkpoint_path: str,
    output_path: str,
):
    # Only support V1
    r = requests.get(
        " https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml"
    )
    io_obj = io.BytesIO(r.content)

    original_config = OmegaConf.load(io_obj)
    image_size = 512
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if checkpoint_path.endswith("safetensors"):
        from safetensors import safe_open

        checkpoint = {}
        with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                checkpoint[key] = f.get_tensor(key)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)["state_dict"]

    # Convert the VAE model.
    vae_config = create_vae_diffusers_config(original_config, image_size=image_size)
    converted_vae_checkpoint = custom_convert_ldm_vae_checkpoint(checkpoint, vae_config)

    vae = AutoencoderKL(**vae_config)
    vae.load_state_dict(converted_vae_checkpoint)
    vae.save_pretrained(output_path)


def convert_images_to_tensors(images: list[Image.Image]):
    return torch.stack([np.transpose(ToTensor()(image), (1, 2, 0)) for image in images])

def convert_tensors_to_images(images: torch.tensor):
    return [Image.fromarray(np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8)) for image in images]

def resize_images(images: list[Image.Image], size: tuple[int, int]):
    return [image.resize(size) for image in images]