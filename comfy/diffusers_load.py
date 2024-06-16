from __future__ import annotations

import os

from . import sd, utils


def first_file(path, filenames) -> str | None:
    for f in filenames:
        p = os.path.join(path, f)
        if os.path.exists(p):
            return str(p)
    return None


def load_diffusers(model_path, output_vae=True, output_clip=True, embedding_directory=None):
    diffusion_model_names = ["diffusion_pytorch_model.fp16.safetensors", "diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.fp16.bin", "diffusion_pytorch_model.bin"]
    unet_path = first_file(os.path.join(model_path, "unet"), diffusion_model_names)
    vae_path = first_file(os.path.join(model_path, "vae"), diffusion_model_names)

    text_encoder_model_names = ["model.fp16.safetensors", "model.safetensors", "pytorch_model.fp16.bin", "pytorch_model.bin"]
    text_encoder1_path = first_file(os.path.join(model_path, "text_encoder"), text_encoder_model_names)
    text_encoder2_path = first_file(os.path.join(model_path, "text_encoder_2"), text_encoder_model_names)

    text_encoder_paths = [text_encoder1_path]
    if text_encoder2_path is not None:
        text_encoder_paths.append(text_encoder2_path)

    unet = None
    if unet_path is not None:
        unet = sd.load_unet(unet_path)

    clip = None
    textmodel_json_config1 = first_file(os.path.join(model_path, "text_encoder"), ["config.json"])
    if output_clip and not all(te is None for te in text_encoder_paths):
        clip = sd.load_clip(text_encoder_paths, embedding_directory=embedding_directory, textmodel_json_config=textmodel_json_config1)

    vae = None
    if output_vae and vae_path is not None:
        _sd = utils.load_torch_file(vae_path)
        vae = sd.VAE(sd=_sd)

    return unet, clip, vae
