import os
import torch
import comfy.sd
import comfy.utils

def first_file(path, filenames):
    for f in filenames:
        p = os.path.join(path, f)
        if os.path.exists(p):
            return p
    return None

def load_diffusers(model_path, output_vae=True, output_clip=True, embedding_directory=None, weight_dtype=torch.float16):
    """
    Load Stable Diffusion model components with custom precision.

    :param model_path: Path to the model directory.
    :param output_vae: Whether to load the VAE model.
    :param output_clip: Whether to load the CLIP model (text encoder).
    :param embedding_directory: Path to embedding directory.
    :param weight_dtype: Data type for model weights (torch.float16, torch.float32, torch.bfloat16).
    :return: (UNet, CLIP, VAE)
    """

    diffusion_model_names = ["diffusion_pytorch_model.fp16.safetensors", "diffusion_pytorch_model.safetensors",
                             "diffusion_pytorch_model.fp16.bin", "diffusion_pytorch_model.bin"]
    unet_path = first_file(os.path.join(model_path, "unet"), diffusion_model_names)
    vae_path = first_file(os.path.join(model_path, "vae"), diffusion_model_names)

    text_encoder_model_names = ["model.fp16.safetensors", "model.safetensors",
                                "pytorch_model.fp16.bin", "pytorch_model.bin"]
    text_encoder1_path = first_file(os.path.join(model_path, "text_encoder"), text_encoder_model_names)
    text_encoder2_path = first_file(os.path.join(model_path, "text_encoder_2"), text_encoder_model_names)

    text_encoder_paths = [text_encoder1_path] if text_encoder1_path else []
    if text_encoder2_path:
        text_encoder_paths.append(text_encoder2_path)

    unet = comfy.sd.load_diffusion_model(unet_path, dtype=weight_dtype)

    clip = None
    if output_clip and text_encoder_paths:
        clip = comfy.sd.load_clip(text_encoder_paths, embedding_directory=embedding_directory, dtype=weight_dtype)

    vae = None
    if output_vae and vae_path:
        sd = comfy.utils.load_torch_file(vae_path, map_location="cpu") 
        vae = comfy.sd.VAE(sd=sd).to(dtype=weight_dtype)  

    return (unet, clip, vae)
