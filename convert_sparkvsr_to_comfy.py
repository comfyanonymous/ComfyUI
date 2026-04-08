#!/usr/bin/env python3
"""Convert SparkVSR/CogVideoX diffusers checkpoint to ComfyUI format.

Usage:
    python convert_sparkvsr_to_comfy.py --model_dir path/to/sparkvsr-checkpoint \
        --output_dir ComfyUI/models/

This creates two files:
    - diffusion_models/cogvideox_sparkvsr.safetensors  (transformer)
    - vae/cogvideox_vae.safetensors                     (VAE)

T5-XXL text encoder does not need conversion — use existing ComfyUI T5 weights.
"""

import argparse
import os
import torch
from safetensors.torch import load_file, save_file


def remap_transformer_keys(state_dict):
    """Remap diffusers transformer keys to ComfyUI CogVideoX naming."""
    new_sd = {}
    for k, v in state_dict.items():
        new_key = k

        # Patch embedding
        new_key = new_key.replace("patch_embed.proj.", "patch_embed.proj.")
        new_key = new_key.replace("patch_embed.text_proj.", "patch_embed.text_proj.")
        new_key = new_key.replace("patch_embed.pos_embedding", "patch_embed.pos_embedding")

        # Time embedding: diffusers uses time_embedding.linear_1/2, we use time_embedding_linear_1/2
        new_key = new_key.replace("time_embedding.linear_1.", "time_embedding_linear_1.")
        new_key = new_key.replace("time_embedding.linear_2.", "time_embedding_linear_2.")

        # OFS embedding
        new_key = new_key.replace("ofs_embedding.linear_1.", "ofs_embedding_linear_1.")
        new_key = new_key.replace("ofs_embedding.linear_2.", "ofs_embedding_linear_2.")

        # Transformer blocks: diffusers uses transformer_blocks, we use blocks
        new_key = new_key.replace("transformer_blocks.", "blocks.")

        # Attention: diffusers uses attn1.to_q/k/v/out, we use q/k/v/attn_out
        new_key = new_key.replace(".attn1.to_q.", ".q.")
        new_key = new_key.replace(".attn1.to_k.", ".k.")
        new_key = new_key.replace(".attn1.to_v.", ".v.")
        new_key = new_key.replace(".attn1.to_out.0.", ".attn_out.")
        new_key = new_key.replace(".attn1.norm_q.", ".norm_q.")
        new_key = new_key.replace(".attn1.norm_k.", ".norm_k.")

        # Feed-forward: diffusers uses ff.net.0.proj/ff.net.2, we use ff_proj/ff_out
        new_key = new_key.replace(".ff.net.0.proj.", ".ff_proj.")
        new_key = new_key.replace(".ff.net.2.", ".ff_out.")

        # Output norms
        new_key = new_key.replace("norm_final.", "norm_final.")
        new_key = new_key.replace("norm_out.linear.", "norm_out.linear.")
        new_key = new_key.replace("norm_out.norm.", "norm_out.norm.")

        new_sd[new_key] = v

    return new_sd


def remap_vae_keys(state_dict):
    """Remap diffusers VAE keys to ComfyUI CogVideoX naming.

    The VAE architecture is directly ported so most keys should match.
    Main differences are in block naming.
    """
    new_sd = {}
    for k, v in state_dict.items():
        new_key = k

        # Encoder blocks
        new_key = new_key.replace("encoder.down_blocks.", "encoder.down_blocks.")
        new_key = new_key.replace("encoder.mid_block.", "encoder.mid_block.")

        # Decoder blocks
        new_key = new_key.replace("decoder.up_blocks.", "decoder.up_blocks.")
        new_key = new_key.replace("decoder.mid_block.", "decoder.mid_block.")

        # Resnet blocks within down/up/mid
        new_key = new_key.replace(".resnets.", ".resnets.")

        # CausalConv3d: diffusers stores as .conv.weight inside CausalConv3d
        # Our CausalConv3d also has .conv.weight, so this should match

        # Downsamplers/Upsamplers
        new_key = new_key.replace(".downsamplers.0.", ".downsamplers.0.")
        new_key = new_key.replace(".upsamplers.0.", ".upsamplers.0.")

        new_sd[new_key] = v

    return new_sd


def main():
    parser = argparse.ArgumentParser(description="Convert SparkVSR/CogVideoX to ComfyUI format")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to diffusers pipeline directory (contains transformer/, vae/, etc.)")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Output base directory (will create diffusion_models/ and vae/ subdirs)")
    args = parser.parse_args()

    # Load transformer
    transformer_dir = os.path.join(args.model_dir, "transformer")
    print(f"Loading transformer from {transformer_dir}...")
    transformer_sd = {}
    for f in sorted(os.listdir(transformer_dir)):
        if f.endswith(".safetensors"):
            sd = load_file(os.path.join(transformer_dir, f))
            transformer_sd.update(sd)

    transformer_sd = remap_transformer_keys(transformer_sd)

    out_dir = os.path.join(args.output_dir, "diffusion_models")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cogvideox_sparkvsr.safetensors")
    print(f"Saving transformer to {out_path} ({len(transformer_sd)} keys)")
    save_file(transformer_sd, out_path)

    # Load VAE
    vae_dir = os.path.join(args.model_dir, "vae")
    print(f"Loading VAE from {vae_dir}...")
    vae_sd = {}
    for f in sorted(os.listdir(vae_dir)):
        if f.endswith(".safetensors"):
            sd = load_file(os.path.join(vae_dir, f))
            vae_sd.update(sd)

    vae_sd = remap_vae_keys(vae_sd)

    out_dir = os.path.join(args.output_dir, "vae")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cogvideox_vae.safetensors")
    print(f"Saving VAE to {out_path} ({len(vae_sd)} keys)")
    save_file(vae_sd, out_path)

    print("Done! T5-XXL text encoder does not need conversion.")


if __name__ == "__main__":
    main()
