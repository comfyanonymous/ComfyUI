#!/usr/bin/env python3
"""
Standalone script to generate music from text using Stable Audio in ComfyUI.
Based on the workflow: user/default/workflows/audio_stable_audio_example.json

This script replicates the workflow:
1. Load checkpoint model (stable-audio-open-1.0.safetensors)
2. Load CLIP text encoder (t5-base.safetensors)
3. Encode positive prompt (music description)
4. Encode negative prompt (empty)
5. Create empty latent audio (47.6 seconds)
6. Sample using KSampler
7. Decode audio from latent using VAE
8. Save as MP3

Requirements:
- stable-audio-open-1.0.safetensors in models/checkpoints/
- t5-base.safetensors in models/text_encoders/
"""

import torch
import sys
import os
import random
import av
from io import BytesIO

# Add ComfyUI to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

import comfy.sd
import comfy.sample
import comfy.samplers
import comfy.model_management
import folder_paths
import latent_preview
import comfy.utils


def load_checkpoint(ckpt_name):
    """Load checkpoint model - returns MODEL, CLIP, VAE"""
    print(f"Loading checkpoint: {ckpt_name}")
    ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
    out = comfy.sd.load_checkpoint_guess_config(
        ckpt_path, 
        output_vae=True, 
        output_clip=True, 
        embedding_directory=folder_paths.get_folder_paths("embeddings")
    )
    return out[:3]  # MODEL, CLIP, VAE


def load_clip(clip_name, clip_type="stable_audio"):
    """Load CLIP text encoder"""
    print(f"Loading CLIP: {clip_name}")
    clip_type_enum = getattr(comfy.sd.CLIPType, clip_type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
    
    clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
    clip = comfy.sd.load_clip(
        ckpt_paths=[clip_path], 
        embedding_directory=folder_paths.get_folder_paths("embeddings"), 
        clip_type=clip_type_enum, 
        model_options={}
    )
    return clip


def encode_text(clip, text):
    """Encode text using CLIP - returns CONDITIONING"""
    print(f"Encoding text: '{text}'")
    if clip is None:
        raise RuntimeError("ERROR: clip input is invalid: None")
    tokens = clip.tokenize(text)
    return clip.encode_from_tokens_scheduled(tokens)


def create_empty_latent_audio(seconds, batch_size=1):
    """Create empty latent audio tensor"""
    print(f"Creating empty latent audio: {seconds} seconds")
    length = round((seconds * 44100 / 2048) / 2) * 2
    latent = torch.zeros(
        [batch_size, 64, length], 
        device=comfy.model_management.intermediate_device()
    )
    return {"samples": latent, "type": "audio"}


def sample_audio(model, seed, steps, cfg, sampler_name, scheduler, 
                positive, negative, latent_image, denoise=1.0):
    """Run KSampler to generate audio latents"""
    print(f"Sampling with seed={seed}, steps={steps}, cfg={cfg}, sampler={sampler_name}, scheduler={scheduler}")
    
    latent_samples = latent_image["samples"]
    latent_samples = comfy.sample.fix_empty_latent_channels(model, latent_samples)
    
    # Prepare noise
    batch_inds = latent_image["batch_index"] if "batch_index" in latent_image else None
    noise = comfy.sample.prepare_noise(latent_samples, seed, batch_inds)
    
    # Check for noise mask
    noise_mask = latent_image.get("noise_mask", None)
    
    # Prepare callback for progress
    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    
    # Sample
    samples = comfy.sample.sample(
        model, noise, steps, cfg, sampler_name, scheduler, 
        positive, negative, latent_samples,
        denoise=denoise, 
        disable_noise=False, 
        start_step=None, 
        last_step=None,
        force_full_denoise=False, 
        noise_mask=noise_mask, 
        callback=callback, 
        disable_pbar=disable_pbar, 
        seed=seed
    )
    
    out = latent_image.copy()
    out["samples"] = samples
    return out


def decode_audio(vae, samples):
    """Decode audio from latent samples using VAE"""
    print("Decoding audio from latents")
    audio = vae.decode(samples["samples"]).movedim(-1, 1)
    
    # Normalize audio
    std = torch.std(audio, dim=[1, 2], keepdim=True) * 5.0
    std[std < 1.0] = 1.0
    audio /= std
    
    return {"waveform": audio, "sample_rate": 44100}


def save_audio_mp3(audio, filename, quality="V0"):
    """Save audio as MP3 file using PyAV (same as ComfyUI)"""
    print(f"Saving audio to: {filename}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]
    
    # Ensure audio is in CPU
    waveform = waveform.cpu()
    
    # Process each audio in batch (usually just 1)
    for batch_number, waveform_item in enumerate(waveform):
        if batch_number > 0:
            # Add batch number to filename if multiple
            base, ext = os.path.splitext(filename)
            output_path = f"{base}_{batch_number}{ext}"
        else:
            output_path = filename
        
        # Create output buffer
        output_buffer = BytesIO()
        output_container = av.open(output_buffer, mode="w", format="mp3")
        
        # Determine audio layout - waveform_item shape is [channels, samples]
        num_channels = waveform_item.shape[0] if waveform_item.dim() > 1 else 1
        layout = "mono" if num_channels == 1 else "stereo"
        
        # Set up the MP3 output stream
        out_stream = output_container.add_stream("libmp3lame", rate=sample_rate, layout=layout)
        
        # Set quality
        if quality == "V0":
            out_stream.codec_context.qscale = 1  # Highest VBR quality
        elif quality == "128k":
            out_stream.bit_rate = 128000
        elif quality == "320k":
            out_stream.bit_rate = 320000
        
        # Prepare waveform for PyAV: needs to be [samples, channels]
        # Use detach() to avoid gradient tracking issues
        if waveform_item.dim() == 1:
            # Mono audio, add channel dimension
            waveform_numpy = waveform_item.unsqueeze(1).float().detach().numpy()
        else:
            # Transpose from [channels, samples] to [samples, channels]
            waveform_numpy = waveform_item.transpose(0, 1).float().detach().numpy()
        
        # Reshape to [1, samples * channels] for PyAV
        waveform_numpy = waveform_numpy.reshape(1, -1)
        
        # Create audio frame
        frame = av.AudioFrame.from_ndarray(
            waveform_numpy,
            format="flt",
            layout=layout,
        )
        frame.sample_rate = sample_rate
        frame.pts = 0
        
        # Encode
        output_container.mux(out_stream.encode(frame))
        
        # Flush encoder
        output_container.mux(out_stream.encode(None))
        
        # Close container
        output_container.close()
        
        # Write to file
        output_buffer.seek(0)
        with open(output_path, "wb") as f:
            f.write(output_buffer.getbuffer())
        
        print(f"Audio saved successfully: {output_path}")


def main():
    # Configuration
    checkpoint_name = "stable-audio-open-1.0.safetensors"
    clip_name = "t5-base.safetensors"
    positive_prompt = "A soft melodious acoustic guitar music"
    negative_prompt = ""
    audio_duration = 47.6  # seconds
    seed = random.randint(0, 0xffffffffffffffff)  # Random seed, or use specific value
    steps = 50
    cfg = 4.98
    sampler_name = "dpmpp_3m_sde_gpu"
    scheduler = "exponential"
    denoise = 1.0
    output_filename = "output/audio/generated_music.mp3"
    quality = "V0"
    
    print("=" * 60)
    print("Stable Audio - Music Generation Script")
    print("=" * 60)
    print(f"Positive Prompt: {positive_prompt}")
    print(f"Duration: {audio_duration} seconds")
    print(f"Seed: {seed}")
    print("=" * 60)
    
    # 1. Load checkpoint (MODEL, CLIP, VAE)
    model, checkpoint_clip, vae = load_checkpoint(checkpoint_name)
    
    # 2. Load separate CLIP text encoder for stable audio
    clip = load_clip(clip_name, "stable_audio")
    
    # 3. Encode positive and negative prompts
    positive_conditioning = encode_text(clip, positive_prompt)
    negative_conditioning = encode_text(clip, negative_prompt)
    
    # 4. Create empty latent audio
    latent_audio = create_empty_latent_audio(audio_duration, batch_size=1)
    
    # 5. Sample using KSampler
    sampled_latent = sample_audio(
        model, seed, steps, cfg, sampler_name, scheduler,
        positive_conditioning, negative_conditioning, latent_audio, denoise
    )
    
    # 6. Decode audio from latent using VAE
    audio = decode_audio(vae, sampled_latent)
    
    # 7. Save as MP3
    save_audio_mp3(audio, output_filename, quality)
    
    print("=" * 60)
    print("Generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
