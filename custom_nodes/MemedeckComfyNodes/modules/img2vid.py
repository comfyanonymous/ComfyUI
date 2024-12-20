import io

import av
import comfy.latent_formats
import comfy.model_base
import comfy.model_management
import comfy.model_patcher
import comfy.sd
import comfy.supported_models_base
import comfy.utils
import numpy as np
import torch
from ltx_video.models.autoencoders.vae_encode import get_vae_size_scale_factor


def encode_single_frame(output_file, image_array: np.ndarray, crf):
    container = av.open(output_file, "w", format="mp4")
    try:
        stream = container.add_stream(
            "h264", rate=1, options={"crf": str(crf), "preset": "veryfast"}
        )
        stream.height = image_array.shape[0]
        stream.width = image_array.shape[1]
        av_frame = av.VideoFrame.from_ndarray(image_array, format="rgb24").reformat(
            format="yuv420p"
        )
        container.mux(stream.encode(av_frame))
        container.mux(stream.encode())
    finally:
        container.close()


def decode_single_frame(video_file):
    container = av.open(video_file)
    try:
        stream = next(s for s in container.streams if s.type == "video")
        frame = next(container.decode(stream))
    finally:
        container.close()
    return frame.to_ndarray(format="rgb24")


def videofy(image: torch.Tensor, crf=29):
    if crf == 0:
        return image

    image_array = (image * 255.0).byte().cpu().numpy()
    with io.BytesIO() as output_file:
        encode_single_frame(output_file, image_array, crf)
        video_bytes = output_file.getvalue()
    with io.BytesIO(video_bytes) as video_file:
        image_array = decode_single_frame(video_file)
    tensor = torch.tensor(image_array, dtype=image.dtype, device=image.device) / 255.0
    return tensor


def pad_tensor(tensor, target_len):
    dim = 2
    repeat_factor = target_len - tensor.shape[dim]  # Ceiling division
    last_element = tensor.select(dim, -1).unsqueeze(dim)
    padding = last_element.repeat(1, 1, repeat_factor, 1, 1)
    return torch.cat([tensor, padding], dim=dim)


def encode_media_conditioning(
    init_media, vae, width, height, frames_number, image_compression, initial_latent
):
    pixels = comfy.utils.common_upscale(
        init_media.movedim(-1, 1), width, height, "bilinear", ""
    ).movedim(1, -1)
    encode_pixels = pixels[:, :, :, :3]
    if image_compression > 0:
        for i in range(encode_pixels.shape[0]):
            image = videofy(encode_pixels[i], image_compression)
            encode_pixels[i] = image

    encoded_latents = vae.encode(encode_pixels).float()

    video_scale_factor, _, _ = get_vae_size_scale_factor(vae.first_stage_model)
    video_scale_factor = video_scale_factor if frames_number > 1 else 1
    target_len = (frames_number // video_scale_factor) + 1
    encoded_latents = encoded_latents[:, :, :target_len]

    if initial_latent is None:
        initial_latent = encoded_latents
    else:
        if encoded_latents.shape[2] > initial_latent.shape[2]:
            initial_latent = pad_tensor(initial_latent, encoded_latents.shape[2])
        initial_latent[:, :, : encoded_latents.shape[2], ...] = encoded_latents

    init_image_frame_number = init_media.shape[0]
    if init_image_frame_number == 1:
        result = pad_tensor(initial_latent, target_len)
    elif init_image_frame_number % 8 != 1:
        result = pad_tensor(initial_latent, target_len)
    else:
        result = initial_latent

    return result
