from .utils import load_torch_file, transformers_convert, state_dict_prefix_replace
import os
import torch
import json
import logging

import comfy.ops
import comfy.model_patcher
import comfy.model_management
import comfy.utils
import comfy.clip_model
import comfy.image_encoders.dino2

class Output:
    def __getitem__(self, key):
        return getattr(self, key)
    def __setitem__(self, key, item):
        setattr(self, key, item)


def cubic_kernel(x, a: float = -0.75):
    absx = x.abs()
    absx2 = absx ** 2
    absx3 = absx ** 3

    w = (a + 2) * absx3 - (a + 3) * absx2 + 1
    w2 = a * absx3 - 5*a * absx2 + 8*a * absx - 4*a

    return torch.where(absx <= 1, w, torch.where(absx < 2, w2, torch.zeros_like(x)))

def get_indices_weights(in_size, out_size, scale):
    # OpenCV-style half-pixel mapping
    x = torch.arange(out_size, dtype=torch.float32)
    x = (x + 0.5) / scale - 0.5

    x0 = x.floor().long()
    dx = x.unsqueeze(1) - (x0.unsqueeze(1) + torch.arange(-1, 3))

    weights = cubic_kernel(dx)
    weights = weights / weights.sum(dim=1, keepdim=True)

    indices = x0.unsqueeze(1) + torch.arange(-1, 3)
    indices = indices.clamp(0, in_size - 1)

    return indices, weights

def resize_cubic_1d(x, out_size, dim):
    b, c, h, w = x.shape
    in_size = h if dim == 2 else w
    scale = out_size / in_size

    indices, weights = get_indices_weights(in_size, out_size, scale)

    if dim == 2:
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, h)
    else:
        x = x.reshape(-1, w)

    gathered = x[:, indices]
    out = (gathered * weights.unsqueeze(0)).sum(dim=2)

    if dim == 2:
        out = out.reshape(b, c, w, out_size).permute(0, 1, 3, 2)
    else:
        out = out.reshape(b, c, h, out_size)

    return out

def resize_cubic(img: torch.Tensor, size: tuple) -> torch.Tensor:
    """
    Resize image using OpenCV-equivalent INTER_CUBIC interpolation.
    Implemented in pure PyTorch
    """

    if img.ndim == 3:
        img = img.unsqueeze(0)

    img = img.permute(0, 3, 1, 2)

    out_h, out_w = size
    img = resize_cubic_1d(img, out_h, dim=2)
    img = resize_cubic_1d(img, out_w, dim=3)
    return img

def resize_area(img: torch.Tensor, size: tuple) -> torch.Tensor:
    # vectorized implementation for OpenCV's INTER_AREA using pure PyTorch
    original_shape = img.shape
    is_hwc = False

    if img.ndim == 3:
        if img.shape[0] <= 4:
            img = img.unsqueeze(0)
        else:
            is_hwc = True
            img = img.permute(2, 0, 1).unsqueeze(0)
    elif img.ndim == 4:
        pass
    else:
        raise ValueError("Expected image with 3 or 4 dims.")

    B, C, H, W = img.shape
    out_h, out_w = size
    scale_y = H / out_h
    scale_x = W / out_w

    device = img.device

    # compute the grid boundries
    y_start = torch.arange(out_h, device=device).float() * scale_y
    y_end = y_start + scale_y
    x_start = torch.arange(out_w, device=device).float() * scale_x
    x_end = x_start + scale_x

    # for each output pixel, we will compute the range for it
    y_start_int = torch.floor(y_start).long()
    y_end_int = torch.ceil(y_end).long()
    x_start_int = torch.floor(x_start).long()
    x_end_int = torch.ceil(x_end).long()

    # We will build the weighted sums by iterating over contributing input pixels once
    output = torch.zeros((B, C, out_h, out_w), dtype=torch.float32, device=device)
    area = torch.zeros((out_h, out_w), dtype=torch.float32, device=device)

    max_kernel_h = int(torch.max(y_end_int - y_start_int).item())
    max_kernel_w = int(torch.max(x_end_int - x_start_int).item())

    for dy in range(max_kernel_h):
        for dx in range(max_kernel_w):
            # compute the weights for this offset for all output pixels

            y_idx = y_start_int.unsqueeze(1) + dy
            x_idx = x_start_int.unsqueeze(0) + dx

            # clamp indices to image boundaries
            y_idx_clamped = torch.clamp(y_idx, 0, H - 1)
            x_idx_clamped = torch.clamp(x_idx, 0, W - 1)

            # compute weights by broadcasting
            y_weight = (torch.min(y_end.unsqueeze(1), y_idx_clamped.float() + 1.0) - torch.max(y_start.unsqueeze(1), y_idx_clamped.float())).clamp(min=0)
            x_weight = (torch.min(x_end.unsqueeze(0), x_idx_clamped.float() + 1.0) - torch.max(x_start.unsqueeze(0), x_idx_clamped.float())).clamp(min=0)

            weight = (y_weight * x_weight)

            y_expand = y_idx_clamped.expand(out_h, out_w)
            x_expand = x_idx_clamped.expand(out_h, out_w)


            pixels = img[:, :, y_expand, x_expand]

            # unsqueeze to broadcast
            w = weight.unsqueeze(0).unsqueeze(0)

            output += pixels * w
            area += weight

    # Normalize by area
    output /= area.unsqueeze(0).unsqueeze(0)

    if is_hwc:
        return output[0].permute(1, 2, 0)
    elif img.shape[0] == 1 and original_shape[0] <= 4:
        return output[0]
    else:
        return output

def recenter(image, border_ratio: float = 0.2):

    if image.shape[-1] == 4:
        mask = image[..., 3]
    else:
        mask = torch.ones_like(image[..., 0:1]) * 255
        image = torch.concatenate([image, mask], axis=-1)
        mask = mask[..., 0]

    H, W, C = image.shape

    size = max(H, W)
    result = torch.zeros((size, size, C), dtype = torch.uint8)

    # as_tuple to match numpy behaviour
    x_coords, y_coords = torch.nonzero(mask, as_tuple=True)

    y_min, y_max = y_coords.min(), y_coords.max()
    x_min, x_max = x_coords.min(), x_coords.max()

    h = x_max - x_min
    w = y_max - y_min

    if h == 0 or w == 0:
        raise ValueError('input image is empty')

    desired_size = int(size * (1 - border_ratio))
    scale = desired_size / max(h, w)

    h2 = int(h * scale)
    w2 = int(w * scale)

    x2_min = (size - h2) // 2
    x2_max = x2_min + h2

    y2_min = (size - w2) // 2
    y2_max = y2_min + w2

    # note: opencv takes columns first (opposite to pytorch and numpy that take the row first)
    result[x2_min:x2_max, y2_min:y2_max] = resize_area(image[x_min:x_max, y_min:y_max], (h2, w2))

    bg = torch.ones((result.shape[0], result.shape[1], 3), dtype = torch.uint8) * 255

    mask = result[..., 3:].to(torch.float32) / 255
    result = result[..., :3] * mask + bg * (1 - mask)

    mask = mask * 255
    result = result.clip(0, 255).to(torch.uint8)
    mask = mask.clip(0, 255).to(torch.uint8)

    return result

def clip_preprocess(image, size=224, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711],
                    crop=True, value_range = (-1, 1), border_ratio: float = None, recenter_size: int = 512):

    if border_ratio is not None:

        image = (image * 255).clamp(0, 255).to(torch.uint8)
        image = [recenter(img, border_ratio = border_ratio) for img in image]

        image = torch.stack(image, dim = 0)
        image = resize_cubic(image, size = (recenter_size, recenter_size))

        image = image / 255 * 2 - 1
        low, high = value_range

        image = (image - low) / (high - low)
        image = image.permute(0, 2, 3, 1)

    image = image[:, :, :, :3] if image.shape[3] > 3 else image

    mean = torch.tensor(mean, device=image.device, dtype=image.dtype)
    std = torch.tensor(std, device=image.device, dtype=image.dtype)

    image = image.movedim(-1, 1)
    if not (image.shape[2] == size and image.shape[3] == size):
        if crop:
            scale = (size / min(image.shape[2], image.shape[3]))
            scale_size = (round(scale * image.shape[2]), round(scale * image.shape[3]))
        else:
            scale_size = (size, size)

        image = torch.nn.functional.interpolate(image, size=scale_size, mode="bilinear" if border_ratio is not None else "bicubic", antialias=True)
        h = (image.shape[2] - size)//2
        w = (image.shape[3] - size)//2
        image = image[:,:,h:h+size,w:w+size]
    image = torch.clip((255. * image), 0, 255).round() / 255.0
    return (image - mean.view([3,1,1])) / std.view([3,1,1])

IMAGE_ENCODERS = {
    "clip_vision_model": comfy.clip_model.CLIPVisionModelProjection,
    "siglip_vision_model": comfy.clip_model.CLIPVisionModelProjection,
    "dinov2": comfy.image_encoders.dino2.Dinov2Model,
}

class ClipVisionModel():
    def __init__(self, json_config):
        with open(json_config) as f:
            config = json.load(f)

        self.image_size = config.get("image_size", 224)
        self.image_mean = config.get("image_mean", [0.48145466, 0.4578275, 0.40821073])
        self.image_std = config.get("image_std", [0.26862954, 0.26130258, 0.27577711])
        model_type = config.get("model_type", "clip_vision_model")
        model_class = IMAGE_ENCODERS.get(model_type)
        if model_type == "siglip_vision_model":
            self.return_all_hidden_states = True
        else:
            self.return_all_hidden_states = False

        self.load_device = comfy.model_management.text_encoder_device()
        offload_device = comfy.model_management.text_encoder_offload_device()
        self.dtype = comfy.model_management.text_encoder_dtype(self.load_device)
        self.model = model_class(config, self.dtype, offload_device, comfy.ops.manual_cast)
        self.model.eval()

        self.patcher = comfy.model_patcher.ModelPatcher(self.model, load_device=self.load_device, offload_device=offload_device)

    def load_sd(self, sd):
        return self.model.load_state_dict(sd, strict=False)

    def get_sd(self):
        return self.model.state_dict()

    def encode_image(self, image, crop=True, border_ratio: float = None):
        comfy.model_management.load_model_gpu(self.patcher)
        pixel_values = clip_preprocess(image.to(self.load_device), size=self.image_size, mean=self.image_mean, std=self.image_std, crop=crop, border_ratio=border_ratio).float()
        out = self.model(pixel_values=pixel_values, intermediate_output='all' if self.return_all_hidden_states else -2)

        outputs = Output()
        outputs["last_hidden_state"] = out[0].to(comfy.model_management.intermediate_device())
        outputs["image_embeds"] = out[2].to(comfy.model_management.intermediate_device())
        if self.return_all_hidden_states:
            all_hs = out[1].to(comfy.model_management.intermediate_device())
            outputs["penultimate_hidden_states"] = all_hs[:, -2]
            outputs["all_hidden_states"] = all_hs
        else:
            outputs["penultimate_hidden_states"] = out[1].to(comfy.model_management.intermediate_device())

        outputs["mm_projected"] = out[3]
        return outputs

def convert_to_transformers(sd, prefix):
    sd_k = sd.keys()
    if "{}transformer.resblocks.0.attn.in_proj_weight".format(prefix) in sd_k:
        keys_to_replace = {
            "{}class_embedding".format(prefix): "vision_model.embeddings.class_embedding",
            "{}conv1.weight".format(prefix): "vision_model.embeddings.patch_embedding.weight",
            "{}positional_embedding".format(prefix): "vision_model.embeddings.position_embedding.weight",
            "{}ln_post.bias".format(prefix): "vision_model.post_layernorm.bias",
            "{}ln_post.weight".format(prefix): "vision_model.post_layernorm.weight",
            "{}ln_pre.bias".format(prefix): "vision_model.pre_layrnorm.bias",
            "{}ln_pre.weight".format(prefix): "vision_model.pre_layrnorm.weight",
        }

        for x in keys_to_replace:
            if x in sd_k:
                sd[keys_to_replace[x]] = sd.pop(x)

        if "{}proj".format(prefix) in sd_k:
            sd['visual_projection.weight'] = sd.pop("{}proj".format(prefix)).transpose(0, 1)

        sd = transformers_convert(sd, prefix, "vision_model.", 48)
    else:
        replace_prefix = {prefix: ""}
        sd = state_dict_prefix_replace(sd, replace_prefix)
    return sd

def load_clipvision_from_sd(sd, prefix="", convert_keys=False):
    if convert_keys:
        sd = convert_to_transformers(sd, prefix)
    if "vision_model.encoder.layers.47.layer_norm1.weight" in sd:
        json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "clip_vision_config_g.json")
    elif "vision_model.encoder.layers.30.layer_norm1.weight" in sd:
        json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "clip_vision_config_h.json")
    elif "vision_model.encoder.layers.22.layer_norm1.weight" in sd:
        embed_shape = sd["vision_model.embeddings.position_embedding.weight"].shape[0]
        if sd["vision_model.encoder.layers.0.layer_norm1.weight"].shape[0] == 1152:
            if embed_shape == 729:
                json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "clip_vision_siglip_384.json")
            elif embed_shape == 1024:
                json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "clip_vision_siglip_512.json")
        elif embed_shape == 577:
            if "multi_modal_projector.linear_1.bias" in sd:
                json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "clip_vision_config_vitl_336_llava.json")
            else:
                json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "clip_vision_config_vitl_336.json")
        else:
            json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "clip_vision_config_vitl.json")

    # Dinov2
    elif 'encoder.layer.39.layer_scale2.lambda1' in sd:
        json_config = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), "image_encoders"), "dino2_giant.json")
    elif 'encoder.layer.23.layer_scale2.lambda1' in sd:
        json_config = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), "image_encoders"), "dino2_large.json")
    else:
        return None

    clip = ClipVisionModel(json_config)
    m, u = clip.load_sd(sd)
    if len(m) > 0:
        logging.warning("missing clip vision: {}".format(m))
    u = set(u)
    keys = list(sd.keys())
    for k in keys:
        if k not in u:
            sd.pop(k)
    return clip

def load(ckpt_path):
    sd = load_torch_file(ckpt_path)
    if "visual.transformer.resblocks.0.attn.in_proj_weight" in sd:
        return load_clipvision_from_sd(sd, prefix="visual.", convert_keys=True)
    else:
        return load_clipvision_from_sd(sd)
