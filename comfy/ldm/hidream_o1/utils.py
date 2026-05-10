"""HiDream-O1 input-prep helpers: image/resolution math and unified-sequence
RoPE position-id assembly. The fix_point offset in get_rope_index_fix_point
lets the target image and patchified ref images share spatial RoPE positions
despite living at different sequence indices — same 2D image plane.
"""

import math
from typing import Optional

import torch
from PIL import Image


PREDEFINED_RESOLUTIONS = [
    (2048, 2048),
    (2304, 1728),
    (1728, 2304),
    (2560, 1440),
    (1440, 2560),
    (2496, 1664),
    (1664, 2496),
    (3104, 1312),
    (1312, 3104),
    (2304, 1792),
    (1792, 2304),
]

PATCH_SIZE = 32
CONDITION_IMAGE_SIZE = 384  # ViT-side base size for ref images


def find_closest_resolution(width, height):
    """Closest (W, H) in PREDEFINED_RESOLUTIONS by aspect ratio."""
    img_ratio = width / height
    best = None
    min_diff = float("inf")
    for w, h in PREDEFINED_RESOLUTIONS:
        diff = abs(w / h - img_ratio)
        if diff < min_diff:
            min_diff = diff
            best = (w, h)
    return best


def resize_pilimage(pil_image, image_size, patch_size=16, resampler=Image.BICUBIC):
    """Resize to fit image_size**2 area, patch-aligned, center-cropped. Pre-halves
    with BOX filter while the image is still very large.
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX,
        )

    m = patch_size
    width, height = pil_image.width, pil_image.height
    s_max = image_size * image_size
    scale = math.sqrt(s_max / (width * height))

    candidates = [
        (round(width * scale) // m * m, round(height * scale) // m * m),
        (round(width * scale) // m * m, math.floor(height * scale) // m * m),
        (math.floor(width * scale) // m * m, round(height * scale) // m * m),
        (math.floor(width * scale) // m * m, math.floor(height * scale) // m * m),
    ]
    candidates = sorted(candidates, key=lambda x: x[0] * x[1], reverse=True)
    new_size = candidates[-1]
    for c in candidates:
        if c[0] * c[1] <= s_max:
            new_size = c
            break

    s1 = width / new_size[0]
    s2 = height / new_size[1]
    if s1 < s2:
        pil_image = pil_image.resize([new_size[0], round(height / s1)], resample=resampler)
        top = (round(height / s1) - new_size[1]) // 2
        pil_image = pil_image.crop((0, top, new_size[0], top + new_size[1]))
    else:
        pil_image = pil_image.resize([round(width / s2), new_size[1]], resample=resampler)
        left = (round(width / s2) - new_size[0]) // 2
        pil_image = pil_image.crop((left, 0, left + new_size[0], new_size[1]))
    return pil_image


def calculate_dimensions(max_size, ratio):
    """(W, H) for an aspect ratio fitting in max_size**2 area, 32-aligned."""
    width = math.sqrt(max_size * max_size * ratio)
    height = width / ratio
    width = int(width / 32) * 32
    height = int(height / 32) * 32
    return width, height


def ref_max_size(target_max_dim, k):
    """K-dependent ref-image max dim before patchifying."""
    if k == 1:
        return target_max_dim
    if k == 2:
        return target_max_dim * 48 // 64
    if k <= 4:
        return target_max_dim // 2
    if k <= 8:
        return target_max_dim * 24 // 64
    return target_max_dim // 4


def cond_image_size(k):
    """K-dependent ViT-side image size."""
    if k <= 4:
        return CONDITION_IMAGE_SIZE
    if k <= 8:
        return CONDITION_IMAGE_SIZE * 48 // 64
    return CONDITION_IMAGE_SIZE // 2


def get_rope_index_fix_point(
    spatial_merge_size: int,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    skip_vision_start_token=None,
    fix_point: int = 4096,
):
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3, input_ids.shape[0], input_ids.shape[1],
            dtype=input_ids.dtype, device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids_b in enumerate(total_input_ids):
            input_ids_b = input_ids_b[attention_mask[i] == 1]
            vision_start_indices = torch.argwhere(input_ids_b == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids_b[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids_b.tolist()
            llm_pos_ids_list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t = image_grid_thw[image_index][0]
                    h = image_grid_thw[image_index][1]
                    w = image_grid_thw[image_index][2]
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t = video_grid_thw[video_index][0]
                    h = video_grid_thw[video_index][1]
                    w = video_grid_thw[video_index][2]
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t = t.item()
                llm_grid_h = h.item() // spatial_merge_size
                llm_grid_w = w.item() // spatial_merge_size
                text_len = ed - st
                text_len -= skip_vision_start_token[image_index - 1]
                text_len = max(0, text_len)
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()

                if skip_vision_start_token[image_index - 1]:
                    if fix_point > 0:
                        fix_point = fix_point - st_idx
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + fix_point + st_idx)
                    fix_point = 0
                else:
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas

    if attention_mask is not None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
        mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
    else:
        position_ids = (
            torch.arange(input_ids.shape[1], device=input_ids.device)
            .view(1, 1, -1).expand(3, input_ids.shape[0], -1)
        )
        mrope_position_deltas = torch.zeros(
            [input_ids.shape[0], 1], device=input_ids.device, dtype=input_ids.dtype,
        )
    return position_ids, mrope_position_deltas
