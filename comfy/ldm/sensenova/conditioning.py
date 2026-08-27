import math

import torch
import torch.nn.functional as F


IMAGE_CONTEXT_ID = 151669
IMAGE_START_ID = 151670
IMAGE_END_ID = 151671
IM_START_ID = 151644
IM_END_ID = 151645
USER_ID = 872
ASSISTANT_ID = 77091
NEWLINE_ID = 198
IMAGE_LABEL_IDS = (
    (1906, 12, 16, 25),
    (1906, 12, 17, 25),
    (1906, 12, 18, 25),
    (1906, 12, 19, 25),
    (1906, 12, 20, 25),
    (1906, 12, 21, 25),
    (1906, 12, 22, 25),
    (1906, 12, 23, 25),
    (1906, 12, 24, 25),
    (1906, 12, 16, 15, 25),
)


def smart_resize(
    height, width, factor=32, min_pixels=512 * 512, max_pixels=2048 * 2048
):
    resized_height = max(factor, round(height / factor) * factor)
    resized_width = max(factor, round(width / factor) * factor)
    if resized_height * resized_width > max_pixels:
        scale = math.sqrt((height * width) / max_pixels)
        resized_height = max(factor, math.floor(height / scale / factor) * factor)
        resized_width = max(factor, math.floor(width / scale / factor) * factor)
    elif resized_height * resized_width < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        resized_height = math.ceil(height * scale / factor) * factor
        resized_width = math.ceil(width * scale / factor) * factor
    return resized_height, resized_width


def preprocess_reference(image, max_pixels=2048 * 2048):
    if image.ndim != 4 or image.shape[-1] < 3:
        raise ValueError("SenseNova reference image must be IMAGE in [B,H,W,C] layout")
    if image.shape[0] != 1:
        raise ValueError(
            "SenseNova Reference Image accepts one image, not an IMAGE batch"
        )
    image = image[:, :, :, :3].movedim(-1, 1).float()
    height, width = smart_resize(
        image.shape[-2], image.shape[-1], max_pixels=max_pixels
    )
    if image.shape[-2:] != (height, width):
        image = F.interpolate(
            image, size=(height, width), mode="bicubic", align_corners=False
        )
    mean = image.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = image.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    return (image - mean) / std


def preprocess_references(images):
    max_pixels = min(2048 * 2048, (4096 * 4096) // len(images))
    return [preprocess_reference(image, max_pixels=max_pixels) for image in images]


def _image_tokens(token_height, token_width):
    return (
        [IMAGE_START_ID]
        + [IMAGE_CONTEXT_ID] * (token_height * token_width)
        + [IMAGE_END_ID]
    )


def conditioned_input_length(input_length, reference_grids, image_only=False):
    image_token_count = sum(height * width for height, width in reference_grids)
    if image_only:
        return image_token_count + 9 + 2 * len(reference_grids)
    label_count = (
        sum(len(IMAGE_LABEL_IDS[index]) for index in range(len(reference_grids)))
        if len(reference_grids) > 1
        else 0
    )
    return input_length + image_token_count + 3 * len(reference_grids) + label_count


def condition_input_ids(input_ids, reference_grids, image_only=False):
    image_blocks = [_image_tokens(height, width) for height, width in reference_grids]
    if image_only:
        values = (
            [IM_START_ID, USER_ID, NEWLINE_ID]
            + [token for block in image_blocks for token in block]
            + [
                IM_END_ID,
                NEWLINE_ID,
                IM_START_ID,
                ASSISTANT_ID,
                NEWLINE_ID,
                IMAGE_START_ID,
            ]
        )
        return torch.tensor([values], dtype=torch.long, device=input_ids.device)

    values = input_ids[0].tolist()
    starts = [index for index, value in enumerate(values) if value == IM_START_ID]
    if len(starts) < 3:
        raise ValueError("SenseNova prompt does not match the bundled chat template")
    insert_at = starts[1] + 3
    inserted = []
    for index, block in enumerate(image_blocks):
        if len(image_blocks) > 1:
            inserted.extend(IMAGE_LABEL_IDS[index])
        inserted.extend(block)
        inserted.append(NEWLINE_ID)
    values[insert_at:insert_at] = inserted
    return torch.tensor([values], dtype=torch.long, device=input_ids.device)


def thw_indexes(input_ids, reference_grids):
    values = input_ids[0]
    image_start_shift = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=values.device),
            (values == IMAGE_START_ID).long(),
        )
    )[:-1]
    not_image = (values != IMAGE_CONTEXT_ID).long()
    time_indexes = (image_start_shift + not_image).cumsum(0) - 1
    height_indexes = torch.zeros_like(time_indexes)
    width_indexes = torch.zeros_like(time_indexes)
    selected = values == IMAGE_CONTEXT_ID
    height_positions = []
    width_positions = []
    for token_height, token_width in reference_grids:
        positions = torch.arange(
            token_height * token_width, dtype=torch.long, device=values.device
        )
        height_positions.append(positions // token_width)
        width_positions.append(positions % token_width)
    height_indexes[selected] = torch.cat(height_positions)
    width_indexes[selected] = torch.cat(width_positions)
    return torch.stack((time_indexes, height_indexes, width_indexes)).unsqueeze(0)


def block_causal_mask(time_indexes):
    values = time_indexes[0, 0]
    length = values.shape[0]
    same_block = values[:, None] == values[None, :]
    causal = (
        torch.arange(length, device=values.device)[None, :]
        <= torch.arange(length, device=values.device)[:, None]
    )
    allowed = same_block | causal
    mask = torch.zeros(
        (1, 1, length, length), dtype=torch.float32, device=values.device
    )
    return mask.masked_fill(~allowed[None, None], float("-inf"))
