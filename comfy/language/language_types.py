from __future__ import annotations

import torch
from transformers import BatchEncoding
from typing_extensions import TypedDict, NotRequired


class ProcessorResult(TypedDict):
    """
    Attributes:
        attention_mask: attention mask
        pixel_values: post image-processed values

        images: used for LLaVA compatibility and points to pixel_values
        inputs: used for LLaVA compatibility and points to input_ids
        images_sizes: used for LLaVA compatibility, stores the (width, height) tuples of the original input images
    """

    attention_mask: NotRequired[torch.Tensor]
    pixel_values: NotRequired[torch.Tensor]

    images: NotRequired[torch.Tensor]
    inputs: BatchEncoding
    image_sizes: NotRequired[torch.Tensor]
