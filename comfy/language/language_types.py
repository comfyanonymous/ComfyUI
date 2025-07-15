from __future__ import annotations

from typing import Union, Callable, List, Optional, Protocol, runtime_checkable, Literal

import numpy as np
import torch
from PIL.Image import Image
from transformers import BatchEncoding, BatchFeature, TensorType
from transformers.tokenization_utils_base import TextInput, PreTokenizedInput, TruncationStrategy
from transformers.utils import PaddingStrategy
from typing_extensions import TypedDict, NotRequired

from ..component_model.tensor_types import RGBImageBatch


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
    images: NotRequired[RGBImageBatch]
    inputs: NotRequired[BatchEncoding | list[str] | LanguagePrompt]
    image_sizes: NotRequired[torch.Tensor]


class GenerationKwargs(TypedDict):
    top_k: NotRequired[int]
    top_p: NotRequired[float]
    temperature: NotRequired[float]
    penalty_alpha: NotRequired[float]
    num_beams: NotRequired[int]
    early_stopping: NotRequired[bool]


GENERATION_KWARGS_TYPE = GenerationKwargs
GENERATION_KWARGS_TYPE_NAME = "SAMPLER"
TOKENS_TYPE = Union[ProcessorResult, BatchFeature]
TOKENS_TYPE_NAME = "TOKENS"


class TransformerStreamedProgress(TypedDict):
    next_token: str


LLaVAProcessor = Callable[
    [
        Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],  # text parameter
        Union[Image, np.ndarray, torch.Tensor, List[Image], List[np.ndarray], List[torch.Tensor]],  # images parameter
        Union[bool, str, PaddingStrategy],  # padding parameter
        Union[bool, str, TruncationStrategy],  # truncation parameter
        Optional[int],  # max_length parameter
        Optional[Union[str, TensorType]]  # return_tensors parameter
    ],
    BatchFeature
]


class LanguageMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str | MessageContent


class MessageContentImage(TypedDict):
    url: NotRequired[str]


class MessageContent(TypedDict):
    type: Literal["text", "image", "video", "image_url"]
    text: NotRequired[str]
    image: NotRequired[str]
    image_url: NotRequired[MessageContentImage]
    min_pixels: NotRequired[int]
    max_pixels: NotRequired[int]


LanguagePrompt = list[LanguageMessage]


@runtime_checkable
class LanguageModel(Protocol):
    @staticmethod
    def from_pretrained(ckpt_name: str, subfolder: Optional[str] = None) -> "LanguageModel":
        ...

    def generate(self, tokens: TOKENS_TYPE = None,
                 max_new_tokens: int = 512,
                 repetition_penalty: float = 0.0,
                 seed: int = 0,
                 sampler: Optional[GENERATION_KWARGS_TYPE] = None,
                 *args,
                 **kwargs) -> str:
        ...

    def tokenize(self, prompt: str | LanguagePrompt, images: RGBImageBatch | None, chat_template: str | None = None) -> ProcessorResult:
        ...

    @property
    def repo_id(self) -> str:
        return ""
