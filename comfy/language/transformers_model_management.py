from __future__ import annotations

import copy
import logging
import warnings
from typing import Optional, Any, Callable, Union, List

import numpy as np
import torch
from PIL.Image import Image
from transformers import PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin, AutoProcessor, AutoTokenizer, \
    TensorType, BatchFeature
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput, TruncationStrategy
from transformers.utils import PaddingStrategy

from .chat_templates import KNOWN_CHAT_TEMPLATES
from .language_types import ProcessorResult
from ..model_management import unet_offload_device, get_torch_device
from ..model_management_types import ModelManageable

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


class TransformersManagedModel(ModelManageable):
    def __init__(
            self,
            repo_id: str,
            model: PreTrainedModel,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            config_dict: Optional[dict] = None,
            processor: Optional[ProcessorMixin | AutoProcessor] = None
    ):
        self.repo_id = repo_id
        self.model = model
        self._tokenizer = tokenizer
        self._processor = processor
        self._parameter_count = sum(param.nelement() for param in self.model.state_dict().values())
        self._size = sum(param.nelement() * param.element_size() for param in self.model.state_dict().values())
        self.load_device = get_torch_device()
        self.offload_device = unet_offload_device()
        self._config_dict = config_dict
        self._on_set_processor(self._processor)
        if model.device != self.offload_device:
            model.to(device=self.offload_device)

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase | AutoTokenizer:
        return self._tokenizer

    @property
    def processor(self) -> AutoProcessor | ProcessorMixin | LLaVAProcessor | None:
        return self._processor

    @property
    def config_dict(self) -> dict:
        """
        The original configuration dictionary located in the Transformers model.

        Many models derive from base models and should inherit their settings like a chat template. This
        config_dict will have the base model's name in _name_or_path, enabling a lookup for the valid
        chat template when it is not specified by the derived model (it almost never is).
        :return: the dict value of the config.json in the HuggingFace model
        """
        if self._config_dict is not None:
            return self._config_dict

        return self.model.config.to_dict()

    @property
    def lowvram_patch_counter(self):
        return 0

    @lowvram_patch_counter.setter
    def lowvram_patch_counter(self, value: int):
        warnings.warn("Not supported")
        pass

    load_device: torch.device
    offload_device: torch.device
    model: PreTrainedModel

    @property
    def current_device(self) -> torch.device:
        return self.model.device

    def is_clone(self, other: Any) -> bool:
        return hasattr(other, "model") and self.model is other.model

    def clone_has_same_weights(self, clone: Any) -> bool:
        if not isinstance(clone, TransformersManagedModel):
            return False

        clone: TransformersManagedModel

        if not self.is_clone(clone):
            return False

        try:
            return frozenset(self.model.active_adapters()) == frozenset(clone.model.active_adapters())
        except ValueError as no_adapters:
            return True

    def model_size(self) -> int:
        return self._size

    def model_patches_to(self, arg: torch.device | torch.dtype):
        if isinstance(arg, torch.device):
            self.model.to(device=arg)
        else:
            self.model.to(arg)

    def model_dtype(self) -> torch.dtype:
        return self.model.dtype

    def patch_model_lowvram(self, device_to: torch.device, lowvram_model_memory: int, force_patch_weights=False) -> torch.nn.Module:
        warnings.warn("Transformers models do not currently support adapters like LoRAs")
        return self.model.to(device=device_to)

    def patch_model(self, device_to: torch.device, patch_weights: bool) -> torch.nn.Module:
        warnings.warn("Transformers models do not currently support adapters like LoRAs")
        return self.model.to(device=device_to)

    def unpatch_model(self, offload_device: torch.device, unpatch_weights: Optional[bool] = False) -> torch.nn.Module:
        warnings.warn("Transformers models do not currently support adapters like LoRAs")
        return self.model.to(device=offload_device)

    def patch_processor(self, processor: Any, overwrite_tokenizer: bool = False) -> TransformersManagedModel:
        model = copy.copy(self)
        model._processor = processor
        if hasattr(processor, "tokenizer") and overwrite_tokenizer:
            model._tokenizer = processor.tokenizer
        self._on_set_processor(model._processor)
        return model

    def _on_set_processor(self, processor: Any):
        if processor is not None and hasattr(processor, "image_processor") and hasattr(processor.image_processor, "do_rescale"):
            processor.image_processor.do_rescale = False

    def tokenize(self, prompt: str, images: List[torch.Tensor] | torch.Tensor, chat_template: str) -> ProcessorResult:
        tokenizer = self.tokenizer
        assert tokenizer is not None
        assert hasattr(tokenizer, "decode")

        # try to retrieve a matching chat template
        chat_template = chat_template or tokenizer.chat_template if hasattr(tokenizer, "chat_template") else None
        if chat_template is None:
            candidate_chat_templates = [(name, template) for name, template in KNOWN_CHAT_TEMPLATES.items() if name in self.config_dict["_name_or_path"] or name in self.model.name_or_path]
            if len(candidate_chat_templates) > 0:
                filename, chat_template = candidate_chat_templates[0]
                logging.debug(f"Selected chat template filename={filename} for {self.model.name_or_path}")
        try:
            if hasattr(tokenizer, "apply_chat_template"):
                # todo: this should come from node inputs
                prompt = tokenizer.apply_chat_template([
                    {"role": "user", "content": prompt},
                ], chat_template=chat_template, add_generation_prompt=True, tokenize=False)
        except Exception as exc:
            logging.debug("Could not apply chat template", exc_info=exc)

        if self.processor is None:
            batch_encoding = tokenizer(prompt, return_tensors="pt").to(device=self.load_device)
            return {**batch_encoding}
        else:
            assert images is not None and len(images) > 0, "When using a multi-modal model, pass at least one, possibly empty, image"
            if hasattr(self.processor, "to"):
                self.processor.to(device=self.load_device)

            assert "<image>" in prompt, "You must specify a &lt;image&gt; token inside the prompt for it to be substituted correctly by a HuggingFace processor"
            batch_feature: BatchFeature = self.processor([prompt], images=images, padding=True, return_tensors="pt")
            if hasattr(self.processor, "to"):
                self.processor.to(device=self.offload_device)
            assert "input_ids" in batch_feature
            batch_feature.to(device=self.load_device, dtype=self.model_dtype())
            # noinspection PyTypeChecker
            return {
                "image_sizes": [(images.shape[-1], image.shape[-2]) for image in images],
                "images": batch_feature["pixel_values"],
                "inputs": batch_feature["input_ids"],
                **batch_feature
            }
