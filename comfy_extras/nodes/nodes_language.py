from __future__ import annotations

import operator
import os.path
from abc import ABC, abstractmethod
from functools import reduce
from typing import Optional, List

import torch
from transformers import AutoProcessor
from transformers.models.m2m_100.tokenization_m2m_100 import \
    FAIRSEQ_LANGUAGE_CODES as tokenization_m2m_100_FAIRSEQ_LANGUAGE_CODES
from transformers.models.nllb.tokenization_nllb import \
    FAIRSEQ_LANGUAGE_CODES as tokenization_nllb_FAIRSEQ_LANGUAGE_CODES

from comfy.cmd import folder_paths
from comfy.component_model.folder_path_types import SaveImagePathTuple
from comfy.language.chat_templates import KNOWN_CHAT_TEMPLATES
from comfy.language.language_types import GENERATION_KWARGS_TYPE, GENERATION_KWARGS_TYPE_NAME, TOKENS_TYPE, \
    TOKENS_TYPE_NAME, LanguageModel
from comfy.language.transformers_model_management import TransformersManagedModel
from comfy.model_downloader import get_huggingface_repo_list, get_or_download_huggingface_repo
from comfy.model_management import get_torch_device_name, unet_dtype, unet_offload_device
from comfy.node_helpers import export_custom_nodes
from comfy.nodes.package_typing import CustomNode, InputTypes, ValidatedNodeResult, Seed

_AUTO_CHAT_TEMPLATE = "default"


class TransformerSamplerBase(CustomNode, ABC):
    RETURN_TYPES = GENERATION_KWARGS_TYPE_NAME,
    RETURN_NAMES = "GENERATION ARGS",
    FUNCTION = "execute"
    CATEGORY = "language/samplers"

    @classmethod
    @abstractmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return ...

    @property
    def do_sample(self):
        return True

    def execute(self, **kwargs):
        return {
            "do_sample": self.do_sample,
            **kwargs
        },


class TransformerTopKSampler(TransformerSamplerBase):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "top_k": ("INT", {"default": 50, "min": 1})
            }
        }


class TransformerTopPSampler(TransformerSamplerBase):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "top_p": ("FLOAT", {"default": 0.9, "min": 0, "max": 1})
            }
        }


class TransformerTemperatureSampler(TransformerSamplerBase):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "temperature": ("FLOAT", {"default": 1.0, "min": 0, "step": 0.001})
            }
        }


class TransformerGreedySampler(TransformerSamplerBase):
    @property
    def do_sample(self):
        return False

    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
            }
        }


class TransformersGenerationConfig(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "model": ("MODEL", {})
            }
        }

    RETURN_TYPES = GENERATION_KWARGS_TYPE_NAME,
    RETURN_NAMES = "GENERATION ARGS",
    FUNCTION = "execute"
    CATEGORY = "language"

    def execute(self, model: TransformersManagedModel):
        if model.model.generation_config is not None:
            return model.model.generation_config

        return {}


class TransformerContrastiveSearchSampler(TransformerTopKSampler):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        top_k = TransformerTopKSampler.INPUT_TYPES()
        top_k["required"] |= {
            "penalty_alpha": ("FLOAT", {"default": 0.6, "min": 0})
        }
        return top_k


class TransformerBeamSearchSampler(TransformerSamplerBase):
    @property
    def do_sample(self):
        return False

    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "num_beams": ("INT", {"default": 1, "min": 0}),
                "early_stopping": ("BOOLEAN", {"default": True})
            }
        }


class TransformerMergeSamplers(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        range_ = {"value0": (GENERATION_KWARGS_TYPE_NAME, {"forceInput": True})}
        range_.update({f"value{i}": (GENERATION_KWARGS_TYPE_NAME, {"forceInput": True}) for i in range(1, 5)})

        return {
            "required": range_
        }

    CATEGORY = "language"
    RETURN_TYPES = GENERATION_KWARGS_TYPE_NAME,
    FUNCTION = "execute"

    def execute(self, **kwargs):
        do_sample = {
            "do_sample": any(k == "do_sample" and v for value in kwargs.values() for k, v in value.items())
        }

        return (reduce(operator.or_, list(kwargs.values()) + [do_sample], {}),)


class TransformersImageProcessorLoader(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "ckpt_name": (get_huggingface_repo_list(),),
                "subfolder": ("STRING", {}),
                "model": ("MODEL", {}),
                "overwrite_tokenizer": ("BOOLEAN", {"default": False}),
            }
        }

    CATEGORY = "language"
    RETURN_TYPES = "MODEL",
    FUNCTION = "execute"

    def execute(self, ckpt_name: str, subfolder: Optional[str] = None, model: TransformersManagedModel = None, overwrite_tokenizer: bool = False):
        hub_kwargs = {}
        if subfolder is not None and subfolder != "":
            hub_kwargs["subfolder"] = subfolder
        ckpt_name = get_or_download_huggingface_repo(ckpt_name)
        processor = AutoProcessor.from_pretrained(ckpt_name, torch_dtype=unet_dtype(), device_map=get_torch_device_name(unet_offload_device()), low_cpu_mem_usage=True, trust_remote_code=True, **hub_kwargs)
        return model.patch_processor(processor, overwrite_tokenizer),


class TransformersLoader(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "ckpt_name": (get_huggingface_repo_list(),),
            },
            "optional": {
                "subfolder": ("STRING", {}),
            }
        }

    CATEGORY = "language"
    RETURN_TYPES = "MODEL",
    RETURN_NAMES = "language model",
    FUNCTION = "execute"

    def execute(self, ckpt_name: str, subfolder: Optional[str] = None, *args, **kwargs) -> tuple[TransformersManagedModel]:
        return TransformersManagedModel.from_pretrained(ckpt_name, subfolder),


class TransformersTokenize(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "model": ("MODEL",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
        }

    CATEGORY = "language"
    RETURN_TYPES = (TOKENS_TYPE_NAME,)
    FUNCTION = "execute"

    def execute(self, model: LanguageModel, prompt: str) -> ValidatedNodeResult:
        return model.tokenize(prompt, [], None),


class TransformersM2M100LanguageCodes(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "lang_id": (tokenization_m2m_100_FAIRSEQ_LANGUAGE_CODES["m2m100"], {"default": "en"}),
            },
        }

    CATEGORY = "language"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"

    def execute(self, lang_id: str) -> ValidatedNodeResult:
        return lang_id,


class TransformersFlores200LanguageCodes(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "lang_id": (tokenization_nllb_FAIRSEQ_LANGUAGE_CODES, {"default": "eng_Latn"}),
            },
        }

    CATEGORY = "language"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"

    def execute(self, lang_id: str) -> ValidatedNodeResult:
        return lang_id,


class TransformersTranslationTokenize(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "model": ("MODEL",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "src_lang": ("STRING", {}),
                "tgt_lang": ("STRING", {}),
            },
        }

    CATEGORY = "language"
    RETURN_TYPES = (TOKENS_TYPE_NAME,)
    FUNCTION = "execute"

    def execute(self, model: TransformersManagedModel, prompt: str, src_lang: str, tgt_lang: str) -> ValidatedNodeResult:
        tokenizer = model.tokenizer

        if hasattr(tokenizer, "src_lang"):
            prev_src_lang = tokenizer.src_lang
        else:
            prev_src_lang = None
        if hasattr(tokenizer, "tgt_lang"):
            prev_tgt_lang = tokenizer.tgt_lang
        else:
            prev_tgt_lang = None

        try:
            if hasattr(tokenizer, "_build_translation_inputs"):
                encoded = tokenizer._build_translation_inputs(
                    prompt, return_tensors="pt", src_lang=src_lang, tgt_lang=tgt_lang
                )
            else:
                tokenizer.src_lang = src_lang
                tokenizer.tgt_lang = tgt_lang

                encoded = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            encoded["input_ids"] = encoded["input_ids"].to(device=model.load_device)
            encoded["attention_mask"] = encoded["attention_mask"].to(device=model.load_device)
            encoded["src_lang"] = src_lang
            encoded["tgt_lang"] = tgt_lang
            return encoded,
        finally:
            if prev_src_lang is not None:
                tokenizer.src_lang = prev_src_lang
            if prev_tgt_lang is not None:
                tokenizer.tgt_lang = prev_tgt_lang


class OneShotInstructTokenize(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "model": ("MODEL",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "chat_template": ([_AUTO_CHAT_TEMPLATE] + list(KNOWN_CHAT_TEMPLATES.keys()), {})
            },
            "optional": {
                "images": ("IMAGE", {}),
            }
        }

    CATEGORY = "language"
    RETURN_TYPES = (TOKENS_TYPE_NAME,)
    FUNCTION = "execute"

    def execute(self, model: LanguageModel, prompt: str, images: List[torch.Tensor] | torch.Tensor = None, chat_template: str = _AUTO_CHAT_TEMPLATE) -> ValidatedNodeResult:
        if chat_template == _AUTO_CHAT_TEMPLATE:
            # use an exact match
            model_name = os.path.basename(model.repo_id)
            if model_name in KNOWN_CHAT_TEMPLATES:
                chat_template = KNOWN_CHAT_TEMPLATES[model_name]
            else:
                chat_template = None
        else:
            chat_template = KNOWN_CHAT_TEMPLATES[chat_template]
        return model.tokenize(prompt, images, chat_template),


class TransformersGenerate(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "model": ("MODEL",),
                "tokens": (TOKENS_TYPE_NAME, {}),
                "max_new_tokens": ("INT", {"default": 512, "min": 1}),
                "repetition_penalty": ("FLOAT", {"default": 0.0, "min": 0}),
                "seed": Seed,
            },
            "optional": {
                "sampler": (GENERATION_KWARGS_TYPE_NAME, {}),
            }
        }

    CATEGORY = "language"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"

    def execute(self,
                model: Optional[LanguageModel] = None,
                tokens: TOKENS_TYPE = None,
                max_new_tokens: int = 512,
                repetition_penalty: float = 0.0,
                seed: int = 0,
                sampler: Optional[GENERATION_KWARGS_TYPE] = None,
                ):
        return model.generate(tokens, max_new_tokens, repetition_penalty, seed, sampler),


class PreviewString(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "value": ("STRING", {"forceInput": True}),
            }
        }

    CATEGORY = "strings"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING",)
    OUTPUT_NODE = True

    def execute(self, value: str):
        return {"ui": {"string": [value]}}


class SaveString(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "value": ("STRING", {"forceInput": True}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."})
            },
            "optional": {
                "extension": ("STRING", {"default": ".json"})
            }
        }

    CATEGORY = "strings"
    FUNCTION = "execute"
    OUTPUT_NODE = True
    RETURN_TYPES = ()

    def get_save_path(self, filename_prefix) -> SaveImagePathTuple:
        return folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory(), 0, 0)

    def execute(self, value: str | list[str], filename_prefix: str, extension: str = ".json"):
        full_output_folder, filename, counter, subfolder, filename_prefix = self.get_save_path(filename_prefix)
        if isinstance(value, str):
            value = [value]

        for i, value_i in enumerate(value):
            # roughly matches the behavior of save image, but does not support batch numbers
            with open(os.path.join(full_output_folder, f"{filename}_{counter:05d}_{extension}" if len(value) == 1 else f"{filename}_{counter:05d}_{i:02d}_{extension}"), "wt+") as f:
                f.write(value_i)
        return {"ui": {"string": value}}


export_custom_nodes()
