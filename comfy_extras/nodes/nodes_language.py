from __future__ import annotations

import copy
import logging
import operator
from functools import reduce
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any, Dict, Optional, List, Callable, TypedDict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, LogitsProcessor, TextStreamer, \
    PreTrainedTokenizerBase, LogitsProcessorList, PretrainedConfig

from comfy.language.transformers_model_management import TransformersManagedModel
from comfy.model_downloader import huggingface_repos
from comfy.model_management import get_torch_device_name, load_model_gpu, unet_dtype, unet_offload_device
from comfy.nodes.package_typing import CustomNode, InputTypes, ValidatedNodeResult
from comfy.utils import comfy_tqdm, seed_for_block, comfy_progress, ProgressBar

# aka kwargs type
_GENERATION_KWARGS_TYPE = Dict[str, Any]
_GENERATION_KWARGS_TYPE_NAME = "SAMPLER"
_TOKENS_TYPE = torch.Tensor
TOKENS_TYPE_NAME = "TOKENS"

KNOWN_CHAT_TEMPLATES = {}


def _update_known_chat_templates():
    try:
        _chat_templates: Traversable
        with files("huggingface_extra_chat_templates") / "chat_templates" as _chat_templates:
            _extra_jinja_templates = {Path(traversable.name).stem: traversable.read_text().replace('    ', '').replace('\n', '') for traversable in _chat_templates.iterdir() if traversable.is_file()}
            KNOWN_CHAT_TEMPLATES.update(_extra_jinja_templates)
    except ImportError as exc:
        logging.warning("Could not load extra chat templates, some text models will fail", exc_info=exc)


class _ProgressTextStreamer(TextStreamer):
    def __init__(self, on_finalized_text: Callable[[str, bool], None], tokenizer: "AutoTokenizer", skip_prompt: bool = False, **decode_kwargs):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.on_finalized_text_handler = on_finalized_text

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.on_finalized_text_handler(text, stream_end)


class _ProgressLogitsProcessor(LogitsProcessor):
    def __init__(self, model: TransformersManagedModel):
        self.eos_token_id = model.tokenizer.eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        probabilities = scores.softmax(dim=-1)
        self.eos_probability = probabilities[:, self.eos_token_id].item()
        return scores


# todo: for per token progress, should this really look like {"ui": {"string": [value]}} ?
class TransformerStreamedProgress(TypedDict):
    next_token: str


class TransformerSamplerBase(CustomNode):
    RETURN_TYPES = _GENERATION_KWARGS_TYPE_NAME,
    RETURN_NAMES = "GENERATION ARGS",
    FUNCTION = "execute"
    CATEGORY = "language/samplers"

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
                "temperature": ("FLOAT", {"default": 1.0, "min": 0})
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
                "model": ("MODEL",)
            }
        }

    RETURN_TYPES = _GENERATION_KWARGS_TYPE_NAME,
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
        range_ = {"value0": (_GENERATION_KWARGS_TYPE_NAME, {"forceInput": True})}
        range_.update({f"value{i}": (_GENERATION_KWARGS_TYPE_NAME, {"forceInput": True}) for i in range(1, 5)})

        return {
            "required": range_
        }

    CATEGORY = "language"
    RETURN_TYPES = _GENERATION_KWARGS_TYPE_NAME,
    FUNCTION = "execute"

    def execute(self, **kwargs):
        do_sample = {
            "do_sample": any(k == "do_sample" and v for value in kwargs.values() for k, v in value.items())
        }

        return (reduce(operator.or_, list(kwargs.values()) + [do_sample], {}),)


class TransformersLoader(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "ckpt_name": (huggingface_repos(),),
                "subfolder": ("STRING", {})
            }
        }

    CATEGORY = "language"
    RETURN_TYPES = "MODEL",
    FUNCTION = "execute"

    def execute(self, ckpt_name: str, subfolder: Optional[str] = None):
        hub_kwargs = {}
        if subfolder is not None and subfolder != "":
            hub_kwargs["subfolder"] = subfolder
        with comfy_tqdm():
            model = AutoModelForCausalLM.from_pretrained(ckpt_name, torch_dtype=unet_dtype(), device_map=get_torch_device_name(unet_offload_device()), low_cpu_mem_usage=True, trust_remote_code=True, **hub_kwargs)
            tokenizer = AutoTokenizer.from_pretrained(ckpt_name)
            config_dict, _ = PretrainedConfig.get_config_dict(ckpt_name, trust_remote_code=True, **hub_kwargs)
        model_managed = TransformersManagedModel(ckpt_name, model, tokenizer, config_dict)
        return model_managed,


class OneShotInstructTokenize(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "model": ("MODEL",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            }
        }

    CATEGORY = "language"
    RETURN_TYPES = (TOKENS_TYPE_NAME,)
    FUNCTION = "execute"

    def execute(self, model: TransformersManagedModel, prompt: str) -> ValidatedNodeResult:
        tokenizer: PreTrainedTokenizerBase | AutoTokenizer = model.tokenizer
        assert tokenizer is not None
        assert hasattr(tokenizer, "decode")

        # try to retrieve a matching chat template
        chat_template = tokenizer.chat_template if hasattr(tokenizer, "chat_template") else None
        if chat_template is None:
            candidate_chat_templates = [(name, template) for name, template in KNOWN_CHAT_TEMPLATES.items() if name in model.config_dict["_name_or_path"] or name in model.model.name_or_path]
            if len(candidate_chat_templates) > 0:
                filename, chat_template = candidate_chat_templates[0]
                logging.debug(f"Selected chat template filename={filename} for {model.model.name_or_path}")
        try:
            # todo: this should come from node inputs
            prompt = tokenizer.apply_chat_template([
                {"role": "user", "content": prompt},
            ], chat_template=chat_template, add_generation_prompt=True, tokenize=False)
        except Exception as exc:
            logging.error("Could not apply chat template", exc_info=exc)

        return tokenizer(prompt, return_tensors="pt"),


class TransformersGenerate(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "model": ("MODEL",),
                "tokens": (TOKENS_TYPE_NAME, {}),
                "max_new_tokens": ("INT", {"default": 512, "min": 1}),
                "repetition_penalty": ("FLOAT", {"default": 0.0, "min": 0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2 ** 32 - 1}),
                "use_cache": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "images": ("IMAGE", {}),
                "sampler": (_GENERATION_KWARGS_TYPE_NAME, {}),
            }
        }

    CATEGORY = "language"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"

    def execute(self,
                model: Optional[TransformersManagedModel] = None,
                tokens: _TOKENS_TYPE = None,
                max_new_tokens: int = 512,
                repetition_penalty: float = 0.0,
                seed: int = 0,
                images: Optional[List[torch.Tensor] | torch.Tensor] = None,
                sampler: Optional[_GENERATION_KWARGS_TYPE] = None,
                *args,
                **kwargs
                ):
        sampler = sampler or {}
        generate_kwargs = copy.copy(sampler)
        # gracefully support LlaVA and others
        if images is not None and not isinstance(images, torch.Tensor):
            images = torch.stack(images, dim=0)
        if images is not None:
            generate_kwargs["images"] = images
            # assuming it's of the form (batch, features..., height, width)
            generate_kwargs["images_sizes"] = [(images.shape[-2], images.shape[-1]) for _ in range(images.shape[0])]
        load_model_gpu(model)
        tokenizer: PreTrainedTokenizerBase | AutoTokenizer = model.tokenizer
        inputs = tokens.to(model.current_device)
        transformers_model: PreTrainedModel = model.model
        progress_logits_processor = _ProgressLogitsProcessor(model)
        progress_bar: ProgressBar
        with comfy_progress(total=max_new_tokens) as progress_bar:
            # todo: deal with batches correctly, don't assume batch size 1
            token_count = 0

            # progress
            def on_finalized_text(next_token: str, stop: bool):
                nonlocal token_count
                nonlocal progress_bar

                # todo: this has to be more mathematically sensible
                eos_token_probability = progress_logits_processor.eos_probability
                token_count += 1
                value = max(eos_token_probability * max_new_tokens, token_count)
                preview = TransformerStreamedProgress(next_token=next_token)
                progress_bar.update_absolute(value, total=max_new_tokens, preview_image_or_output=preview)

            text_streamer = _ProgressTextStreamer(on_finalized_text, tokenizer, True)

            with seed_for_block(seed):
                # load the model as close to the actual generation as possible
                output_ids = transformers_model.generate(
                    inputs.input_ids,
                    logits_processor=LogitsProcessorList([progress_logits_processor]),
                    streamer=text_streamer,
                    max_new_tokens=max_new_tokens,
                    repetition_penalty=repetition_penalty if repetition_penalty != 0 else None,
                    **generate_kwargs
                )

                if transformers_model.config.is_encoder_decoder:
                    start_position = 1
                else:
                    start_position = inputs.input_ids.shape[1]
                output_ids = output_ids[:, start_position:]

        # todo: is this redundant consider I'm decoding in the on_finalized_text block?
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return outputs,


class PreviewString(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "value": ("STRING", {"forceInput": True}),
            }
        }

    CATEGORY = "language"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING",)
    OUTPUT_NODE = True

    def execute(self, value: str):
        return {"ui": {"string": [value]}}


NODE_CLASS_MAPPINGS = {}
for cls in (
        TransformerTopKSampler,
        TransformerTopPSampler,
        TransformerTemperatureSampler,
        TransformerGreedySampler,
        TransformerContrastiveSearchSampler,
        TransformerBeamSearchSampler,
        TransformerMergeSamplers,
        TransformersLoader,
        TransformersGenerate,
        OneShotInstructTokenize,
        PreviewString,
):
    NODE_CLASS_MAPPINGS[cls.__name__] = cls

_update_known_chat_templates()
