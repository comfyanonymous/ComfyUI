from __future__ import annotations

import copy
import inspect
import logging
import operator
import os.path
from functools import reduce
from typing import Any, Dict, Optional, List, Callable, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, LogitsProcessor, TextStreamer, \
    PreTrainedTokenizerBase, LogitsProcessorList, PretrainedConfig, AutoProcessor, BatchFeature, ProcessorMixin, \
    LlavaNextForConditionalGeneration, LlavaNextProcessor
from typing_extensions import TypedDict

from comfy.language.chat_templates import KNOWN_CHAT_TEMPLATES
from comfy.language.language_types import ProcessorResult
from comfy.language.transformers_model_management import TransformersManagedModel
from comfy.model_downloader import huggingface_repos
from comfy.model_management import get_torch_device_name, load_model_gpu, unet_dtype, unet_offload_device
from comfy.nodes.package_typing import CustomNode, InputTypes, ValidatedNodeResult
from comfy.utils import comfy_tqdm, seed_for_block, comfy_progress, ProgressBar

_AUTO_CHAT_TEMPLATE = "default"

# add llava support
try:
    from llava import model

    logging.info("Additional LLaVA models are now supported")
except ImportError as exc:
    logging.info(f"Install LLavA with `pip install git+https://github.com/AppMana/appmana-comfyui-llava` for additional LLaVA support")

# aka kwargs type
_GENERATION_KWARGS_TYPE = Dict[str, Any]
_GENERATION_KWARGS_TYPE_NAME = "SAMPLER"

_TOKENS_TYPE = Union[ProcessorResult, BatchFeature]
TOKENS_TYPE_NAME = "TOKENS"


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


class TransformersImageProcessorLoader(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "ckpt_name": (huggingface_repos(),),
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
        processor = AutoProcessor.from_pretrained(ckpt_name, torch_dtype=unet_dtype(), device_map=get_torch_device_name(unet_offload_device()), low_cpu_mem_usage=True, trust_remote_code=True, **hub_kwargs)
        return model.patch_processor(processor, overwrite_tokenizer),


class TransformersLoader(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "ckpt_name": (huggingface_repos(),),
                "subfolder": ("STRING", {})
            },
        }

    CATEGORY = "language"
    RETURN_TYPES = "MODEL",
    FUNCTION = "execute"

    def execute(self, ckpt_name: str, subfolder: Optional[str] = None, *args, **kwargs):
        hub_kwargs = {}
        if subfolder is not None and subfolder != "":
            hub_kwargs["subfolder"] = subfolder
        with comfy_tqdm():
            from_pretrained_kwargs = {
                "pretrained_model_name_or_path": ckpt_name,
                "torch_dtype": unet_dtype(),
                "device_map": get_torch_device_name(unet_offload_device()),
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
                **hub_kwargs
            }

            try:
                model = AutoModelForCausalLM.from_pretrained(**from_pretrained_kwargs)
            except:
                model = LlavaNextForConditionalGeneration.from_pretrained(**from_pretrained_kwargs)

            config_dict, _ = PretrainedConfig.get_config_dict(ckpt_name, trust_remote_code=True, **hub_kwargs)
            try:
                try:
                    processor = AutoProcessor.from_pretrained(**from_pretrained_kwargs)
                except:
                    processor = LlavaNextProcessor.from_pretrained(**from_pretrained_kwargs)
            except:
                processor = None
            if not isinstance(processor, ProcessorMixin):
                processor = None
            tokenizer = getattr(processor, "tokenizer") if processor is not None and hasattr(processor, "tokenizer") else AutoTokenizer.from_pretrained(ckpt_name, **hub_kwargs)

        model_managed = TransformersManagedModel(
            repo_id=ckpt_name,
            model=model,
            tokenizer=tokenizer,
            config_dict=config_dict,
            processor=processor
        )
        return model_managed,


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

    def execute(self, model: TransformersManagedModel, prompt: str, images: List[torch.Tensor] | torch.Tensor = None, chat_template: str = "__auto__") -> ValidatedNodeResult:
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
                "seed": ("INT", {"default": 0, "min": 0, "max": 2 ** 32 - 1}),
                "use_cache": ("BOOLEAN", {"default": True}),
            },
            "optional": {
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
                sampler: Optional[_GENERATION_KWARGS_TYPE] = None,
                *args,
                **kwargs
                ):
        tokens = copy.copy(tokens)
        sampler = sampler or {}
        generate_kwargs = copy.copy(sampler)
        load_model_gpu(model)
        transformers_model: PreTrainedModel = model.model
        tokenizer: PreTrainedTokenizerBase | AutoTokenizer = model.tokenizer
        # remove unused inputs
        # maximizes compatibility with different models
        generate_signature = inspect.signature(transformers_model.generate).parameters
        prepare_signature = inspect.signature(transformers_model.prepare_inputs_for_generation).parameters
        to_delete = set(reduce(operator.sub, map(lambda x: x.keys(), [tokens, generate_signature, prepare_signature])))
        gen_sig_keys = generate_signature.keys()
        if "input_ids" in tokens and "inputs" in tokens:
            if "input_ids" in gen_sig_keys:
                to_delete.add("inputs")
            elif "inputs" in gen_sig_keys:
                to_delete.add("input_ids")
        for unused_kwarg in to_delete:
            tokens.pop(unused_kwarg)
            logging.info(f"{transformers_model.name_or_path}.generate does not accept {unused_kwarg}, removing")

        # images should be moved to model
        for key in ("images", "pixel_values"):
            if key in tokens:
                tokens[key] = tokens[key].to(device=model.current_device, dtype=model.model_dtype())
        inputs = tokens
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
                output_ids = transformers_model.generate(
                    **inputs,
                    logits_processor=LogitsProcessorList([progress_logits_processor]),
                    streamer=text_streamer,
                    max_new_tokens=max_new_tokens,
                    repetition_penalty=repetition_penalty if repetition_penalty != 0 else None,
                    **generate_kwargs
                )

                if transformers_model.config.is_encoder_decoder:
                    start_position = 1
                else:
                    start_position = inputs["input_ids" if "input_ids" in inputs else "inputs"].shape[1]
                output_ids = output_ids[:, start_position:]

        # todo: is this redundant consider I'm decoding in the on_finalized_text block?
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        # gpu-loaded stuff like images can now be unloaded
        if hasattr(tokens, "to"):
            del tokens
        else:
            for to_delete in tokens.values():
                del to_delete
            del tokens

        # todo: better support batches
        return outputs[0],


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
        TransformersImageProcessorLoader,
        TransformersGenerate,
        OneShotInstructTokenize,
        PreviewString,
):
    NODE_CLASS_MAPPINGS[cls.__name__] = cls
