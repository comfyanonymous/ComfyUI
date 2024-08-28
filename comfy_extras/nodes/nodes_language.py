from __future__ import annotations

import copy
import inspect
import logging
import operator
import os.path
from functools import reduce
from typing import Any, Dict, Optional, List, Callable, Union

import torch
from transformers import AutoTokenizer, PreTrainedModel, LogitsProcessor, TextStreamer, \
    PreTrainedTokenizerBase, PretrainedConfig, AutoProcessor, BatchFeature, AutoModel, AutoModelForCausalLM, \
    AutoModelForSeq2SeqLM
from transformers.models.auto.modeling_auto import MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES, \
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES, AutoModelForVision2Seq
from transformers.models.m2m_100.tokenization_m2m_100 import \
    FAIRSEQ_LANGUAGE_CODES as tokenization_m2m_100_FAIRSEQ_LANGUAGE_CODES
from transformers.models.nllb.tokenization_nllb import \
    FAIRSEQ_LANGUAGE_CODES as tokenization_nllb_FAIRSEQ_LANGUAGE_CODES
from typing_extensions import TypedDict

from comfy import model_management
from comfy.cmd import folder_paths
from comfy.component_model.folder_path_types import SaveImagePathResponse
from comfy.language.chat_templates import KNOWN_CHAT_TEMPLATES
from comfy.language.language_types import ProcessorResult
from comfy.language.transformers_model_management import TransformersManagedModel
from comfy.model_downloader import get_huggingface_repo_list, get_or_download_huggingface_repo
from comfy.model_management import get_torch_device_name, unet_dtype, unet_offload_device, load_models_gpu
from comfy.nodes.package_typing import CustomNode, InputTypes, ValidatedNodeResult
from comfy.utils import comfy_tqdm, seed_for_block, comfy_progress, ProgressBar

_AUTO_CHAT_TEMPLATE = "default"

# add llava support
try:
    from llava import model as _llava_model_side_effects

    logging.debug("Additional LLaVA models are now supported")
except ImportError as exc:
    logging.debug(f"Install LLavA with `pip install git+https://github.com/AppMana/appmana-comfyui-llava` for additional LLaVA support")

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
                "model": ("MODEL", {})
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

        ckpt_name = get_or_download_huggingface_repo(ckpt_name)
        with comfy_tqdm():
            from_pretrained_kwargs = {
                "pretrained_model_name_or_path": ckpt_name,
                "trust_remote_code": True,
                **hub_kwargs
            }

            # if flash attention exists, use it

            # compute bitsandbytes configuration
            try:
                import bitsandbytes
            except ImportError:
                pass

            config_dict, _ = PretrainedConfig.get_config_dict(ckpt_name, **hub_kwargs)
            model_type = config_dict["model_type"]
            # language models prefer to use bfloat16 over float16
            kwargs_to_try = ({"torch_dtype": unet_dtype(supported_dtypes=(torch.bfloat16, torch.float16, torch.float32)),
                              "low_cpu_mem_usage": True,
                              "device_map": str(unet_offload_device()), }, {})

            # if we have flash-attn installed, try to use it
            try:
                import flash_attn
                attn_override_kwargs = {
                    "attn_implementation": "flash_attention_2",
                    **kwargs_to_try[0]
                }
                kwargs_to_try = (attn_override_kwargs, *kwargs_to_try)
                logging.debug(f"while loading model {ckpt_name}, flash_attn was installed, so the flash_attention_2 implementation will be tried")
            except ImportError:
                pass
            for i, props in enumerate(kwargs_to_try):
                try:
                    if model_type in MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES:
                        model = AutoModelForVision2Seq.from_pretrained(**from_pretrained_kwargs, **props)
                    elif model_type in MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES:
                        model = AutoModelForSeq2SeqLM.from_pretrained(**from_pretrained_kwargs, **props)
                    elif model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
                        model = AutoModelForCausalLM.from_pretrained(**from_pretrained_kwargs, **props)
                    else:
                        model = AutoModel.from_pretrained(**from_pretrained_kwargs, **props)
                    if model is not None:
                        break
                except Exception as exc_info:
                    if i == len(kwargs_to_try) - 1:
                        raise exc_info
                    else:
                        logging.warning(f"tried to import transformers model {ckpt_name} but got exception when trying additional import args {props}", exc_info=exc_info)
                finally:
                    torch.set_default_dtype(torch.float32)

            for i, props in enumerate(kwargs_to_try):
                try:
                    try:
                        processor = AutoProcessor.from_pretrained(**from_pretrained_kwargs, **props)
                    except:
                        processor = None
                    if isinstance(processor, PreTrainedTokenizerBase):
                        tokenizer = processor
                        processor = None
                    else:
                        tokenizer = getattr(processor, "tokenizer") if processor is not None and hasattr(processor, "tokenizer") else AutoTokenizer.from_pretrained(ckpt_name, **hub_kwargs, **props)
                    if tokenizer is not None or processor is not None:
                        break
                except Exception as exc_info:
                    if i == len(kwargs_to_try) - 1:
                        raise exc_info
                finally:
                    torch.set_default_dtype(torch.float32)

        if model_management.xformers_enabled() and hasattr(model, "enable_xformers_memory_efficient_attention"):
            model.enable_xformers_memory_efficient_attention()
            logging.debug("enabled xformers memory efficient attention")

        model_managed = TransformersManagedModel(
            repo_id=ckpt_name,
            model=model,
            tokenizer=tokenizer,
            config_dict=config_dict,
            processor=processor
        )
        return model_managed,


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

    def execute(self, model: TransformersManagedModel, prompt: str) -> ValidatedNodeResult:
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
        tokens_original = copy.copy(tokens)
        sampler = sampler or {}
        generate_kwargs = copy.copy(sampler)
        load_models_gpu([model])
        transformers_model: PreTrainedModel = model.model
        tokenizer: PreTrainedTokenizerBase | AutoTokenizer = model.tokenizer
        # remove unused inputs
        # maximizes compatibility with different models
        generate_signature = inspect.signature(transformers_model.generate).parameters
        prepare_signature = inspect.signature(transformers_model.prepare_inputs_for_generation).parameters
        to_delete = set(reduce(operator.sub, map(lambda x: x.keys(), [tokens, generate_signature, prepare_signature])))
        gen_sig_keys = generate_signature.keys()
        if "tgt_lang" in tokens:
            to_delete.add("tgt_lang")
            to_delete.add("src_lang")
            to_delete.discard("input_ids")
            if "forced_bos_token_id" in tokens:
                to_delete.discard("forced_bos_token_id")
            elif hasattr(tokenizer, "convert_tokens_to_ids"):
                generate_kwargs["forced_bos_token_id"] = tokenizer.convert_tokens_to_ids(tokens["tgt_lang"])
            else:
                logging.warning(f"tokenizer {tokenizer} unexpected for translation task")
        if "input_ids" in tokens and "inputs" in tokens:
            if "input_ids" in gen_sig_keys:
                to_delete.add("inputs")
            elif "inputs" in gen_sig_keys:
                to_delete.add("input_ids")
        for unused_kwarg in to_delete:
            tokens.pop(unused_kwarg)
            logging.debug(f"{transformers_model.name_or_path}.generate does not accept {unused_kwarg}, removing")

        # images should be moved to model
        for key in ("images", "pixel_values"):
            if key in tokens:
                tokens[key] = tokens[key].to(device=model.current_device, dtype=model.model_dtype())

        # sets up inputs
        inputs = tokens

        # used to determine if text streaming is supported
        num_beams = generate_kwargs.get("num_beams", transformers_model.generation_config.num_beams)

        progress_bar: ProgressBar
        with comfy_progress(total=max_new_tokens) as progress_bar:
            # todo: deal with batches correctly, don't assume batch size 1
            token_count = 0

            # progress
            def on_finalized_text(next_token: str, stop: bool):
                nonlocal token_count
                nonlocal progress_bar

                token_count += 1
                preview = TransformerStreamedProgress(next_token=next_token)
                progress_bar.update_absolute(token_count, total=max_new_tokens, preview_image_or_output=preview)

            text_streamer = _ProgressTextStreamer(on_finalized_text, tokenizer, True)

            with seed_for_block(seed):
                if hasattr(inputs, "encodings") and inputs.encodings is not None and all(hasattr(encoding, "attention_mask") for encoding in inputs.encodings) and "attention_mask" in inputs:
                    inputs.pop("attention_mask")
                output_ids = transformers_model.generate(
                    **inputs,
                    streamer=text_streamer if num_beams <= 1 else None,
                    max_new_tokens=max_new_tokens,
                    repetition_penalty=repetition_penalty if repetition_penalty != 0 else None,
                    **generate_kwargs
                )

                if not transformers_model.config.is_encoder_decoder:
                    start_position = inputs["input_ids" if "input_ids" in inputs else "inputs"].shape[1]
                    output_ids = output_ids[:, start_position:]

        if hasattr(tokenizer, "src_lang") and "src_lang" in tokens_original:
            prev_src_lang = tokenizer.src_lang
            tokenizer.src_lang = tokens_original["src_lang"]
        else:
            prev_src_lang = None
        # todo: is this redundant consider I'm decoding in the on_finalized_text block?
        try:
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        finally:
            if prev_src_lang is not None:
                tokenizer.src_lang = prev_src_lang
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

    CATEGORY = "language"
    FUNCTION = "execute"
    OUTPUT_NODE = True
    RETURN_TYPES = ()

    def get_save_path(self, filename_prefix) -> SaveImagePathResponse:
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
        TransformersM2M100LanguageCodes,
        TransformersTokenize,
        TransformersFlores200LanguageCodes,
        TransformersTranslationTokenize,
        PreviewString,
        SaveString,
):
    NODE_CLASS_MAPPINGS[cls.__name__] = cls
