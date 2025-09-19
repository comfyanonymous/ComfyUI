from __future__ import annotations

import contextlib
import copy
import inspect
import logging
import operator
import pathlib
import weakref
from functools import reduce
from typing import Optional, Any, Callable

import torch
import transformers
from huggingface_hub.errors import EntryNotFoundError
from transformers import PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin, AutoProcessor, AutoTokenizer, \
    BatchFeature, AutoModelForVision2Seq, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoModel, \
    PretrainedConfig, TextStreamer, LogitsProcessor
from huggingface_hub import hf_api
from huggingface_hub.file_download import hf_hub_download
from transformers.models.auto.modeling_auto import MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES, \
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

from .chat_templates import KNOWN_CHAT_TEMPLATES
from .language_types import ProcessorResult, TOKENS_TYPE, GENERATION_KWARGS_TYPE, TransformerStreamedProgress, \
    LLaVAProcessor, LanguageModel, LanguagePrompt
from .. import model_management
from ..component_model.tensor_types import RGBImageBatch
from ..model_downloader import get_or_download_huggingface_repo
from ..model_management import unet_offload_device, get_torch_device, unet_dtype, load_models_gpu
from ..model_management_types import ModelManageableStub
from ..utils import comfy_tqdm, ProgressBar, comfy_progress, seed_for_block

logger = logging.getLogger(__name__)

# tweaks to support florence 2
_OVERRIDDEN_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = list(MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.keys()) + ['florence2']

# should be added if the expectation is that this model emits special tokens
_DO_NOT_SKIP_SPECIAL_TOKENS = {'florence2', 'paligemma'}


class TransformersManagedModel(ModelManageableStub, LanguageModel):
    def __init__(
            self,
            repo_id: str,
            model: PreTrainedModel,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            config_dict: Optional[dict] = None,
            processor: Optional[ProcessorMixin | AutoProcessor] = None
    ):
        self._repo_id = repo_id
        self._model = model
        self._tokenizer = tokenizer
        self._processor = processor
        self._object_patches: dict[str, Any] = {}
        self._parameter_count = sum(param.nelement() for param in self.model.state_dict().values())
        self._size = sum(param.nelement() * param.element_size() for param in self.model.state_dict().values())
        self.load_device = get_torch_device()
        self.offload_device = unet_offload_device()
        self._config_dict = config_dict
        self._on_set_processor(self._processor)
        self._model_type = ""
        self._original_transformers_managed_model: weakref.ReferenceType["TransformersManagedModel"] = weakref.ref(self)
        if model.device != self.offload_device:
            model.to(device=self.offload_device)

    @staticmethod
    def from_pretrained(ckpt_name: str, subfolder: Optional[str] = None, config_dict: PretrainedConfig | dict | None = None) -> "TransformersManagedModel":
        hub_kwargs = {}
        if subfolder is not None and subfolder.strip() != "":
            hub_kwargs["subfolder"] = subfolder
        repo_id = ckpt_name
        with comfy_tqdm():
            ckpt_name = get_or_download_huggingface_repo(repo_id)

            if config_dict is None:
                config_dict, _ = PretrainedConfig.get_config_dict(ckpt_name, **hub_kwargs)
            elif isinstance(config_dict, PretrainedConfig):
                config_dict: dict = config_dict.to_dict()
            else:
                config_dict = {}

            try:
                model_type = config_dict["model_type"]
            except KeyError:
                logger.debug(f"Configuration was missing for repo_id={repo_id}")
                model_type = ""

            from_pretrained_kwargs = {
                "pretrained_model_name_or_path": ckpt_name,
                **hub_kwargs
            }

            # language models prefer to use bfloat16 over float16
            default_kwargs = {
                "dtype": unet_dtype(supported_dtypes=(torch.bfloat16, torch.float16, torch.float32)),
                "low_cpu_mem_usage": True,
                "device_map": str(unet_offload_device()),
                # transformers usually has a better upstream implementation than whatever is put into the author's repos
                "trust_remote_code": False,
            }

            default_kwargs_trust_remote = {
                **default_kwargs,
                "trust_remote_code": True
            }

            kwargses_to_try = (default_kwargs, default_kwargs_trust_remote, {})

            # if we have flash-attn installed, try to use it
            try:
                if model_management.flash_attn_enabled():
                    attn_override_kwargs = {
                        "attn_implementation": "flash_attention_2",
                        **kwargses_to_try[0]
                    }
                    kwargses_to_try = (attn_override_kwargs, *kwargses_to_try)
                    logger.debug(f"while loading model {ckpt_name}, flash_attn was installed, so the flash_attention_2 implementation will be tried")
            except ImportError:
                pass
            for i, kwargs_to_try in enumerate(kwargses_to_try):
                try:
                    if model_type in MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES:
                        model = AutoModelForVision2Seq.from_pretrained(**from_pretrained_kwargs, **kwargs_to_try)
                    elif model_type in MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES:
                        model = AutoModelForSeq2SeqLM.from_pretrained(**from_pretrained_kwargs, **kwargs_to_try)
                    elif model_type in _OVERRIDDEN_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
                        model = AutoModelForCausalLM.from_pretrained(**from_pretrained_kwargs, **kwargs_to_try)
                    else:
                        model = AutoModel.from_pretrained(**from_pretrained_kwargs, **kwargs_to_try)
                    if model is not None:
                        break
                except Exception as exc_info:
                    if i == len(kwargses_to_try) - 1:
                        raise exc_info
                    else:
                        logger.warning(f"tried to import transformers model {ckpt_name} but got exception when trying additional import args {kwargs_to_try}", exc_info=exc_info)
                finally:
                    torch.set_default_dtype(torch.float32)

            for i, kwargs_to_try in enumerate(kwargses_to_try):
                try:
                    try:
                        processor = AutoProcessor.from_pretrained(**from_pretrained_kwargs, **kwargs_to_try)
                    except:
                        processor = None
                    if isinstance(processor, PreTrainedTokenizerBase):
                        tokenizer = processor
                        processor = None
                    else:
                        tokenizer = getattr(processor, "tokenizer") if processor is not None and hasattr(processor, "tokenizer") else AutoTokenizer.from_pretrained(ckpt_name, **hub_kwargs, **kwargs_to_try)
                    if tokenizer is not None or processor is not None:
                        break
                except Exception as exc_info:
                    if i == len(kwargses_to_try) - 1:
                        raise exc_info
                finally:
                    torch.set_default_dtype(torch.float32)

        if model_management.xformers_enabled() and hasattr(model, "enable_xformers_memory_efficient_attention"):
            model.enable_xformers_memory_efficient_attention()
            logger.debug("enabled xformers memory efficient attention")

        model_managed = TransformersManagedModel(
            repo_id=repo_id,
            model=model,
            tokenizer=tokenizer,
            config_dict=config_dict,
            processor=processor
        )

        model_managed._model_type = model_type

        return model_managed

    def generate(self, tokens: TOKENS_TYPE = None,
                 max_new_tokens: int = 512,
                 repetition_penalty: float = 0.0,
                 seed: int = 0,
                 sampler: Optional[GENERATION_KWARGS_TYPE] = None,
                 *args,
                 **kwargs) -> str:
        tokens = copy.copy(tokens)
        tokens_original = copy.copy(tokens)
        sampler = sampler or {}
        generate_kwargs = copy.copy(sampler)
        load_models_gpu([self])
        transformers_model: PreTrainedModel = self.model
        tokenizer: PreTrainedTokenizerBase | AutoTokenizer = self.tokenizer
        # remove unused inputs
        # maximizes compatibility with different models
        generate_signature = inspect.signature(transformers_model.generate).parameters
        prepare_signature = inspect.signature(transformers_model.prepare_inputs_for_generation).parameters
        if hasattr(transformers_model, "forward"):
            forward_signature = inspect.signature(transformers_model.forward).parameters
        else:
            forward_signature = {}
        to_delete = set(reduce(operator.sub, map(lambda x: x.keys(), [tokens, generate_signature, prepare_signature, forward_signature])))
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
                logger.warning(f"tokenizer {tokenizer} unexpected for translation task")
        if "input_ids" in tokens and "inputs" in tokens:
            if "input_ids" in gen_sig_keys:
                to_delete.add("inputs")
            elif "inputs" in gen_sig_keys:
                to_delete.add("input_ids")
        for unused_kwarg in to_delete:
            tokens.pop(unused_kwarg)
            logger.debug(f"{transformers_model.name_or_path}.generate does not accept {unused_kwarg}, removing")

        # images should be moved to model
        for key in ("images", "pixel_values"):
            if key in tokens:
                tokens[key] = tokens[key].to(device=self.current_device, dtype=self.model_dtype())

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

            try:
                import triton  # pylint: disable=import-error
                has_triton = True
            except (ImportError, ModuleNotFoundError):
                has_triton = False

            with seed_for_block(seed), torch.inference_mode(mode=True) if has_triton else contextlib.nullcontext():
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
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=self._model_type not in _DO_NOT_SKIP_SPECIAL_TOKENS, clean_up_tokenization_spaces=False)
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
        return outputs[0]

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

    def lowvram_patch_counter(self):
        return 0

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

    def patch_model(self, device_to: torch.device | None = None, lowvram_model_memory: int = 0, load_weights: bool = True, force_patch_weights: bool = False) -> torch.nn.Module:
        return self.model.to(device=device_to)

    def unpatch_model(self, device_to: torch.device | None = None, unpatch_weights: Optional[bool] = False) -> torch.nn.Module:
        return self.model.to(device=device_to)

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

    def tokenize(self, prompt: str | LanguagePrompt, images: RGBImageBatch | None, chat_template: str | None = None) -> ProcessorResult:
        tokenizer = self.processor if self.processor is not None else self.tokenizer
        assert tokenizer is not None
        assert hasattr(tokenizer, "decode")

        # try to retrieve a matching chat template
        chat_template = chat_template or tokenizer.chat_template if hasattr(tokenizer, "chat_template") else None
        if chat_template is None and self.config_dict is not None and "_name_or_path" in self.config_dict:
            candidate_chat_templates = [(name, template) for name, template in KNOWN_CHAT_TEMPLATES.items() if name in self.config_dict["_name_or_path"] or name in self.model.name_or_path]
            if len(candidate_chat_templates) > 0:
                filename, chat_template = candidate_chat_templates[0]
                logger.debug(f"Selected chat template filename={filename} for {self.model.name_or_path}")
        if isinstance(images, list):
            images = torch.stack(images, dim=0)
        if images is not None:
            image_sizes = [(image.shape[-2], image.shape[-3]) for image in images]
        else:
            image_sizes = []
            # todo: what is the best choice for this?
            # probably select a size that related to the vision tower?
            images = torch.zeros((0, 0, 0, 3))

        try:
            if hasattr(tokenizer, "apply_chat_template"):
                messages: LanguagePrompt
                if isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], dict):
                    messages = prompt
                elif "content[" in chat_template:
                    messages = [
                        {"role": "user",
                         "content": [
                                        {
                                            "type": "text",
                                            "text": prompt
                                        }
                                    ] + [
                                        {"type": "image"} for _ in range(len(images))
                                    ]

                         }
                    ]
                else:
                    messages = [
                        {"role": "user", "content": prompt},
                    ]
                prompt = tokenizer.apply_chat_template(messages, chat_template=chat_template, add_generation_prompt=True, tokenize=False)
        except Exception as exc:
            logger.debug("Could not apply chat template", exc_info=exc)

        if self.processor is None and isinstance(prompt, str):
            batch_encoding = tokenizer(prompt, return_tensors="pt").to(device=self.load_device)
            return {**batch_encoding}
        else:
            if hasattr(self.processor, "to"):
                self.processor.to(device=self.load_device)
            # convert tuple to list from images.unbind() for paligemma workaround
            image_tensor_list = list(images.unbind()) if images is not None and len(images) > 0 else None
            try:
                batch_feature: BatchFeature = self.processor(text=[prompt], images=image_tensor_list, return_tensors="pt", padding=True)
            except TypeError as exc_info:
                logger.warning(f"Exception while trying to run processor. Your transformers package is version {transformers.__version__} and may need to be updated")
                raise exc_info
            if hasattr(self.processor, "to"):
                self.processor.to(device=self.offload_device)
            assert "input_ids" in batch_feature
            try:
                batch_feature.to(device=self.load_device, dtype=self.model_dtype())
            except TypeError:
                # works around Pixtral processor bug
                batch_feature.to(self.load_device)
                batch_feature.to(self.model_dtype())
            # noinspection PyTypeChecker
            batch_feature_dict = {
                "inputs": batch_feature["input_ids"],
                **batch_feature
            }
            if "pixel_values" in batch_feature and "image_sizes" not in batch_feature_dict:
                batch_feature_dict["image_sizes"] = image_sizes
            if "pixel_values" in batch_feature and "images" not in batch_feature_dict:
                batch_feature_dict["images"] = batch_feature["pixel_values"]
            return batch_feature_dict

    @property
    def repo_id(self) -> str:
        return self._repo_id

    def __str__(self):
        if self.repo_id is not None:
            repo_id_as_path = pathlib.PurePath(self.repo_id)
            return f"<TransformersManagedModel for {'/'.join(repo_id_as_path.parts[-2:])} ({self.model.__class__.__name__})>"
        else:
            return f"<TransformersManagedModel for {self.model.__class__.__name__}>"

    def clone(self) -> TransformersManagedModel:
        m = copy.copy(self)
        # deep copy a few objects
        m._object_patches = copy.copy(self._object_patches)
        return m

    def add_object_patch(self, name: str, obj: Any):
        # for the sake of compatibility, rewrite the name to the actual model field
        if name == "diffusion_model":
            name = "model"

        self._object_patches[name] = obj

    def get_model_object(self, name: str) -> torch.nn.Module:
        if name == "diffusion_model":
            name = "model"
        return super().get_model_object(name)

    @property
    def model(self) -> PreTrainedModel | torch.nn.Module:
        return self._object_patches.get("model", self._model)


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
