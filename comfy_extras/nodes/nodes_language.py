from __future__ import annotations

from typing import Any, List, Dict

import torch
from fastchat.model import get_conversation_template
from transformers import AutoModelForCausalLM, AutoTokenizer

from comfy.language.language_types import ProcArgsRes
from comfy.language.transformers_model_management import TransformersManagedModel
from comfy.model_downloader import huggingface_repos
from comfy.model_management import get_torch_device_name, load_model_gpu, unet_dtype, unet_offload_device
from comfy.nodes.package_typing import CustomNode, InputTypes
from comfy.utils import comfy_tqdm, seed_for_block

_transformer_args_deterministic_decoding = {
    "max_length": ("INT", {"default": 4096, "min": 1}),
    "temperature": ("FLOAT", {"default": 0.7, "min": 0}),
    "repetition_penalty": ("FLOAT", {"default": 1.0, "min": 0}),
}


def proc_args(kwargs: Dict[str, Any]) -> ProcArgsRes:
    generate_kwargs = {k: v for k, v in kwargs.items() if k in _transformer_args_deterministic_decoding}
    seed = generate_kwargs.pop("seed", 0)
    return ProcArgsRes(seed, generate_kwargs)


class TransformersLoader(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "ckpt_name": (huggingface_repos(),)
            }
        }

    RETURN_TYPES = "MODEL",
    FUNCTION = "execute"

    def execute(self, ckpt_name: str):
        with comfy_tqdm():
            model = AutoModelForCausalLM.from_pretrained(ckpt_name, torch_dtype=unet_dtype(), device_map=get_torch_device_name(unet_offload_device()), low_cpu_mem_usage=True, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(ckpt_name)
        model_managed = TransformersManagedModel(ckpt_name, model, tokenizer)
        return model_managed,


class SimpleBatchDecode(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "model": ("MODEL",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                **_transformer_args_deterministic_decoding
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"

    def execute(self, model: TransformersManagedModel, prompt: str, **kwargs):
        load_model_gpu(model)
        seed, generate_kwargs = proc_args(kwargs)

        tokenizer = model.tokenizer
        inputs = tokenizer(prompt, return_tensors="pt").to(model.current_device)
        with comfy_tqdm():
            with seed_for_block(seed):
                generate_ids = model.model.generate(inputs.input_ids, **generate_kwargs)
        outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return outputs,


class SimpleInstruct(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "model": ("MODEL",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                **_transformer_args_deterministic_decoding
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"

    def execute(self, model: TransformersManagedModel, prompt: str, **kwargs):
        load_model_gpu(model)
        seed, generate_kwargs = proc_args(kwargs)
        conv = get_conversation_template(model.repo_id)
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        inputs = model.tokenizer([prompt], return_token_type_ids=False)
        inputs = {k: torch.tensor(v).to(model.current_device) for k, v in inputs.items()}
        with seed_for_block(seed):
            output_ids = model.model.generate(
                **inputs,
                do_sample=True,
                **generate_kwargs
            )
        if model.model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(inputs["input_ids"][0]):]
        outputs = model.tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )
        return outputs,


class PreviewString(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "value": ("STRING", {}),
            }
        }

    FUNCTION = "execute"
    RETURN_TYPES = ("STRING",)
    OUTPUT_NODE = True

    def execute(self, value: str):
        return {"ui": {"string": [value]}}


NODE_CLASS_MAPPINGS = {}
for cls in (
        TransformersLoader,
        SimpleBatchDecode,
        SimpleInstruct,
        PreviewString,
):
    NODE_CLASS_MAPPINGS[cls.__name__] = cls
