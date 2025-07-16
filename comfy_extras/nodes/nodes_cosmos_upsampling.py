import logging
import re
from pathlib import Path
from typing import Optional

import torch

from comfy import model_management
from comfy.component_model.empty_init import EmptyInitOnDevice
from comfy.component_model.tensor_types import RGBImageBatch
from comfy.language.language_types import LanguageModel, ProcessorResult, LanguagePrompt, GENERATION_KWARGS_TYPE, \
    TOKENS_TYPE
from comfy.model_downloader import get_or_download_huggingface_repo
from comfy.model_management import load_models_gpu
from comfy.model_patcher import ModelPatcher
from comfy.node_helpers import export_custom_nodes
from comfy.nodes.package_typing import InputTypes, ValidatedNodeResult, CustomNode
from .nodes_language import TransformersLoader, TransformersTokenize, OneShotInstructTokenize, _AUTO_CHAT_TEMPLATE

# from https://github.com/NVIDIA/Cosmos/blob/b867572b99d08f450ddb8bcd6661d8c35bf6b967/cosmos1/models/diffusion/nemo/inference/inference_utils.py#L54
COSMOS_TEXT_TO_WORLD_UPSAMPLE_TASK = "Upsample the short caption to a long caption: "
COSMOS_VIDEO_TO_WORLD_UPSAMPLE_TASK = """
Your task is to transform a given prompt into a refined and concise video description, no more than 150 words.
Focus only on the content, no filler words or descriptions on the style. Never mention things outside the video.
"""

logger = logging.getLogger(__name__)


def _log_install_cosmos():
    logger.error("""
Cosmos was not installed. Install it using 

pip install loguru pynvml
pip install --no-deps git+https://github.com/NVIDIA/Cosmos.git

then restart this instance.

""")


# from cosmos repository, obviously written by Claude
def clean_text(text: str) -> str:
    """Clean the text by removing prefixes, suffixes, formatting markers, and normalizing whitespace."""
    # Replace all variations of newlines with a space
    text = text.replace("\n", " ").replace("\r", " ")

    # Use a regex to find sections of the form '- **...**'
    pattern = r"(- \*\*)(.*?)(\*\*)"

    def replacement(match: re.Match[str]) -> str:
        content = match.group(2)  # The text inside - ** and **
        words = re.findall(r"\w+", content)
        if len(words) < 10:
            # If fewer than 10 words, remove the entire '- **...**' portion
            return ""
        else:
            # If 10 or more words, keep the entire section as it is
            return match.group(0)

    text = re.sub(pattern, replacement, text)

    # Remove common prefixes
    prefixes = ["Caption:", "#####", "####", "- ", "* ", ","]
    for prefix in prefixes:
        # lstrip(prefix) won't strip entire strings, but character sets.
        # For more reliable prefix removal, do:
        if text.startswith(prefix):
            text = text[len(prefix):].lstrip()

    # Remove extra spaces
    text = " ".join(text.split())

    # Strip any remaining leading/trailing punctuation, whitespace, and quotes
    text = text.strip(' -,*:"\'"“”')  # pylint: disable=bad-str-strip-call

    return text


class PixtralTransformersLoader(TransformersLoader):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "ckpt_name": (["unsloth/Pixtral-12B-2409"], {}),
            },
        }


class Mistral12b(LanguageModel):
    def __init__(self, model: ModelPatcher, ckpt_name: str):
        self.model = model
        self.ckpt_name = ckpt_name

    @staticmethod
    def from_pretrained(ckpt_name: str, subfolder: Optional[str] = None) -> "Mistral12b":
        try:
            from cosmos1.models.autoregressive.configs.base.model_config import create_text_model_config  # pylint: disable=import-error
            from cosmos1.models.autoregressive.model import AutoRegressiveModel  # pylint: disable=import-error
        except (ImportError, ModuleNotFoundError) as exc_info:
            _log_install_cosmos()
            raise exc_info
        checkpoint_dir = get_or_download_huggingface_repo(ckpt_name)
        assert checkpoint_dir is not None, f"did not successfully download {ckpt_name}"
        checkpoint_dir = Path(checkpoint_dir)
        model_config, tokenizer_config = create_text_model_config(
            model_ckpt_path=str(checkpoint_dir / "model.pt"),
            tokenizer_path=str(checkpoint_dir),
            model_family="mistral",
            model_size="12b",
            is_instruct_model=True,
            max_batch_size=1,
            rope_dim="1D",
            add_special_tokens=True,
            max_seq_len=1024,
            pytorch_rope_version="v1",
        )

        try:
            with EmptyInitOnDevice(device=model_management.unet_offload_device()):
                completion_instance_cpu = AutoRegressiveModel.build(model_config=model_config, tokenizer_config=tokenizer_config)
        finally:
            torch.set_default_dtype(torch.float32)

        patchable_completion_instance_cpu = ModelPatcher(completion_instance_cpu, load_device=model_management.get_torch_device(), offload_device=model_management.unet_offload_device(), size=completion_instance_cpu.get_num_params() * 2, ckpt_name=ckpt_name)
        return Mistral12b(patchable_completion_instance_cpu, ckpt_name=ckpt_name)

    def generate(self,
                 tokens: TOKENS_TYPE = None,
                 max_new_tokens: int = 512,
                 repetition_penalty: float = 0.0,
                 seed: int = 0,
                 sampler: Optional[GENERATION_KWARGS_TYPE] = None,
                 *args,
                 **kwargs) -> str:
        sampler = sampler or {}
        prompt = tokens.get("inputs", [])
        prompt = "".join(prompt)

        dialogs = [[{"role": "user", "content": prompt}]]

        try:
            from cosmos1.models.diffusion.prompt_upsampler.inference import chat_completion  # pylint: disable=import-error
            from cosmos1.models.autoregressive.model import AutoRegressiveModel  # pylint: disable=import-error
        except (ImportError, ModuleNotFoundError) as exc_info:
            _log_install_cosmos()
            raise exc_info

        load_models_gpu([self.model])

        # noinspection PyTypeChecker
        model: AutoRegressiveModel = self.model.model
        assert isinstance(model, AutoRegressiveModel)

        results = chat_completion(
            model,
            dialogs,
            seed=seed,
            max_gen_len=max_new_tokens,
            temperature=sampler.get("temperature", 0.01),
            top_p=sampler.get("top_p", None),
            top_k=sampler.get("top_k", None),
            logprobs=False,
        )

        upsampled_prompt = str(clean_text(results[0]["generation"]["content"]))
        return upsampled_prompt

    def tokenize(self, prompt: str | LanguagePrompt, images: RGBImageBatch | None, chat_template: str | None = None) -> ProcessorResult:
        # Return prompts and image as is
        return {
            "inputs": [prompt],
            "attention_mask": torch.ones(1, len(prompt)),  # Dummy attention mask
            "images": images
        }

    @property
    def repo_id(self) -> str:
        return self.ckpt_name


class CosmosPromptUpsamplerLoader(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "ckpt_name": (["nvidia/Cosmos-1.0-Prompt-Upsampler-12B-Text2World"], {}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("upsampler model",)
    CATEGORY = "cosmos"
    FUNCTION = "execute"

    def execute(self, ckpt_name: str) -> tuple[LanguageModel]:
        return Mistral12b.from_pretrained(ckpt_name),


class CosmosText2WorldTokenize(TransformersTokenize):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "model": ("MODEL",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
        }

    def execute(self, model: LanguageModel, prompt: str) -> ValidatedNodeResult:
        return super().execute(model, f"{COSMOS_TEXT_TO_WORLD_UPSAMPLE_TASK}{prompt}")


class CosmosVideo2WorldTokenize(OneShotInstructTokenize):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "model": ("MODEL", {}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "images": ("IMAGE", {}),
            }
        }

    def execute(self, model: LanguageModel, prompt: str, images: list[torch.Tensor] | torch.Tensor = None, chat_template: str = _AUTO_CHAT_TEMPLATE, system_prompt: str = "") -> ValidatedNodeResult:
        return super().execute(model, prompt, images, chat_template=None, system_prompt=COSMOS_VIDEO_TO_WORLD_UPSAMPLE_TASK)


export_custom_nodes()
