import base64
import io
import os
from io import BytesIO
from typing import Literal, Optional

import numpy as np
import requests
import torch
from PIL import Image
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from comfy.cli_args import args
from comfy.component_model.tensor_types import RGBImageBatch
from comfy.language.language_types import LanguageModel, ProcessorResult, GENERATION_KWARGS_TYPE, TOKENS_TYPE, \
    TransformerStreamedProgress, LanguagePrompt
from comfy.nodes.package_typing import CustomNode, InputTypes
from comfy.utils import comfy_progress, ProgressBar, seed_for_block


class _Client:
    _client: Optional[OpenAI] = None

    @staticmethod
    def instance() -> OpenAI:
        if _Client._client is None:
            open_ai_api_key = args.openai_api_key
            _Client._client = OpenAI(
                api_key=open_ai_api_key,
            )

        return _Client._client


def validate_has_key():
    open_api_key = os.environ.get("OPENAI_API_KEY", args.openai_api_key)
    if open_api_key is None or open_api_key == "":
        return "set OPENAI_API_KEY environment variable"
    return True


def image_to_base64(image: RGBImageBatch) -> str:
    pil_image = Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


class OpenAILanguageModelWrapper(LanguageModel):
    def __init__(self, model: str):
        self.model = model
        self.client = _Client.instance()

    @staticmethod
    def from_pretrained(ckpt_name: str, subfolder: Optional[str] = None) -> "OpenAILanguageModelWrapper":
        return OpenAILanguageModelWrapper(ckpt_name)

    def generate(self, tokens: TOKENS_TYPE = None,
                 max_new_tokens: int = 512,
                 repetition_penalty: float = 0.0,
                 seed: int = 0,
                 sampler: Optional[GENERATION_KWARGS_TYPE] = None,
                 *args,
                 **kwargs) -> str:
        sampler = sampler or {}
        prompt = tokens.get("inputs", [])
        prompt = "".join(prompt)
        images = tokens.get("images", [])
        images = images if images is not None else []
        images = [image for image in images if image is not None]
        messages: list[ChatCompletionMessageParam] = [
            {
                "role": "user",
                "content": [
                               {"type": "text", "text": prompt},
                           ] + [
                               {
                                   "type": "image_url",
                                   "image_url": {
                                       "url": f"data:image/jpeg;base64,{image_to_base64(image)}"
                                   }
                               } for image in images
                           ]
            }
        ]

        progress_bar: ProgressBar
        with comfy_progress(total=max_new_tokens) as progress_bar:
            token_count = 0
            full_response = ""

            def on_finalized_text(next_token: str, stop: bool):
                nonlocal token_count
                nonlocal progress_bar
                nonlocal full_response

                token_count += 1
                full_response += next_token
                preview = TransformerStreamedProgress(next_token=next_token)
                progress_bar.update_absolute(max_new_tokens if stop else token_count, total=max_new_tokens, preview_image_or_output=preview)

            with seed_for_block(seed):
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=sampler.get("temperature", 1.0),
                    top_p=sampler.get("top_p", 1.0),
                    # n=1,
                    # stop=None,
                    # presence_penalty=repetition_penalty,
                    seed=seed,
                    stream=True
                )

                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        on_finalized_text(chunk.choices[0].delta.content, False)

                on_finalized_text("", True)  # Signal the end of streaming

        return full_response

    def tokenize(self, prompt: str | LanguagePrompt, images: RGBImageBatch | None, chat_template: str | None = None) -> ProcessorResult:
        # OpenAI API doesn't require explicit tokenization, so we'll just return the prompt and images as is
        return {
            "inputs": [prompt],
            "attention_mask": torch.ones(1, len(prompt)),  # Dummy attention mask
            "images": images
        }

    @property
    def repo_id(self) -> str:
        return f"openai/{self.model}"


class OpenAILanguageModelLoader(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "model": (["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"], {"default": "gpt-3.5-turbo"})
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("language model",)

    FUNCTION = "execute"
    CATEGORY = "openai"

    def execute(self, model: str) -> tuple[LanguageModel]:
        return OpenAILanguageModelWrapper(model),

    @classmethod
    def VALIDATE_INPUTS(cls):
        return validate_has_key()


class DallEGenerate(CustomNode):

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": (["dall-e-2", "dall-e-3"], {"default": "dall-e-3"}),
            "text": ("STRING", {"multiline": True}),
            "size": (["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"], {"default": "1024x1024"}),
            "quality": (["standard", "hd"], {"default": "standard"}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("images", "revised prompt")
    FUNCTION = "generate"

    CATEGORY = "openai"

    def generate(self,
                 model: Literal["dall-e-2", "dall-e-3"],
                 text: str,
                 size: Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"],
                 quality: Literal["standard", "hd"]) -> tuple[RGBImageBatch, str]:
        response = _Client.instance().images.generate(
            model=model,
            prompt=text,
            size=size,
            quality=quality,
            n=1,
        )

        image_url = response.data[0].url
        image_response = requests.get(image_url)

        img = Image.open(BytesIO(image_response.content))

        image = np.array(img).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        return image, response.data[0].revised_prompt

    @classmethod
    def VALIDATE_INPUTS(cls):
        return validate_has_key()


NODE_CLASS_MAPPINGS = {
    "DallEGenerate": DallEGenerate,
    "OpenAILanguageModelLoader": OpenAILanguageModelLoader
}
