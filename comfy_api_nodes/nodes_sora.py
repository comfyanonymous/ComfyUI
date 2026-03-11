from typing import Optional

import torch
from pydantic import BaseModel, Field
from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension
from comfy_api_nodes.util import (
    download_url_to_video_output,
    get_number_of_images,
    tensor_to_bytesio,
)
from comfy_api_nodes.util.client import fal_run

FAL_SORA_2 = "fal-ai/sora-2/text-to-video"


class Sora2GenerationRequest(BaseModel):
    prompt: str = Field(...)
    model: str = Field(...)
    seconds: str = Field(...)
    size: str = Field(...)


class Sora2GenerationResponse(BaseModel):
    id: str = Field(...)
    error: Optional[dict] = Field(None)
    status: Optional[str] = Field(None)


class OpenAIVideoSora2(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="OpenAIVideoSora2",
            display_name="OpenAI Sora - Video",
            category="api node/video/Sora",
            description="OpenAI video and audio generation.",
            inputs=[
                IO.Combo.Input(
                    "model",
                    options=["sora-2", "sora-2-pro"],
                    default="sora-2",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Guiding text; may be empty if an input image is present.",
                ),
                IO.Combo.Input(
                    "size",
                    options=[
                        "720x1280",
                        "1280x720",
                        "1024x1792",
                        "1792x1024",
                    ],
                    default="1280x720",
                ),
                IO.Combo.Input(
                    "duration",
                    options=[4, 8, 12],
                    default=8,
                ),
                IO.Image.Input(
                    "image",
                    optional=True,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    optional=True,
                    tooltip="Seed to determine if node should re-run; "
                    "actual results are nondeterministic regardless of seed.",
                ),
            ],
            outputs=[
                IO.Video.Output(),
            ],
            hidden=[
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        prompt: str,
        size: str = "1280x720",
        duration: int = 8,
        seed: int = 0,
        image: Optional[torch.Tensor] = None,
    ):
        if model == "sora-2" and size not in ("720x1280", "1280x720"):
            raise ValueError("Invalid size for sora-2 model, only 720x1280 and 1280x720 are supported.")
        data = {
            "model": model,
            "prompt": prompt,
            "seconds": str(duration),
            "size": size,
        }
        if image is not None:
            if get_number_of_images(image) != 1:
                raise ValueError("Currently only one input image is supported.")
            from comfy_api_nodes.util.upload_helpers import upload_image_to_fal
            data["input_reference"] = await upload_image_to_fal(
                image[0] if len(image.shape) > 3 else image, "image/png"
            )
        result = await fal_run(cls, FAL_SORA_2, data)  # TODO: verify fal.ai field names
        video_url = result["video"]["url"]
        return IO.NodeOutput(
            await download_url_to_video_output(video_url),
        )


class OpenAISoraExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            OpenAIVideoSora2,
        ]


async def comfy_entrypoint() -> OpenAISoraExtension:
    return OpenAISoraExtension()
