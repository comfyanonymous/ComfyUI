from typing import Optional

import torch
from pydantic import BaseModel, Field
from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_to_video_output,
    get_number_of_images,
    poll_op,
    sync_op,
    tensor_to_bytesio,
)


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
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
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
        files_input = None
        if image is not None:
            if get_number_of_images(image) != 1:
                raise ValueError("Currently only one input image is supported.")
            files_input = {"input_reference": ("image.png", tensor_to_bytesio(image), "image/png")}
        initial_response = await sync_op(
            cls,
            endpoint=ApiEndpoint(path="/proxy/openai/v1/videos", method="POST"),
            data=Sora2GenerationRequest(
                model=model,
                prompt=prompt,
                seconds=str(duration),
                size=size,
            ),
            files=files_input,
            response_model=Sora2GenerationResponse,
            content_type="multipart/form-data",
        )
        if initial_response.error:
            raise Exception(initial_response.error["message"])

        model_time_multiplier = 1 if model == "sora-2" else 2
        await poll_op(
            cls,
            poll_endpoint=ApiEndpoint(path=f"/proxy/openai/v1/videos/{initial_response.id}"),
            response_model=Sora2GenerationResponse,
            status_extractor=lambda x: x.status,
            poll_interval=8.0,
            max_poll_attempts=160,
            estimated_duration=int(45 * (duration / 4) * model_time_multiplier),
        )
        return IO.NodeOutput(
            await download_url_to_video_output(f"/proxy/openai/v1/videos/{initial_response.id}/content", cls=cls),
        )


class OpenAISoraExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            OpenAIVideoSora2,
        ]


async def comfy_entrypoint() -> OpenAISoraExtension:
    return OpenAISoraExtension()
