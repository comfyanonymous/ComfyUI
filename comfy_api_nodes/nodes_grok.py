import torch
from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api_nodes.apis.grok import (
    ImageEditRequest,
    ImageGenerationRequest,
    ImageGenerationResponse,
    InputUrlObject,
    VideoEditRequest,
    VideoGenerationRequest,
    VideoGenerationResponse,
    VideoStatusResponse,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_to_image_tensor,
    download_url_to_video_output,
    get_fs_object_size,
    get_number_of_images,
    poll_op,
    sync_op,
    tensor_to_base64_string,
    upload_video_to_comfyapi,
    validate_string,
    validate_video_duration,
)


class GrokImageNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="GrokImageNode",
            display_name="Grok Image",
            category="api node/image/Grok",
            description="Generate images using Grok based on a text prompt",
            inputs=[
                IO.Combo.Input("model", options=["grok-imagine-image-beta"]),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="The text prompt used to generate the image",
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=[
                        "1:1",
                        "2:3",
                        "3:2",
                        "3:4",
                        "4:3",
                        "9:16",
                        "16:9",
                        "9:19.5",
                        "19.5:9",
                        "9:20",
                        "20:9",
                        "1:2",
                        "2:1",
                    ],
                ),
                IO.Int.Input(
                    "number_of_images",
                    default=1,
                    min=1,
                    max=10,
                    step=1,
                    tooltip="Number of images to generate",
                    display_mode=IO.NumberDisplay.number,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to determine if node should re-run; "
                    "actual results are nondeterministic regardless of seed.",
                ),
            ],
            outputs=[
                IO.Image.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["number_of_images"]),
                expr="""{"type":"usd","usd":0.033 * widgets.number_of_images}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        prompt: str,
        aspect_ratio: str,
        number_of_images: int,
        seed: int,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1)
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/xai/v1/images/generations", method="POST"),
            data=ImageGenerationRequest(
                model=model,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                n=number_of_images,
                seed=seed,
            ),
            response_model=ImageGenerationResponse,
        )
        if len(response.data) == 1:
            return IO.NodeOutput(await download_url_to_image_tensor(response.data[0].url))
        return IO.NodeOutput(
            torch.cat(
                [await download_url_to_image_tensor(i) for i in [str(d.url) for d in response.data if d.url]],
            )
        )


class GrokImageEditNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="GrokImageEditNode",
            display_name="Grok Image Edit",
            category="api node/image/Grok",
            description="Modify an existing image based on a text prompt",
            inputs=[
                IO.Combo.Input("model", options=["grok-imagine-image-beta"]),
                IO.Image.Input("image"),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="The text prompt used to generate the image",
                ),
                IO.Combo.Input("resolution", options=["1K"]),
                IO.Int.Input(
                    "number_of_images",
                    default=1,
                    min=1,
                    max=10,
                    step=1,
                    tooltip="Number of edited images to generate",
                    display_mode=IO.NumberDisplay.number,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to determine if node should re-run; "
                    "actual results are nondeterministic regardless of seed.",
                ),
            ],
            outputs=[
                IO.Image.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["number_of_images"]),
                expr="""{"type":"usd","usd":0.002 + 0.033 * widgets.number_of_images}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        image: Input.Image,
        prompt: str,
        resolution: str,
        number_of_images: int,
        seed: int,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1)
        if get_number_of_images(image) != 1:
            raise ValueError("Only one input image is supported.")
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/xai/v1/images/edits", method="POST"),
            data=ImageEditRequest(
                model=model,
                image=InputUrlObject(url=f"data:image/png;base64,{tensor_to_base64_string(image)}"),
                prompt=prompt,
                resolution=resolution.lower(),
                n=number_of_images,
                seed=seed,
            ),
            response_model=ImageGenerationResponse,
        )
        if len(response.data) == 1:
            return IO.NodeOutput(await download_url_to_image_tensor(response.data[0].url))
        return IO.NodeOutput(
            torch.cat(
                [await download_url_to_image_tensor(i) for i in [str(d.url) for d in response.data if d.url]],
            )
        )


class GrokVideoNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="GrokVideoNode",
            display_name="Grok Video",
            category="api node/video/Grok",
            description="Generate video from a prompt or an image",
            inputs=[
                IO.Combo.Input("model", options=["grok-imagine-video-beta"]),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="Text description of the desired video.",
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["480p", "720p"],
                    tooltip="The resolution of the output video.",
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=["auto", "16:9", "4:3", "3:2", "1:1", "2:3", "3:4", "9:16"],
                    tooltip="The aspect ratio of the output video.",
                ),
                IO.Int.Input(
                    "duration",
                    default=6,
                    min=1,
                    max=15,
                    step=1,
                    tooltip="The duration of the output video in seconds.",
                    display_mode=IO.NumberDisplay.slider,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to determine if node should re-run; "
                    "actual results are nondeterministic regardless of seed.",
                ),
                IO.Image.Input("image", optional=True),
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
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["duration"], inputs=["image"]),
                expr="""
                (
                  $base := 0.181 * widgets.duration;
                  {"type":"usd","usd": inputs.image.connected ? $base + 0.002 : $base}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        prompt: str,
        resolution: str,
        aspect_ratio: str,
        duration: int,
        seed: int,
        image: Input.Image | None = None,
    ) -> IO.NodeOutput:
        image_url = None
        if image is not None:
            if get_number_of_images(image) != 1:
                raise ValueError("Only one input image is supported.")
            image_url = InputUrlObject(url=f"data:image/png;base64,{tensor_to_base64_string(image)}")
        validate_string(prompt, strip_whitespace=True, min_length=1)
        initial_response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/xai/v1/videos/generations", method="POST"),
            data=VideoGenerationRequest(
                model=model,
                image=image_url,
                prompt=prompt,
                resolution=resolution,
                duration=duration,
                aspect_ratio=None if aspect_ratio == "auto" else aspect_ratio,
                seed=seed,
            ),
            response_model=VideoGenerationResponse,
        )
        response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/xai/v1/videos/{initial_response.request_id}"),
            status_extractor=lambda r: r.status if r.status is not None else "complete",
            response_model=VideoStatusResponse,
        )
        return IO.NodeOutput(await download_url_to_video_output(response.video.url))


class GrokVideoEditNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="GrokVideoEditNode",
            display_name="Grok Video Edit",
            category="api node/video/Grok",
            description="Edit an existing video based on a text prompt.",
            inputs=[
                IO.Combo.Input("model", options=["grok-imagine-video-beta"]),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="Text description of the desired video.",
                ),
                IO.Video.Input("video", tooltip="Maximum supported duration is 8.7 seconds and 50MB file size."),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
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
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd": 0.191, "format": {"suffix": "/sec", "approximate": true}}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        prompt: str,
        video: Input.Video,
        seed: int,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1)
        validate_video_duration(video, min_duration=1, max_duration=8.7)
        video_stream = video.get_stream_source()
        video_size = get_fs_object_size(video_stream)
        if video_size > 50 * 1024 * 1024:
            raise ValueError(f"Video size ({video_size / 1024 / 1024:.1f}MB) exceeds 50MB limit.")
        initial_response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/xai/v1/videos/edits", method="POST"),
            data=VideoEditRequest(
                model=model,
                video=InputUrlObject(url=await upload_video_to_comfyapi(cls, video)),
                prompt=prompt,
                seed=seed,
            ),
            response_model=VideoGenerationResponse,
        )
        response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/xai/v1/videos/{initial_response.request_id}"),
            status_extractor=lambda r: r.status if r.status is not None else "complete",
            response_model=VideoStatusResponse,
        )
        return IO.NodeOutput(await download_url_to_video_output(response.video.url))


class GrokExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            GrokImageNode,
            GrokImageEditNode,
            GrokVideoNode,
            GrokVideoEditNode,
        ]


async def comfy_entrypoint() -> GrokExtension:
    return GrokExtension()
