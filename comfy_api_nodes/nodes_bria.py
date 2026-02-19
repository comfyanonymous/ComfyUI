from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api_nodes.apis.bria import (
    BriaEditImageRequest,
    BriaRemoveBackgroundRequest,
    BriaRemoveBackgroundResponse,
    BriaRemoveVideoBackgroundRequest,
    BriaRemoveVideoBackgroundResponse,
    BriaImageEditResponse,
    BriaStatusResponse,
    InputModerationSettings,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    convert_mask_to_image,
    download_url_to_image_tensor,
    download_url_to_video_output,
    poll_op,
    sync_op,
    upload_image_to_comfyapi,
    upload_video_to_comfyapi,
    validate_video_duration,
)


class BriaImageEditNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BriaImageEditNode",
            display_name="Bria FIBO Image Edit",
            category="api node/image/Bria",
            description="Edit images using Bria latest model",
            inputs=[
                IO.Combo.Input("model", options=["FIBO"]),
                IO.Image.Input("image"),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Instruction to edit image",
                ),
                IO.String.Input("negative_prompt", multiline=True, default=""),
                IO.String.Input(
                    "structured_prompt",
                    multiline=True,
                    default="",
                    tooltip="A string containing the structured edit prompt in JSON format. "
                    "Use this instead of usual prompt for precise, programmatic control.",
                ),
                IO.Int.Input(
                    "seed",
                    default=1,
                    min=1,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                ),
                IO.Float.Input(
                    "guidance_scale",
                    default=3,
                    min=3,
                    max=5,
                    step=0.01,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="Higher value makes the image follow the prompt more closely.",
                ),
                IO.Int.Input(
                    "steps",
                    default=50,
                    min=20,
                    max=50,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                ),
                IO.DynamicCombo.Input(
                    "moderation",
                    options=[
                        IO.DynamicCombo.Option("false", []),
                        IO.DynamicCombo.Option(
                            "true",
                            [
                                IO.Boolean.Input("prompt_content_moderation", default=False),
                                IO.Boolean.Input("visual_input_moderation", default=False),
                                IO.Boolean.Input("visual_output_moderation", default=True),
                            ],
                        ),
                    ],
                    tooltip="Moderation settings",
                ),
                IO.Mask.Input(
                    "mask",
                    tooltip="If omitted, the edit applies to the entire image.",
                    optional=True,
                ),
            ],
            outputs=[
                IO.Image.Output(),
                IO.String.Output(display_name="structured_prompt"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd":0.04}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        image: Input.Image,
        prompt: str,
        negative_prompt: str,
        structured_prompt: str,
        seed: int,
        guidance_scale: float,
        steps: int,
        moderation: InputModerationSettings,
        mask: Input.Image | None = None,
    ) -> IO.NodeOutput:
        if not prompt and not structured_prompt:
            raise ValueError("One of prompt or structured_prompt is required to be non-empty.")
        mask_url = None
        if mask is not None:
            mask_url = await upload_image_to_comfyapi(cls, convert_mask_to_image(mask), wait_label="Uploading mask")
        response = await sync_op(
            cls,
            ApiEndpoint(path="proxy/bria/v2/image/edit", method="POST"),
            data=BriaEditImageRequest(
                instruction=prompt if prompt else None,
                structured_instruction=structured_prompt if structured_prompt else None,
                images=[await upload_image_to_comfyapi(cls, image, wait_label="Uploading image")],
                mask=mask_url,
                negative_prompt=negative_prompt if negative_prompt else None,
                guidance_scale=guidance_scale,
                seed=seed,
                model_version=model,
                steps_num=steps,
                prompt_content_moderation=moderation.get("prompt_content_moderation", False),
                visual_input_content_moderation=moderation.get("visual_input_moderation", False),
                visual_output_content_moderation=moderation.get("visual_output_moderation", False),
            ),
            response_model=BriaStatusResponse,
        )
        response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/bria/v2/status/{response.request_id}"),
            status_extractor=lambda r: r.status,
            response_model=BriaImageEditResponse,
        )
        return IO.NodeOutput(
            await download_url_to_image_tensor(response.result.image_url),
            response.result.structured_prompt,
        )


class BriaRemoveImageBackground(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BriaRemoveImageBackground",
            display_name="Bria Remove Image Background",
            category="api node/image/Bria",
            description="Remove the background from an image using Bria RMBG 2.0.",
            inputs=[
                IO.Image.Input("image"),
                IO.DynamicCombo.Input(
                    "moderation",
                    options=[
                        IO.DynamicCombo.Option("false", []),
                        IO.DynamicCombo.Option(
                            "true",
                            [
                                IO.Boolean.Input("visual_input_moderation", default=False),
                                IO.Boolean.Input("visual_output_moderation", default=True),
                            ],
                        ),
                    ],
                    tooltip="Moderation settings",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed controls whether the node should re-run; "
                    "results are non-deterministic regardless of seed.",
                ),
            ],
            outputs=[IO.Image.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd":0.018}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        image: Input.Image,
        moderation: dict,
        seed: int,
    ) -> IO.NodeOutput:
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/bria/v2/image/edit/remove_background", method="POST"),
            data=BriaRemoveBackgroundRequest(
                image=await upload_image_to_comfyapi(cls, image, wait_label="Uploading image"),
                sync=False,
                visual_input_content_moderation=moderation.get("visual_input_moderation", False),
                visual_output_content_moderation=moderation.get("visual_output_moderation", False),
                seed=seed,
            ),
            response_model=BriaStatusResponse,
        )
        response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/bria/v2/status/{response.request_id}"),
            status_extractor=lambda r: r.status,
            response_model=BriaRemoveBackgroundResponse,
        )
        return IO.NodeOutput(await download_url_to_image_tensor(response.result.image_url))


class BriaRemoveVideoBackground(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BriaRemoveVideoBackground",
            display_name="Bria Remove Video Background",
            category="api node/video/Bria",
            description="Remove the background from a video using Bria. ",
            inputs=[
                IO.Video.Input("video"),
                IO.Combo.Input(
                    "background_color",
                    options=[
                        "Black",
                        "White",
                        "Gray",
                        "Red",
                        "Green",
                        "Blue",
                        "Yellow",
                        "Cyan",
                        "Magenta",
                        "Orange",
                    ],
                    tooltip="Background color for the output video.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed controls whether the node should re-run; "
                    "results are non-deterministic regardless of seed.",
                ),
            ],
            outputs=[IO.Video.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd":0.14,"format":{"suffix":"/second"}}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        video: Input.Video,
        background_color: str,
        seed: int,
    ) -> IO.NodeOutput:
        validate_video_duration(video, max_duration=60.0)
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/bria/v2/video/edit/remove_background", method="POST"),
            data=BriaRemoveVideoBackgroundRequest(
                video=await upload_video_to_comfyapi(cls, video),
                background_color=background_color,
                output_container_and_codec="mp4_h264",
                seed=seed,
            ),
            response_model=BriaStatusResponse,
        )
        response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/bria/v2/status/{response.request_id}"),
            status_extractor=lambda r: r.status,
            response_model=BriaRemoveVideoBackgroundResponse,
        )
        return IO.NodeOutput(await download_url_to_video_output(response.result.video_url))


class BriaExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            BriaImageEditNode,
            BriaRemoveImageBackground,
            BriaRemoveVideoBackground,
        ]


async def comfy_entrypoint() -> BriaExtension:
    return BriaExtension()
