from __future__ import annotations
from inspect import cleandoc
from typing import Optional
from typing_extensions import override
from comfy_api.latest import ComfyExtension, IO
from comfy_api.input_impl.video_types import VideoFromFile
from comfy_api_nodes.apis.luma_api import (
    LumaImageModel,
    LumaVideoModel,
    LumaVideoOutputResolution,
    LumaVideoModelOutputDuration,
    LumaAspectRatio,
    LumaState,
    LumaImageGenerationRequest,
    LumaGenerationRequest,
    LumaGeneration,
    LumaCharacterRef,
    LumaModifyImageRef,
    LumaImageIdentity,
    LumaReference,
    LumaReferenceChain,
    LumaImageReference,
    LumaKeyframes,
    LumaConceptChain,
    LumaIO,
    get_luma_concepts,
)
from comfy_api_nodes.apis.client import (
    ApiEndpoint,
    HttpMethod,
    SynchronousOperation,
    PollingOperation,
    EmptyRequest,
)
from comfy_api_nodes.apinode_utils import (
    upload_images_to_comfyapi,
    process_image_response,
)
from server import PromptServer
from comfy_api_nodes.util import validate_string

import aiohttp
import torch
from io import BytesIO

LUMA_T2V_AVERAGE_DURATION = 105
LUMA_I2V_AVERAGE_DURATION = 100

def image_result_url_extractor(response: LumaGeneration):
    return response.assets.image if hasattr(response, "assets") and hasattr(response.assets, "image") else None

def video_result_url_extractor(response: LumaGeneration):
    return response.assets.video if hasattr(response, "assets") and hasattr(response.assets, "video") else None

class LumaReferenceNode(IO.ComfyNode):
    """
    Holds an image and weight for use with Luma Generate Image node.
    """

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="LumaReferenceNode",
            display_name="Luma Reference",
            category="api node/image/Luma",
            description=cleandoc(cls.__doc__ or ""),
            inputs=[
                IO.Image.Input(
                    "image",
                    tooltip="Image to use as reference.",
                ),
                IO.Float.Input(
                    "weight",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Weight of image reference.",
                ),
                IO.Custom(LumaIO.LUMA_REF).Input(
                    "luma_ref",
                    optional=True,
                ),
            ],
            outputs=[IO.Custom(LumaIO.LUMA_REF).Output(display_name="luma_ref")],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
        )

    @classmethod
    def execute(
        cls, image: torch.Tensor, weight: float, luma_ref: LumaReferenceChain = None
    ) -> IO.NodeOutput:
        if luma_ref is not None:
            luma_ref = luma_ref.clone()
        else:
            luma_ref = LumaReferenceChain()
        luma_ref.add(LumaReference(image=image, weight=round(weight, 2)))
        return IO.NodeOutput(luma_ref)


class LumaConceptsNode(IO.ComfyNode):
    """
    Holds one or more Camera Concepts for use with Luma Text to Video and Luma Image to Video nodes.
    """

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="LumaConceptsNode",
            display_name="Luma Concepts",
            category="api node/video/Luma",
            description=cleandoc(cls.__doc__ or ""),
            inputs=[
                IO.Combo.Input(
                    "concept1",
                    options=get_luma_concepts(include_none=True),
                ),
                IO.Combo.Input(
                    "concept2",
                    options=get_luma_concepts(include_none=True),
                ),
                IO.Combo.Input(
                    "concept3",
                    options=get_luma_concepts(include_none=True),
                ),
                IO.Combo.Input(
                    "concept4",
                    options=get_luma_concepts(include_none=True),
                ),
                IO.Custom(LumaIO.LUMA_CONCEPTS).Input(
                    "luma_concepts",
                    tooltip="Optional Camera Concepts to add to the ones chosen here.",
                    optional=True,
                ),
            ],
            outputs=[IO.Custom(LumaIO.LUMA_CONCEPTS).Output(display_name="luma_concepts")],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
        )

    @classmethod
    def execute(
        cls,
        concept1: str,
        concept2: str,
        concept3: str,
        concept4: str,
        luma_concepts: LumaConceptChain = None,
    ) -> IO.NodeOutput:
        chain = LumaConceptChain(str_list=[concept1, concept2, concept3, concept4])
        if luma_concepts is not None:
            chain = luma_concepts.clone_and_merge(chain)
        return IO.NodeOutput(chain)


class LumaImageGenerationNode(IO.ComfyNode):
    """
    Generates images synchronously based on prompt and aspect ratio.
    """

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="LumaImageNode",
            display_name="Luma Text to Image",
            category="api node/image/Luma",
            description=cleandoc(cls.__doc__ or ""),
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Prompt for the image generation",
                ),
                IO.Combo.Input(
                    "model",
                    options=LumaImageModel,
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=LumaAspectRatio,
                    default=LumaAspectRatio.ratio_16_9,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="Seed to determine if node should re-run; actual results are nondeterministic regardless of seed.",
                ),
                IO.Float.Input(
                    "style_image_weight",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Weight of style image. Ignored if no style_image provided.",
                ),
                IO.Custom(LumaIO.LUMA_REF).Input(
                    "image_luma_ref",
                    tooltip="Luma Reference node connection to influence generation with input images; up to 4 images can be considered.",
                    optional=True,
                ),
                IO.Image.Input(
                    "style_image",
                    tooltip="Style reference image; only 1 image will be used.",
                    optional=True,
                ),
                IO.Image.Input(
                    "character_image",
                    tooltip="Character reference images; can be a batch of multiple, up to 4 images can be considered.",
                    optional=True,
                ),
            ],
            outputs=[IO.Image.Output()],
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
        prompt: str,
        model: str,
        aspect_ratio: str,
        seed,
        style_image_weight: float,
        image_luma_ref: LumaReferenceChain = None,
        style_image: torch.Tensor = None,
        character_image: torch.Tensor = None,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=3)
        auth_kwargs = {
            "auth_token": cls.hidden.auth_token_comfy_org,
            "comfy_api_key": cls.hidden.api_key_comfy_org,
        }
        # handle image_luma_ref
        api_image_ref = None
        if image_luma_ref is not None:
            api_image_ref = await cls._convert_luma_refs(
                image_luma_ref, max_refs=4, auth_kwargs=auth_kwargs,
            )
        # handle style_luma_ref
        api_style_ref = None
        if style_image is not None:
            api_style_ref = await cls._convert_style_image(
                style_image, weight=style_image_weight, auth_kwargs=auth_kwargs,
            )
        # handle character_ref images
        character_ref = None
        if character_image is not None:
            download_urls = await upload_images_to_comfyapi(
                character_image, max_images=4, auth_kwargs=auth_kwargs,
            )
            character_ref = LumaCharacterRef(
                identity0=LumaImageIdentity(images=download_urls)
            )

        operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/luma/generations/image",
                method=HttpMethod.POST,
                request_model=LumaImageGenerationRequest,
                response_model=LumaGeneration,
            ),
            request=LumaImageGenerationRequest(
                prompt=prompt,
                model=model,
                aspect_ratio=aspect_ratio,
                image_ref=api_image_ref,
                style_ref=api_style_ref,
                character_ref=character_ref,
            ),
            auth_kwargs=auth_kwargs,
        )
        response_api: LumaGeneration = await operation.execute()

        operation = PollingOperation(
            poll_endpoint=ApiEndpoint(
                path=f"/proxy/luma/generations/{response_api.id}",
                method=HttpMethod.GET,
                request_model=EmptyRequest,
                response_model=LumaGeneration,
            ),
            completed_statuses=[LumaState.completed],
            failed_statuses=[LumaState.failed],
            status_extractor=lambda x: x.state,
            result_url_extractor=image_result_url_extractor,
            node_id=cls.hidden.unique_id,
            auth_kwargs=auth_kwargs,
        )
        response_poll = await operation.execute()

        async with aiohttp.ClientSession() as session:
            async with session.get(response_poll.assets.image) as img_response:
                img = process_image_response(await img_response.content.read())
        return IO.NodeOutput(img)

    @classmethod
    async def _convert_luma_refs(
        cls, luma_ref: LumaReferenceChain, max_refs: int, auth_kwargs: Optional[dict[str,str]] = None
    ):
        luma_urls = []
        ref_count = 0
        for ref in luma_ref.refs:
            download_urls = await upload_images_to_comfyapi(
                ref.image, max_images=1, auth_kwargs=auth_kwargs
            )
            luma_urls.append(download_urls[0])
            ref_count += 1
            if ref_count >= max_refs:
                break
        return luma_ref.create_api_model(download_urls=luma_urls, max_refs=max_refs)

    @classmethod
    async def _convert_style_image(
        cls, style_image: torch.Tensor, weight: float, auth_kwargs: Optional[dict[str,str]] = None
    ):
        chain = LumaReferenceChain(
            first_ref=LumaReference(image=style_image, weight=weight)
        )
        return await cls._convert_luma_refs(chain, max_refs=1, auth_kwargs=auth_kwargs)


class LumaImageModifyNode(IO.ComfyNode):
    """
    Modifies images synchronously based on prompt and aspect ratio.
    """

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="LumaImageModifyNode",
            display_name="Luma Image to Image",
            category="api node/image/Luma",
            description=cleandoc(cls.__doc__ or ""),
            inputs=[
                IO.Image.Input(
                    "image",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Prompt for the image generation",
                ),
                IO.Float.Input(
                    "image_weight",
                    default=0.1,
                    min=0.0,
                    max=0.98,
                    step=0.01,
                    tooltip="Weight of the image; the closer to 1.0, the less the image will be modified.",
                ),
                IO.Combo.Input(
                    "model",
                    options=LumaImageModel,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="Seed to determine if node should re-run; actual results are nondeterministic regardless of seed.",
                ),
            ],
            outputs=[IO.Image.Output()],
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
        prompt: str,
        model: str,
        image: torch.Tensor,
        image_weight: float,
        seed,
    ) -> IO.NodeOutput:
        auth_kwargs = {
            "auth_token": cls.hidden.auth_token_comfy_org,
            "comfy_api_key": cls.hidden.api_key_comfy_org,
        }
        # first, upload image
        download_urls = await upload_images_to_comfyapi(
            image, max_images=1, auth_kwargs=auth_kwargs,
        )
        image_url = download_urls[0]
        # next, make Luma call with download url provided
        operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/luma/generations/image",
                method=HttpMethod.POST,
                request_model=LumaImageGenerationRequest,
                response_model=LumaGeneration,
            ),
            request=LumaImageGenerationRequest(
                prompt=prompt,
                model=model,
                modify_image_ref=LumaModifyImageRef(
                    url=image_url, weight=round(max(min(1.0-image_weight, 0.98), 0.0), 2)
                ),
            ),
            auth_kwargs=auth_kwargs,
        )
        response_api: LumaGeneration = await operation.execute()

        operation = PollingOperation(
            poll_endpoint=ApiEndpoint(
                path=f"/proxy/luma/generations/{response_api.id}",
                method=HttpMethod.GET,
                request_model=EmptyRequest,
                response_model=LumaGeneration,
            ),
            completed_statuses=[LumaState.completed],
            failed_statuses=[LumaState.failed],
            status_extractor=lambda x: x.state,
            result_url_extractor=image_result_url_extractor,
            node_id=cls.hidden.unique_id,
            auth_kwargs=auth_kwargs,
        )
        response_poll = await operation.execute()

        async with aiohttp.ClientSession() as session:
            async with session.get(response_poll.assets.image) as img_response:
                img = process_image_response(await img_response.content.read())
        return IO.NodeOutput(img)


class LumaTextToVideoGenerationNode(IO.ComfyNode):
    """
    Generates videos synchronously based on prompt and output_size.
    """

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="LumaVideoNode",
            display_name="Luma Text to Video",
            category="api node/video/Luma",
            description=cleandoc(cls.__doc__ or ""),
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Prompt for the video generation",
                ),
                IO.Combo.Input(
                    "model",
                    options=LumaVideoModel,
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=LumaAspectRatio,
                    default=LumaAspectRatio.ratio_16_9,
                ),
                IO.Combo.Input(
                    "resolution",
                    options=LumaVideoOutputResolution,
                    default=LumaVideoOutputResolution.res_540p,
                ),
                IO.Combo.Input(
                    "duration",
                    options=LumaVideoModelOutputDuration,
                ),
                IO.Boolean.Input(
                    "loop",
                    default=False,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="Seed to determine if node should re-run; actual results are nondeterministic regardless of seed.",
                ),
                IO.Custom(LumaIO.LUMA_CONCEPTS).Input(
                    "luma_concepts",
                    tooltip="Optional Camera Concepts to dictate camera motion via the Luma Concepts node.",
                    optional=True,
                )
            ],
            outputs=[IO.Video.Output()],
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
        prompt: str,
        model: str,
        aspect_ratio: str,
        resolution: str,
        duration: str,
        loop: bool,
        seed,
        luma_concepts: LumaConceptChain = None,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=False, min_length=3)
        duration = duration if model != LumaVideoModel.ray_1_6 else None
        resolution = resolution if model != LumaVideoModel.ray_1_6 else None

        auth_kwargs = {
            "auth_token": cls.hidden.auth_token_comfy_org,
            "comfy_api_key": cls.hidden.api_key_comfy_org,
        }
        operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/luma/generations",
                method=HttpMethod.POST,
                request_model=LumaGenerationRequest,
                response_model=LumaGeneration,
            ),
            request=LumaGenerationRequest(
                prompt=prompt,
                model=model,
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                duration=duration,
                loop=loop,
                concepts=luma_concepts.create_api_model() if luma_concepts else None,
            ),
            auth_kwargs=auth_kwargs,
        )
        response_api: LumaGeneration = await operation.execute()

        if cls.hidden.unique_id:
            PromptServer.instance.send_progress_text(f"Luma video generation started: {response_api.id}", cls.hidden.unique_id)

        operation = PollingOperation(
            poll_endpoint=ApiEndpoint(
                path=f"/proxy/luma/generations/{response_api.id}",
                method=HttpMethod.GET,
                request_model=EmptyRequest,
                response_model=LumaGeneration,
            ),
            completed_statuses=[LumaState.completed],
            failed_statuses=[LumaState.failed],
            status_extractor=lambda x: x.state,
            result_url_extractor=video_result_url_extractor,
            node_id=cls.hidden.unique_id,
            estimated_duration=LUMA_T2V_AVERAGE_DURATION,
            auth_kwargs=auth_kwargs,
        )
        response_poll = await operation.execute()

        async with aiohttp.ClientSession() as session:
            async with session.get(response_poll.assets.video) as vid_response:
                return IO.NodeOutput(VideoFromFile(BytesIO(await vid_response.content.read())))


class LumaImageToVideoGenerationNode(IO.ComfyNode):
    """
    Generates videos synchronously based on prompt, input images, and output_size.
    """

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="LumaImageToVideoNode",
            display_name="Luma Image to Video",
            category="api node/video/Luma",
            description=cleandoc(cls.__doc__ or ""),
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Prompt for the video generation",
                ),
                IO.Combo.Input(
                    "model",
                    options=LumaVideoModel,
                ),
                # IO.Combo.Input(
                #     "aspect_ratio",
                #     options=[ratio.value for ratio in LumaAspectRatio],
                #     default=LumaAspectRatio.ratio_16_9,
                # ),
                IO.Combo.Input(
                    "resolution",
                    options=LumaVideoOutputResolution,
                    default=LumaVideoOutputResolution.res_540p,
                ),
                IO.Combo.Input(
                    "duration",
                    options=[dur.value for dur in LumaVideoModelOutputDuration],
                ),
                IO.Boolean.Input(
                    "loop",
                    default=False,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="Seed to determine if node should re-run; actual results are nondeterministic regardless of seed.",
                ),
                IO.Image.Input(
                    "first_image",
                    tooltip="First frame of generated video.",
                    optional=True,
                ),
                IO.Image.Input(
                    "last_image",
                    tooltip="Last frame of generated video.",
                    optional=True,
                ),
                IO.Custom(LumaIO.LUMA_CONCEPTS).Input(
                    "luma_concepts",
                    tooltip="Optional Camera Concepts to dictate camera motion via the Luma Concepts node.",
                    optional=True,
                )
            ],
            outputs=[IO.Video.Output()],
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
        prompt: str,
        model: str,
        resolution: str,
        duration: str,
        loop: bool,
        seed,
        first_image: torch.Tensor = None,
        last_image: torch.Tensor = None,
        luma_concepts: LumaConceptChain = None,
    ) -> IO.NodeOutput:
        if first_image is None and last_image is None:
            raise Exception(
                "At least one of first_image and last_image requires an input."
            )
        auth_kwargs = {
            "auth_token": cls.hidden.auth_token_comfy_org,
            "comfy_api_key": cls.hidden.api_key_comfy_org,
        }
        keyframes = await cls._convert_to_keyframes(first_image, last_image, auth_kwargs=auth_kwargs)
        duration = duration if model != LumaVideoModel.ray_1_6 else None
        resolution = resolution if model != LumaVideoModel.ray_1_6 else None

        operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/luma/generations",
                method=HttpMethod.POST,
                request_model=LumaGenerationRequest,
                response_model=LumaGeneration,
            ),
            request=LumaGenerationRequest(
                prompt=prompt,
                model=model,
                aspect_ratio=LumaAspectRatio.ratio_16_9,  # ignored, but still needed by the API for some reason
                resolution=resolution,
                duration=duration,
                loop=loop,
                keyframes=keyframes,
                concepts=luma_concepts.create_api_model() if luma_concepts else None,
            ),
            auth_kwargs=auth_kwargs,
        )
        response_api: LumaGeneration = await operation.execute()

        if cls.hidden.unique_id:
            PromptServer.instance.send_progress_text(f"Luma video generation started: {response_api.id}", cls.hidden.unique_id)

        operation = PollingOperation(
            poll_endpoint=ApiEndpoint(
                path=f"/proxy/luma/generations/{response_api.id}",
                method=HttpMethod.GET,
                request_model=EmptyRequest,
                response_model=LumaGeneration,
            ),
            completed_statuses=[LumaState.completed],
            failed_statuses=[LumaState.failed],
            status_extractor=lambda x: x.state,
            result_url_extractor=video_result_url_extractor,
            node_id=cls.hidden.unique_id,
            estimated_duration=LUMA_I2V_AVERAGE_DURATION,
            auth_kwargs=auth_kwargs,
        )
        response_poll = await operation.execute()

        async with aiohttp.ClientSession() as session:
            async with session.get(response_poll.assets.video) as vid_response:
                return IO.NodeOutput(VideoFromFile(BytesIO(await vid_response.content.read())))

    @classmethod
    async def _convert_to_keyframes(
        cls,
        first_image: torch.Tensor = None,
        last_image: torch.Tensor = None,
        auth_kwargs: Optional[dict[str,str]] = None,
    ):
        if first_image is None and last_image is None:
            return None
        frame0 = None
        frame1 = None
        if first_image is not None:
            download_urls = await upload_images_to_comfyapi(
                first_image, max_images=1, auth_kwargs=auth_kwargs,
            )
            frame0 = LumaImageReference(type="image", url=download_urls[0])
        if last_image is not None:
            download_urls = await upload_images_to_comfyapi(
                last_image, max_images=1, auth_kwargs=auth_kwargs,
            )
            frame1 = LumaImageReference(type="image", url=download_urls[0])
        return LumaKeyframes(frame0=frame0, frame1=frame1)


class LumaExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            LumaImageGenerationNode,
            LumaImageModifyNode,
            LumaTextToVideoGenerationNode,
            LumaImageToVideoGenerationNode,
            LumaReferenceNode,
            LumaConceptsNode,
        ]


async def comfy_entrypoint() -> LumaExtension:
    return LumaExtension()
