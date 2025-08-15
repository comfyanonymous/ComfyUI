from __future__ import annotations
from inspect import cleandoc
from typing import Optional
from comfy.comfy_types.node_typing import IO, ComfyNodeABC
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
    validate_string,
)
from server import PromptServer

import aiohttp
import torch
from io import BytesIO

LUMA_T2V_AVERAGE_DURATION = 105
LUMA_I2V_AVERAGE_DURATION = 100

def image_result_url_extractor(response: LumaGeneration):
    return response.assets.image if hasattr(response, "assets") and hasattr(response.assets, "image") else None

def video_result_url_extractor(response: LumaGeneration):
    return response.assets.video if hasattr(response, "assets") and hasattr(response.assets, "video") else None

class LumaReferenceNode(ComfyNodeABC):
    """
    Holds an image and weight for use with Luma Generate Image node.
    """

    RETURN_TYPES = (LumaIO.LUMA_REF,)
    RETURN_NAMES = ("luma_ref",)
    DESCRIPTION = cleandoc(__doc__ or "")  # Handle potential None value
    FUNCTION = "create_luma_reference"
    CATEGORY = "api node/image/Luma"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": (
                    IO.IMAGE,
                    {
                        "tooltip": "Image to use as reference.",
                    },
                ),
                "weight": (
                    IO.FLOAT,
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Weight of image reference.",
                    },
                ),
            },
            "optional": {"luma_ref": (LumaIO.LUMA_REF,)},
        }

    def create_luma_reference(
        self, image: torch.Tensor, weight: float, luma_ref: LumaReferenceChain = None
    ):
        if luma_ref is not None:
            luma_ref = luma_ref.clone()
        else:
            luma_ref = LumaReferenceChain()
        luma_ref.add(LumaReference(image=image, weight=round(weight, 2)))
        return (luma_ref,)


class LumaConceptsNode(ComfyNodeABC):
    """
    Holds one or more Camera Concepts for use with Luma Text to Video and Luma Image to Video nodes.
    """

    RETURN_TYPES = (LumaIO.LUMA_CONCEPTS,)
    RETURN_NAMES = ("luma_concepts",)
    DESCRIPTION = cleandoc(__doc__ or "")  # Handle potential None value
    FUNCTION = "create_concepts"
    CATEGORY = "api node/video/Luma"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "concept1": (get_luma_concepts(include_none=True),),
                "concept2": (get_luma_concepts(include_none=True),),
                "concept3": (get_luma_concepts(include_none=True),),
                "concept4": (get_luma_concepts(include_none=True),),
            },
            "optional": {
                "luma_concepts": (
                    LumaIO.LUMA_CONCEPTS,
                    {
                        "tooltip": "Optional Camera Concepts to add to the ones chosen here."
                    },
                ),
            },
        }

    def create_concepts(
        self,
        concept1: str,
        concept2: str,
        concept3: str,
        concept4: str,
        luma_concepts: LumaConceptChain = None,
    ):
        chain = LumaConceptChain(str_list=[concept1, concept2, concept3, concept4])
        if luma_concepts is not None:
            chain = luma_concepts.clone_and_merge(chain)
        return (chain,)


class LumaImageGenerationNode(ComfyNodeABC):
    """
    Generates images synchronously based on prompt and aspect ratio.
    """

    RETURN_TYPES = (IO.IMAGE,)
    DESCRIPTION = cleandoc(__doc__ or "")  # Handle potential None value
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node/image/Luma"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Prompt for the image generation",
                    },
                ),
                "model": ([model.value for model in LumaImageModel],),
                "aspect_ratio": (
                    [ratio.value for ratio in LumaAspectRatio],
                    {
                        "default": LumaAspectRatio.ratio_16_9,
                    },
                ),
                "seed": (
                    IO.INT,
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "Seed to determine if node should re-run; actual results are nondeterministic regardless of seed.",
                    },
                ),
                "style_image_weight": (
                    IO.FLOAT,
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Weight of style image. Ignored if no style_image provided.",
                    },
                ),
            },
            "optional": {
                "image_luma_ref": (
                    LumaIO.LUMA_REF,
                    {
                        "tooltip": "Luma Reference node connection to influence generation with input images; up to 4 images can be considered."
                    },
                ),
                "style_image": (
                    IO.IMAGE,
                    {"tooltip": "Style reference image; only 1 image will be used."},
                ),
                "character_image": (
                    IO.IMAGE,
                    {
                        "tooltip": "Character reference images; can be a batch of multiple, up to 4 images can be considered."
                    },
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    async def api_call(
        self,
        prompt: str,
        model: str,
        aspect_ratio: str,
        seed,
        style_image_weight: float,
        image_luma_ref: LumaReferenceChain = None,
        style_image: torch.Tensor = None,
        character_image: torch.Tensor = None,
        unique_id: str = None,
        **kwargs,
    ):
        validate_string(prompt, strip_whitespace=True, min_length=3)
        # handle image_luma_ref
        api_image_ref = None
        if image_luma_ref is not None:
            api_image_ref = await self._convert_luma_refs(
                image_luma_ref, max_refs=4, auth_kwargs=kwargs,
            )
        # handle style_luma_ref
        api_style_ref = None
        if style_image is not None:
            api_style_ref = await self._convert_style_image(
                style_image, weight=style_image_weight, auth_kwargs=kwargs,
            )
        # handle character_ref images
        character_ref = None
        if character_image is not None:
            download_urls = await upload_images_to_comfyapi(
                character_image, max_images=4, auth_kwargs=kwargs,
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
            auth_kwargs=kwargs,
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
            node_id=unique_id,
            auth_kwargs=kwargs,
        )
        response_poll = await operation.execute()

        async with aiohttp.ClientSession() as session:
            async with session.get(response_poll.assets.image) as img_response:
                img = process_image_response(await img_response.content.read())
        return (img,)

    async def _convert_luma_refs(
        self, luma_ref: LumaReferenceChain, max_refs: int, auth_kwargs: Optional[dict[str,str]] = None
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

    async def _convert_style_image(
        self, style_image: torch.Tensor, weight: float, auth_kwargs: Optional[dict[str,str]] = None
    ):
        chain = LumaReferenceChain(
            first_ref=LumaReference(image=style_image, weight=weight)
        )
        return await self._convert_luma_refs(chain, max_refs=1, auth_kwargs=auth_kwargs)


class LumaImageModifyNode(ComfyNodeABC):
    """
    Modifies images synchronously based on prompt and aspect ratio.
    """

    RETURN_TYPES = (IO.IMAGE,)
    DESCRIPTION = cleandoc(__doc__ or "")  # Handle potential None value
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node/image/Luma"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": (IO.IMAGE,),
                "prompt": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Prompt for the image generation",
                    },
                ),
                "image_weight": (
                    IO.FLOAT,
                    {
                        "default": 0.1,
                        "min": 0.0,
                        "max": 0.98,
                        "step": 0.01,
                        "tooltip": "Weight of the image; the closer to 1.0, the less the image will be modified.",
                    },
                ),
                "model": ([model.value for model in LumaImageModel],),
                "seed": (
                    IO.INT,
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "Seed to determine if node should re-run; actual results are nondeterministic regardless of seed.",
                    },
                ),
            },
            "optional": {},
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    async def api_call(
        self,
        prompt: str,
        model: str,
        image: torch.Tensor,
        image_weight: float,
        seed,
        unique_id: str = None,
        **kwargs,
    ):
        # first, upload image
        download_urls = await upload_images_to_comfyapi(
            image, max_images=1, auth_kwargs=kwargs,
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
            auth_kwargs=kwargs,
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
            node_id=unique_id,
            auth_kwargs=kwargs,
        )
        response_poll = await operation.execute()

        async with aiohttp.ClientSession() as session:
            async with session.get(response_poll.assets.image) as img_response:
                img = process_image_response(await img_response.content.read())
        return (img,)


class LumaTextToVideoGenerationNode(ComfyNodeABC):
    """
    Generates videos synchronously based on prompt and output_size.
    """

    RETURN_TYPES = (IO.VIDEO,)
    DESCRIPTION = cleandoc(__doc__ or "")  # Handle potential None value
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node/video/Luma"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Prompt for the video generation",
                    },
                ),
                "model": ([model.value for model in LumaVideoModel],),
                "aspect_ratio": (
                    [ratio.value for ratio in LumaAspectRatio],
                    {
                        "default": LumaAspectRatio.ratio_16_9,
                    },
                ),
                "resolution": (
                    [resolution.value for resolution in LumaVideoOutputResolution],
                    {
                        "default": LumaVideoOutputResolution.res_540p,
                    },
                ),
                "duration": ([dur.value for dur in LumaVideoModelOutputDuration],),
                "loop": (
                    IO.BOOLEAN,
                    {
                        "default": False,
                    },
                ),
                "seed": (
                    IO.INT,
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "Seed to determine if node should re-run; actual results are nondeterministic regardless of seed.",
                    },
                ),
            },
            "optional": {
                "luma_concepts": (
                    LumaIO.LUMA_CONCEPTS,
                    {
                        "tooltip": "Optional Camera Concepts to dictate camera motion via the Luma Concepts node."
                    },
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    async def api_call(
        self,
        prompt: str,
        model: str,
        aspect_ratio: str,
        resolution: str,
        duration: str,
        loop: bool,
        seed,
        luma_concepts: LumaConceptChain = None,
        unique_id: str = None,
        **kwargs,
    ):
        validate_string(prompt, strip_whitespace=False, min_length=3)
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
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                duration=duration,
                loop=loop,
                concepts=luma_concepts.create_api_model() if luma_concepts else None,
            ),
            auth_kwargs=kwargs,
        )
        response_api: LumaGeneration = await operation.execute()

        if unique_id:
            PromptServer.instance.send_progress_text(f"Luma video generation started: {response_api.id}", unique_id)

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
            node_id=unique_id,
            estimated_duration=LUMA_T2V_AVERAGE_DURATION,
            auth_kwargs=kwargs,
        )
        response_poll = await operation.execute()

        async with aiohttp.ClientSession() as session:
            async with session.get(response_poll.assets.video) as vid_response:
                return (VideoFromFile(BytesIO(await vid_response.content.read())),)


class LumaImageToVideoGenerationNode(ComfyNodeABC):
    """
    Generates videos synchronously based on prompt, input images, and output_size.
    """

    RETURN_TYPES = (IO.VIDEO,)
    DESCRIPTION = cleandoc(__doc__ or "")  # Handle potential None value
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node/video/Luma"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Prompt for the video generation",
                    },
                ),
                "model": ([model.value for model in LumaVideoModel],),
                # "aspect_ratio": ([ratio.value for ratio in LumaAspectRatio], {
                #     "default": LumaAspectRatio.ratio_16_9,
                # }),
                "resolution": (
                    [resolution.value for resolution in LumaVideoOutputResolution],
                    {
                        "default": LumaVideoOutputResolution.res_540p,
                    },
                ),
                "duration": ([dur.value for dur in LumaVideoModelOutputDuration],),
                "loop": (
                    IO.BOOLEAN,
                    {
                        "default": False,
                    },
                ),
                "seed": (
                    IO.INT,
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "Seed to determine if node should re-run; actual results are nondeterministic regardless of seed.",
                    },
                ),
            },
            "optional": {
                "first_image": (
                    IO.IMAGE,
                    {"tooltip": "First frame of generated video."},
                ),
                "last_image": (IO.IMAGE, {"tooltip": "Last frame of generated video."}),
                "luma_concepts": (
                    LumaIO.LUMA_CONCEPTS,
                    {
                        "tooltip": "Optional Camera Concepts to dictate camera motion via the Luma Concepts node."
                    },
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    async def api_call(
        self,
        prompt: str,
        model: str,
        resolution: str,
        duration: str,
        loop: bool,
        seed,
        first_image: torch.Tensor = None,
        last_image: torch.Tensor = None,
        luma_concepts: LumaConceptChain = None,
        unique_id: str = None,
        **kwargs,
    ):
        if first_image is None and last_image is None:
            raise Exception(
                "At least one of first_image and last_image requires an input."
            )
        keyframes = await self._convert_to_keyframes(first_image, last_image, auth_kwargs=kwargs)
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
            auth_kwargs=kwargs,
        )
        response_api: LumaGeneration = await operation.execute()

        if unique_id:
            PromptServer.instance.send_progress_text(f"Luma video generation started: {response_api.id}", unique_id)

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
            node_id=unique_id,
            estimated_duration=LUMA_I2V_AVERAGE_DURATION,
            auth_kwargs=kwargs,
        )
        response_poll = await operation.execute()

        async with aiohttp.ClientSession() as session:
            async with session.get(response_poll.assets.video) as vid_response:
                return (VideoFromFile(BytesIO(await vid_response.content.read())),)

    async def _convert_to_keyframes(
        self,
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


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "LumaImageNode": LumaImageGenerationNode,
    "LumaImageModifyNode": LumaImageModifyNode,
    "LumaVideoNode": LumaTextToVideoGenerationNode,
    "LumaImageToVideoNode": LumaImageToVideoGenerationNode,
    "LumaReferenceNode": LumaReferenceNode,
    "LumaConceptsNode": LumaConceptsNode,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "LumaImageNode": "Luma Text to Image",
    "LumaImageModifyNode": "Luma Image to Image",
    "LumaVideoNode": "Luma Text to Video",
    "LumaImageToVideoNode": "Luma Image to Video",
    "LumaReferenceNode": "Luma Reference",
    "LumaConceptsNode": "Luma Concepts",
}
