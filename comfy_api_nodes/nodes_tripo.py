import os
from folder_paths import get_output_directory
from comfy_api_nodes.mapper_utils import model_field_to_node_input
from comfy.comfy_types.node_typing import IO
from comfy_api_nodes.apis import (
    TripoOrientation,
    TripoModelVersion,
)
from comfy_api_nodes.apis.tripo_api import (
    TripoTaskType,
    TripoStyle,
    TripoFileReference,
    TripoFileEmptyReference,
    TripoUrlReference,
    TripoTaskResponse,
    TripoTaskStatus,
    TripoTextToModelRequest,
    TripoImageToModelRequest,
    TripoMultiviewToModelRequest,
    TripoTextureModelRequest,
    TripoRefineModelRequest,
    TripoAnimateRigRequest,
    TripoAnimateRetargetRequest,
    TripoConvertModelRequest,
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
    download_url_to_bytesio,
)


async def upload_image_to_tripo(image, **kwargs):
    urls = await upload_images_to_comfyapi(image, max_images=1, auth_kwargs=kwargs)
    return TripoFileReference(TripoUrlReference(url=urls[0], type="jpeg"))

def get_model_url_from_response(response: TripoTaskResponse) -> str:
    if response.data is not None:
        for key in ["pbr_model", "model", "base_model"]:
            if getattr(response.data.output, key, None) is not None:
                return getattr(response.data.output, key)
    raise RuntimeError(f"Failed to get model url from response: {response}")


async def poll_until_finished(
    kwargs: dict[str, str],
    response: TripoTaskResponse,
) -> tuple[str, str]:
    """Polls the Tripo API endpoint until the task reaches a terminal state, then returns the response."""
    if response.code != 0:
        raise RuntimeError(f"Failed to generate mesh: {response.error}")
    task_id = response.data.task_id
    response_poll = await PollingOperation(
        poll_endpoint=ApiEndpoint(
            path=f"/proxy/tripo/v2/openapi/task/{task_id}",
            method=HttpMethod.GET,
            request_model=EmptyRequest,
            response_model=TripoTaskResponse,
        ),
        completed_statuses=[TripoTaskStatus.SUCCESS],
        failed_statuses=[
            TripoTaskStatus.FAILED,
            TripoTaskStatus.CANCELLED,
            TripoTaskStatus.UNKNOWN,
            TripoTaskStatus.BANNED,
            TripoTaskStatus.EXPIRED,
        ],
        status_extractor=lambda x: x.data.status,
        auth_kwargs=kwargs,
        node_id=kwargs["unique_id"],
        result_url_extractor=get_model_url_from_response,
        progress_extractor=lambda x: x.data.progress,
    ).execute()
    if response_poll.data.status == TripoTaskStatus.SUCCESS:
        url = get_model_url_from_response(response_poll)
        bytesio = await download_url_to_bytesio(url)
        # Save the downloaded model file
        model_file = f"tripo_model_{task_id}.glb"
        with open(os.path.join(get_output_directory(), model_file), "wb") as f:
            f.write(bytesio.getvalue())
        return model_file, task_id
    raise RuntimeError(f"Failed to generate mesh: {response_poll}")


class TripoTextToModelNode:
    """
    Generates 3D models synchronously based on a text prompt using Tripo's API.
    """
    AVERAGE_DURATION = 80
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True}),
                "model_version": model_field_to_node_input(IO.COMBO, TripoTextToModelRequest, "model_version", enum_type=TripoModelVersion),
                "style": model_field_to_node_input(IO.COMBO, TripoTextToModelRequest, "style", enum_type=TripoStyle, default="None"),
                "texture": ("BOOLEAN", {"default": True}),
                "pbr": ("BOOLEAN", {"default": True}),
                "image_seed": ("INT", {"default": 42}),
                "model_seed": ("INT", {"default": 42}),
                "texture_seed": ("INT", {"default": 42}),
                "texture_quality": (["standard", "detailed"], {"default": "standard"}),
                "face_limit": ("INT", {"min": -1, "max": 500000, "default": -1}),
                "quad": ("BOOLEAN", {"default": False})
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING", "MODEL_TASK_ID",)
    RETURN_NAMES = ("model_file", "model task_id")
    FUNCTION = "generate_mesh"
    CATEGORY = "api node/3d/Tripo"
    API_NODE = True
    OUTPUT_NODE = True

    async def generate_mesh(self, prompt, negative_prompt=None, model_version=None, style=None, texture=None, pbr=None, image_seed=None, model_seed=None, texture_seed=None, texture_quality=None, face_limit=None, quad=None, **kwargs):
        style_enum = None if style == "None" else style
        if not prompt:
            raise RuntimeError("Prompt is required")
        response = await SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/tripo/v2/openapi/task",
                method=HttpMethod.POST,
                request_model=TripoTextToModelRequest,
                response_model=TripoTaskResponse,
            ),
            request=TripoTextToModelRequest(
                type=TripoTaskType.TEXT_TO_MODEL,
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                model_version=model_version,
                style=style_enum,
                texture=texture,
                pbr=pbr,
                image_seed=image_seed,
                model_seed=model_seed,
                texture_seed=texture_seed,
                texture_quality=texture_quality,
                face_limit=face_limit,
                auto_size=True,
                quad=quad
            ),
            auth_kwargs=kwargs,
        ).execute()
        return await poll_until_finished(kwargs, response)


class TripoImageToModelNode:
    """
    Generates 3D models synchronously based on a single image using Tripo's API.
    """
    AVERAGE_DURATION = 80
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "model_version": model_field_to_node_input(IO.COMBO, TripoImageToModelRequest, "model_version", enum_type=TripoModelVersion),
                "style": model_field_to_node_input(IO.COMBO, TripoTextToModelRequest, "style", enum_type=TripoStyle, default="None"),
                "texture": ("BOOLEAN", {"default": True}),
                "pbr": ("BOOLEAN", {"default": True}),
                "model_seed": ("INT", {"default": 42}),
                "orientation": model_field_to_node_input(IO.COMBO, TripoImageToModelRequest, "orientation", enum_type=TripoOrientation),
                "texture_seed": ("INT", {"default": 42}),
                "texture_quality": (["standard", "detailed"], {"default": "standard"}),
                "texture_alignment": (["original_image", "geometry"], {"default": "original_image"}),
                "face_limit": ("INT", {"min": -1, "max": 500000, "default": -1}),
                "quad": ("BOOLEAN", {"default": False})
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING", "MODEL_TASK_ID",)
    RETURN_NAMES = ("model_file", "model task_id")
    FUNCTION = "generate_mesh"
    CATEGORY = "api node/3d/Tripo"
    API_NODE = True
    OUTPUT_NODE = True

    async def generate_mesh(self, image, model_version=None, style=None, texture=None, pbr=None, model_seed=None, orientation=None, texture_alignment=None, texture_seed=None, texture_quality=None, face_limit=None, quad=None, **kwargs):
        style_enum = None if style == "None" else style
        if image is None:
            raise RuntimeError("Image is required")
        tripo_file = await upload_image_to_tripo(image, **kwargs)
        response = await SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/tripo/v2/openapi/task",
                method=HttpMethod.POST,
                request_model=TripoImageToModelRequest,
                response_model=TripoTaskResponse,
            ),
            request=TripoImageToModelRequest(
                type=TripoTaskType.IMAGE_TO_MODEL,
                file=tripo_file,
                model_version=model_version,
                style=style_enum,
                texture=texture,
                pbr=pbr,
                model_seed=model_seed,
                orientation=orientation,
                texture_alignment=texture_alignment,
                texture_seed=texture_seed,
                texture_quality=texture_quality,
                face_limit=face_limit,
                auto_size=True,
                quad=quad
            ),
            auth_kwargs=kwargs,
        ).execute()
        return await poll_until_finished(kwargs, response)


class TripoMultiviewToModelNode:
    """
    Generates 3D models synchronously based on up to four images (front, left, back, right) using Tripo's API.
    """
    AVERAGE_DURATION = 80
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "image_left": ("IMAGE",),
                "image_back": ("IMAGE",),
                "image_right": ("IMAGE",),
                "model_version": model_field_to_node_input(IO.COMBO, TripoMultiviewToModelRequest, "model_version", enum_type=TripoModelVersion),
                "orientation": model_field_to_node_input(IO.COMBO, TripoImageToModelRequest, "orientation", enum_type=TripoOrientation),
                "texture": ("BOOLEAN", {"default": True}),
                "pbr": ("BOOLEAN", {"default": True}),
                "model_seed": ("INT", {"default": 42}),
                "texture_seed": ("INT", {"default": 42}),
                "texture_quality": (["standard", "detailed"], {"default": "standard"}),
                "texture_alignment": (["original_image", "geometry"], {"default": "original_image"}),
                "face_limit": ("INT", {"min": -1, "max": 500000, "default": -1}),
                "quad": ("BOOLEAN", {"default": False})
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING", "MODEL_TASK_ID",)
    RETURN_NAMES = ("model_file", "model task_id")
    FUNCTION = "generate_mesh"
    CATEGORY = "api node/3d/Tripo"
    API_NODE = True
    OUTPUT_NODE = True

    async def generate_mesh(self, image, image_left=None, image_back=None, image_right=None, model_version=None, orientation=None, texture=None, pbr=None, model_seed=None, texture_seed=None, texture_quality=None, texture_alignment=None, face_limit=None, quad=None, **kwargs):
        if image is None:
            raise RuntimeError("front image for multiview is required")
        images = []
        image_dict = {
            "image": image,
            "image_left": image_left,
            "image_back": image_back,
            "image_right": image_right
        }
        if image_left is None and image_back is None and image_right is None:
            raise RuntimeError("At least one of left, back, or right image must be provided for multiview")
        for image_name in ["image", "image_left", "image_back", "image_right"]:
            image_ = image_dict[image_name]
            if image_ is not None:
                tripo_file = await upload_image_to_tripo(image_, **kwargs)
                images.append(tripo_file)
            else:
                images.append(TripoFileEmptyReference())
        response = await SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/tripo/v2/openapi/task",
                method=HttpMethod.POST,
                request_model=TripoMultiviewToModelRequest,
                response_model=TripoTaskResponse,
            ),
            request=TripoMultiviewToModelRequest(
                type=TripoTaskType.MULTIVIEW_TO_MODEL,
                files=images,
                model_version=model_version,
                orientation=orientation,
                texture=texture,
                pbr=pbr,
                model_seed=model_seed,
                texture_seed=texture_seed,
                texture_quality=texture_quality,
                texture_alignment=texture_alignment,
                face_limit=face_limit,
                quad=quad,
            ),
            auth_kwargs=kwargs,
        ).execute()
        return await poll_until_finished(kwargs, response)


class TripoTextureNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_task_id": ("MODEL_TASK_ID",),
            },
            "optional": {
                "texture": ("BOOLEAN", {"default": True}),
                "pbr": ("BOOLEAN", {"default": True}),
                "texture_seed": ("INT", {"default": 42}),
                "texture_quality": (["standard", "detailed"], {"default": "standard"}),
                "texture_alignment": (["original_image", "geometry"], {"default": "original_image"}),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING", "MODEL_TASK_ID",)
    RETURN_NAMES = ("model_file", "model task_id")
    FUNCTION = "generate_mesh"
    CATEGORY = "api node/3d/Tripo"
    API_NODE = True
    OUTPUT_NODE = True
    AVERAGE_DURATION = 80

    async def generate_mesh(self, model_task_id, texture=None, pbr=None, texture_seed=None, texture_quality=None, texture_alignment=None, **kwargs):
        response = await SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/tripo/v2/openapi/task",
                method=HttpMethod.POST,
                request_model=TripoTextureModelRequest,
                response_model=TripoTaskResponse,
            ),
            request=TripoTextureModelRequest(
                original_model_task_id=model_task_id,
                texture=texture,
                pbr=pbr,
                texture_seed=texture_seed,
                texture_quality=texture_quality,
                texture_alignment=texture_alignment
            ),
            auth_kwargs=kwargs,
        ).execute()
        return await poll_until_finished(kwargs, response)


class TripoRefineNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_task_id": ("MODEL_TASK_ID", {
                    "tooltip": "Must be a v1.4 Tripo model"
                }),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    DESCRIPTION = "Refine a draft model created by v1.4 Tripo models only."

    RETURN_TYPES = ("STRING", "MODEL_TASK_ID",)
    RETURN_NAMES = ("model_file", "model task_id")
    FUNCTION = "generate_mesh"
    CATEGORY = "api node/3d/Tripo"
    API_NODE = True
    OUTPUT_NODE = True
    AVERAGE_DURATION = 240

    async def generate_mesh(self, model_task_id, **kwargs):
        response = await SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/tripo/v2/openapi/task",
                method=HttpMethod.POST,
                request_model=TripoRefineModelRequest,
                response_model=TripoTaskResponse,
            ),
            request=TripoRefineModelRequest(
                draft_model_task_id=model_task_id
            ),
            auth_kwargs=kwargs,
        ).execute()
        return await poll_until_finished(kwargs, response)


class TripoRigNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original_model_task_id": ("MODEL_TASK_ID",),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING", "RIG_TASK_ID")
    RETURN_NAMES = ("model_file", "rig task_id")
    FUNCTION = "generate_mesh"
    CATEGORY = "api node/3d/Tripo"
    API_NODE = True
    OUTPUT_NODE = True
    AVERAGE_DURATION = 180

    async def generate_mesh(self, original_model_task_id, **kwargs):
        response = await SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/tripo/v2/openapi/task",
                method=HttpMethod.POST,
                request_model=TripoAnimateRigRequest,
                response_model=TripoTaskResponse,
            ),
            request=TripoAnimateRigRequest(
                original_model_task_id=original_model_task_id,
                out_format="glb",
                spec="tripo"
            ),
            auth_kwargs=kwargs,
        ).execute()
        return await poll_until_finished(kwargs, response)


class TripoRetargetNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original_model_task_id": ("RIG_TASK_ID",),
                "animation": ([
                    "preset:idle",
                    "preset:walk",
                    "preset:climb",
                    "preset:jump",
                    "preset:slash",
                    "preset:shoot",
                    "preset:hurt",
                    "preset:fall",
                    "preset:turn",
                ],),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING", "RETARGET_TASK_ID")
    RETURN_NAMES = ("model_file", "retarget task_id")
    FUNCTION = "generate_mesh"
    CATEGORY = "api node/3d/Tripo"
    API_NODE = True
    OUTPUT_NODE = True
    AVERAGE_DURATION = 30

    async def generate_mesh(self, animation, original_model_task_id, **kwargs):
        response = await SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/tripo/v2/openapi/task",
                method=HttpMethod.POST,
                request_model=TripoAnimateRetargetRequest,
                response_model=TripoTaskResponse,
            ),
            request=TripoAnimateRetargetRequest(
                original_model_task_id=original_model_task_id,
                animation=animation,
                out_format="glb",
                bake_animation=True
            ),
            auth_kwargs=kwargs,
        ).execute()
        return await poll_until_finished(kwargs, response)


class TripoConversionNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original_model_task_id": ("MODEL_TASK_ID,RIG_TASK_ID,RETARGET_TASK_ID",),
                "format": (["GLTF", "USDZ", "FBX", "OBJ", "STL", "3MF"],),
            },
            "optional": {
                "quad": ("BOOLEAN", {"default": False}),
                "face_limit": ("INT", {"min": -1, "max": 500000, "default": -1}),
                "texture_size": ("INT", {"min": 128, "max": 4096, "default": 4096}),
                "texture_format": (["BMP", "DPX", "HDR", "JPEG", "OPEN_EXR", "PNG", "TARGA", "TIFF", "WEBP"], {"default": "JPEG"})
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        # The min and max of input1 and input2 are still validated because
        # we didn't take `input1` or `input2` as arguments
        if input_types["original_model_task_id"] not in ("MODEL_TASK_ID", "RIG_TASK_ID", "RETARGET_TASK_ID"):
            return "original_model_task_id must be MODEL_TASK_ID, RIG_TASK_ID or RETARGET_TASK_ID type"
        return True

    RETURN_TYPES = ()
    FUNCTION = "generate_mesh"
    CATEGORY = "api node/3d/Tripo"
    API_NODE = True
    OUTPUT_NODE = True
    AVERAGE_DURATION = 30

    async def generate_mesh(self, original_model_task_id, format, quad, face_limit, texture_size, texture_format, **kwargs):
        if not original_model_task_id:
            raise RuntimeError("original_model_task_id is required")
        response = await SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/tripo/v2/openapi/task",
                method=HttpMethod.POST,
                request_model=TripoConvertModelRequest,
                response_model=TripoTaskResponse,
            ),
            request=TripoConvertModelRequest(
                original_model_task_id=original_model_task_id,
                format=format,
                quad=quad if quad else None,
                face_limit=face_limit if face_limit != -1 else None,
                texture_size=texture_size if texture_size != 4096 else None,
                texture_format=texture_format if texture_format != "JPEG" else None
            ),
            auth_kwargs=kwargs,
        ).execute()
        return await poll_until_finished(kwargs, response)


NODE_CLASS_MAPPINGS = {
    "TripoTextToModelNode": TripoTextToModelNode,
    "TripoImageToModelNode": TripoImageToModelNode,
    "TripoMultiviewToModelNode": TripoMultiviewToModelNode,
    "TripoTextureNode": TripoTextureNode,
    "TripoRefineNode": TripoRefineNode,
    "TripoRigNode": TripoRigNode,
    "TripoRetargetNode": TripoRetargetNode,
    "TripoConversionNode": TripoConversionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TripoTextToModelNode": "Tripo: Text to Model",
    "TripoImageToModelNode": "Tripo: Image to Model",
    "TripoMultiviewToModelNode": "Tripo: Multiview to Model",
    "TripoTextureNode": "Tripo: Texture model",
    "TripoRefineNode": "Tripo: Refine Draft model",
    "TripoRigNode": "Tripo: Rig model",
    "TripoRetargetNode": "Tripo: Retarget rigged model",
    "TripoConversionNode": "Tripo: Convert model",
}
