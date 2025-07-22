import os
from folder_paths import get_output_directory
from comfy_api_nodes.mapper_utils import model_field_to_node_input
from comfy.comfy_types.node_typing import IO
from comfy_api_nodes.apis.meshy_api import (
    MeshyArtStyle,
    MeshyAIModel,
    MeshySymmetryMode,
    MeshyTopology,
    MeshyTextToModelPreviewRequest,
    MeshyTaskResponse,
    MeshyTaskResponse,
    MeshyTaskStatus,
    MeshyTextToModelRefineRequest,
    MeshyImageToModelRequest,
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


def get_model_url_from_response(response: MeshyTaskResponse) -> str:
    if response.data is not None:
        return response.data.model_urls.glb
    raise RuntimeError(f"Failed to get model url from response: {response}")


def poll_until_finished(
    kwargs: dict[str, str],
    response: MeshyTaskResponse,
) -> tuple[str, str]:
    """Polls the Meshy API endpoint until the task reaches a terminal state, then returns the response."""
    if response.code != 0:
        raise RuntimeError(f"Failed to generate mesh: {response.error}")
    task_id = response.data.task_id
    response_poll = PollingOperation(
        poll_endpoint=ApiEndpoint(
            path=f"/proxy/meshy/openapi/v2/task/{task_id}",
            method=HttpMethod.GET,
            request_model=EmptyRequest,
            response_model=MeshyTaskResponse,
        ),
        completed_statuses=[MeshyTaskStatus.completed],
        failed_statuses=[
            MeshyTaskStatus.failed,
            MeshyTaskStatus.canceled,
        ],
        status_extractor=lambda x: x.data.status,
        auth_kwargs=kwargs,
        node_id=kwargs["unique_id"],
        result_url_extractor=get_model_url_from_response,
        progress_extractor=lambda x: x.data.progress,
    ).execute()
    if response_poll.data.status == MeshyTaskStatus.completed:
        url = get_model_url_from_response(response_poll)
        bytesio = download_url_to_bytesio(url)
        # Save the downloaded model file
        model_file = f"meshy_model_{task_id}.glb"
        with open(os.path.join(get_output_directory(), model_file), "wb") as f:
            f.write(bytesio.getvalue())
        return model_file, task_id
    raise RuntimeError(f"Failed to generate mesh: {response_poll}")


class MeshyTextToModelPreviewNode:
    """
    Generates 3D models preview based on a text prompt using Meshy's API.
    """

    AVERAGE_DURATION = 80

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
            },
            "optional": {
                "art_style": model_field_to_node_input(
                    IO.COMBO,
                    MeshyTextToModelPreviewRequest,
                    "art_style",
                    enum_type=MeshyArtStyle,
                    default="realistic",
                ),
                "seed": ("INT", {"default": 42}),
                "ai_model": model_field_to_node_input(
                    IO.COMBO,
                    MeshyTextToModelPreviewRequest,
                    "ai_model",
                    enum_type=MeshyAIModel,
                ),
                "topology": model_field_to_node_input(
                    IO.COMBO,
                    MeshyTextToModelPreviewRequest,
                    "topology",
                    enum_type=MeshyTopology,
                ),
                "target_polycount": (
                    "INT",
                    {"min": 100, "max": 300000, "default": 30000},
                ),
                "should_remesh": ("BOOLEAN", {"default": True}),
                "symmetry_mode": model_field_to_node_input(
                    IO.COMBO,
                    MeshyTextToModelPreviewRequest,
                    "symmetry_mode",
                    enum_type=MeshySymmetryMode,
                ),
                "should_simplify": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (
        "STRING",
        "MODEL_TASK_ID",
    )
    RETURN_NAMES = ("model_file", "model task_id")
    FUNCTION = "generate_mesh"
    CATEGORY = "api node/3d/Meshy"
    API_NODE = True
    OUTPUT_NODE = True

    def generate_mesh(
        self,
        prompt,
        art_style=None,
        seed=None,
        ai_model=None,
        topology=None,
        target_polycount=None,
        should_remesh=None,
        symmetry_mode=None,
        should_simplify=None,
        **kwargs,
    ):
        if not prompt:
            raise RuntimeError("Prompt is required")
        response = SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/meshy/openapi/v2/text-to-3d",
                method=HttpMethod.POST,
                request_model=MeshyTextToModelPreviewRequest,
                response_model=MeshyTaskResponse,
            ),
            request=MeshyTextToModelPreviewRequest(
                mode="preview",
                prompt=prompt,
                art_style=art_style,
                seed=seed,
                ai_model=ai_model,
                topology=topology,
                target_polycount=target_polycount,
                should_remesh=should_remesh,
                symmetry_mode=symmetry_mode,
                should_simplify=should_simplify,
            ),
            auth_kwargs=kwargs.get("auth_kwargs", {}),
        ).execute()
        return poll_until_finished(kwargs, response)


class MeshyTextToModelRefineNode:
    """
    Refines a 3D model based on a text prompt using Meshy's API.
    """

    AVERAGE_DURATION = 80

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "preview_task_id": ("MODEL_TASK_ID",),
            },
            "optional": {
                "enable_pbr": ("BOOLEAN", {"default": True}),
                "texture_prompt": ("STRING", {"multiline": True}),
                "ai_model": model_field_to_node_input(
                    IO.COMBO,
                    MeshyTextToModelRefineRequest,
                    "ai_model",
                    enum_type=MeshyAIModel,
                ),
                "moderation": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
    )
    RETURN_NAMES = ("model_file", "preview task_id")
    FUNCTION = "generate_mesh"
    CATEGORY = "api node/3d/Meshy"
    API_NODE = True
    OUTPUT_NODE = True

    def generate_mesh(
        self,
        preview_task_id,
        enable_pbr=None,
        texture_prompt=None,
        ai_model=None,
        moderation=None,
        **kwargs,
    ):
        if not preview_task_id:
            raise RuntimeError("Preview task ID is required")

        response = SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/meshy/openapi/v2/text-to-3d",
                method=HttpMethod.POST,
                request_model=MeshyTextToModelRefineRequest,
                response_model=MeshyTaskResponse,
            ),
            request=MeshyTextToModelRefineRequest(
                mode="refine",
                preview_task_id=preview_task_id,
                enable_pbr=enable_pbr,
                texture_prompt=texture_prompt,
                ai_model=ai_model,
                moderation=moderation,
            ),
            auth_kwargs=kwargs,
        ).execute()
        return poll_until_finished(kwargs, response)


class MeshyImageToModelNode:
    """
    Generates 3D models synchronously based on a single image using Meshy's API.
    """

    AVERAGE_DURATION = 80

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "ai_model": model_field_to_node_input(
                    IO.COMBO,
                    MeshyImageToModelRequest,
                    "ai_model",
                    enum_type=MeshyAIModel,
                ),
                "topology": model_field_to_node_input(
                    IO.COMBO,
                    MeshyImageToModelRequest,
                    "topology",
                    enum_type=MeshyTopology,
                ),
                "target_polycount": (
                    "INT",
                    {"min": 100, "max": 300000, "default": 30000},
                ),
                "should_remesh": ("BOOLEAN", {"default": True}),
                "symmetry_mode": model_field_to_node_input(
                    IO.COMBO,
                    MeshyImageToModelRequest,
                    "symmetry_mode",
                    enum_type=MeshySymmetryMode,
                ),
                "should_remesh": ("BOOLEAN", {"default": True}),
                "should_texture": ("BOOLEAN", {"default": True}),
                "enable_pbr": ("BOOLEAN", {"default": True}),
                "texture_prompt": ("STRING", {"multiline": True}),
                "moderation": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
    )
    RETURN_NAMES = ("model_file", "model task_id")
    FUNCTION = "generate_mesh"
    CATEGORY = "api node/3d/Meshy"
    API_NODE = True
    OUTPUT_NODE = True

    def generate_mesh(
        self,
        image,
        ai_model=None,
        topology=None,
        target_polycount=None,
        should_remesh=None,
        symmetry_mode=None,
        should_texture=None,
        enable_pbr=None,
        texture_prompt=None,
        moderation=None,
        **kwargs,
    ):
        if not image:
            raise RuntimeError("Image is required")
        response = SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/meshy/openapi/v2/image-to-3d",
                method=HttpMethod.POST,
                request_model=MeshyImageToModelRequest,
                response_model=MeshyTaskResponse,
            ),
            request=MeshyImageToModelRequest(
                mode="image",
                image_url=image,
                ai_model=ai_model,
                topology=topology,
                target_polycount=target_polycount,
                should_remesh=should_remesh,
                symmetry_mode=symmetry_mode,
                should_texture=should_texture,
                enable_pbr=enable_pbr,
                texture_prompt=texture_prompt,
                moderation=moderation,
            ),
            auth_kwargs=kwargs,
        ).execute()
        return poll_until_finished(kwargs, response)


# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "MeshyTextToModelPreviewNode": MeshyTextToModelPreviewNode,
    "MeshyTextToModelRefineNode": MeshyTextToModelRefineNode,
    "MeshyImageToModelNode": MeshyImageToModelNode,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "MeshyTextToModelPreviewNode": "Meshy: Text to Model (Preview)",
    "MeshyTextToModelRefineNode": "Meshy: Text to Model (Refine)",
    "MeshyImageToModelNode": "Meshy: Image to Model",
}
