import os
from typing import Optional

import torch
from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension
from comfy_api_nodes.apis.tripo_api import (
    TripoAnimateRetargetRequest,
    TripoAnimateRigRequest,
    TripoConvertModelRequest,
    TripoFileEmptyReference,
    TripoFileReference,
    TripoImageToModelRequest,
    TripoModelVersion,
    TripoMultiviewToModelRequest,
    TripoOrientation,
    TripoRefineModelRequest,
    TripoStyle,
    TripoTaskResponse,
    TripoTaskStatus,
    TripoTaskType,
    TripoTextToModelRequest,
    TripoTextureModelRequest,
    TripoUrlReference,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_as_bytesio,
    poll_op,
    sync_op,
    upload_images_to_comfyapi,
)
from folder_paths import get_output_directory


def get_model_url_from_response(response: TripoTaskResponse) -> str:
    if response.data is not None:
        for key in ["pbr_model", "model", "base_model"]:
            if getattr(response.data.output, key, None) is not None:
                return getattr(response.data.output, key)
    raise RuntimeError(f"Failed to get model url from response: {response}")


async def poll_until_finished(
    node_cls: type[IO.ComfyNode],
    response: TripoTaskResponse,
    average_duration: Optional[int] = None,
) -> IO.NodeOutput:
    """Polls the Tripo API endpoint until the task reaches a terminal state, then returns the response."""
    if response.code != 0:
        raise RuntimeError(f"Failed to generate mesh: {response.error}")
    task_id = response.data.task_id
    response_poll = await poll_op(
        node_cls,
        poll_endpoint=ApiEndpoint(path=f"/proxy/tripo/v2/openapi/task/{task_id}"),
        response_model=TripoTaskResponse,
        completed_statuses=[TripoTaskStatus.SUCCESS],
        failed_statuses=[
            TripoTaskStatus.FAILED,
            TripoTaskStatus.CANCELLED,
            TripoTaskStatus.UNKNOWN,
            TripoTaskStatus.BANNED,
            TripoTaskStatus.EXPIRED,
        ],
        status_extractor=lambda x: x.data.status,
        progress_extractor=lambda x: x.data.progress,
        estimated_duration=average_duration,
    )
    if response_poll.data.status == TripoTaskStatus.SUCCESS:
        url = get_model_url_from_response(response_poll)
        bytesio = await download_url_as_bytesio(url)
        # Save the downloaded model file
        model_file = f"tripo_model_{task_id}.glb"
        with open(os.path.join(get_output_directory(), model_file), "wb") as f:
            f.write(bytesio.getvalue())
        return IO.NodeOutput(model_file, task_id)
    raise RuntimeError(f"Failed to generate mesh: {response_poll}")


class TripoTextToModelNode(IO.ComfyNode):
    """
    Generates 3D models synchronously based on a text prompt using Tripo's API.
    """

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TripoTextToModelNode",
            display_name="Tripo: Text to Model",
            category="api node/3d/Tripo",
            inputs=[
                IO.String.Input("prompt", multiline=True),
                IO.String.Input("negative_prompt", multiline=True, optional=True),
                IO.Combo.Input(
                    "model_version", options=TripoModelVersion, default=TripoModelVersion.v2_5_20250123, optional=True
                ),
                IO.Combo.Input("style", options=TripoStyle, default="None", optional=True),
                IO.Boolean.Input("texture", default=True, optional=True),
                IO.Boolean.Input("pbr", default=True, optional=True),
                IO.Int.Input("image_seed", default=42, optional=True),
                IO.Int.Input("model_seed", default=42, optional=True),
                IO.Int.Input("texture_seed", default=42, optional=True),
                IO.Combo.Input("texture_quality", default="standard", options=["standard", "detailed"], optional=True),
                IO.Int.Input("face_limit", default=-1, min=-1, max=500000, optional=True),
                IO.Boolean.Input("quad", default=False, optional=True),
            ],
            outputs=[
                IO.String.Output(display_name="model_file"),
                IO.Custom("MODEL_TASK_ID").Output(display_name="model task_id"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            is_output_node=True,
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        negative_prompt: Optional[str] = None,
        model_version=None,
        style: Optional[str] = None,
        texture: Optional[bool] = None,
        pbr: Optional[bool] = None,
        image_seed: Optional[int] = None,
        model_seed: Optional[int] = None,
        texture_seed: Optional[int] = None,
        texture_quality: Optional[str] = None,
        face_limit: Optional[int] = None,
        quad: Optional[bool] = None,
    ) -> IO.NodeOutput:
        style_enum = None if style == "None" else style
        if not prompt:
            raise RuntimeError("Prompt is required")
        response = await sync_op(
            cls,
            endpoint=ApiEndpoint(path="/proxy/tripo/v2/openapi/task", method="POST"),
            response_model=TripoTaskResponse,
            data=TripoTextToModelRequest(
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
                quad=quad,
            ),
        )
        return await poll_until_finished(cls, response, average_duration=80)


class TripoImageToModelNode(IO.ComfyNode):
    """
    Generates 3D models synchronously based on a single image using Tripo's API.
    """

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TripoImageToModelNode",
            display_name="Tripo: Image to Model",
            category="api node/3d/Tripo",
            inputs=[
                IO.Image.Input("image"),
                IO.Combo.Input(
                    "model_version",
                    options=TripoModelVersion,
                    tooltip="The model version to use for generation",
                    optional=True,
                ),
                IO.Combo.Input("style", options=TripoStyle, default="None", optional=True),
                IO.Boolean.Input("texture", default=True, optional=True),
                IO.Boolean.Input("pbr", default=True, optional=True),
                IO.Int.Input("model_seed", default=42, optional=True),
                IO.Combo.Input(
                    "orientation", options=TripoOrientation, default=TripoOrientation.DEFAULT, optional=True
                ),
                IO.Int.Input("texture_seed", default=42, optional=True),
                IO.Combo.Input("texture_quality", default="standard", options=["standard", "detailed"], optional=True),
                IO.Combo.Input(
                    "texture_alignment", default="original_image", options=["original_image", "geometry"], optional=True
                ),
                IO.Int.Input("face_limit", default=-1, min=-1, max=500000, optional=True),
                IO.Boolean.Input("quad", default=False, optional=True),
            ],
            outputs=[
                IO.String.Output(display_name="model_file"),
                IO.Custom("MODEL_TASK_ID").Output(display_name="model task_id"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            is_output_node=True,
        )

    @classmethod
    async def execute(
        cls,
        image: torch.Tensor,
        model_version: Optional[str] = None,
        style: Optional[str] = None,
        texture: Optional[bool] = None,
        pbr: Optional[bool] = None,
        model_seed: Optional[int] = None,
        orientation=None,
        texture_seed: Optional[int] = None,
        texture_quality: Optional[str] = None,
        texture_alignment: Optional[str] = None,
        face_limit: Optional[int] = None,
        quad: Optional[bool] = None,
    ) -> IO.NodeOutput:
        style_enum = None if style == "None" else style
        if image is None:
            raise RuntimeError("Image is required")
        tripo_file = TripoFileReference(
            root=TripoUrlReference(
                url=(await upload_images_to_comfyapi(cls, image, max_images=1))[0],
                type="jpeg",
            )
        )
        response = await sync_op(
            cls,
            endpoint=ApiEndpoint(path="/proxy/tripo/v2/openapi/task", method="POST"),
            response_model=TripoTaskResponse,
            data=TripoImageToModelRequest(
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
                quad=quad,
            ),
        )
        return await poll_until_finished(cls, response, average_duration=80)


class TripoMultiviewToModelNode(IO.ComfyNode):
    """
    Generates 3D models synchronously based on up to four images (front, left, back, right) using Tripo's API.
    """

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TripoMultiviewToModelNode",
            display_name="Tripo: Multiview to Model",
            category="api node/3d/Tripo",
            inputs=[
                IO.Image.Input("image"),
                IO.Image.Input("image_left", optional=True),
                IO.Image.Input("image_back", optional=True),
                IO.Image.Input("image_right", optional=True),
                IO.Combo.Input(
                    "model_version",
                    options=TripoModelVersion,
                    optional=True,
                    tooltip="The model version to use for generation",
                ),
                IO.Combo.Input(
                    "orientation",
                    options=TripoOrientation,
                    default=TripoOrientation.DEFAULT,
                    optional=True,
                ),
                IO.Boolean.Input("texture", default=True, optional=True),
                IO.Boolean.Input("pbr", default=True, optional=True),
                IO.Int.Input("model_seed", default=42, optional=True),
                IO.Int.Input("texture_seed", default=42, optional=True),
                IO.Combo.Input("texture_quality", default="standard", options=["standard", "detailed"], optional=True),
                IO.Combo.Input(
                    "texture_alignment", default="original_image", options=["original_image", "geometry"], optional=True
                ),
                IO.Int.Input("face_limit", default=-1, min=-1, max=500000, optional=True),
                IO.Boolean.Input("quad", default=False, optional=True),
            ],
            outputs=[
                IO.String.Output(display_name="model_file"),
                IO.Custom("MODEL_TASK_ID").Output(display_name="model task_id"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            is_output_node=True,
        )

    @classmethod
    async def execute(
        cls,
        image: torch.Tensor,
        image_left: Optional[torch.Tensor] = None,
        image_back: Optional[torch.Tensor] = None,
        image_right: Optional[torch.Tensor] = None,
        model_version: Optional[str] = None,
        orientation: Optional[str] = None,
        texture: Optional[bool] = None,
        pbr: Optional[bool] = None,
        model_seed: Optional[int] = None,
        texture_seed: Optional[int] = None,
        texture_quality: Optional[str] = None,
        texture_alignment: Optional[str] = None,
        face_limit: Optional[int] = None,
        quad: Optional[bool] = None,
    ) -> IO.NodeOutput:
        if image is None:
            raise RuntimeError("front image for multiview is required")
        images = []
        image_dict = {"image": image, "image_left": image_left, "image_back": image_back, "image_right": image_right}
        if image_left is None and image_back is None and image_right is None:
            raise RuntimeError("At least one of left, back, or right image must be provided for multiview")
        for image_name in ["image", "image_left", "image_back", "image_right"]:
            image_ = image_dict[image_name]
            if image_ is not None:
                images.append(
                    TripoFileReference(
                        root=TripoUrlReference(
                            url=(await upload_images_to_comfyapi(cls, image_, max_images=1))[0], type="jpeg"
                        )
                    )
                )
            else:
                images.append(TripoFileEmptyReference())
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/tripo/v2/openapi/task", method="POST"),
            response_model=TripoTaskResponse,
            data=TripoMultiviewToModelRequest(
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
        )
        return await poll_until_finished(cls, response, average_duration=80)


class TripoTextureNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TripoTextureNode",
            display_name="Tripo: Texture model",
            category="api node/3d/Tripo",
            inputs=[
                IO.Custom("MODEL_TASK_ID").Input("model_task_id"),
                IO.Boolean.Input("texture", default=True, optional=True),
                IO.Boolean.Input("pbr", default=True, optional=True),
                IO.Int.Input("texture_seed", default=42, optional=True),
                IO.Combo.Input("texture_quality", default="standard", options=["standard", "detailed"], optional=True),
                IO.Combo.Input(
                    "texture_alignment", default="original_image", options=["original_image", "geometry"], optional=True
                ),
            ],
            outputs=[
                IO.String.Output(display_name="model_file"),
                IO.Custom("MODEL_TASK_ID").Output(display_name="model task_id"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            is_output_node=True,
        )

    @classmethod
    async def execute(
        cls,
        model_task_id,
        texture: Optional[bool] = None,
        pbr: Optional[bool] = None,
        texture_seed: Optional[int] = None,
        texture_quality: Optional[str] = None,
        texture_alignment: Optional[str] = None,
    ) -> IO.NodeOutput:
        response = await sync_op(
            cls,
            endpoint=ApiEndpoint(path="/proxy/tripo/v2/openapi/task", method="POST"),
            response_model=TripoTaskResponse,
            data=TripoTextureModelRequest(
                original_model_task_id=model_task_id,
                texture=texture,
                pbr=pbr,
                texture_seed=texture_seed,
                texture_quality=texture_quality,
                texture_alignment=texture_alignment,
            ),
        )
        return await poll_until_finished(cls, response, average_duration=80)


class TripoRefineNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TripoRefineNode",
            display_name="Tripo: Refine Draft model",
            category="api node/3d/Tripo",
            description="Refine a draft model created by v1.4 Tripo models only.",
            inputs=[
                IO.Custom("MODEL_TASK_ID").Input("model_task_id", tooltip="Must be a v1.4 Tripo model"),
            ],
            outputs=[
                IO.String.Output(display_name="model_file"),
                IO.Custom("MODEL_TASK_ID").Output(display_name="model task_id"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            is_output_node=True,
        )

    @classmethod
    async def execute(cls, model_task_id) -> IO.NodeOutput:
        response = await sync_op(
            cls,
            endpoint=ApiEndpoint(path="/proxy/tripo/v2/openapi/task", method="POST"),
            response_model=TripoTaskResponse,
            data=TripoRefineModelRequest(draft_model_task_id=model_task_id),
        )
        return await poll_until_finished(cls, response, average_duration=240)


class TripoRigNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TripoRigNode",
            display_name="Tripo: Rig model",
            category="api node/3d/Tripo",
            inputs=[IO.Custom("MODEL_TASK_ID").Input("original_model_task_id")],
            outputs=[
                IO.String.Output(display_name="model_file"),
                IO.Custom("RIG_TASK_ID").Output(display_name="rig task_id"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            is_output_node=True,
        )

    @classmethod
    async def execute(cls, original_model_task_id) -> IO.NodeOutput:
        response = await sync_op(
            cls,
            endpoint=ApiEndpoint(path="/proxy/tripo/v2/openapi/task", method="POST"),
            response_model=TripoTaskResponse,
            data=TripoAnimateRigRequest(original_model_task_id=original_model_task_id, out_format="glb", spec="tripo"),
        )
        return await poll_until_finished(cls, response, average_duration=180)


class TripoRetargetNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TripoRetargetNode",
            display_name="Tripo: Retarget rigged model",
            category="api node/3d/Tripo",
            inputs=[
                IO.Custom("RIG_TASK_ID").Input("original_model_task_id"),
                IO.Combo.Input(
                    "animation",
                    options=[
                        "preset:idle",
                        "preset:walk",
                        "preset:climb",
                        "preset:jump",
                        "preset:slash",
                        "preset:shoot",
                        "preset:hurt",
                        "preset:fall",
                        "preset:turn",
                    ],
                ),
            ],
            outputs=[
                IO.String.Output(display_name="model_file"),
                IO.Custom("RETARGET_TASK_ID").Output(display_name="retarget task_id"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            is_output_node=True,
        )

    @classmethod
    async def execute(cls, original_model_task_id, animation: str) -> IO.NodeOutput:
        response = await sync_op(
            cls,
            endpoint=ApiEndpoint(path="/proxy/tripo/v2/openapi/task", method="POST"),
            response_model=TripoTaskResponse,
            data=TripoAnimateRetargetRequest(
                original_model_task_id=original_model_task_id,
                animation=animation,
                out_format="glb",
                bake_animation=True,
            ),
        )
        return await poll_until_finished(cls, response, average_duration=30)


class TripoConversionNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TripoConversionNode",
            display_name="Tripo: Convert model",
            category="api node/3d/Tripo",
            inputs=[
                IO.Custom("MODEL_TASK_ID,RIG_TASK_ID,RETARGET_TASK_ID").Input("original_model_task_id"),
                IO.Combo.Input("format", options=["GLTF", "USDZ", "FBX", "OBJ", "STL", "3MF"]),
                IO.Boolean.Input("quad", default=False, optional=True),
                IO.Int.Input(
                    "face_limit",
                    default=-1,
                    min=-1,
                    max=500000,
                    optional=True,
                ),
                IO.Int.Input(
                    "texture_size",
                    default=4096,
                    min=128,
                    max=4096,
                    optional=True,
                ),
                IO.Combo.Input(
                    "texture_format",
                    options=["BMP", "DPX", "HDR", "JPEG", "OPEN_EXR", "PNG", "TARGA", "TIFF", "WEBP"],
                    default="JPEG",
                    optional=True,
                ),
            ],
            outputs=[],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            is_output_node=True,
        )

    @classmethod
    def validate_inputs(cls, input_types):
        # The min and max of input1 and input2 are still validated because
        # we didn't take `input1` or `input2` as arguments
        if input_types["original_model_task_id"] not in ("MODEL_TASK_ID", "RIG_TASK_ID", "RETARGET_TASK_ID"):
            return "original_model_task_id must be MODEL_TASK_ID, RIG_TASK_ID or RETARGET_TASK_ID type"
        return True

    @classmethod
    async def execute(
        cls,
        original_model_task_id,
        format: str,
        quad: bool,
        face_limit: int,
        texture_size: int,
        texture_format: str,
    ) -> IO.NodeOutput:
        if not original_model_task_id:
            raise RuntimeError("original_model_task_id is required")
        response = await sync_op(
            cls,
            endpoint=ApiEndpoint(path="/proxy/tripo/v2/openapi/task", method="POST"),
            response_model=TripoTaskResponse,
            data=TripoConvertModelRequest(
                original_model_task_id=original_model_task_id,
                format=format,
                quad=quad if quad else None,
                face_limit=face_limit if face_limit != -1 else None,
                texture_size=texture_size if texture_size != 4096 else None,
                texture_format=texture_format if texture_format != "JPEG" else None,
            ),
        )
        return await poll_until_finished(cls, response, average_duration=30)


class TripoExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            TripoTextToModelNode,
            TripoImageToModelNode,
            TripoMultiviewToModelNode,
            TripoTextureNode,
            TripoRefineNode,
            TripoRigNode,
            TripoRetargetNode,
            TripoConversionNode,
        ]


async def comfy_entrypoint() -> TripoExtension:
    return TripoExtension()
