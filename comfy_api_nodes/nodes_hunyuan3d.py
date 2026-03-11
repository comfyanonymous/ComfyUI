from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input, Types
from comfy_api_nodes.apis.hunyuan3d import (
    InputGenerateType,
    ResultFile3D,
)
from comfy_api_nodes.util import (
    download_url_to_file_3d,
    downscale_image_tensor_by_max_side,
    upload_3d_model_to_fal,
    validate_image_dimensions,
    validate_string,
)
from comfy_api_nodes.util.client import fal_run
from comfy_api_nodes.util.upload_helpers import upload_image_to_fal

FAL_HUNYUAN3D_V2 = "fal-ai/hunyuan3d/v2"


def get_file_from_response(
    response_objs: list, file_type: str, raise_if_not_found: bool = True
):
    """Extract file of given type from response list (works with dicts or ResultFile3D objects)."""
    for i in response_objs:
        if isinstance(i, dict):
            if i.get("Type", "").lower() == file_type.lower():
                return i
        else:
            if i.Type.lower() == file_type.lower():
                return i
    if raise_if_not_found:
        raise ValueError(f"'{file_type}' file type is not found in the response.")
    return None


def _get_url(file_obj) -> str:
    """Get URL from a file object (dict or ResultFile3D)."""
    if isinstance(file_obj, dict):
        return file_obj["Url"]
    return file_obj.Url


class TencentTextToModelNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TencentTextToModelNode",
            display_name="Hunyuan3D: Text to Model",
            category="api node/3d/Tencent",
            essentials_category="3D",
            inputs=[
                IO.Combo.Input(
                    "model",
                    options=["3.0", "3.1"],
                    tooltip="The LowPoly option is unavailable for the `3.1` model.",
                ),
                IO.String.Input("prompt", multiline=True, default="", tooltip="Supports up to 1024 characters."),
                IO.Int.Input("face_count", default=500000, min=40000, max=1500000),
                IO.DynamicCombo.Input(
                    "generate_type",
                    options=[
                        IO.DynamicCombo.Option("Normal", [IO.Boolean.Input("pbr", default=False)]),
                        IO.DynamicCombo.Option(
                            "LowPoly",
                            [
                                IO.Combo.Input("polygon_type", options=["triangle", "quadrilateral"]),
                                IO.Boolean.Input("pbr", default=False),
                            ],
                        ),
                        IO.DynamicCombo.Option("Geometry", []),
                    ],
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
            outputs=[
                IO.String.Output(display_name="model_file"),  # for backward compatibility only
                IO.File3DGLB.Output(display_name="GLB"),
                IO.File3DOBJ.Output(display_name="OBJ"),
            ],
            hidden=[
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            is_output_node=True,
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        prompt: str,
        face_count: int,
        generate_type: InputGenerateType,
        seed: int,
    ) -> IO.NodeOutput:
        _ = seed
        validate_string(prompt, field_name="prompt", min_length=1, max_length=1024)
        if model == "3.1" and generate_type["generate_type"].lower() == "lowpoly":
            raise ValueError("The LowPoly option is currently unavailable for the 3.1 model.")
        result = await fal_run(cls, FAL_HUNYUAN3D_V2, {
            "Model": model,
            "Prompt": prompt,
            "FaceCount": face_count,
            "GenerateType": generate_type["generate_type"],
            "EnablePBR": generate_type.get("pbr", None),
            "PolygonType": generate_type.get("polygon_type", None),
        })
        # TODO: verify fal.ai field names
        if result.get("Error"):
            raise ValueError(f"Task creation failed with code {result['Error']['Code']}: {result['Error']['Message']}")
        task_id = result.get("JobId", "hunyuan_task")
        file_3ds = result["ResultFile3Ds"]
        return IO.NodeOutput(
            f"{task_id}.glb",
            await download_url_to_file_3d(
                get_file_from_response(file_3ds, "glb")["Url"], "glb", task_id=task_id
            ),
            await download_url_to_file_3d(
                get_file_from_response(file_3ds, "obj")["Url"], "obj", task_id=task_id
            ),
        )


class TencentImageToModelNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TencentImageToModelNode",
            display_name="Hunyuan3D: Image(s) to Model",
            category="api node/3d/Tencent",
            essentials_category="3D",
            inputs=[
                IO.Combo.Input(
                    "model",
                    options=["3.0", "3.1"],
                    tooltip="The LowPoly option is unavailable for the `3.1` model.",
                ),
                IO.Image.Input("image"),
                IO.Image.Input("image_left", optional=True),
                IO.Image.Input("image_right", optional=True),
                IO.Image.Input("image_back", optional=True),
                IO.Int.Input("face_count", default=500000, min=40000, max=1500000),
                IO.DynamicCombo.Input(
                    "generate_type",
                    options=[
                        IO.DynamicCombo.Option("Normal", [IO.Boolean.Input("pbr", default=False)]),
                        IO.DynamicCombo.Option(
                            "LowPoly",
                            [
                                IO.Combo.Input("polygon_type", options=["triangle", "quadrilateral"]),
                                IO.Boolean.Input("pbr", default=False),
                            ],
                        ),
                        IO.DynamicCombo.Option("Geometry", []),
                    ],
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
            outputs=[
                IO.String.Output(display_name="model_file"),  # for backward compatibility only
                IO.File3DGLB.Output(display_name="GLB"),
                IO.File3DOBJ.Output(display_name="OBJ"),
            ],
            hidden=[
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            is_output_node=True,
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        image: Input.Image,
        face_count: int,
        generate_type: InputGenerateType,
        seed: int,
        image_left: Input.Image | None = None,
        image_right: Input.Image | None = None,
        image_back: Input.Image | None = None,
    ) -> IO.NodeOutput:
        _ = seed
        if model == "3.1" and generate_type["generate_type"].lower() == "lowpoly":
            raise ValueError("The LowPoly option is currently unavailable for the 3.1 model.")
        validate_image_dimensions(image, min_width=128, min_height=128)
        multiview_images = []
        for k, v in {
            "left": image_left,
            "right": image_right,
            "back": image_back,
        }.items():
            if v is None:
                continue
            validate_image_dimensions(v, min_width=128, min_height=128)
            view_image_url = await upload_image_to_fal(
                downscale_image_tensor_by_max_side(v, max_side=4900),
            )
            multiview_images.append({
                "ViewType": k,
                "ViewImageUrl": view_image_url,
            })
        image_url = await upload_image_to_fal(
            downscale_image_tensor_by_max_side(image, max_side=4900),
        )
        result = await fal_run(cls, FAL_HUNYUAN3D_V2, {
            "Model": model,
            "FaceCount": face_count,
            "GenerateType": generate_type["generate_type"],
            "ImageUrl": image_url,
            "MultiViewImages": multiview_images if multiview_images else None,
            "EnablePBR": generate_type.get("pbr", None),
            "PolygonType": generate_type.get("polygon_type", None),
        })
        # TODO: verify fal.ai field names
        if result.get("Error"):
            raise ValueError(f"Task creation failed with code {result['Error']['Code']}: {result['Error']['Message']}")
        task_id = result.get("JobId", "hunyuan_task")
        file_3ds = result["ResultFile3Ds"]
        return IO.NodeOutput(
            f"{task_id}.glb",
            await download_url_to_file_3d(
                get_file_from_response(file_3ds, "glb")["Url"], "glb", task_id=task_id
            ),
            await download_url_to_file_3d(
                get_file_from_response(file_3ds, "obj")["Url"], "obj", task_id=task_id
            ),
        )


class TencentModelTo3DUVNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TencentModelTo3DUVNode",
            display_name="Hunyuan3D: Model to UV",
            category="api node/3d/Tencent",
            description="Perform UV unfolding on a 3D model to generate UV texture. "
            "Input model must have less than 30000 faces.",
            inputs=[
                IO.MultiType.Input(
                    "model_3d",
                    types=[IO.File3DGLB, IO.File3DOBJ, IO.File3DFBX, IO.File3DAny],
                    tooltip="Input 3D model (GLB, OBJ, or FBX)",
                ),
                IO.Int.Input(
                    "seed",
                    default=1,
                    min=0,
                    max=2147483647,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed controls whether the node should re-run; "
                    "results are non-deterministic regardless of seed.",
                ),
            ],
            outputs=[
                IO.File3DOBJ.Output(display_name="OBJ"),
                IO.File3DFBX.Output(display_name="FBX"),
            ],
            hidden=[
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    SUPPORTED_FORMATS = {"glb", "obj", "fbx"}

    @classmethod
    async def execute(
        cls,
        model_3d: Types.File3D,
        seed: int,
    ) -> IO.NodeOutput:
        _ = seed
        file_format = model_3d.format.lower()
        if file_format not in cls.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported file format: '{file_format}'. "
                f"Supported formats: {', '.join(sorted(cls.SUPPORTED_FORMATS))}."
            )
        model_url = await upload_3d_model_to_fal(model_3d, file_format)
        result = await fal_run(cls, FAL_HUNYUAN3D_V2, {
            "File": {
                "Type": file_format.upper(),
                "Url": model_url,
            },
        })
        # TODO: verify fal.ai field names
        if result.get("Error"):
            raise ValueError(f"Task creation failed with code {result['Error']['Code']}: {result['Error']['Message']}")
        file_3ds = result["ResultFile3Ds"]
        return IO.NodeOutput(
            await download_url_to_file_3d(get_file_from_response(file_3ds, "obj")["Url"], "obj"),
            await download_url_to_file_3d(get_file_from_response(file_3ds, "fbx")["Url"], "fbx"),
        )


class Tencent3DTextureEditNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Tencent3DTextureEditNode",
            display_name="Hunyuan3D: 3D Texture Edit",
            category="api node/3d/Tencent",
            description="After inputting the 3D model, perform 3D model texture redrawing.",
            inputs=[
                IO.MultiType.Input(
                    "model_3d",
                    types=[IO.File3DFBX, IO.File3DAny],
                    tooltip="3D model in FBX format. Model should have less than 100000 faces.",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Describes texture editing. Supports up to 1024 UTF-8 characters.",
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
            outputs=[
                IO.File3DGLB.Output(display_name="GLB"),
                IO.File3DFBX.Output(display_name="FBX"),
            ],
            hidden=[
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        model_3d: Types.File3D,
        prompt: str,
        seed: int,
    ) -> IO.NodeOutput:
        _ = seed
        file_format = model_3d.format.lower()
        if file_format != "fbx":
            raise ValueError(f"Unsupported file format: '{file_format}'. Only FBX format is supported.")
        validate_string(prompt, field_name="prompt", min_length=1, max_length=1024)
        model_url = await upload_3d_model_to_fal(model_3d, file_format)
        result = await fal_run(cls, FAL_HUNYUAN3D_V2, {
            "File3D": {
                "Type": file_format.upper(),
                "Url": model_url,
            },
            "Prompt": prompt,
            "EnablePBR": True,
        })
        # TODO: verify fal.ai field names
        if result.get("Error"):
            raise ValueError(f"Task creation failed with code {result['Error']['Code']}: {result['Error']['Message']}")
        file_3ds = result["ResultFile3Ds"]
        return IO.NodeOutput(
            await download_url_to_file_3d(get_file_from_response(file_3ds, "glb")["Url"], "glb"),
            await download_url_to_file_3d(get_file_from_response(file_3ds, "fbx")["Url"], "fbx"),
        )


class Tencent3DPartNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Tencent3DPartNode",
            display_name="Hunyuan3D: 3D Part",
            category="api node/3d/Tencent",
            description="Automatically perform component identification and generation based on the model structure.",
            inputs=[
                IO.MultiType.Input(
                    "model_3d",
                    types=[IO.File3DFBX, IO.File3DAny],
                    tooltip="3D model in FBX format. Model should have less than 30000 faces.",
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
            outputs=[
                IO.File3DFBX.Output(display_name="FBX"),
            ],
            hidden=[
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        model_3d: Types.File3D,
        seed: int,
    ) -> IO.NodeOutput:
        _ = seed
        file_format = model_3d.format.lower()
        if file_format != "fbx":
            raise ValueError(f"Unsupported file format: '{file_format}'. Only FBX format is supported.")
        model_url = await upload_3d_model_to_fal(model_3d, file_format)
        result = await fal_run(cls, FAL_HUNYUAN3D_V2, {
            "File": {
                "Type": file_format.upper(),
                "Url": model_url,
            },
        })
        # TODO: verify fal.ai field names
        if result.get("Error"):
            raise ValueError(f"Task creation failed with code {result['Error']['Code']}: {result['Error']['Message']}")
        file_3ds = result["ResultFile3Ds"]
        return IO.NodeOutput(
            await download_url_to_file_3d(get_file_from_response(file_3ds, "fbx")["Url"], "fbx"),
        )


class TencentSmartTopologyNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TencentSmartTopologyNode",
            display_name="Hunyuan3D: Smart Topology",
            category="api node/3d/Tencent",
            description="Perform smart retopology on a 3D model. "
            "Supports GLB/OBJ formats; max 200MB; recommended for high-poly models.",
            inputs=[
                IO.MultiType.Input(
                    "model_3d",
                    types=[IO.File3DGLB, IO.File3DOBJ, IO.File3DAny],
                    tooltip="Input 3D model (GLB or OBJ)",
                ),
                IO.Combo.Input(
                    "polygon_type",
                    options=["triangle", "quadrilateral"],
                    tooltip="Surface composition type.",
                ),
                IO.Combo.Input(
                    "face_level",
                    options=["medium", "high", "low"],
                    tooltip="Polygon reduction level.",
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
            outputs=[
                IO.File3DOBJ.Output(display_name="OBJ"),
            ],
            hidden=[
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    SUPPORTED_FORMATS = {"glb", "obj"}

    @classmethod
    async def execute(
        cls,
        model_3d: Types.File3D,
        polygon_type: str,
        face_level: str,
        seed: int,
    ) -> IO.NodeOutput:
        _ = seed
        file_format = model_3d.format.lower()
        if file_format not in cls.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported file format: '{file_format}'. " f"Supported: {', '.join(sorted(cls.SUPPORTED_FORMATS))}."
            )
        model_url = await upload_3d_model_to_fal(model_3d, file_format)
        result = await fal_run(cls, FAL_HUNYUAN3D_V2, {
            "File3D": {
                "Type": file_format.upper(),
                "Url": model_url,
            },
            "PolygonType": polygon_type,
            "FaceLevel": face_level,
        })
        # TODO: verify fal.ai field names
        if result.get("Error"):
            raise ValueError(f"Task creation failed: [{result['Error']['Code']}] {result['Error']['Message']}")
        file_3ds = result["ResultFile3Ds"]
        return IO.NodeOutput(
            await download_url_to_file_3d(get_file_from_response(file_3ds, "obj")["Url"], "obj"),
        )


class TencentHunyuan3DExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            TencentTextToModelNode,
            TencentImageToModelNode,
            TencentModelTo3DUVNode,
            # Tencent3DTextureEditNode,
            Tencent3DPartNode,
            TencentSmartTopologyNode,
        ]


async def comfy_entrypoint() -> TencentHunyuan3DExtension:
    return TencentHunyuan3DExtension()
