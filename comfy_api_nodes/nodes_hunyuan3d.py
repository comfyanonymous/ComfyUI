import os

from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api_nodes.apis.hunyuan3d import (
    Hunyuan3DViewImage,
    InputGenerateType,
    ResultFile3D,
    To3DProTaskCreateResponse,
    To3DProTaskQueryRequest,
    To3DProTaskRequest,
    To3DProTaskResultResponse,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_to_bytesio,
    downscale_image_tensor_by_max_side,
    poll_op,
    sync_op,
    upload_image_to_comfyapi,
    validate_image_dimensions,
    validate_string,
)
from folder_paths import get_output_directory


def get_glb_obj_from_response(response_objs: list[ResultFile3D]) -> ResultFile3D:
    for i in response_objs:
        if i.Type.lower() == "glb":
            return i
    raise ValueError("No GLB file found in response. Please report this to the developers.")


class TencentTextToModelNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TencentTextToModelNode",
            display_name="Hunyuan3D: Text to Model (Pro)",
            category="api node/3d/Tencent",
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
                IO.String.Output(display_name="model_file"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            is_output_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["generate_type", "generate_type.pbr", "face_count"]),
                expr="""
                (
                  $base := widgets.generate_type = "normal" ? 25 : widgets.generate_type = "lowpoly" ? 30 : 15;
                  $pbr := $lookup(widgets, "generate_type.pbr") ? 10 : 0;
                  $face := widgets.face_count != 500000 ? 10 : 0;
                  {"type":"usd","usd": ($base + $pbr + $face) * 0.02}
                )
                """,
            ),
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
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/tencent/hunyuan/3d-pro", method="POST"),
            response_model=To3DProTaskCreateResponse,
            data=To3DProTaskRequest(
                Model=model,
                Prompt=prompt,
                FaceCount=face_count,
                GenerateType=generate_type["generate_type"],
                EnablePBR=generate_type.get("pbr", None),
                PolygonType=generate_type.get("polygon_type", None),
            ),
        )
        if response.Error:
            raise ValueError(f"Task creation failed with code {response.Error.Code}: {response.Error.Message}")
        result = await poll_op(
            cls,
            ApiEndpoint(path="/proxy/tencent/hunyuan/3d-pro/query", method="POST"),
            data=To3DProTaskQueryRequest(JobId=response.JobId),
            response_model=To3DProTaskResultResponse,
            status_extractor=lambda r: r.Status,
        )
        model_file = f"hunyuan_model_{response.JobId}.glb"
        await download_url_to_bytesio(
            get_glb_obj_from_response(result.ResultFile3Ds).Url,
            os.path.join(get_output_directory(), model_file),
        )
        return IO.NodeOutput(model_file)


class TencentImageToModelNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TencentImageToModelNode",
            display_name="Hunyuan3D: Image(s) to Model (Pro)",
            category="api node/3d/Tencent",
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
                IO.String.Output(display_name="model_file"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            is_output_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(
                    widgets=["generate_type", "generate_type.pbr", "face_count"],
                    inputs=["image_left", "image_right", "image_back"],
                ),
                expr="""
                (
                  $base := widgets.generate_type = "normal" ? 25 : widgets.generate_type = "lowpoly" ? 30 : 15;
                  $multiview := (
                    inputs.image_left.connected or inputs.image_right.connected or inputs.image_back.connected
                  ) ? 10 : 0;
                  $pbr := $lookup(widgets, "generate_type.pbr") ? 10 : 0;
                  $face := widgets.face_count != 500000 ? 10 : 0;
                  {"type":"usd","usd": ($base + $multiview + $pbr + $face) * 0.02}
                )
                """,
            ),
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
            multiview_images.append(
                Hunyuan3DViewImage(
                    ViewType=k,
                    ViewImageUrl=await upload_image_to_comfyapi(
                        cls,
                        downscale_image_tensor_by_max_side(v, max_side=4900),
                        mime_type="image/webp",
                        total_pixels=24_010_000,
                    ),
                )
            )
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/tencent/hunyuan/3d-pro", method="POST"),
            response_model=To3DProTaskCreateResponse,
            data=To3DProTaskRequest(
                Model=model,
                FaceCount=face_count,
                GenerateType=generate_type["generate_type"],
                ImageUrl=await upload_image_to_comfyapi(
                    cls,
                    downscale_image_tensor_by_max_side(image, max_side=4900),
                    mime_type="image/webp",
                    total_pixels=24_010_000,
                ),
                MultiViewImages=multiview_images if multiview_images else None,
                EnablePBR=generate_type.get("pbr", None),
                PolygonType=generate_type.get("polygon_type", None),
            ),
        )
        if response.Error:
            raise ValueError(f"Task creation failed with code {response.Error.Code}: {response.Error.Message}")
        result = await poll_op(
            cls,
            ApiEndpoint(path="/proxy/tencent/hunyuan/3d-pro/query", method="POST"),
            data=To3DProTaskQueryRequest(JobId=response.JobId),
            response_model=To3DProTaskResultResponse,
            status_extractor=lambda r: r.Status,
        )
        model_file = f"hunyuan_model_{response.JobId}.glb"
        await download_url_to_bytesio(
            get_glb_obj_from_response(result.ResultFile3Ds).Url,
            os.path.join(get_output_directory(), model_file),
        )
        return IO.NodeOutput(model_file)


class TencentHunyuan3DExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            TencentTextToModelNode,
            TencentImageToModelNode,
        ]


async def comfy_entrypoint() -> TencentHunyuan3DExtension:
    return TencentHunyuan3DExtension()
