"""
ComfyUI X Rodin3D(Deemos) API Nodes

Rodin API docs: https://developer.hyper3d.ai/

"""

from inspect import cleandoc
import folder_paths as comfy_paths
import os
import logging
import math
from io import BytesIO
from typing_extensions import override
from PIL import Image
from comfy_api_nodes.util import (
    download_url_to_bytesio,
    download_url_to_file_3d,
)
from comfy_api_nodes.util.client import fal_run
from comfy_api_nodes.util.upload_helpers import upload_file_to_fal

FAL_RODIN_V2 = "fal-ai/hyper3d/rodin/v2"
from comfy_api.latest import ComfyExtension, IO, Types


COMMON_PARAMETERS = [
    IO.Int.Input(
        "Seed",
        default=0,
        min=0,
        max=65535,
        display_mode=IO.NumberDisplay.number,
        optional=True,
    ),
    IO.Combo.Input("Material_Type", options=["PBR", "Shaded"], default="PBR", optional=True),
    IO.Combo.Input(
        "Polygon_count",
        options=["4K-Quad", "8K-Quad", "18K-Quad", "50K-Quad", "200K-Triangle"],
        default="18K-Quad",
        optional=True,
    ),
]


def get_quality_mode(poly_count):
    polycount = poly_count.split("-")
    poly = polycount[1]
    count = polycount[0]
    if poly == "Triangle":
        mesh_mode = "Raw"
    elif poly == "Quad":
        mesh_mode = "Quad"
    else:
        mesh_mode = "Quad"

    if count == "4K":
        quality_override = 4000
    elif count == "8K":
        quality_override = 8000
    elif count == "18K":
        quality_override = 18000
    elif count == "50K":
        quality_override = 50000
    elif count == "2K":
        quality_override = 2000
    elif count == "20K":
        quality_override = 20000
    elif count == "150K":
        quality_override = 150000
    elif count == "500K":
        quality_override = 500000
    else:
        quality_override = 18000

    return mesh_mode, quality_override


def tensor_to_filelike(tensor, max_pixels: int = 2048*2048):
    """
    Converts a PyTorch tensor to a file-like object.

    Args:
    - tensor (torch.Tensor): A tensor representing an image of shape (H, W, C)
      where C is the number of channels (3 for RGB), H is height, and W is width.

    Returns:
    - io.BytesIO: A file-like object containing the image data.
    """
    array = tensor.cpu().numpy()
    array = (array * 255).astype('uint8')
    image = Image.fromarray(array, 'RGB')

    original_width, original_height = image.size
    original_pixels = original_width * original_height
    if original_pixels > max_pixels:
        scale = math.sqrt(max_pixels / original_pixels)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
    else:
        new_width, new_height = original_width, original_height

    if new_width != original_width or new_height != original_height:
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')  # PNG is used for lossless compression
    img_byte_arr.seek(0)
    return img_byte_arr


async def create_generate_task(
    cls: type[IO.ComfyNode],
    images=None,
    seed=1,
    material="PBR",
    quality_override=18000,
    tier="Regular",
    mesh_mode="Quad",
    ta_pose: bool = False,
):
    if images is None:
        raise Exception("Rodin 3D generate requires at least 1 image.")
    if len(images) > 5:
        raise Exception("Rodin 3D generate requires up to 5 image.")

    image_urls = []
    for image in images:
        if image is None:
            continue
        if isinstance(image, str):
            with open(image, "rb") as f:
                bio = BytesIO(f.read())
        else:
            bio = tensor_to_filelike(image)
        url = await upload_file_to_fal(bio, "image/png")
        image_urls.append(url)

    result = await fal_run(cls, FAL_RODIN_V2, {
        "image_urls": image_urls,
        "seed": seed,
        "tier": tier,
        "material": material,
        "quality_override": quality_override,
        "mesh_mode": mesh_mode,
        "TAPose": ta_pose,
    })

    # TODO: verify fal.ai field names
    if "error" in result:
        error_message = f"Rodin3D Create 3D generate Task Failed. Message: {result.get('message', '')}, error: {result['error']}"
        logging.error(error_message)
        raise Exception(error_message)

    logging.info("[ Rodin3D API - Submit Jobs ] Submit Generate Task Success!")
    task_uuid = result.get("uuid", result.get("id", "rodin_task"))
    logging.info("[ Rodin3D API - Submit Jobs ] UUID: %s", task_uuid)
    return task_uuid, result


async def download_files(result: dict, task_uuid: str) -> tuple[str | None, Types.File3D | None]:
    """Download files from the fal_run result."""
    result_folder_name = f"Rodin3D_{task_uuid}"
    save_path = os.path.join(comfy_paths.get_output_directory(), result_folder_name)
    os.makedirs(save_path, exist_ok=True)
    model_file_path = None
    file_3d = None

    # TODO: verify fal.ai field names
    download_list = result.get("list", result.get("downloads", []))
    for i in download_list:
        name = i.get("name", i.get("filename", ""))
        url = i.get("url", "")
        file_path = os.path.join(save_path, name)
        if name.lower().endswith(".glb"):
            model_file_path = os.path.join(result_folder_name, name)
            file_3d = await download_url_to_file_3d(url, "glb")
            # Save to disk for backward compatibility
            with open(file_path, "wb") as f:
                f.write(file_3d.get_bytes())
        else:
            await download_url_to_bytesio(url, file_path)

    return model_file_path, file_3d


class Rodin3D_Regular(IO.ComfyNode):
    """Generate 3D Assets using Rodin API"""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="Rodin3D_Regular",
            display_name="Rodin 3D Generate - Regular Generate",
            category="api node/3d/Rodin",
            description=cleandoc(cls.__doc__ or ""),
            inputs=[
                IO.Image.Input("Images"),
                *COMMON_PARAMETERS,
            ],
            outputs=[
                IO.String.Output(display_name="3D Model Path"),  # for backward compatibility only
                IO.File3DGLB.Output(display_name="GLB"),
            ],
            hidden=[
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        Images,
        Seed,
        Material_Type,
        Polygon_count,
    ) -> IO.NodeOutput:
        tier = "Regular"
        num_images = Images.shape[0]
        m_images = []
        for i in range(num_images):
            m_images.append(Images[i])
        mesh_mode, quality_override = get_quality_mode(Polygon_count)
        task_uuid, result = await create_generate_task(
            cls,
            images=m_images,
            seed=Seed,
            material=Material_Type,
            quality_override=quality_override,
            tier=tier,
            mesh_mode=mesh_mode,
        )
        model_path, file_3d = await download_files(result, task_uuid)

        return IO.NodeOutput(model_path, file_3d)


class Rodin3D_Detail(IO.ComfyNode):
    """Generate 3D Assets using Rodin API"""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="Rodin3D_Detail",
            display_name="Rodin 3D Generate - Detail Generate",
            category="api node/3d/Rodin",
            description=cleandoc(cls.__doc__ or ""),
            inputs=[
                IO.Image.Input("Images"),
                *COMMON_PARAMETERS,
            ],
            outputs=[
                IO.String.Output(display_name="3D Model Path"),  # for backward compatibility only
                IO.File3DGLB.Output(display_name="GLB"),
            ],
            hidden=[
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        Images,
        Seed,
        Material_Type,
        Polygon_count,
    ) -> IO.NodeOutput:
        tier = "Detail"
        num_images = Images.shape[0]
        m_images = []
        for i in range(num_images):
            m_images.append(Images[i])
        mesh_mode, quality_override = get_quality_mode(Polygon_count)
        task_uuid, result = await create_generate_task(
            cls,
            images=m_images,
            seed=Seed,
            material=Material_Type,
            quality_override=quality_override,
            tier=tier,
            mesh_mode=mesh_mode,
        )
        model_path, file_3d = await download_files(result, task_uuid)

        return IO.NodeOutput(model_path, file_3d)


class Rodin3D_Smooth(IO.ComfyNode):
    """Generate 3D Assets using Rodin API"""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="Rodin3D_Smooth",
            display_name="Rodin 3D Generate - Smooth Generate",
            category="api node/3d/Rodin",
            description=cleandoc(cls.__doc__ or ""),
            inputs=[
                IO.Image.Input("Images"),
                *COMMON_PARAMETERS,
            ],
            outputs=[
                IO.String.Output(display_name="3D Model Path"),  # for backward compatibility only
                IO.File3DGLB.Output(display_name="GLB"),
            ],
            hidden=[
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        Images,
        Seed,
        Material_Type,
        Polygon_count,
    ) -> IO.NodeOutput:
        num_images = Images.shape[0]
        m_images = []
        for i in range(num_images):
            m_images.append(Images[i])
        mesh_mode, quality_override = get_quality_mode(Polygon_count)
        task_uuid, result = await create_generate_task(
            cls,
            images=m_images,
            seed=Seed,
            material=Material_Type,
            quality_override=quality_override,
            tier="Smooth",
            mesh_mode=mesh_mode,
        )
        model_path, file_3d = await download_files(result, task_uuid)

        return IO.NodeOutput(model_path, file_3d)


class Rodin3D_Sketch(IO.ComfyNode):
    """Generate 3D Assets using Rodin API"""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="Rodin3D_Sketch",
            display_name="Rodin 3D Generate - Sketch Generate",
            category="api node/3d/Rodin",
            description=cleandoc(cls.__doc__ or ""),
            inputs=[
                IO.Image.Input("Images"),
                IO.Int.Input(
                    "Seed",
                    default=0,
                    min=0,
                    max=65535,
                    display_mode=IO.NumberDisplay.number,
                    optional=True,
                ),
            ],
            outputs=[
                IO.String.Output(display_name="3D Model Path"),  # for backward compatibility only
                IO.File3DGLB.Output(display_name="GLB"),
            ],
            hidden=[
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        Images,
        Seed,
    ) -> IO.NodeOutput:
        num_images = Images.shape[0]
        m_images = []
        for i in range(num_images):
            m_images.append(Images[i])
        task_uuid, result = await create_generate_task(
            cls,
            images=m_images,
            seed=Seed,
            material="PBR",
            quality_override=18000,
            tier="Sketch",
            mesh_mode="Quad",
        )
        model_path, file_3d = await download_files(result, task_uuid)

        return IO.NodeOutput(model_path, file_3d)


class Rodin3D_Gen2(IO.ComfyNode):
    """Generate 3D Assets using Rodin API"""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="Rodin3D_Gen2",
            display_name="Rodin 3D Generate - Gen-2 Generate",
            category="api node/3d/Rodin",
            description=cleandoc(cls.__doc__ or ""),
            inputs=[
                IO.Image.Input("Images"),
                IO.Int.Input(
                    "Seed",
                    default=0,
                    min=0,
                    max=65535,
                    display_mode=IO.NumberDisplay.number,
                    optional=True,
                ),
                IO.Combo.Input("Material_Type", options=["PBR", "Shaded"], default="PBR", optional=True),
                IO.Combo.Input(
                    "Polygon_count",
                    options=["4K-Quad", "8K-Quad", "18K-Quad", "50K-Quad", "2K-Triangle", "20K-Triangle", "150K-Triangle", "500K-Triangle"],
                    default="500K-Triangle",
                    optional=True,
                ),
                IO.Boolean.Input("TAPose", default=False, advanced=True),
            ],
            outputs=[
                IO.String.Output(display_name="3D Model Path"),  # for backward compatibility only
                IO.File3DGLB.Output(display_name="GLB"),
            ],
            hidden=[
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        Images,
        Seed,
        Material_Type,
        Polygon_count,
        TAPose,
    ) -> IO.NodeOutput:
        tier = "Gen-2"
        num_images = Images.shape[0]
        m_images = []
        for i in range(num_images):
            m_images.append(Images[i])
        mesh_mode, quality_override = get_quality_mode(Polygon_count)
        task_uuid, result = await create_generate_task(
            cls,
            images=m_images,
            seed=Seed,
            material=Material_Type,
            quality_override=quality_override,
            tier=tier,
            mesh_mode=mesh_mode,
            ta_pose=TAPose,
        )
        model_path, file_3d = await download_files(result, task_uuid)

        return IO.NodeOutput(model_path, file_3d)


class Rodin3DExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            Rodin3D_Regular,
            Rodin3D_Detail,
            Rodin3D_Smooth,
            Rodin3D_Sketch,
            Rodin3D_Gen2,
        ]


async def comfy_entrypoint() -> Rodin3DExtension:
    return Rodin3DExtension()
