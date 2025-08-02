"""
ComfyUI X Rodin3D(Deemos) API Nodes

Rodin API docs: https://developer.hyper3d.ai/

"""

from __future__ import annotations
from inspect import cleandoc
from comfy.comfy_types.node_typing import IO
import folder_paths as comfy_paths
import aiohttp
import os
import datetime
import asyncio
import io
import logging
import math
from PIL import Image
from comfy_api_nodes.apis.rodin_api import (
    Rodin3DGenerateRequest,
    Rodin3DGenerateResponse,
    Rodin3DCheckStatusRequest,
    Rodin3DCheckStatusResponse,
    Rodin3DDownloadRequest,
    Rodin3DDownloadResponse,
    JobStatus,
)
from comfy_api_nodes.apis.client import (
    ApiEndpoint,
    HttpMethod,
    SynchronousOperation,
    PollingOperation,
)


COMMON_PARAMETERS = {
    "Seed": (
        IO.INT,
        {
            "default":0,
            "min":0,
            "max":65535,
            "display":"number"
        }
    ),
    "Material_Type": (
        IO.COMBO,
        {
            "options": ["PBR", "Shaded"],
            "default": "PBR"
        }
    ),
    "Polygon_count": (
        IO.COMBO,
        {
            "options": ["4K-Quad", "8K-Quad", "18K-Quad", "50K-Quad", "200K-Triangle"],
            "default": "18K-Quad"
        }
    )
}

def create_task_error(response: Rodin3DGenerateResponse):
    """Check if the response has error"""
    return hasattr(response, "error")


class Rodin3DAPI:
    """
    Generate 3D Assets using Rodin API
    """
    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("3D Model Path",)
    CATEGORY = "api node/3d/Rodin"
    DESCRIPTION = cleandoc(__doc__ or "")
    FUNCTION = "api_call"
    API_NODE = True

    def tensor_to_filelike(self, tensor, max_pixels: int = 2048*2048):
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

        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')  # PNG is used for lossless compression
        img_byte_arr.seek(0)
        return img_byte_arr

    def check_rodin_status(self, response: Rodin3DCheckStatusResponse) -> str:
        has_failed = any(job.status == JobStatus.Failed for job in response.jobs)
        all_done = all(job.status == JobStatus.Done for job in response.jobs)
        status_list = [str(job.status) for job in response.jobs]
        logging.info(f"[ Rodin3D API - CheckStatus ] Generate Status: {status_list}")
        if has_failed:
            logging.error(f"[ Rodin3D API - CheckStatus ] Generate Failed: {status_list}, Please try again.")
            raise Exception("[ Rodin3D API ] Generate Failed, Please Try again.")
        elif all_done:
            return "DONE"
        else:
            return "Generating"

    async def create_generate_task(self, images=None, seed=1, material="PBR", quality="medium", tier="Regular", mesh_mode="Quad", **kwargs):
        if images is None:
            raise Exception("Rodin 3D generate requires at least 1 image.")
        if len(images) >= 5:
            raise Exception("Rodin 3D generate requires up to 5 image.")

        path = "/proxy/rodin/api/v2/rodin"
        operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path=path,
                method=HttpMethod.POST,
                request_model=Rodin3DGenerateRequest,
                response_model=Rodin3DGenerateResponse,
            ),
            request=Rodin3DGenerateRequest(
                seed=seed,
                tier=tier,
                material=material,
                quality=quality,
                mesh_mode=mesh_mode
            ),
            files=[
                (
                    "images",
                    open(image, "rb") if isinstance(image, str) else self.tensor_to_filelike(image)
                )
                for image in images if image is not None
            ],
            content_type = "multipart/form-data",
            auth_kwargs=kwargs,
        )

        response = await operation.execute()

        if create_task_error(response):
            error_message = f"Rodin3D Create 3D generate Task Failed. Message: {response.message}, error: {response.error}"
            logging.error(error_message)
            raise Exception(error_message)

        logging.info("[ Rodin3D API - Submit Jobs ] Submit Generate Task Success!")
        subscription_key = response.jobs.subscription_key
        task_uuid = response.uuid
        logging.info(f"[ Rodin3D API - Submit Jobs ] UUID: {task_uuid}")
        return task_uuid, subscription_key

    async def poll_for_task_status(self, subscription_key, **kwargs) -> Rodin3DCheckStatusResponse:

        path = "/proxy/rodin/api/v2/status"

        poll_operation = PollingOperation(
            poll_endpoint=ApiEndpoint(
                path = path,
                method=HttpMethod.POST,
                request_model=Rodin3DCheckStatusRequest,
                response_model=Rodin3DCheckStatusResponse,
            ),
            request=Rodin3DCheckStatusRequest(
                subscription_key = subscription_key
            ),
            completed_statuses=["DONE"],
            failed_statuses=["FAILED"],
            status_extractor=self.check_rodin_status,
            poll_interval=3.0,
            auth_kwargs=kwargs,
        )

        logging.info("[ Rodin3D API - CheckStatus ] Generate Start!")

        return await poll_operation.execute()

    async def get_rodin_download_list(self, uuid, **kwargs) -> Rodin3DDownloadResponse:
        logging.info("[ Rodin3D API - Downloading ] Generate Successfully!")

        path = "/proxy/rodin/api/v2/download"
        operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path=path,
                method=HttpMethod.POST,
                request_model=Rodin3DDownloadRequest,
                response_model=Rodin3DDownloadResponse,
            ),
            request=Rodin3DDownloadRequest(
                task_uuid=uuid
            ),
            auth_kwargs=kwargs
        )

        return await operation.execute()

    def get_quality_mode(self, poly_count):
        if poly_count == "200K-Triangle":
            mesh_mode = "Raw"
            quality = "medium"
        else:
            mesh_mode = "Quad"
            if poly_count == "4K-Quad":
                quality = "extra-low"
            elif poly_count == "8K-Quad":
                quality = "low"
            elif poly_count == "18K-Quad":
                quality = "medium"
            elif poly_count == "50K-Quad":
                quality = "high"
            else:
                quality = "medium"

        return mesh_mode, quality

    async def download_files(self, url_list):
        save_path = os.path.join(comfy_paths.get_output_directory(), "Rodin3D", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(save_path, exist_ok=True)
        model_file_path = None
        async with aiohttp.ClientSession() as session:
            for i in url_list.list:
                url = i.url
                file_name = i.name
                file_path = os.path.join(save_path, file_name)
                if file_path.endswith(".glb"):
                    model_file_path = file_path
                logging.info(f"[ Rodin3D API - download_files ] Downloading file: {file_path}")
                max_retries = 5
                for attempt in range(max_retries):
                    try:
                        async with session.get(url) as resp:
                            resp.raise_for_status()
                            with open(file_path, "wb") as f:
                                async for chunk in resp.content.iter_chunked(32 * 1024):
                                    f.write(chunk)
                        break
                    except Exception as e:
                        logging.info(f"[ Rodin3D API - download_files ] Error downloading {file_path}:{e}")
                        if attempt < max_retries - 1:
                            logging.info("Retrying...")
                            await asyncio.sleep(2)
                        else:
                            logging.info(
                                "[ Rodin3D API - download_files ] Failed to download %s after %s attempts.",
                                file_path,
                                max_retries,
                            )

        return model_file_path


class Rodin3D_Regular(Rodin3DAPI):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Images":
                (
                    IO.IMAGE,
                    {
                        "forceInput":True,
                    }
                )
            },
            "optional": {
                **COMMON_PARAMETERS
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
            },
        }

    async def api_call(
        self,
        Images,
        Seed,
        Material_Type,
        Polygon_count,
        **kwargs
    ):
        tier = "Regular"
        num_images = Images.shape[0]
        m_images = []
        for i in range(num_images):
            m_images.append(Images[i])
        mesh_mode, quality = self.get_quality_mode(Polygon_count)
        task_uuid, subscription_key = await self.create_generate_task(images=m_images, seed=Seed, material=Material_Type,
                                                                quality=quality, tier=tier, mesh_mode=mesh_mode,
                                                                **kwargs)
        await self.poll_for_task_status(subscription_key, **kwargs)
        download_list = await self.get_rodin_download_list(task_uuid, **kwargs)
        model = await self.download_files(download_list)

        return (model,)


class Rodin3D_Detail(Rodin3DAPI):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Images":
                (
                    IO.IMAGE,
                    {
                        "forceInput":True,
                    }
                )
            },
            "optional": {
                **COMMON_PARAMETERS
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
            },
        }

    async def api_call(
        self,
        Images,
        Seed,
        Material_Type,
        Polygon_count,
        **kwargs
    ):
        tier = "Detail"
        num_images = Images.shape[0]
        m_images = []
        for i in range(num_images):
            m_images.append(Images[i])
        mesh_mode, quality = self.get_quality_mode(Polygon_count)
        task_uuid, subscription_key = await self.create_generate_task(images=m_images, seed=Seed, material=Material_Type,
                                                                quality=quality, tier=tier, mesh_mode=mesh_mode,
                                                                **kwargs)
        await self.poll_for_task_status(subscription_key, **kwargs)
        download_list = await self.get_rodin_download_list(task_uuid, **kwargs)
        model = await self.download_files(download_list)

        return (model,)


class Rodin3D_Smooth(Rodin3DAPI):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Images":
                (
                    IO.IMAGE,
                    {
                        "forceInput":True,
                    }
                )
            },
            "optional": {
                **COMMON_PARAMETERS
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
            },
        }

    async def api_call(
        self,
        Images,
        Seed,
        Material_Type,
        Polygon_count,
        **kwargs
    ):
        tier = "Smooth"
        num_images = Images.shape[0]
        m_images = []
        for i in range(num_images):
            m_images.append(Images[i])
        mesh_mode, quality = self.get_quality_mode(Polygon_count)
        task_uuid, subscription_key = await self.create_generate_task(images=m_images, seed=Seed, material=Material_Type,
                                                                quality=quality, tier=tier, mesh_mode=mesh_mode,
                                                                **kwargs)
        await self.poll_for_task_status(subscription_key, **kwargs)
        download_list = await self.get_rodin_download_list(task_uuid, **kwargs)
        model = await self.download_files(download_list)

        return (model,)


class Rodin3D_Sketch(Rodin3DAPI):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Images":
                (
                    IO.IMAGE,
                    {
                        "forceInput":True,
                    }
                )
            },
            "optional": {
                "Seed":
                (
                    IO.INT,
                    {
                        "default":0,
                        "min":0,
                        "max":65535,
                        "display":"number"
                    }
                )
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
            },
        }

    async def api_call(
        self,
        Images,
        Seed,
        **kwargs
    ):
        tier = "Sketch"
        num_images = Images.shape[0]
        m_images = []
        for i in range(num_images):
            m_images.append(Images[i])
        material_type = "PBR"
        quality = "medium"
        mesh_mode = "Quad"
        task_uuid, subscription_key = await self.create_generate_task(
            images=m_images, seed=Seed, material=material_type, quality=quality, tier=tier, mesh_mode=mesh_mode, **kwargs
        )
        await self.poll_for_task_status(subscription_key, **kwargs)
        download_list = await self.get_rodin_download_list(task_uuid, **kwargs)
        model = await self.download_files(download_list)

        return (model,)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Rodin3D_Regular": Rodin3D_Regular,
    "Rodin3D_Detail": Rodin3D_Detail,
    "Rodin3D_Smooth": Rodin3D_Smooth,
    "Rodin3D_Sketch": Rodin3D_Sketch,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Rodin3D_Regular": "Rodin 3D Generate - Regular Generate",
    "Rodin3D_Detail": "Rodin 3D Generate - Detail Generate",
    "Rodin3D_Smooth": "Rodin 3D Generate - Smooth Generate",
    "Rodin3D_Sketch": "Rodin 3D Generate - Sketch Generate",
}
