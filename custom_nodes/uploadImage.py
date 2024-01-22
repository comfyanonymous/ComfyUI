import hashlib
import os
import sys
from io import BytesIO

import numpy as np
import requests
import torch
from PIL import Image, ImageOps, ImageSequence
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

from comfy import clip_vision, controlnet, model_management, sample, samplers, sd, utils
import folder_paths

class LoadImageCloud:
    """
    LoadImageCloud is a class responsible for loading images from Azure Blob Storage.
    It also processes these images and their masks for further use.

    Class variables:
    - account_url: URL of the Azure Blob Storage account.
    - container_name: Name of the blob container.
    - default_credential: Credentials for accessing the Azure Blob Storage.
    - blob_service_client: Client for the Azure Blob Service.
    """

    # Class variables
    account_url = "https://comfyimgstore.blob.core.windows.net"
    container_name = "comfyblob"
    default_credential = DefaultAzureCredential()
    blob_service_client = BlobServiceClient(account_url, credential=default_credential)

    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"

    @classmethod
    def INPUT_TYPES(cls):
        """
        Returns the expected input types for the LoadImageCloud node.

        :return: Dictionary specifying required input type.
        """
        file_names = cls._list_blob_names()
        return {"required": {"file_name": (file_names,)}}

    @classmethod
    def _list_blob_names(cls):
        """
        Lists the names of the blobs in the specified Azure container.

        :return: List of blob names.
        """
        blob_service_client = BlobServiceClient(cls.account_url, credential=cls.default_credential)
        container_client = blob_service_client.get_container_client(cls.container_name)
        return [blob.name for blob in container_client.list_blobs()]

    def load_image(self, file_name):
        """
        Loads and processes an image and its mask from Azure Blob Storage.

        :param file_name: Name of the file to be loaded.
        :return: Tuple of processed image and mask.
        """
        # Get the blob client
        blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=file_name)
        blob_url = blob_client.url  # Get the blob URL

        # Download the image from the blob URL
        response = requests.get(blob_url)
        img = Image.open(BytesIO(response.content))

        # Process image and generate masks
        output_images, output_masks = self._process_image(img)

        # Combine images and masks if there are multiple frames
        output_image, output_mask = self._combine_outputs(output_images, output_masks)

        return output_image, output_mask

    def _process_image(self, img):
        """
        Processes the given image by handling orientation, converting to RGB, normalizing,
        and preparing masks.

        :param img: PIL Image object to be processed.
        :return: Tuple of lists containing processed images and masks.
        """

        output_images = []
        output_masks = []
        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]

            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        return output_images, output_masks

    def _combine_outputs(self, output_images, output_masks):
        """
        Combines multiple images and masks into single tensors.

        :param output_images: List of image tensors.
        :param output_masks: List of mask tensors.
        :return: Combined image and mask tensors.
        """
        if len(output_images) > 1:
            return torch.cat(output_images, dim=0), torch.cat(output_masks, dim=0)
        return output_images[0], output_masks[0]


# Node and display name mappings
NODE_CLASS_MAPPINGS = {
    "LoadImageCloud": LoadImageCloud
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageCloud": "Load Image Cloud"
}
