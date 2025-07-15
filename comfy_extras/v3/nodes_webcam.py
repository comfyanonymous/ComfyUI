import hashlib

import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence

import folder_paths
import node_helpers
import nodes
from comfy_api.v3 import io

MAX_RESOLUTION = nodes.MAX_RESOLUTION


class WebcamCapture_V3(io.ComfyNodeV3):
    @classmethod
    def DEFINE_SCHEMA(cls):
        return io.SchemaV3(
            node_id="WebcamCapture_V3",
            display_name="Webcam Capture _V3",
            category="image",
            inputs=[
                io.Webcam.Input("image"),
                io.Int.Input(
                    "width",
                    default=0,
                    min=0,
                    max=MAX_RESOLUTION,
                    step=1,
                ),
                io.Int.Input(
                    "height",
                    default=0,
                    min=0,
                    max=MAX_RESOLUTION,
                    step=1,
                ),
                io.Boolean.Input("capture_on_queue", default=True),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, image, **kwargs) -> io.NodeOutput:
        img = node_helpers.pillow(Image.open, folder_paths.get_annotated_filepath(image))

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ["MPO"]

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == "I":
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if "A" in i.getbands():
                mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask)
            elif i.mode == "P" and "transparency" in i.info:
                mask = np.array(i.convert("RGBA").getchannel("A")).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return io.NodeOutput(output_image, output_mask)

    @classmethod
    def fingerprint_inputs(s, image, width, height, capture_on_queue):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def validate_inputs(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True


NODES_LIST: list[type[io.ComfyNodeV3]] = [WebcamCapture_V3]
