import logging
import os
import pickle
import math

import numpy as np
import torch
from PIL import Image
from typing_extensions import override

import folder_paths
import node_helpers
from comfy_api.latest import ComfyExtension, io


def load_and_process_images(image_files, input_dir):
    """Utility function to load and process a list of images.

    Args:
        image_files: List of image filenames
        input_dir: Base directory containing the images
        resize_method: How to handle images of different sizes ("None", "Stretch", "Crop", "Pad")

    Returns:
        torch.Tensor: Batch of processed images
    """
    if not image_files:
        raise ValueError("No valid images found in input")

    output_images = []

    for file in image_files:
        image_path = os.path.join(input_dir, file)
        img = node_helpers.pillow(Image.open, image_path)

        if img.mode == "I":
            img = img.point(lambda i: i * (1 / 255))
        img = img.convert("RGB")
        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)[None,]
        output_images.append(img_tensor)

    return output_images


class LoadImageDataSetFromFolderNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadImageDataSetFromFolder",
            display_name="Load Simple Image Dataset from Folder",
            category="dataset",
            is_experimental=True,
            inputs=[
                io.Combo.Input(
                    "folder",
                    options=folder_paths.get_input_subfolders(),
                    tooltip="The folder to load images from.",
                )
            ],
            outputs=[
                io.Image.Output(is_output_list=True, tooltip="List of loaded images")
            ],
        )

    @classmethod
    def execute(cls, folder):
        sub_input_dir = os.path.join(folder_paths.get_input_directory(), folder)
        valid_extensions = [".png", ".jpg", ".jpeg", ".webp"]
        image_files = [
            f
            for f in os.listdir(sub_input_dir)
            if any(f.lower().endswith(ext) for ext in valid_extensions)
        ]
        output_tensor = load_and_process_images(image_files, sub_input_dir)
        return io.NodeOutput(output_tensor)


class LoadImageTextDataSetFromFolderNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadImageTextDataSetFromFolder",
            display_name="Load Simple Image and Text Dataset from Folder",
            category="dataset",
            is_experimental=True,
            inputs=[
                io.Combo.Input(
                    "folder",
                    options=folder_paths.get_input_subfolders(),
                    tooltip="The folder to load images from.",
                )
            ],
            outputs=[
                io.Image.Output(is_output_list=True, tooltip="List of loaded images"),
                io.String.Output(is_output_list=True, tooltip="List of text captions"),
            ],
        )

    @classmethod
    def execute(cls, folder):
        logging.info(f"Loading images from folder: {folder}")

        sub_input_dir = os.path.join(folder_paths.get_input_directory(), folder)
        valid_extensions = [".png", ".jpg", ".jpeg", ".webp"]

        image_files = []
        for item in os.listdir(sub_input_dir):
            path = os.path.join(sub_input_dir, item)
            if any(item.lower().endswith(ext) for ext in valid_extensions):
                image_files.append(path)
            elif os.path.isdir(path):
                # Support kohya-ss/sd-scripts folder structure
                repeat = 1
                if item.split("_")[0].isdigit():
                    repeat = int(item.split("_")[0])
                image_files.extend(
                    [
                        os.path.join(path, f)
                        for f in os.listdir(path)
                        if any(f.lower().endswith(ext) for ext in valid_extensions)
                    ]
                    * repeat
                )

        caption_file_path = [
            f.replace(os.path.splitext(f)[1], ".txt") for f in image_files
        ]
        captions = []
        for caption_file in caption_file_path:
            caption_path = os.path.join(sub_input_dir, caption_file)
            if os.path.exists(caption_path):
                with open(caption_path, "r", encoding="utf-8") as f:
                    caption = f.read().strip()
                    captions.append(caption)
            else:
                captions.append("")

        output_tensor = load_and_process_images(image_files, sub_input_dir)

        logging.info(f"Loaded {len(output_tensor)} images from {sub_input_dir}.")
        return io.NodeOutput(output_tensor, captions)


def save_images_to_folder(image_list, output_dir, prefix="image"):
    """Utility function to save a list of image tensors to disk.

    Args:
        image_list: List of image tensors (each [1, H, W, C] or [H, W, C] or [C, H, W])
        output_dir: Directory to save images to
        prefix: Filename prefix

    Returns:
        List of saved filenames
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []

    for idx, img_tensor in enumerate(image_list):
        # Handle different tensor shapes
        if isinstance(img_tensor, torch.Tensor):
            # Remove batch dimension if present [1, H, W, C] -> [H, W, C]
            if img_tensor.dim() == 4 and img_tensor.shape[0] == 1:
                img_tensor = img_tensor.squeeze(0)

            # If tensor is [C, H, W], permute to [H, W, C]
            if img_tensor.dim() == 3 and img_tensor.shape[0] in [1, 3, 4]:
                if (
                    img_tensor.shape[0] <= 4
                    and img_tensor.shape[1] > 4
                    and img_tensor.shape[2] > 4
                ):
                    img_tensor = img_tensor.permute(1, 2, 0)

            # Convert to numpy and scale to 0-255
            img_array = img_tensor.cpu().numpy()
            img_array = np.clip(img_array * 255.0, 0, 255).astype(np.uint8)

            # Convert to PIL Image
            img = Image.fromarray(img_array)
        else:
            raise ValueError(f"Expected torch.Tensor, got {type(img_tensor)}")

        # Save image
        filename = f"{prefix}_{idx:05d}.png"
        filepath = os.path.join(output_dir, filename)
        img.save(filepath)
        saved_files.append(filename)

    return saved_files


class SaveImageDataSetToFolderNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveImageDataSetToFolder",
            display_name="Save Simple Image Dataset to Folder",
            category="dataset",
            is_experimental=True,
            is_output_node=True,
            is_input_list=True,  # Receive images as list
            inputs=[
                io.Image.Input("images", tooltip="List of images to save."),
                io.String.Input(
                    "folder_name",
                    default="dataset",
                    tooltip="Name of the folder to save images to (inside output directory).",
                ),
                io.String.Input(
                    "filename_prefix",
                    default="image",
                    tooltip="Prefix for saved image filenames.",
                ),
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, images, folder_name, filename_prefix):
        # Extract scalar values
        folder_name = folder_name[0]
        filename_prefix = filename_prefix[0]

        output_dir = os.path.join(folder_paths.get_output_directory(), folder_name)
        saved_files = save_images_to_folder(images, output_dir, filename_prefix)

        logging.info(f"Saved {len(saved_files)} images to {output_dir}.")
        return io.NodeOutput()


class SaveImageTextDataSetToFolderNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveImageTextDataSetToFolder",
            display_name="Save Simple Image and Text Dataset to Folder",
            category="dataset",
            is_experimental=True,
            is_output_node=True,
            is_input_list=True,  # Receive both images and texts as lists
            inputs=[
                io.Image.Input("images", tooltip="List of images to save."),
                io.String.Input("texts", tooltip="List of text captions to save."),
                io.String.Input(
                    "folder_name",
                    default="dataset",
                    tooltip="Name of the folder to save images to (inside output directory).",
                ),
                io.String.Input(
                    "filename_prefix",
                    default="image",
                    tooltip="Prefix for saved image filenames.",
                ),
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, images, texts, folder_name, filename_prefix):
        # Extract scalar values
        folder_name = folder_name[0]
        filename_prefix = filename_prefix[0]

        output_dir = os.path.join(folder_paths.get_output_directory(), folder_name)
        saved_files = save_images_to_folder(images, output_dir, filename_prefix)

        # Save captions
        for idx, (filename, caption) in enumerate(zip(saved_files, texts)):
            caption_filename = filename.replace(".png", ".txt")
            caption_path = os.path.join(output_dir, caption_filename)
            with open(caption_path, "w", encoding="utf-8") as f:
                f.write(caption)

        logging.info(f"Saved {len(saved_files)} images and captions to {output_dir}.")
        return io.NodeOutput()


# ========== Helper Functions for Transform Nodes ==========


def tensor_to_pil(img_tensor):
    """Convert tensor to PIL Image."""
    if img_tensor.dim() == 4 and img_tensor.shape[0] == 1:
        img_tensor = img_tensor.squeeze(0)
    img_array = (img_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img_array)


def pil_to_tensor(img):
    """Convert PIL Image to tensor."""
    img_array = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(img_array)[None,]


# ========== Base Classes for Transform Nodes ==========


class ImageProcessingNode(io.ComfyNode):
    """Base class for image processing nodes that operate on lists of images.

    Child classes should set:
        node_id: Unique node identifier (required)
        display_name: Display name (optional, defaults to node_id)
        description: Node description (optional)
        extra_inputs: List of additional io.Input objects beyond "images" (optional)

    Child classes must implement:
        _process(cls, images, **kwargs) -> list[tensor]
    """

    node_id = None
    display_name = None
    description = None
    extra_inputs = []

    @classmethod
    def define_schema(cls):
        if cls.node_id is None:
            raise NotImplementedError(f"{cls.__name__} must set node_id class variable")

        inputs = [io.Image.Input("images", tooltip="List of images to process.")]
        inputs.extend(cls.extra_inputs)

        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name or cls.node_id,
            category="dataset/image",
            is_experimental=True,
            is_input_list=True,
            inputs=inputs,
            outputs=[io.Image.Output(is_output_list=True, tooltip="Processed images")],
        )

    @classmethod
    def execute(cls, images, **kwargs):
        """Execute the node. Extracts scalar values and calls _process."""
        # Extract scalar values from lists (due to is_input_list=True)
        params = {}
        for k, v in kwargs.items():
            if isinstance(v, list) and len(v) == 1:
                params[k] = v[0]
            else:
                params[k] = v

        result = cls._process(images, **params)
        return io.NodeOutput(result)

    @classmethod
    def _process(cls, images, **kwargs):
        """Override this method in subclasses to implement specific processing.

        Args:
            images: list[tensor] - List of image tensors
            **kwargs: Additional parameters (already extracted from lists)

        Returns:
            list[tensor] - Processed images
        """
        raise NotImplementedError(f"{cls.__name__} must implement _process method")


class TextProcessingNode(io.ComfyNode):
    """Base class for text processing nodes that operate on lists of texts.

    Child classes should set:
        node_id: Unique node identifier (required)
        display_name: Display name (optional, defaults to node_id)
        description: Node description (optional)
        extra_inputs: List of additional io.Input objects beyond "texts" (optional)

    Child classes must implement:
        _process(cls, texts, **kwargs) -> list[str]
    """

    node_id = None
    display_name = None
    description = None
    extra_inputs = []

    @classmethod
    def define_schema(cls):
        if cls.node_id is None:
            raise NotImplementedError(f"{cls.__name__} must set node_id class variable")

        inputs = [io.String.Input("texts", tooltip="List of texts to process.")]
        inputs.extend(cls.extra_inputs)

        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name or cls.node_id,
            category="dataset/text",
            is_experimental=True,
            is_input_list=True,
            inputs=inputs,
            outputs=[io.String.Output(is_output_list=True, tooltip="Processed texts")],
        )

    @classmethod
    def execute(cls, texts, **kwargs):
        """Execute the node. Extracts scalar values and calls _process."""
        # Extract scalar values from lists (due to is_input_list=True)
        params = {}
        for k, v in kwargs.items():
            if isinstance(v, list) and len(v) == 1:
                params[k] = v[0]
            else:
                params[k] = v

        result = cls._process(texts, **params)
        return io.NodeOutput(result)

    @classmethod
    def _process(cls, texts, **kwargs):
        """Override this method in subclasses to implement specific processing.

        Args:
            texts: list[str] - List of text strings
            **kwargs: Additional parameters (already extracted from lists)

        Returns:
            list[str] - Processed texts
        """
        raise NotImplementedError(f"{cls.__name__} must implement _process method")


# ========== Image Transform Nodes ==========


class ResizeImagesToSameSizeNode(ImageProcessingNode):
    node_id = "ResizeImagesToSameSize"
    display_name = "Resize Images to Same Size"
    description = "Resize all images to the same width and height."
    extra_inputs = [
        io.Int.Input("width", default=512, min=1, max=8192, tooltip="Target width."),
        io.Int.Input("height", default=512, min=1, max=8192, tooltip="Target height."),
        io.Combo.Input(
            "mode",
            options=["stretch", "crop_center", "pad"],
            default="stretch",
            tooltip="Resize mode.",
        ),
    ]

    @classmethod
    def _process(cls, images, width, height, mode):
        output_images = []
        for img_tensor in images:
            img = tensor_to_pil(img_tensor)

            if mode == "stretch":
                img = img.resize((width, height), Image.Resampling.LANCZOS)
            elif mode == "crop_center":
                left = max(0, (img.width - width) // 2)
                top = max(0, (img.height - height) // 2)
                right = min(img.width, left + width)
                bottom = min(img.height, top + height)
                img = img.crop((left, top, right, bottom))
                if img.width != width or img.height != height:
                    img = img.resize((width, height), Image.Resampling.LANCZOS)
            elif mode == "pad":
                img.thumbnail((width, height), Image.Resampling.LANCZOS)
                new_img = Image.new("RGB", (width, height), (0, 0, 0))
                paste_x = (width - img.width) // 2
                paste_y = (height - img.height) // 2
                new_img.paste(img, (paste_x, paste_y))
                img = new_img

            output_images.append(pil_to_tensor(img))
        return output_images


class ResizeImagesToPixelCountNode(ImageProcessingNode):
    node_id = "ResizeImagesToPixelCount"
    display_name = "Resize Images to Pixel Count"
    description = "Resize images so that the total pixel count matches the specified number while preserving aspect ratio."
    extra_inputs = [
        io.Int.Input(
            "pixel_count",
            default=512 * 512,
            min=1,
            max=8192 * 8192,
            tooltip="Target pixel count.",
        ),
        io.Int.Input(
            "steps",
            default=64,
            min=1,
            max=128,
            tooltip="The stepping for resize width/height.",
        ),
    ]

    @classmethod
    def _process(cls, images, pixel_count, steps):
        output_images = []
        for img_tensor in images:
            img = tensor_to_pil(img_tensor)
            w, h = img.size
            pixel_count_ratio = math.sqrt(pixel_count / (w * h))
            new_w = int(h * pixel_count_ratio / steps) * steps
            new_h = int(w * pixel_count_ratio / steps) * steps
            logging.info(f"Resizing from {w}x{h} to {new_w}x{new_h}")
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            output_images.append(pil_to_tensor(img))
        return output_images


class ResizeImagesByShorterEdgeNode(ImageProcessingNode):
    node_id = "ResizeImagesByShorterEdge"
    display_name = "Resize Images by Shorter Edge"
    description = "Resize images so that the shorter edge matches the specified length while preserving aspect ratio."
    extra_inputs = [
        io.Int.Input(
            "shorter_edge",
            default=512,
            min=1,
            max=8192,
            tooltip="Target length for the shorter edge.",
        ),
    ]

    @classmethod
    def _process(cls, images, shorter_edge):
        output_images = []
        for img_tensor in images:
            img = tensor_to_pil(img_tensor)
            w, h = img.size
            if w < h:
                new_w = shorter_edge
                new_h = int(h * (shorter_edge / w))
            else:
                new_h = shorter_edge
                new_w = int(w * (shorter_edge / h))
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            output_images.append(pil_to_tensor(img))
        return output_images


class ResizeImagesByLongerEdgeNode(ImageProcessingNode):
    node_id = "ResizeImagesByLongerEdge"
    display_name = "Resize Images by Longer Edge"
    description = "Resize images so that the longer edge matches the specified length while preserving aspect ratio."
    extra_inputs = [
        io.Int.Input(
            "longer_edge",
            default=1024,
            min=1,
            max=8192,
            tooltip="Target length for the longer edge.",
        ),
    ]

    @classmethod
    def _process(cls, images, longer_edge):
        output_images = []
        for img_tensor in images:
            img = tensor_to_pil(img_tensor)
            w, h = img.size
            if w > h:
                new_w = longer_edge
                new_h = int(h * (longer_edge / w))
            else:
                new_h = longer_edge
                new_w = int(w * (longer_edge / h))
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            output_images.append(pil_to_tensor(img))
        return output_images


class CenterCropImagesNode(ImageProcessingNode):
    node_id = "CenterCropImages"
    display_name = "Center Crop Images"
    description = "Center crop all images to the specified dimensions."
    extra_inputs = [
        io.Int.Input("width", default=512, min=1, max=8192, tooltip="Crop width."),
        io.Int.Input("height", default=512, min=1, max=8192, tooltip="Crop height."),
    ]

    @classmethod
    def _process(cls, images, width, height):
        output_images = []
        for img_tensor in images:
            img = tensor_to_pil(img_tensor)
            left = max(0, (img.width - width) // 2)
            top = max(0, (img.height - height) // 2)
            right = min(img.width, left + width)
            bottom = min(img.height, top + height)
            img = img.crop((left, top, right, bottom))
            output_images.append(pil_to_tensor(img))
        return output_images


class RandomCropImagesNode(ImageProcessingNode):
    node_id = "RandomCropImages"
    display_name = "Random Crop Images"
    description = (
        "Randomly crop all images to the specified dimensions (for data augmentation)."
    )
    extra_inputs = [
        io.Int.Input("width", default=512, min=1, max=8192, tooltip="Crop width."),
        io.Int.Input("height", default=512, min=1, max=8192, tooltip="Crop height."),
        io.Int.Input(
            "seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, tooltip="Random seed."
        ),
    ]

    @classmethod
    def _process(cls, images, width, height, seed):
        np.random.seed(seed % (2**32 - 1))
        output_images = []
        for img_tensor in images:
            img = tensor_to_pil(img_tensor)
            max_left = max(0, img.width - width)
            max_top = max(0, img.height - height)
            left = np.random.randint(0, max_left + 1) if max_left > 0 else 0
            top = np.random.randint(0, max_top + 1) if max_top > 0 else 0
            right = min(img.width, left + width)
            bottom = min(img.height, top + height)
            img = img.crop((left, top, right, bottom))
            output_images.append(pil_to_tensor(img))
        return output_images


class FlipImagesNode(ImageProcessingNode):
    node_id = "FlipImages"
    display_name = "Flip Images"
    description = "Flip all images horizontally or vertically."
    extra_inputs = [
        io.Combo.Input(
            "direction",
            options=["horizontal", "vertical"],
            default="horizontal",
            tooltip="Flip direction.",
        ),
    ]

    @classmethod
    def _process(cls, images, direction):
        output_images = []
        for img_tensor in images:
            img = tensor_to_pil(img_tensor)
            if direction == "horizontal":
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            output_images.append(pil_to_tensor(img))
        return output_images


class NormalizeImagesNode(ImageProcessingNode):
    node_id = "NormalizeImages"
    display_name = "Normalize Images"
    description = "Normalize images using mean and standard deviation."
    extra_inputs = [
        io.Float.Input(
            "mean",
            default=0.5,
            min=0.0,
            max=1.0,
            tooltip="Mean value for normalization.",
        ),
        io.Float.Input(
            "std",
            default=0.5,
            min=0.001,
            max=1.0,
            tooltip="Standard deviation for normalization.",
        ),
    ]

    @classmethod
    def _process(cls, images, mean, std):
        return [(img - mean) / std for img in images]


class AdjustBrightnessNode(ImageProcessingNode):
    node_id = "AdjustBrightness"
    display_name = "Adjust Brightness"
    description = "Adjust brightness of all images."
    extra_inputs = [
        io.Float.Input(
            "factor",
            default=1.0,
            min=0.0,
            max=2.0,
            tooltip="Brightness factor. 1.0 = no change, <1.0 = darker, >1.0 = brighter.",
        ),
    ]

    @classmethod
    def _process(cls, images, factor):
        return [(img * factor).clamp(0.0, 1.0) for img in images]


class AdjustContrastNode(ImageProcessingNode):
    node_id = "AdjustContrast"
    display_name = "Adjust Contrast"
    description = "Adjust contrast of all images."
    extra_inputs = [
        io.Float.Input(
            "factor",
            default=1.0,
            min=0.0,
            max=2.0,
            tooltip="Contrast factor. 1.0 = no change, <1.0 = less contrast, >1.0 = more contrast.",
        ),
    ]

    @classmethod
    def _process(cls, images, factor):
        return [((img - 0.5) * factor + 0.5).clamp(0.0, 1.0) for img in images]


class ShuffleDatasetNode(ImageProcessingNode):
    node_id = "ShuffleDataset"
    display_name = "Shuffle Image Dataset"
    description = "Randomly shuffle the order of images in the dataset."
    extra_inputs = [
        io.Int.Input(
            "seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, tooltip="Random seed."
        ),
    ]

    @classmethod
    def _process(cls, images, seed):
        np.random.seed(seed % (2**32 - 1))
        indices = np.random.permutation(len(images))
        return [images[i] for i in indices]


class ShuffleImageTextDatasetNode(io.ComfyNode):
    """Special node that shuffles both images and texts together."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ShuffleImageTextDataset",
            display_name="Shuffle Image-Text Dataset",
            category="dataset/image",
            is_experimental=True,
            is_input_list=True,
            inputs=[
                io.Image.Input("images", tooltip="List of images to shuffle."),
                io.String.Input("texts", tooltip="List of texts to shuffle."),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    tooltip="Random seed.",
                ),
            ],
            outputs=[
                io.Image.Output(is_output_list=True, tooltip="Shuffled images"),
                io.String.Output(is_output_list=True, tooltip="Shuffled texts"),
            ],
        )

    @classmethod
    def execute(cls, images, texts, seed):
        seed = seed[0]  # Extract scalar
        np.random.seed(seed % (2**32 - 1))
        indices = np.random.permutation(len(images))
        shuffled_images = [images[i] for i in indices]
        shuffled_texts = [texts[i] for i in indices]
        return io.NodeOutput(shuffled_images, shuffled_texts)


# ========== Text Transform Nodes ==========


class TextToLowercaseNode(TextProcessingNode):
    node_id = "TextToLowercase"
    display_name = "Text to Lowercase"
    description = "Convert all texts to lowercase."

    @classmethod
    def _process(cls, texts):
        return [text.lower() for text in texts]


class TextToUppercaseNode(TextProcessingNode):
    node_id = "TextToUppercase"
    display_name = "Text to Uppercase"
    description = "Convert all texts to uppercase."

    @classmethod
    def _process(cls, texts):
        return [text.upper() for text in texts]


class TruncateTextNode(TextProcessingNode):
    node_id = "TruncateText"
    display_name = "Truncate Text"
    description = "Truncate all texts to a maximum length."
    extra_inputs = [
        io.Int.Input(
            "max_length", default=77, min=1, max=10000, tooltip="Maximum text length."
        ),
    ]

    @classmethod
    def _process(cls, texts, max_length):
        return [text[:max_length] for text in texts]


class AddTextPrefixNode(TextProcessingNode):
    node_id = "AddTextPrefix"
    display_name = "Add Text Prefix"
    description = "Add a prefix to all texts."
    extra_inputs = [
        io.String.Input("prefix", default="", tooltip="Prefix to add."),
    ]

    @classmethod
    def _process(cls, texts, prefix):
        return [prefix + text for text in texts]


class AddTextSuffixNode(TextProcessingNode):
    node_id = "AddTextSuffix"
    display_name = "Add Text Suffix"
    description = "Add a suffix to all texts."
    extra_inputs = [
        io.String.Input("suffix", default="", tooltip="Suffix to add."),
    ]

    @classmethod
    def _process(cls, texts, suffix):
        return [text + suffix for text in texts]


class ReplaceTextNode(TextProcessingNode):
    node_id = "ReplaceText"
    display_name = "Replace Text"
    description = "Replace text in all texts."
    extra_inputs = [
        io.String.Input("find", default="", tooltip="Text to find."),
        io.String.Input("replace", default="", tooltip="Text to replace with."),
    ]

    @classmethod
    def _process(cls, texts, find, replace):
        return [text.replace(find, replace) for text in texts]


class StripWhitespaceNode(TextProcessingNode):
    node_id = "StripWhitespace"
    display_name = "Strip Whitespace"
    description = "Strip leading and trailing whitespace from all texts."

    @classmethod
    def _process(cls, texts):
        return [text.strip() for text in texts]


# ========== Training Dataset Nodes ==========


class MakeTrainingDataset(io.ComfyNode):
    """Encode images with VAE and texts with CLIP to create a training dataset."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MakeTrainingDataset",
            display_name="Make Training Dataset",
            category="dataset",
            is_experimental=True,
            is_input_list=True,  # images and texts as lists
            inputs=[
                io.Image.Input("images", tooltip="List of images to encode."),
                io.Vae.Input(
                    "vae", tooltip="VAE model for encoding images to latents."
                ),
                io.Clip.Input(
                    "clip", tooltip="CLIP model for encoding text to conditioning."
                ),
                io.String.Input(
                    "texts",
                    optional=True,
                    tooltip="List of text captions. Can be length n (matching images), 1 (repeated for all), or omitted (uses empty string).",
                ),
            ],
            outputs=[
                io.Latent.Output(is_output_list=True, tooltip="List of latent dicts"),
                io.Conditioning.Output(
                    is_output_list=True, tooltip="List of conditioning lists"
                ),
            ],
        )

    @classmethod
    def execute(cls, images, vae, clip, texts=None):
        # Extract scalars (vae and clip are single values wrapped in lists)
        vae = vae[0]
        clip = clip[0]

        # Handle text list
        num_images = len(images)

        if texts is None or len(texts) == 0:
            # Treat as [""] for unconditional training
            texts = [""]

        if len(texts) == 1 and num_images > 1:
            # Repeat single text for all images
            texts = texts * num_images
        elif len(texts) != num_images:
            raise ValueError(
                f"Number of texts ({len(texts)}) does not match number of images ({num_images}). "
                f"Text list should have length {num_images}, 1, or 0."
            )

        # Encode images with VAE
        logging.info(f"Encoding {num_images} images with VAE...")
        latents_list = []  # list[{"samples": tensor}]
        for img_tensor in images:
            # img_tensor is [1, H, W, 3]
            latent_tensor = vae.encode(img_tensor[:, :, :, :3])
            latents_list.append({"samples": latent_tensor})

        # Encode texts with CLIP
        logging.info(f"Encoding {len(texts)} texts with CLIP...")
        conditioning_list = []  # list[list[cond]]
        for text in texts:
            if text == "":
                cond = clip.encode_from_tokens_scheduled(clip.tokenize(""))
            else:
                tokens = clip.tokenize(text)
                cond = clip.encode_from_tokens_scheduled(tokens)
            conditioning_list.append(cond)

        logging.info(
            f"Created dataset with {len(latents_list)} latents and {len(conditioning_list)} conditioning."
        )
        return io.NodeOutput(latents_list, conditioning_list)


class SaveTrainingDataset(io.ComfyNode):
    """Save encoded training dataset (latents + conditioning) to disk."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveTrainingDataset",
            display_name="Save Training Dataset",
            category="dataset",
            is_experimental=True,
            is_output_node=True,
            is_input_list=True,  # Receive lists
            inputs=[
                io.Latent.Input(
                    "latents",
                    tooltip="List of latent dicts from MakeTrainingDataset.",
                ),
                io.Conditioning.Input(
                    "conditioning",
                    tooltip="List of conditioning lists from MakeTrainingDataset.",
                ),
                io.String.Input(
                    "folder_name",
                    default="training_dataset",
                    tooltip="Name of folder to save dataset (inside output directory).",
                ),
                io.Int.Input(
                    "shard_size",
                    default=1000,
                    min=1,
                    max=100000,
                    tooltip="Number of samples per shard file.",
                ),
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, latents, conditioning, folder_name, shard_size):
        # Extract scalars
        folder_name = folder_name[0]
        shard_size = shard_size[0]

        # latents: list[{"samples": tensor}]
        # conditioning: list[list[cond]]

        # Validate lengths match
        if len(latents) != len(conditioning):
            raise ValueError(
                f"Number of latents ({len(latents)}) does not match number of conditions ({len(conditioning)}). "
                f"Something went wrong in dataset preparation."
            )

        # Create output directory
        output_dir = os.path.join(folder_paths.get_output_directory(), folder_name)
        os.makedirs(output_dir, exist_ok=True)

        # Prepare data pairs
        num_samples = len(latents)
        num_shards = (num_samples + shard_size - 1) // shard_size  # Ceiling division

        logging.info(
            f"Saving {num_samples} samples to {num_shards} shards in {output_dir}..."
        )

        # Save data in shards
        for shard_idx in range(num_shards):
            start_idx = shard_idx * shard_size
            end_idx = min(start_idx + shard_size, num_samples)

            # Get shard data (list of latent dicts and conditioning lists)
            shard_data = {
                "latents": latents[start_idx:end_idx],
                "conditioning": conditioning[start_idx:end_idx],
            }

            # Save shard
            shard_filename = f"shard_{shard_idx:04d}.pkl"
            shard_path = os.path.join(output_dir, shard_filename)

            with open(shard_path, "wb") as f:
                pickle.dump(shard_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            logging.info(
                f"Saved shard {shard_idx + 1}/{num_shards}: {shard_filename} ({end_idx - start_idx} samples)"
            )

        # Save metadata
        metadata = {
            "num_samples": num_samples,
            "num_shards": num_shards,
            "shard_size": shard_size,
        }
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            import json

            json.dump(metadata, f, indent=2)

        logging.info(f"Successfully saved {num_samples} samples to {output_dir}.")
        return io.NodeOutput()


class LoadTrainingDataset(io.ComfyNode):
    """Load encoded training dataset from disk."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadTrainingDataset",
            display_name="Load Training Dataset",
            category="dataset",
            is_experimental=True,
            inputs=[
                io.String.Input(
                    "folder_name",
                    default="training_dataset",
                    tooltip="Name of folder containing the saved dataset (inside output directory).",
                ),
            ],
            outputs=[
                io.Latent.Output(is_output_list=True, tooltip="List of latent dicts"),
                io.Conditioning.Output(
                    is_output_list=True, tooltip="List of conditioning lists"
                ),
            ],
        )

    @classmethod
    def execute(cls, folder_name):
        # Get dataset directory
        dataset_dir = os.path.join(folder_paths.get_output_directory(), folder_name)

        if not os.path.exists(dataset_dir):
            raise ValueError(f"Dataset directory not found: {dataset_dir}")

        # Find all shard files
        shard_files = sorted(
            [
                f
                for f in os.listdir(dataset_dir)
                if f.startswith("shard_") and f.endswith(".pkl")
            ]
        )

        if not shard_files:
            raise ValueError(f"No shard files found in {dataset_dir}")

        logging.info(f"Loading {len(shard_files)} shards from {dataset_dir}...")

        # Load all shards
        all_latents = []  # list[{"samples": tensor}]
        all_conditioning = []  # list[list[cond]]

        for shard_file in shard_files:
            shard_path = os.path.join(dataset_dir, shard_file)

            with open(shard_path, "rb") as f:
                shard_data = pickle.load(f)

            all_latents.extend(shard_data["latents"])
            all_conditioning.extend(shard_data["conditioning"])

            logging.info(f"Loaded {shard_file}: {len(shard_data['latents'])} samples")

        logging.info(
            f"Successfully loaded {len(all_latents)} samples from {dataset_dir}."
        )
        return io.NodeOutput(all_latents, all_conditioning)


# ========== Extension Setup ==========


class DatasetExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            # Data loading/saving nodes
            LoadImageDataSetFromFolderNode,
            LoadImageTextDataSetFromFolderNode,
            SaveImageDataSetToFolderNode,
            SaveImageTextDataSetToFolderNode,
            # Image transform nodes
            ResizeImagesToSameSizeNode,
            ResizeImagesToPixelCountNode,
            ResizeImagesByShorterEdgeNode,
            ResizeImagesByLongerEdgeNode,
            CenterCropImagesNode,
            RandomCropImagesNode,
            FlipImagesNode,
            NormalizeImagesNode,
            AdjustBrightnessNode,
            AdjustContrastNode,
            ShuffleDatasetNode,
            ShuffleImageTextDatasetNode,
            # Text transform nodes
            TextToLowercaseNode,
            TextToUppercaseNode,
            TruncateTextNode,
            AddTextPrefixNode,
            AddTextSuffixNode,
            ReplaceTextNode,
            StripWhitespaceNode,
            # Training dataset nodes
            MakeTrainingDataset,
            SaveTrainingDataset,
            LoadTrainingDataset,
        ]


async def comfy_entrypoint() -> DatasetExtension:
    return DatasetExtension()
