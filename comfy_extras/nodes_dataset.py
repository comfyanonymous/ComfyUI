import logging
import os

import numpy as np
import torch
from PIL import Image

import folder_paths
import node_helpers


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


class LoadImageDataSetFromFolderNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder": (folder_paths.get_input_subfolders(), {"tooltip": "The folder to load images from."})
            },
        }

    RETURN_TYPES = ("IMAGE_LIST",)
    FUNCTION = "load_images"
    CATEGORY = "loaders"
    EXPERIMENTAL = True
    DESCRIPTION = "Loads a batch of images from a directory for training."

    def load_images(self, folder):
        sub_input_dir = os.path.join(folder_paths.get_input_directory(), folder)
        valid_extensions = [".png", ".jpg", ".jpeg", ".webp"]
        image_files = [
            f
            for f in os.listdir(sub_input_dir)
            if any(f.lower().endswith(ext) for ext in valid_extensions)
        ]
        output_tensor = load_and_process_images(image_files, sub_input_dir)
        return (output_tensor,)


class LoadImageTextDataSetFromFolderNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder": (folder_paths.get_input_subfolders(), {"tooltip": "The folder to load images from."}),
            },
        }

    RETURN_TYPES = ("IMAGE_LIST", "TEXT_LIST",)
    FUNCTION = "load_images"
    CATEGORY = "loaders"
    EXPERIMENTAL = True
    DESCRIPTION = "Loads a batch of images and caption from a directory for training."

    def load_images(self, folder):
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
                image_files.extend([
                    os.path.join(path, f) for f in os.listdir(path) if any(f.lower().endswith(ext) for ext in valid_extensions)
                ] * repeat)

        caption_file_path = [
            f.replace(os.path.splitext(f)[1], ".txt")
            for f in image_files
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

        width = width if width != -1 else None
        height = height if height != -1 else None
        output_tensor = load_and_process_images(image_files, sub_input_dir)

        logging.info(f"Loaded {len(output_tensor)} images from {sub_input_dir}.")
        return (output_tensor, captions)


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
                if img_tensor.shape[0] <= 4 and img_tensor.shape[1] > 4 and img_tensor.shape[2] > 4:
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


class SaveImageDataSetToFolderNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE_LIST", {"tooltip": "List of images to save."}),
                "folder_name": ("STRING", {"default": "dataset", "tooltip": "Name of the folder to save images to (inside output directory)."}),
                "filename_prefix": ("STRING", {"default": "image", "tooltip": "Prefix for saved image filenames."}),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save_images"
    CATEGORY = "loaders"
    EXPERIMENTAL = True
    DESCRIPTION = "Saves a batch of images to a directory."

    def save_images(self, images, folder_name, filename_prefix):
        output_dir = os.path.join(folder_paths.get_output_directory(), folder_name)
        saved_files = save_images_to_folder(images, output_dir, filename_prefix)

        logging.info(f"Saved {len(saved_files)} images to {output_dir}.")
        return {}


class SaveImageTextDataSetToFolderNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE_LIST", {"tooltip": "List of images to save."}),
                "texts": ("TEXT_LIST", {"tooltip": "List of text captions to save."}),
                "folder_name": ("STRING", {"default": "dataset", "tooltip": "Name of the folder to save images to (inside output directory)."}),
                "filename_prefix": ("STRING", {"default": "image", "tooltip": "Prefix for saved image filenames."}),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save_images"
    CATEGORY = "loaders"
    EXPERIMENTAL = True
    DESCRIPTION = "Saves a batch of images and captions to a directory."

    def save_images(self, images, texts, folder_name, filename_prefix):
        output_dir = os.path.join(folder_paths.get_output_directory(), folder_name)
        saved_files = save_images_to_folder(images, output_dir, filename_prefix)

        # Save captions
        for idx, (filename, caption) in enumerate(zip(saved_files, texts)):
            caption_filename = filename.replace(".png", ".txt")
            caption_path = os.path.join(output_dir, caption_filename)
            with open(caption_path, "w", encoding="utf-8") as f:
                f.write(caption)

        logging.info(f"Saved {len(saved_files)} images and captions to {output_dir}.")
        return {}


# ========== Base Classes for Transform Nodes ==========

class ImageProcessingNode:
    """Base class for image processing nodes that operate on IMAGE_LIST."""

    CATEGORY = "image/transforms"
    EXPERIMENTAL = True
    RETURN_TYPES = ("IMAGE_LIST",)
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE_LIST", {"tooltip": "List of images to process."}),
            },
        }

    def process(self, images, **kwargs):
        """Default process function that calls _process for each image."""
        return (self._process(images, **kwargs),)

    def _process(self, images, **kwargs):
        """Override this method in subclasses to implement specific processing."""
        raise NotImplementedError("Subclasses must implement _process method")

    def _tensor_to_pil(self, img_tensor):
        """Convert tensor to PIL Image."""
        if img_tensor.dim() == 4 and img_tensor.shape[0] == 1:
            img_tensor = img_tensor.squeeze(0)
        img_array = (img_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(img_array)

    def _pil_to_tensor(self, img):
        """Convert PIL Image to tensor."""
        img_array = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(img_array)[None,]


class TextProcessingNode:
    """Base class for text processing nodes that operate on TEXT_LIST."""

    CATEGORY = "text/transforms"
    EXPERIMENTAL = True
    RETURN_TYPES = ("TEXT_LIST",)
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "texts": ("TEXT_LIST", {"tooltip": "List of texts to process."}),
            },
        }

    def process(self, texts, **kwargs):
        """Default process function that calls _process."""
        return (self._process(texts, **kwargs),)

    def _process(self, texts, **kwargs):
        """Override this method in subclasses to implement specific processing."""
        raise NotImplementedError("Subclasses must implement _process method")


# ========== Image Transform Nodes ==========

class ResizeImagesToSameSizeNode(ImageProcessingNode):
    DESCRIPTION = "Resize all images to the same width and height."

    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"].update({
            "width": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1, "tooltip": "Target width."}),
            "height": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1, "tooltip": "Target height."}),
            "mode": (["stretch", "crop_center", "pad"], {"default": "stretch", "tooltip": "Resize mode."}),
        })
        return base_inputs

    def _process(self, images, width, height, mode):
        output_images = []
        for img_tensor in images:
            img = self._tensor_to_pil(img_tensor)

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

            output_images.append(self._pil_to_tensor(img))
        return output_images


class ResizeImagesByShorterEdgeNode(ImageProcessingNode):
    DESCRIPTION = "Resize images so that the shorter edge matches the specified length while preserving aspect ratio."

    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"]["shorter_edge"] = ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1, "tooltip": "Target length for the shorter edge."})
        return base_inputs

    def _process(self, images, shorter_edge):
        output_images = []
        for img_tensor in images:
            img = self._tensor_to_pil(img_tensor)
            w, h = img.size
            if w < h:
                new_w = shorter_edge
                new_h = int(h * (shorter_edge / w))
            else:
                new_h = shorter_edge
                new_w = int(w * (shorter_edge / h))
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            output_images.append(self._pil_to_tensor(img))
        return output_images


class ResizeImagesByLongerEdgeNode(ImageProcessingNode):
    DESCRIPTION = "Resize images so that the longer edge matches the specified length while preserving aspect ratio."

    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"]["longer_edge"] = ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1, "tooltip": "Target length for the longer edge."})
        return base_inputs

    def _process(self, images, longer_edge):
        output_images = []
        for img_tensor in images:
            img = self._tensor_to_pil(img_tensor)
            w, h = img.size
            if w > h:
                new_w = longer_edge
                new_h = int(h * (longer_edge / w))
            else:
                new_h = longer_edge
                new_w = int(w * (longer_edge / h))
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            output_images.append(self._pil_to_tensor(img))
        return output_images


class CenterCropImagesNode(ImageProcessingNode):
    DESCRIPTION = "Center crop all images to the specified dimensions."

    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"].update({
            "width": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1, "tooltip": "Crop width."}),
            "height": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1, "tooltip": "Crop height."}),
        })
        return base_inputs

    def _process(self, images, width, height):
        output_images = []
        for img_tensor in images:
            img = self._tensor_to_pil(img_tensor)
            left = max(0, (img.width - width) // 2)
            top = max(0, (img.height - height) // 2)
            right = min(img.width, left + width)
            bottom = min(img.height, top + height)
            img = img.crop((left, top, right, bottom))
            output_images.append(self._pil_to_tensor(img))
        return output_images


class RandomCropImagesNode(ImageProcessingNode):
    DESCRIPTION = "Randomly crop all images to the specified dimensions (for data augmentation)."

    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"].update({
            "width": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1, "tooltip": "Crop width."}),
            "height": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1, "tooltip": "Crop height."}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "Random seed."}),
        })
        return base_inputs

    def _process(self, images, width, height, seed):
        np.random.seed(seed)
        output_images = []
        for img_tensor in images:
            img = self._tensor_to_pil(img_tensor)
            max_left = max(0, img.width - width)
            max_top = max(0, img.height - height)
            left = np.random.randint(0, max_left + 1) if max_left > 0 else 0
            top = np.random.randint(0, max_top + 1) if max_top > 0 else 0
            right = min(img.width, left + width)
            bottom = min(img.height, top + height)
            img = img.crop((left, top, right, bottom))
            output_images.append(self._pil_to_tensor(img))
        return output_images


class FlipImagesNode(ImageProcessingNode):
    DESCRIPTION = "Flip all images horizontally or vertically."

    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"]["direction"] = (["horizontal", "vertical"], {"default": "horizontal", "tooltip": "Flip direction."})
        return base_inputs

    def _process(self, images, direction):
        output_images = []
        for img_tensor in images:
            img = self._tensor_to_pil(img_tensor)
            if direction == "horizontal":
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            output_images.append(self._pil_to_tensor(img))
        return output_images


class NormalizeImagesNode(ImageProcessingNode):
    DESCRIPTION = "Normalize images using mean and standard deviation."

    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"].update({
            "mean": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Mean value for normalization."}),
            "std": ("FLOAT", {"default": 0.5, "min": 0.001, "max": 1.0, "step": 0.01, "tooltip": "Standard deviation for normalization."}),
        })
        return base_inputs

    def _process(self, images, mean, std):
        return [(img - mean) / std for img in images]


class AdjustBrightnessNode(ImageProcessingNode):
    DESCRIPTION = "Adjust brightness of all images."

    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"]["factor"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Brightness factor. 1.0 = no change, <1.0 = darker, >1.0 = brighter."})
        return base_inputs

    def _process(self, images, factor):
        return [(img * factor).clamp(0.0, 1.0) for img in images]


class AdjustContrastNode(ImageProcessingNode):
    DESCRIPTION = "Adjust contrast of all images."

    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"]["factor"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Contrast factor. 1.0 = no change, <1.0 = less contrast, >1.0 = more contrast."})
        return base_inputs

    def _process(self, images, factor):
        return [((img - 0.5) * factor + 0.5).clamp(0.0, 1.0) for img in images]


class ShuffleDatasetNode(ImageProcessingNode):
    DESCRIPTION = "Randomly shuffle the order of images in the dataset."

    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"]["seed"] = ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "Random seed."})
        return base_inputs

    def _process(self, images, seed):
        np.random.seed(seed)
        indices = np.random.permutation(len(images))
        return [images[i] for i in indices]


class ShuffleImageTextDatasetNode:
    """Special node that shuffles both images and texts together (doesn't inherit from base class)."""

    CATEGORY = "image/transforms"
    EXPERIMENTAL = True
    RETURN_TYPES = ("IMAGE_LIST", "TEXT_LIST")
    FUNCTION = "process"
    DESCRIPTION = "Randomly shuffle the order of images and texts in the dataset together."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE_LIST", {"tooltip": "List of images to shuffle."}),
                "texts": ("TEXT_LIST", {"tooltip": "List of texts to shuffle."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "Random seed."}),
            },
        }

    def process(self, images, texts, seed):
        np.random.seed(seed)
        indices = np.random.permutation(len(images))
        shuffled_images = [images[i] for i in indices]
        shuffled_texts = [texts[i] for i in indices]
        return (shuffled_images, shuffled_texts)


# ========== Text Transform Nodes ==========

class TextToLowercaseNode(TextProcessingNode):
    DESCRIPTION = "Convert all texts to lowercase."

    def _process(self, texts):
        return [text.lower() for text in texts]


class TextToUppercaseNode(TextProcessingNode):
    DESCRIPTION = "Convert all texts to uppercase."

    def _process(self, texts):
        return [text.upper() for text in texts]


class TruncateTextNode(TextProcessingNode):
    DESCRIPTION = "Truncate all texts to a maximum length."

    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"]["max_length"] = ("INT", {"default": 77, "min": 1, "max": 10000, "step": 1, "tooltip": "Maximum text length."})
        return base_inputs

    def _process(self, texts, max_length):
        return [text[:max_length] for text in texts]


class AddTextPrefixNode(TextProcessingNode):
    DESCRIPTION = "Add a prefix to all texts."

    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"]["prefix"] = ("STRING", {"default": "", "multiline": False, "tooltip": "Prefix to add."})
        return base_inputs

    def _process(self, texts, prefix):
        return [prefix + text for text in texts]


class AddTextSuffixNode(TextProcessingNode):
    DESCRIPTION = "Add a suffix to all texts."

    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"]["suffix"] = ("STRING", {"default": "", "multiline": False, "tooltip": "Suffix to add."})
        return base_inputs

    def _process(self, texts, suffix):
        return [text + suffix for text in texts]


class ReplaceTextNode(TextProcessingNode):
    DESCRIPTION = "Replace text in all texts."

    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"].update({
            "find": ("STRING", {"default": "", "multiline": False, "tooltip": "Text to find."}),
            "replace": ("STRING", {"default": "", "multiline": False, "tooltip": "Text to replace with."}),
        })
        return base_inputs

    def _process(self, texts, find, replace):
        return [text.replace(find, replace) for text in texts]


class StripWhitespaceNode(TextProcessingNode):
    DESCRIPTION = "Strip leading and trailing whitespace from all texts."

    def _process(self, texts):
        return [text.strip() for text in texts]


NODE_CLASS_MAPPINGS = {
    "LoadImageDataSetFromFolderNode": LoadImageDataSetFromFolderNode,
    "LoadImageTextDataSetFromFolderNode": LoadImageTextDataSetFromFolderNode,
    "SaveImageDataSetToFolderNode": SaveImageDataSetToFolderNode,
    "SaveImageTextDataSetToFolderNode": SaveImageTextDataSetToFolderNode,
    # Image transforms
    "ResizeImagesToSameSizeNode": ResizeImagesToSameSizeNode,
    "ResizeImagesByShorterEdgeNode": ResizeImagesByShorterEdgeNode,
    "ResizeImagesByLongerEdgeNode": ResizeImagesByLongerEdgeNode,
    "CenterCropImagesNode": CenterCropImagesNode,
    "RandomCropImagesNode": RandomCropImagesNode,
    "FlipImagesNode": FlipImagesNode,
    "NormalizeImagesNode": NormalizeImagesNode,
    "AdjustBrightnessNode": AdjustBrightnessNode,
    "AdjustContrastNode": AdjustContrastNode,
    "ShuffleDatasetNode": ShuffleDatasetNode,
    "ShuffleImageTextDatasetNode": ShuffleImageTextDatasetNode,
    # Text transforms
    "TextToLowercaseNode": TextToLowercaseNode,
    "TextToUppercaseNode": TextToUppercaseNode,
    "TruncateTextNode": TruncateTextNode,
    "AddTextPrefixNode": AddTextPrefixNode,
    "AddTextSuffixNode": AddTextSuffixNode,
    "ReplaceTextNode": ReplaceTextNode,
    "StripWhitespaceNode": StripWhitespaceNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageDataSetFromFolderNode": "Load Simple Image Dataset from Folder",
    "LoadImageTextDataSetFromFolderNode": "Load Simple Image and Text Dataset from Folder",
    "SaveImageDataSetToFolderNode": "Save Simple Image Dataset to Folder",
    "SaveImageTextDataSetToFolderNode": "Save Simple Image and Text Dataset to Folder",
    # Image transforms
    "ResizeImagesToSameSizeNode": "Resize Images to Same Size",
    "ResizeImagesByShorterEdgeNode": "Resize Images by Shorter Edge",
    "ResizeImagesByLongerEdgeNode": "Resize Images by Longer Edge",
    "CenterCropImagesNode": "Center Crop Images",
    "RandomCropImagesNode": "Random Crop Images",
    "FlipImagesNode": "Flip Images",
    "NormalizeImagesNode": "Normalize Images",
    "AdjustBrightnessNode": "Adjust Brightness",
    "AdjustContrastNode": "Adjust Contrast",
    "ShuffleDatasetNode": "Shuffle Image Dataset",
    "ShuffleImageTextDatasetNode": "Shuffle Image-Text Dataset",
    # Text transforms
    "TextToLowercaseNode": "Text to Lowercase",
    "TextToUppercaseNode": "Text to Uppercase",
    "TruncateTextNode": "Truncate Text",
    "AddTextPrefixNode": "Add Text Prefix",
    "AddTextSuffixNode": "Add Text Suffix",
    "ReplaceTextNode": "Replace Text",
    "StripWhitespaceNode": "Strip Whitespace",
}
