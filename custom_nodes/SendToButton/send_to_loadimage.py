from PIL import Image
import numpy as np
import os

import folder_paths  # ComfyUI helper for input/output dirs

class TaggedLoadImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # STRING + image_upload avoids "value not in list" and supports subfolders (e.g., pasted/...)
                "image_file": ("STRING", {
                    "default": "",
                    "image_upload": True,
                    "placeholder": "Select or upload an image (relative to input/)",
                }),
                "tag": ("STRING", {"default": "default"}),
            },
            "optional": {
                # still allow an upstream IMAGE to override file selection if connected
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"
    CATEGORY = "Load"

    def load_image(self, image_file, tag="default", image=None):
        # If an upstream IMAGE is provided, prefer it
        if image is not None:
            if isinstance(image, np.ndarray):
                mask = np.ones((image.shape[0], image.shape[1]), dtype=np.float32)
                return (image, mask)
            raise TypeError("TaggedLoadImage: unsupported IMAGE type; expected numpy array")

        # Otherwise, load from the provided path (supports subfolders like pasted/..)
        if not image_file or image_file.strip() == "":
            raise ValueError("TaggedLoadImage: no image provided. Upload/select a file or connect an IMAGE input.")

        # Resolve to full path: allow absolute paths; otherwise treat as relative to input dir
        if os.path.isabs(image_file):
            full_path = image_file
        else:
            input_dir = folder_paths.get_input_directory()
            full_path = os.path.join(input_dir, image_file)

        if not os.path.exists(full_path) or not os.path.isfile(full_path):
            raise ValueError(f"TaggedLoadImage: file not found: {full_path}")

        with Image.open(full_path) as im:
            pil_image = im.convert("RGB")
        arr = np.array(pil_image).astype(np.float32) / 255.0
        mask = np.ones((arr.shape[0], arr.shape[1]), dtype=np.float32)
        return (arr, mask)