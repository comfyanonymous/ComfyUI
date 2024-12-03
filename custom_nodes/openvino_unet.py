import torch

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

# Test OpenVINO Execution
from openvino.runtime import Core
from optimum.intel import OVStableDiffusionPipeline
from PIL import Image, ImageOps
from io import BytesIO
import numpy as np
import struct
import comfy.utils
import time
from pathlib import Path

MODEL_ID = "helenai/stabilityai-stable-diffusion-2-1-ov"
MODEL_DIR = Path("diffusion_pipeline")
DEVICE="CPU"

batch_size = 1
num_images_per_prompt = 1
height =512
width = 512

class OpenVINOUNetInference:
    def __init__(self):
        if not MODEL_DIR.exists():
            self.pipe = OVStableDiffusionPipeline.from_pretrained(
                MODEL_ID,
                compile=False,
                device=DEVICE,
            )
            self.pipe.save_pretrained(MODEL_DIR)
        else:
            self.pipe = OVStableDiffusionPipeline.from_pretrained(
                MODEL_ID,
                compile=False,
                device=DEVICE,
            )
        self.pipe.reshape(batch_size=batch_size, height=height, width=width, num_images_per_prompt=num_images_per_prompt)
        self.pipe.compile()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"tooltip": "The text prompt for generation."}),
                "num_inference_steps": ("INT", {"tooltip": "The number of inference steps."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_TOOLTIPS = ("The generated image from the prompt.",)
    FUNCTION = "generate_image"
    CATEGORY = "generation"

    def generate_image(self, prompt, num_inference_steps=50, guidance_scale=7.5):
        """
        Generates an image from the given prompt using the OpenVINO optimized Stable Diffusion model.
        """
        # Generate the image from the prompt using the pipeline
        #image = self.pipe(prompt, num_inference_steps=num_inference_steps).images[0]
        image = self.pipe(prompt, num_inference_steps=num_inference_steps)
        final_image = image["images"][0]        
        return (final_image,)  # Return the generated image as output


NODE_CLASS_MAPPINGS = {
    "OpenVINOUNetInference": OpenVINOUNetInference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenVINOUnetInference": "OpenVINO Inference"
}
