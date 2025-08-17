import numpy as np
import time
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import io
import base64
import random
import math
import os
import re
import json
from PIL.PngImagePlugin import PngInfo
try:
    import cv2
except:
    print("OpenCV not installed")
    pass
from PIL import ImageGrab, ImageDraw, ImageFont, Image, ImageOps

from nodes import MAX_RESOLUTION, SaveImage
from comfy_extras.nodes_mask import ImageCompositeMasked
from comfy.cli_args import args
from comfy.utils import ProgressBar, common_upscale
import folder_paths
from comfy import model_management
try:
    from server import PromptServer
except:
    PromptServer = None

script_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class ImagePass:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {               
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "passthrough"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Passes the image through without modifying it.
"""

    def passthrough(self, image=None):
        return image,

class ColorMatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_ref": ("IMAGE",),
                "image_target": ("IMAGE",),
                "method": (
            [   
                'mkl',
                'hm', 
                'reinhard', 
                'mvgd', 
                'hm-mvgd-hm', 
                'hm-mkl-hm',
            ], {
               "default": 'mkl'
            }),
            },
            "optional": {
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            }
        }
    
    CATEGORY = "KJNodes/image"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "colormatch"
    DESCRIPTION = """
color-matcher enables color transfer across images which comes in handy for automatic  
color-grading of photographs, paintings and film sequences as well as light-field  
and stopmotion corrections.  

The methods behind the mappings are based on the approach from Reinhard et al.,  
the Monge-Kantorovich Linearization (MKL) as proposed by Pitie et al. and our analytical solution  
to a Multi-Variate Gaussian Distribution (MVGD) transfer in conjunction with classical histogram   
matching. As shown below our HM-MVGD-HM compound outperforms existing methods.   
https://github.com/hahnec/color-matcher/

"""
    
    def colormatch(self, image_ref, image_target, method, strength=1.0):
        try:
            from color_matcher import ColorMatcher
        except:
            raise Exception("Can't import color-matcher, did you install requirements.txt? Manual install: pip install color-matcher")
        cm = ColorMatcher()
        image_ref = image_ref.cpu()
        image_target = image_target.cpu()
        batch_size = image_target.size(0)
        out = []
        images_target = image_target.squeeze()
        images_ref = image_ref.squeeze()

        image_ref_np = images_ref.numpy()
        images_target_np = images_target.numpy()

        if image_ref.size(0) > 1 and image_ref.size(0) != batch_size:
            raise ValueError("ColorMatch: Use either single reference image or a matching batch of reference images.")

        for i in range(batch_size):
            image_target_np = images_target_np if batch_size == 1 else images_target[i].numpy()
            image_ref_np_i = image_ref_np if image_ref.size(0) == 1 else images_ref[i].numpy()
            try:
                image_result = cm.transfer(src=image_target_np, ref=image_ref_np_i, method=method)
            except BaseException as e:
                print(f"Error occurred during transfer: {e}")
                break
            # Apply the strength multiplier
            image_result = image_target_np + strength * (image_result - image_target_np)
            out.append(torch.from_numpy(image_result))
            
        out = torch.stack(out, dim=0).to(torch.float32)
        out.clamp_(0, 1)
        return (out,)
    
class SaveImageWithAlpha:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"images": ("IMAGE", ),
                    "mask": ("MASK", ),
                    "filename_prefix": ("STRING", {"default": "ComfyUI"})},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_images_alpha"
    OUTPUT_NODE = True
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Saves an image and mask as .PNG with the mask as the alpha channel. 
"""

    def save_images_alpha(self, images, mask, filename_prefix="ComfyUI_image_with_alpha", prompt=None, extra_pnginfo=None):
        from PIL.PngImagePlugin import PngInfo
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        if mask.dtype == torch.float16:
            mask = mask.to(torch.float32)
        def file_counter():
            max_counter = 0
            # Loop through the existing files
            for existing_file in os.listdir(full_output_folder):
                # Check if the file matches the expected format
                match = re.fullmatch(fr"{filename}_(\d+)_?\.[a-zA-Z0-9]+", existing_file)
                if match:
                    # Extract the numeric portion of the filename
                    file_counter = int(match.group(1))
                    # Update the maximum counter value if necessary
                    if file_counter > max_counter:
                        max_counter = file_counter
            return max_counter

        for image, alpha in zip(images, mask):
            i = 255. * image.cpu().numpy()
            a = 255. * alpha.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
             # Resize the mask to match the image size
            a_resized = Image.fromarray(a).resize(img.size, Image.LANCZOS)
            a_resized = np.clip(a_resized, 0, 255).astype(np.uint8)
            img.putalpha(Image.fromarray(a_resized, mode='L'))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))
           
            # Increment the counter by 1 to get the next available value
            counter = file_counter() + 1
            file = f"{filename}_{counter:05}.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=4)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })

        return { "ui": { "images": results } }

class ImageConcanate:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image1": ("IMAGE",),
            "image2": ("IMAGE",),
            "direction": (
            [   'right',
                'down',
                'left',
                'up',
            ],
            {
            "default": 'right'
             }),
            "match_image_size": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "concatenate"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Concatenates the image2 to image1 in the specified direction.
"""

    def concatenate(self, image1, image2, direction, match_image_size, first_image_shape=None):
        # Check if the batch sizes are different
        batch_size1 = image1.shape[0]
        batch_size2 = image2.shape[0]

        if batch_size1 != batch_size2:
            # Calculate the number of repetitions needed
            max_batch_size = max(batch_size1, batch_size2)
            repeats1 = max_batch_size - batch_size1
            repeats2 = max_batch_size - batch_size2
            
            # Repeat the last image to match the largest batch size
            if repeats1 > 0:
                last_image1 = image1[-1].unsqueeze(0).repeat(repeats1, 1, 1, 1)
                image1 = torch.cat([image1.clone(), last_image1], dim=0)
            if repeats2 > 0:
                last_image2 = image2[-1].unsqueeze(0).repeat(repeats2, 1, 1, 1)
                image2 = torch.cat([image2.clone(), last_image2], dim=0)

        if match_image_size:
            # Use first_image_shape if provided; otherwise, default to image1's shape
            target_shape = first_image_shape if first_image_shape is not None else image1.shape

            original_height = image2.shape[1]
            original_width = image2.shape[2]
            original_aspect_ratio = original_width / original_height

            if direction in ['left', 'right']:
                # Match the height and adjust the width to preserve aspect ratio
                target_height = target_shape[1]  # B, H, W, C format
                target_width = int(target_height * original_aspect_ratio)
            elif direction in ['up', 'down']:
                # Match the width and adjust the height to preserve aspect ratio
                target_width = target_shape[2]  # B, H, W, C format
                target_height = int(target_width / original_aspect_ratio)
            
            # Adjust image2 to the expected format for common_upscale
            image2_for_upscale = image2.movedim(-1, 1)  # Move C to the second position (B, C, H, W)
            
            # Resize image2 to match the target size while preserving aspect ratio
            image2_resized = common_upscale(image2_for_upscale, target_width, target_height, "lanczos", "disabled")
            
            # Adjust image2 back to the original format (B, H, W, C) after resizing
            image2_resized = image2_resized.movedim(1, -1)
        else:
            image2_resized = image2

        # Ensure both images have the same number of channels
        channels_image1 = image1.shape[-1]
        channels_image2 = image2_resized.shape[-1]

        if channels_image1 != channels_image2:
            if channels_image1 < channels_image2:
                # Add alpha channel to image1 if image2 has it
                alpha_channel = torch.ones((*image1.shape[:-1], channels_image2 - channels_image1), device=image1.device)
                image1 = torch.cat((image1, alpha_channel), dim=-1)
            else:
                # Add alpha channel to image2 if image1 has it
                alpha_channel = torch.ones((*image2_resized.shape[:-1], channels_image1 - channels_image2), device=image2_resized.device)
                image2_resized = torch.cat((image2_resized, alpha_channel), dim=-1)


        # Concatenate based on the specified direction
        if direction == 'right':
            concatenated_image = torch.cat((image1, image2_resized), dim=2)  # Concatenate along width
        elif direction == 'down':
            concatenated_image = torch.cat((image1, image2_resized), dim=1)  # Concatenate along height
        elif direction == 'left':
            concatenated_image = torch.cat((image2_resized, image1), dim=2)  # Concatenate along width
        elif direction == 'up':
            concatenated_image = torch.cat((image2_resized, image1), dim=1)  # Concatenate along height
        return concatenated_image,

import torch  # Make sure you have PyTorch installed

class ImageConcatFromBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "images": ("IMAGE",),
            "num_columns": ("INT", {"default": 3, "min": 1, "max": 255, "step": 1}),
            "match_image_size": ("BOOLEAN", {"default": False}),
            "max_resolution": ("INT", {"default": 4096}), 
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "concat"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
    Concatenates images from a batch into a grid with a specified number of columns.
    """

    def concat(self, images, num_columns, match_image_size, max_resolution):
        # Assuming images is a batch of images (B, H, W, C)
        batch_size, height, width, channels = images.shape
        num_rows = (batch_size + num_columns - 1) // num_columns  # Calculate number of rows

        print(f"Initial dimensions: batch_size={batch_size}, height={height}, width={width}, channels={channels}")
        print(f"num_rows={num_rows}, num_columns={num_columns}")

        if match_image_size:
            target_shape = images[0].shape

            resized_images = []
            for image in images:
                original_height = image.shape[0]
                original_width = image.shape[1]
                original_aspect_ratio = original_width / original_height

                if original_aspect_ratio > 1:
                    target_height = target_shape[0]
                    target_width = int(target_height * original_aspect_ratio)
                else:
                    target_width = target_shape[1]
                    target_height = int(target_width / original_aspect_ratio)

                print(f"Resizing image from ({original_height}, {original_width}) to ({target_height}, {target_width})")

                # Resize the image to match the target size while preserving aspect ratio
                resized_image = common_upscale(image.movedim(-1, 0), target_width, target_height, "lanczos", "disabled")
                resized_image = resized_image.movedim(0, -1)  # Move channels back to the last dimension
                resized_images.append(resized_image)

            # Convert the list of resized images back to a tensor
            images = torch.stack(resized_images)

            height, width = target_shape[:2]  # Update height and width

        # Initialize an empty grid
        grid_height = num_rows * height
        grid_width = num_columns * width

        print(f"Grid dimensions before scaling: grid_height={grid_height}, grid_width={grid_width}")

        # Original scale factor calculation remains unchanged
        scale_factor = min(max_resolution / grid_height, max_resolution / grid_width, 1.0)

        # Apply scale factor to height and width
        scaled_height = height * scale_factor
        scaled_width = width * scale_factor

        # Round scaled dimensions to the nearest number divisible by 8
        height = max(1, int(round(scaled_height / 8) * 8))
        width = max(1, int(round(scaled_width / 8) * 8))

        if abs(scaled_height - height) > 4:
            height = max(1, int(round((scaled_height + 4) / 8) * 8))
        if abs(scaled_width - width) > 4:
            width = max(1, int(round((scaled_width + 4) / 8) * 8))

        # Recalculate grid dimensions with adjusted height and width
        grid_height = num_rows * height
        grid_width = num_columns * width
        print(f"Grid dimensions after scaling: grid_height={grid_height}, grid_width={grid_width}")
        print(f"Final image dimensions: height={height}, width={width}")

        grid = torch.zeros((grid_height, grid_width, channels), dtype=images.dtype)

        for idx, image in enumerate(images):
            resized_image = torch.nn.functional.interpolate(image.unsqueeze(0).permute(0, 3, 1, 2), size=(height, width), mode="bilinear").squeeze().permute(1, 2, 0)
            row = idx // num_columns
            col = idx % num_columns
            grid[row*height:(row+1)*height, col*width:(col+1)*width, :] = resized_image

        return grid.unsqueeze(0),

    
class ImageGridComposite2x2:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image1": ("IMAGE",),
            "image2": ("IMAGE",),
            "image3": ("IMAGE",),
            "image4": ("IMAGE",),   
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "compositegrid"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Concatenates the 4 input images into a 2x2 grid. 
"""

    def compositegrid(self, image1, image2, image3, image4):
        top_row = torch.cat((image1, image2), dim=2)
        bottom_row = torch.cat((image3, image4), dim=2)
        grid = torch.cat((top_row, bottom_row), dim=1)
        return (grid,)
    
class ImageGridComposite3x3:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image1": ("IMAGE",),
            "image2": ("IMAGE",),
            "image3": ("IMAGE",),
            "image4": ("IMAGE",),
            "image5": ("IMAGE",),
            "image6": ("IMAGE",),
            "image7": ("IMAGE",),
            "image8": ("IMAGE",),
            "image9": ("IMAGE",),     
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "compositegrid"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Concatenates the 9 input images into a 3x3 grid. 
"""

    def compositegrid(self, image1, image2, image3, image4, image5, image6, image7, image8, image9):
        top_row = torch.cat((image1, image2, image3), dim=2)
        mid_row = torch.cat((image4, image5, image6), dim=2)
        bottom_row = torch.cat((image7, image8, image9), dim=2)
        grid = torch.cat((top_row, mid_row, bottom_row), dim=1)
        return (grid,)
    
class ImageBatchTestPattern:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "batch_size": ("INT", {"default": 1,"min": 1, "max": 255, "step": 1}),
            "start_from": ("INT", {"default": 0,"min": 0, "max": 255, "step": 1}),
            "text_x": ("INT", {"default": 256,"min": 0, "max": 4096, "step": 1}),
            "text_y": ("INT", {"default": 256,"min": 0, "max": 4096, "step": 1}),
            "width": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
            "height": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
            "font": (folder_paths.get_filename_list("kjnodes_fonts"), ),
            "font_size": ("INT", {"default": 255,"min": 8, "max": 4096, "step": 1}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generatetestpattern"
    CATEGORY = "KJNodes/text"

    def generatetestpattern(self, batch_size, font, font_size, start_from, width, height, text_x, text_y):
        out = []
        # Generate the sequential numbers for each image
        numbers = np.arange(start_from, start_from + batch_size)
        font_path = folder_paths.get_full_path("kjnodes_fonts", font)

        for number in numbers:
            # Create a black image with the number as a random color text
            image = Image.new("RGB", (width, height), color='black')
            draw = ImageDraw.Draw(image)
            
            # Generate a random color for the text
            font_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            
            font = ImageFont.truetype(font_path, font_size)
            
            # Get the size of the text and position it in the center
            text = str(number)
           
            try:
                draw.text((text_x, text_y), text, font=font, fill=font_color, features=['-liga'])
            except:
                draw.text((text_x, text_y), text, font=font, fill=font_color,)
            
            # Convert the image to a numpy array and normalize the pixel values
            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np).unsqueeze(0)
            out.append(image_tensor)
        out_tensor = torch.cat(out, dim=0)
  
        return (out_tensor,)

class ImageGrabPIL:

    @classmethod
    def IS_CHANGED(cls):

        return

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "screencap"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Captures an area specified by screen coordinates.  
Can be used for realtime diffusion with autoqueue.
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "x": ("INT", {"default": 0,"min": 0, "max": 4096, "step": 1}),
                 "y": ("INT", {"default": 0,"min": 0, "max": 4096, "step": 1}),
                 "width": ("INT", {"default": 512,"min": 0, "max": 4096, "step": 1}),
                 "height": ("INT", {"default": 512,"min": 0, "max": 4096, "step": 1}),
                 "num_frames": ("INT", {"default": 1,"min": 1, "max": 255, "step": 1}),
                 "delay": ("FLOAT", {"default": 0.1,"min": 0.0, "max": 10.0, "step": 0.01}),
        },
    } 

    def screencap(self, x, y, width, height, num_frames, delay):
        start_time = time.time()
        captures = []
        bbox = (x, y, x + width, y + height)
        
        for _ in range(num_frames):
            # Capture screen
            screen_capture = ImageGrab.grab(bbox=bbox)
            screen_capture_torch = torch.from_numpy(np.array(screen_capture, dtype=np.float32) / 255.0).unsqueeze(0)
            captures.append(screen_capture_torch)
            
            # Wait for a short delay if more than one frame is to be captured
            if num_frames > 1:
                time.sleep(delay)

        elapsed_time = time.time() - start_time
        print(f"screengrab took {elapsed_time} seconds.")
        
        return (torch.cat(captures, dim=0),)
    
class Screencap_mss:

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("NaN")

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "screencap"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Captures an area specified by screen coordinates.  
Can be used for realtime diffusion with autoqueue.
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "x": ("INT", {"default": 0,"min": 0, "max": 10000, "step": 1}),
                 "y": ("INT", {"default": 0,"min": 0, "max": 10000, "step": 1}),
                 "width": ("INT", {"default": 512,"min": 0, "max": 10000, "step": 1}),
                 "height": ("INT", {"default": 512,"min": 0, "max": 10000, "step": 1}),
                 "num_frames": ("INT", {"default": 1,"min": 1, "max": 255, "step": 1}),
                 "delay": ("FLOAT", {"default": 0.1,"min": 0.0, "max": 10.0, "step": 0.01}),
        },
    } 

    def screencap(self, x, y, width, height, num_frames, delay):
        from mss import mss
        captures = []
        with mss() as sct:
            bbox = {'top': y, 'left': x, 'width': width, 'height': height}
            
            for _ in range(num_frames):
                sct_img = sct.grab(bbox)
                img_np = np.array(sct_img)
                img_torch = torch.from_numpy(img_np[..., [2, 1, 0]]).float() / 255.0
                captures.append(img_torch)
                
                if num_frames > 1:
                    time.sleep(delay)
        
        return (torch.stack(captures, 0),)
    
class WebcamCaptureCV2:

    @classmethod
    def IS_CHANGED(cls):
        return

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "capture"
    CATEGORY = "KJNodes/experimental"
    DESCRIPTION = """
Captures a frame from a webcam using CV2.  
Can be used for realtime diffusion with autoqueue.
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "x": ("INT", {"default": 0,"min": 0, "max": 4096, "step": 1}),
                 "y": ("INT", {"default": 0,"min": 0, "max": 4096, "step": 1}),
                 "width": ("INT", {"default": 512,"min": 0, "max": 4096, "step": 1}),
                 "height": ("INT", {"default": 512,"min": 0, "max": 4096, "step": 1}),
                 "cam_index": ("INT", {"default": 0,"min": 0, "max": 255, "step": 1}),
                 "release": ("BOOLEAN", {"default": False}),
            },
        } 

    def capture(self, x, y, cam_index, width, height, release):
        # Check if the camera index has changed or the capture object doesn't exist
        if not hasattr(self, "cap") or self.cap is None or self.current_cam_index != cam_index:
            if hasattr(self, "cap") and self.cap is not None:
                self.cap.release()
            self.current_cam_index = cam_index
            self.cap = cv2.VideoCapture(cam_index)
            try:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            except:
                pass
            if not self.cap.isOpened():
                raise Exception("Could not open webcam")
    
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Failed to capture image from webcam")
    
        # Crop the frame to the specified bbox
        frame = frame[y:y+height, x:x+width]
        img_torch = torch.from_numpy(frame[..., [2, 1, 0]]).float() / 255.0
    
        if release:
            self.cap.release()
            self.cap = None
    
        return (img_torch.unsqueeze(0),)
    
class AddLabel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image":("IMAGE",),  
            "text_x": ("INT", {"default": 10, "min": 0, "max": 4096, "step": 1}),
            "text_y": ("INT", {"default": 2, "min": 0, "max": 4096, "step": 1}),
            "height": ("INT", {"default": 48, "min": -1, "max": 4096, "step": 1}),
            "font_size": ("INT", {"default": 32, "min": 0, "max": 4096, "step": 1}),
            "font_color": ("STRING", {"default": "white"}),
            "label_color": ("STRING", {"default": "black"}),
            "font": (folder_paths.get_filename_list("kjnodes_fonts"), ),
            "text": ("STRING", {"default": "Text"}),
            "direction": (
            [   'up',
                'down',
                'left',
                'right',
                'overlay'
            ],
            {
            "default": 'up'
             }),
            },
            "optional":{
                "caption": ("STRING", {"default": "", "forceInput": True}),
            }
            }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "addlabel"
    CATEGORY = "KJNodes/text"
    DESCRIPTION = """
Creates a new with the given text, and concatenates it to  
either above or below the input image.  
Note that this changes the input image's height!  
Fonts are loaded from this folder:  
ComfyUI/custom_nodes/ComfyUI-KJNodes/fonts
"""
        
    def addlabel(self, image, text_x, text_y, text, height, font_size, font_color, label_color, font, direction, caption=""):
        batch_size = image.shape[0]
        width = image.shape[2]
        
        font_path = os.path.join(script_directory, "fonts", "TTNorms-Black.otf") if font == "TTNorms-Black.otf" else folder_paths.get_full_path("kjnodes_fonts", font)
        
        def process_image(input_image, caption_text):
            font = ImageFont.truetype(font_path, font_size)
            words = caption_text.split()
            lines = []
            current_line = []
            current_line_width = 0

            for word in words:
                word_width = font.getbbox(word)[2]
                if current_line_width + word_width <= width - 2 * text_x:
                    current_line.append(word)
                    current_line_width += word_width + font.getbbox(" ")[2]  # Add space width
                else:
                    lines.append(" ".join(current_line))
                    current_line = [word]
                    current_line_width = word_width

            if current_line:
                lines.append(" ".join(current_line))

            if direction == 'overlay':
                pil_image = Image.fromarray((input_image.cpu().numpy() * 255).astype(np.uint8))
            else:
                if height == -1:
                    # Adjust the image height automatically
                    margin = 8
                    required_height = (text_y + len(lines) * font_size) + margin # Calculate required height
                    pil_image = Image.new("RGB", (width, required_height), label_color)
                else:
                    # Initialize with a minimal height
                    label_image = Image.new("RGB", (width, height), label_color)
                    pil_image = label_image

            draw = ImageDraw.Draw(pil_image)
            

            y_offset = text_y
            for line in lines:
                try:
                    draw.text((text_x, y_offset), line, font=font, fill=font_color, features=['-liga'])
                except:
                    draw.text((text_x, y_offset), line, font=font, fill=font_color)
                y_offset += font_size

            processed_image = torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0).unsqueeze(0)
            return processed_image
        
        if caption == "":
            processed_images = [process_image(img, text) for img in image]
        else:
            assert len(caption) == batch_size, f"Number of captions {(len(caption))} does not match number of images"
            processed_images = [process_image(img, cap) for img, cap in zip(image, caption)]
        processed_batch = torch.cat(processed_images, dim=0)

        # Combine images based on direction
        if direction == 'down':
            combined_images = torch.cat((image, processed_batch), dim=1)
        elif direction == 'up':
            combined_images = torch.cat((processed_batch, image), dim=1)
        elif direction == 'left':
            processed_batch = torch.rot90(processed_batch, 3, (2, 3)).permute(0, 3, 1, 2)
            combined_images = torch.cat((processed_batch, image), dim=2)
        elif direction == 'right':
            processed_batch = torch.rot90(processed_batch, 3, (2, 3)).permute(0, 3, 1, 2)
            combined_images = torch.cat((image, processed_batch), dim=2)
        else:
            combined_images = processed_batch
        
        return (combined_images,)
    
class GetImageSizeAndCount:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
        }}

    RETURN_TYPES = ("IMAGE","INT", "INT", "INT",)
    RETURN_NAMES = ("image", "width", "height", "count",)
    FUNCTION = "getsize"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Returns width, height and batch size of the image,  
and passes it through unchanged.  

"""

    def getsize(self, image):
        width = image.shape[2]
        height = image.shape[1]
        count = image.shape[0]
        return {"ui": {
            "text": [f"{count}x{width}x{height}"]}, 
            "result": (image, width, height, count) 
        }
    
class ImageBatchRepeatInterleaving:
    
    RETURN_TYPES = ("IMAGE", "MASK",)
    FUNCTION = "repeat"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Repeats each image in a batch by the specified number of times.  
Example batch of 5 images: 0, 1 ,2, 3, 4  
with repeats 2 becomes batch of 10 images: 0, 0, 1, 1, 2, 2, 3, 3, 4, 4  
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "images": ("IMAGE",),
                 "repeats": ("INT", {"default": 1, "min": 1, "max": 4096}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }
    
    def repeat(self, images, repeats, mask=None):
        original_count = images.shape[0]
        total_count = original_count * repeats
       
        repeated_images = torch.repeat_interleave(images, repeats=repeats, dim=0)
        if mask is not None:
            mask = torch.repeat_interleave(mask, repeats=repeats, dim=0)
        else:
            mask = torch.zeros((total_count, images.shape[1], images.shape[2]), 
                              device=images.device, dtype=images.dtype)
            for i in range(original_count):
                mask[i * repeats] = 1.0

        print("mask shape", mask.shape)
        return (repeated_images, mask)
    
class ImageUpscaleWithModelBatched:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "upscale_model": ("UPSCALE_MODEL",),
                              "images": ("IMAGE",),
                              "per_batch": ("INT", {"default": 16, "min": 1, "max": 4096, "step": 1}),
                              }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Same as ComfyUI native model upscaling node,  
but allows setting sub-batches for reduced VRAM usage.
"""
    def upscale(self, upscale_model, images, per_batch):
        
        device = model_management.get_torch_device()
        upscale_model.to(device)
        in_img = images.movedim(-1,-3)
        
        steps = in_img.shape[0]
        pbar = ProgressBar(steps)
        t = []
        
        for start_idx in range(0, in_img.shape[0], per_batch):
            sub_images = upscale_model(in_img[start_idx:start_idx+per_batch].to(device))
            t.append(sub_images.cpu())
            # Calculate the number of images processed in this batch
            batch_count = sub_images.shape[0]
            # Update the progress bar by the number of images processed in this batch
            pbar.update(batch_count)
        upscale_model.cpu()
        
        t = torch.cat(t, dim=0).permute(0, 2, 3, 1).cpu()

        return (t,)

class ImageNormalize_Neg1_To_1:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                              "images": ("IMAGE",),
    
                              }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "normalize"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Normalize the images to be in the range [-1, 1]  
"""

    def normalize(self,images):
        images = images * 2.0 - 1.0
        return (images,)

class RemapImageRange:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "image": ("IMAGE",),
            "min": ("FLOAT", {"default": 0.0,"min": -10.0, "max": 1.0, "step": 0.01}),
            "max": ("FLOAT", {"default": 1.0,"min": 0.0, "max": 10.0, "step": 0.01}),
            "clamp": ("BOOLEAN", {"default": True}),
            },
            }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "remap"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Remaps the image values to the specified range. 
"""
        
    def remap(self, image, min, max, clamp):
        if image.dtype == torch.float16:
            image = image.to(torch.float32)
        image = min + image * (max - min)
        if clamp:
            image = torch.clamp(image, min=0.0, max=1.0)
        return (image, )

class SplitImageChannels:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "image": ("IMAGE",),
            },
            }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("red", "green", "blue", "mask")
    FUNCTION = "split"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Splits image channels into images where the selected channel  
is repeated for all channels, and the alpha as a mask. 
"""
        
    def split(self, image):
        red = image[:, :, :, 0:1] # Red channel
        green = image[:, :, :, 1:2] # Green channel
        blue = image[:, :, :, 2:3] # Blue channel
        alpha = image[:, :, :, 3:4] # Alpha channel
        alpha = alpha.squeeze(-1)

        # Repeat the selected channel for all channels
        red = torch.cat([red, red, red], dim=3)
        green = torch.cat([green, green, green], dim=3)
        blue = torch.cat([blue, blue, blue], dim=3)
        return (red, green, blue, alpha)
    
class MergeImageChannels:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "red": ("IMAGE",),
            "green": ("IMAGE",),
            "blue": ("IMAGE",),
            
            },
            "optional": {
                "alpha": ("MASK", {"default": None}),
                },
            }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "merge"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Merges channel data into an image.  
"""
        
    def merge(self, red, green, blue, alpha=None):
        image = torch.stack([
        red[..., 0, None], # Red channel
        green[..., 1, None], # Green channel
        blue[..., 2, None]   # Blue channel
        ], dim=-1)
        image = image.squeeze(-2)
        if alpha is not None:
            image = torch.cat([image, alpha.unsqueeze(-1)], dim=-1)
        return (image,)

class ImagePadForOutpaintMasked:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "left": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "top": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "right": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "bottom": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "feathering": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "expand_image"

    CATEGORY = "image"

    def expand_image(self, image, left, top, right, bottom, feathering, mask=None):
        if mask is not None:
            if torch.allclose(mask, torch.zeros_like(mask)):
                    print("Warning: The incoming mask is fully black. Handling it as None.")
                    mask = None
        B, H, W, C = image.size()

        new_image = torch.ones(
            (B, H + top + bottom, W + left + right, C),
            dtype=torch.float32,
        ) * 0.5

        new_image[:, top:top + H, left:left + W, :] = image

        if mask is None:
            new_mask = torch.ones(
                (B, H + top + bottom, W + left + right),
                dtype=torch.float32,
            )

            t = torch.zeros(
            (B, H, W),
            dtype=torch.float32
            )
        else:
            # If a mask is provided, pad it to fit the new image size
            mask = F.pad(mask, (left, right, top, bottom), mode='constant', value=0)
            mask = 1 - mask
            t = torch.zeros_like(mask)
        
        if feathering > 0 and feathering * 2 < H and feathering * 2 < W:

            for i in range(H):
                for j in range(W):
                    dt = i if top != 0 else H
                    db = H - i if bottom != 0 else H

                    dl = j if left != 0 else W
                    dr = W - j if right != 0 else W

                    d = min(dt, db, dl, dr)

                    if d >= feathering:
                        continue

                    v = (feathering - d) / feathering

                    if mask is None:
                        t[:, i, j] = v * v
                    else:
                        t[:, top + i, left + j] = v * v
        
        if mask is None:
            new_mask[:, top:top + H, left:left + W] = t
            return (new_image, new_mask,)
        else:
            return (new_image, mask,)

class ImagePadForOutpaintTargetSize:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_width": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "target_height": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "feathering": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "upscale_method": (s.upscale_methods,),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "expand_image"

    CATEGORY = "image"

    def expand_image(self, image, target_width, target_height, feathering, upscale_method, mask=None):
        B, H, W, C = image.size()
        new_height = H
        new_width = W
         # Calculate the scaling factor while maintaining aspect ratio
        scaling_factor = min(target_width / W, target_height / H)
        
        # Check if the image needs to be downscaled
        if scaling_factor < 1:
            image = image.movedim(-1,1)
            # Calculate the new width and height after downscaling
            new_width = int(W * scaling_factor)
            new_height = int(H * scaling_factor)
            
            # Downscale the image
            image_scaled = common_upscale(image, new_width, new_height, upscale_method, "disabled").movedim(1,-1)
            if mask is not None:
                mask_scaled = mask.unsqueeze(0)  # Add an extra dimension for batch size
                mask_scaled = F.interpolate(mask_scaled, size=(new_height, new_width), mode="nearest")
                mask_scaled = mask_scaled.squeeze(0)  # Remove the extra dimension after interpolation
            else:
                mask_scaled = mask
        else:
            # If downscaling is not needed, use the original image dimensions
            image_scaled = image
            mask_scaled = mask

        # Calculate how much padding is needed to reach the target dimensions
        pad_top = max(0, (target_height - new_height) // 2)
        pad_bottom = max(0, target_height - new_height - pad_top)
        pad_left = max(0, (target_width - new_width) // 2)
        pad_right = max(0, target_width - new_width - pad_left)

        # Now call the original expand_image with the calculated padding
        return ImagePadForOutpaintMasked.expand_image(self, image_scaled, pad_left, pad_top, pad_right, pad_bottom, feathering, mask_scaled)

class ImagePrepForICLora:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "reference_image": ("IMAGE",),
                "output_width": ("INT", {"default": 1024, "min": 1, "max": 4096, "step": 1}),
                "output_height": ("INT", {"default": 1024, "min": 1, "max": 4096, "step": 1}),
                "border_width": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            },
            "optional": {
                "latent_image": ("IMAGE",),
                "latent_mask": ("MASK",),
                "reference_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "expand_image"

    CATEGORY = "image"

    def expand_image(self, reference_image, output_width, output_height, border_width, latent_image=None, reference_mask=None, latent_mask=None):

        if reference_mask is not None:
            if torch.allclose(reference_mask, torch.zeros_like(reference_mask)):
                    print("Warning: The incoming mask is fully black. Handling it as None.")
                    reference_mask = None
        image = reference_image
        if latent_image is not None:
            if image.shape[0] != latent_image.shape[0]:
                image = image.repeat(latent_image.shape[0], 1, 1, 1)
        B, H, W, C = image.size()

        # Handle mask
        if reference_mask is not None:
            resized_mask = torch.nn.functional.interpolate(
                reference_mask.unsqueeze(1), 
                size=(H, W),
                mode='nearest'
            ).squeeze(1)
            print(resized_mask.shape)
            image = image * resized_mask.unsqueeze(-1)

        # Calculate new width maintaining aspect ratio
        new_width = int((W / H) * output_height)
        
        # Resize image to new height while maintaining aspect ratio
        resized_image = common_upscale(image.movedim(-1,1), new_width, output_height, "lanczos", "disabled").movedim(1,-1)

        # Create padded image
        if latent_image is None:
            pad_image = torch.zeros((B, output_height, output_width, C), device=image.device)
        else:
            resized_latent_image = common_upscale(latent_image.movedim(-1,1), output_width, output_height, "lanczos", "disabled").movedim(1,-1)
            pad_image = resized_latent_image
            if latent_mask is not None:
                resized_latent_mask = torch.nn.functional.interpolate(
                    latent_mask.unsqueeze(1), 
                    size=(pad_image.shape[1], pad_image.shape[2]), 
                    mode='nearest'
                ).squeeze(1)

        if border_width > 0:
            border = torch.zeros((B, output_height, border_width, C), device=image.device)
            padded_image = torch.cat((resized_image, border, pad_image), dim=2)
            if latent_mask is not None:
                padded_mask = torch.zeros((B, padded_image.shape[1], padded_image.shape[2]), device=image.device)
                padded_mask[:, :, (new_width + border_width):] = resized_latent_mask
            else:
                padded_mask = torch.ones((B, padded_image.shape[1], padded_image.shape[2]), device=image.device)
                padded_mask[:, :, :new_width + border_width] = 0
        else:
            padded_image = torch.cat((resized_image, pad_image), dim=2)
            if latent_mask is not None:
                padded_mask = torch.zeros((B, padded_image.shape[1], padded_image.shape[2]), device=image.device)
                padded_mask[:, :, new_width:] = resized_latent_mask
            else:
                padded_mask = torch.ones((B, padded_image.shape[1], padded_image.shape[2]), device=image.device)
                padded_mask[:, :, :new_width] = 0

        return (padded_image, padded_mask)
        

class ImageAndMaskPreview(SaveImage):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mask_color": ("STRING", {"default": "255, 255, 255"}),
                "pass_through": ("BOOLEAN", {"default": False}),
             },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),                
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("composite",)
    FUNCTION = "execute"
    CATEGORY = "KJNodes/masking"
    DESCRIPTION = """
Preview an image or a mask, when both inputs are used  
composites the mask on top of the image.
with pass_through on the preview is disabled and the  
composite is returned from the composite slot instead,  
this allows for the preview to be passed for video combine  
nodes for example.
"""

    def execute(self, mask_opacity, mask_color, pass_through, filename_prefix="ComfyUI", image=None, mask=None, prompt=None, extra_pnginfo=None):
        if mask is not None and image is None:
            preview = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
        elif mask is None and image is not None:
            preview = image
        elif mask is not None and image is not None:
            mask_adjusted = mask * mask_opacity
            mask_image = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3).clone()

            if ',' in mask_color:
                color_list = np.clip([int(channel) for channel in mask_color.split(',')], 0, 255) # RGB format
            else:
                mask_color = mask_color.lstrip('#')
                color_list = [int(mask_color[i:i+2], 16) for i in (0, 2, 4)] # Hex format
            mask_image[:, :, :, 0] = color_list[0] / 255 # Red channel
            mask_image[:, :, :, 1] = color_list[1] / 255 # Green channel
            mask_image[:, :, :, 2] = color_list[2] / 255 # Blue channel
            
            preview, = ImageCompositeMasked.composite(self, image, mask_image, 0, 0, True, mask_adjusted)
        if pass_through:
            return (preview, )
        return(self.save_images(preview, filename_prefix, prompt, extra_pnginfo))
        
class CrossFadeImages:
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crossfadeimages"
    CATEGORY = "KJNodes/image"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "images_1": ("IMAGE",),
                 "images_2": ("IMAGE",),
                 "interpolation": (["linear", "ease_in", "ease_out", "ease_in_out", "bounce", "elastic", "glitchy", "exponential_ease_out"],),
                 "transition_start_index": ("INT", {"default": 1,"min": 0, "max": 4096, "step": 1}),
                 "transitioning_frames": ("INT", {"default": 1,"min": 0, "max": 4096, "step": 1}),
                 "start_level": ("FLOAT", {"default": 0.0,"min": 0.0, "max": 1.0, "step": 0.01}),
                 "end_level": ("FLOAT", {"default": 1.0,"min": 0.0, "max": 1.0, "step": 0.01}),
        },
    } 
    
    def crossfadeimages(self, images_1, images_2, transition_start_index, transitioning_frames, interpolation, start_level, end_level):

        def crossfade(images_1, images_2, alpha):
            crossfade = (1 - alpha) * images_1 + alpha * images_2
            return crossfade
        def ease_in(t):
            return t * t
        def ease_out(t):
            return 1 - (1 - t) * (1 - t)
        def ease_in_out(t):
            return 3 * t * t - 2 * t * t * t
        def bounce(t):
            if t < 0.5:
                return self.ease_out(t * 2) * 0.5
            else:
                return self.ease_in((t - 0.5) * 2) * 0.5 + 0.5
        def elastic(t):
            return math.sin(13 * math.pi / 2 * t) * math.pow(2, 10 * (t - 1))
        def glitchy(t):
            return t + 0.1 * math.sin(40 * t)
        def exponential_ease_out(t):
            return 1 - (1 - t) ** 4

        easing_functions = {
            "linear": lambda t: t,
            "ease_in": ease_in,
            "ease_out": ease_out,
            "ease_in_out": ease_in_out,
            "bounce": bounce,
            "elastic": elastic,
            "glitchy": glitchy,
            "exponential_ease_out": exponential_ease_out,
        }

        crossfade_images = []

        alphas = torch.linspace(start_level, end_level, transitioning_frames)
        for i in range(transitioning_frames):
            alpha = alphas[i]
            image1 = images_1[i + transition_start_index]
            image2 = images_2[i + transition_start_index]
            easing_function = easing_functions.get(interpolation)
            alpha = easing_function(alpha)  # Apply the easing function to the alpha value

            crossfade_image = crossfade(image1, image2, alpha)
            crossfade_images.append(crossfade_image)
            
        # Convert crossfade_images to tensor
        crossfade_images = torch.stack(crossfade_images, dim=0)
        # Get the last frame result of the interpolation
        last_frame = crossfade_images[-1]
        # Calculate the number of remaining frames from images_2
        remaining_frames = len(images_2) - (transition_start_index + transitioning_frames)
        # Crossfade the remaining frames with the last used alpha value
        for i in range(remaining_frames):
            alpha = alphas[-1]
            image1 = images_1[i + transition_start_index + transitioning_frames]
            image2 = images_2[i + transition_start_index + transitioning_frames]
            easing_function = easing_functions.get(interpolation)
            alpha = easing_function(alpha)  # Apply the easing function to the alpha value

            crossfade_image = crossfade(image1, image2, alpha)
            crossfade_images = torch.cat([crossfade_images, crossfade_image.unsqueeze(0)], dim=0)
        # Append the beginning of images_1
        beginning_images_1 = images_1[:transition_start_index]
        crossfade_images = torch.cat([beginning_images_1, crossfade_images], dim=0)
        return (crossfade_images, )
    
class CrossFadeImagesMulti:
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crossfadeimages"
    CATEGORY = "KJNodes/image"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "inputcount": ("INT", {"default": 2, "min": 2, "max": 1000, "step": 1}),
                 "image_1": ("IMAGE",),
                 "image_2": ("IMAGE",),
                 "interpolation": (["linear", "ease_in", "ease_out", "ease_in_out", "bounce", "elastic", "glitchy", "exponential_ease_out"],),
                 "transitioning_frames": ("INT", {"default": 1,"min": 0, "max": 4096, "step": 1}),
        },
    } 
    
    def crossfadeimages(self, inputcount, transitioning_frames, interpolation, **kwargs):

        def crossfade(images_1, images_2, alpha):
            crossfade = (1 - alpha) * images_1 + alpha * images_2
            return crossfade
        def ease_in(t):
            return t * t
        def ease_out(t):
            return 1 - (1 - t) * (1 - t)
        def ease_in_out(t):
            return 3 * t * t - 2 * t * t * t
        def bounce(t):
            if t < 0.5:
                return self.ease_out(t * 2) * 0.5
            else:
                return self.ease_in((t - 0.5) * 2) * 0.5 + 0.5
        def elastic(t):
            return math.sin(13 * math.pi / 2 * t) * math.pow(2, 10 * (t - 1))
        def glitchy(t):
            return t + 0.1 * math.sin(40 * t)
        def exponential_ease_out(t):
            return 1 - (1 - t) ** 4

        easing_functions = {
            "linear": lambda t: t,
            "ease_in": ease_in,
            "ease_out": ease_out,
            "ease_in_out": ease_in_out,
            "bounce": bounce,
            "elastic": elastic,
            "glitchy": glitchy,
            "exponential_ease_out": exponential_ease_out,
        }

        image_1 = kwargs["image_1"]
        height = image_1.shape[1]
        width = image_1.shape[2]

        easing_function = easing_functions[interpolation]
       
        for c in range(1, inputcount):
            frames = []
            new_image = kwargs[f"image_{c + 1}"]
            new_image_height = new_image.shape[1]
            new_image_width = new_image.shape[2]

            if new_image_height != height or new_image_width != width:
                new_image = common_upscale(new_image.movedim(-1, 1), width, height, "lanczos", "disabled")
                new_image = new_image.movedim(1, -1)  # Move channels back to the last dimension

            last_frame_image_1 = image_1[-1]
            first_frame_image_2 = new_image[0]

            for frame in range(transitioning_frames):
                t = frame / (transitioning_frames - 1)
                alpha = easing_function(t)
                alpha_tensor = torch.tensor(alpha, dtype=last_frame_image_1.dtype, device=last_frame_image_1.device)
                frame_image = crossfade(last_frame_image_1, first_frame_image_2, alpha_tensor)
                frames.append(frame_image)
        
            frames = torch.stack(frames)
            image_1 = torch.cat((image_1, frames, new_image), dim=0)
        
        return image_1,

def transition_images(images_1, images_2, alpha, transition_type, blur_radius, reverse):        
            width = images_1.shape[1]
            height = images_1.shape[0]

            mask = torch.zeros_like(images_1, device=images_1.device)
          
            alpha = alpha.item()
            if reverse:
                alpha = 1 - alpha

            #transitions from matteo's essential nodes
            if "horizontal slide" in transition_type:
                pos = round(width * alpha)
                mask[:, :pos, :] = 1.0
            elif "vertical slide" in transition_type:
                pos = round(height * alpha)
                mask[:pos, :, :] = 1.0
            elif "box" in transition_type:
                box_w = round(width * alpha)
                box_h = round(height * alpha)
                x1 = (width - box_w) // 2
                y1 = (height - box_h) // 2
                x2 = x1 + box_w
                y2 = y1 + box_h
                mask[y1:y2, x1:x2, :] = 1.0
            elif "circle" in transition_type:
                radius = math.ceil(math.sqrt(pow(width, 2) + pow(height, 2)) * alpha / 2)
                c_x = width // 2
                c_y = height // 2
                x = torch.arange(0, width, dtype=torch.float32, device="cpu")
                y = torch.arange(0, height, dtype=torch.float32, device="cpu")
                y, x = torch.meshgrid((y, x), indexing="ij")
                circle = ((x - c_x) ** 2 + (y - c_y) ** 2) <= (radius ** 2)
                mask[circle] = 1.0
            elif "horizontal door" in transition_type:
                bar = math.ceil(height * alpha / 2)
                if bar > 0:
                    mask[:bar, :, :] = 1.0
                    mask[-bar:,:, :] = 1.0
            elif "vertical door" in transition_type:
                bar = math.ceil(width * alpha / 2)
                if bar > 0:
                    mask[:, :bar,:] = 1.0
                    mask[:, -bar:,:] = 1.0
            elif "fade" in transition_type:
                mask[:, :, :] = alpha

            mask = gaussian_blur(mask, blur_radius)

            return images_1 * (1 - mask) + images_2 * mask
        
def ease_in(t):
    return t * t
def ease_out(t):
    return 1 - (1 - t) * (1 - t)
def ease_in_out(t):
    return 3 * t * t - 2 * t * t * t
def bounce(t):
    if t < 0.5:
        return ease_out(t * 2) * 0.5
    else:
        return ease_in((t - 0.5) * 2) * 0.5 + 0.5
def elastic(t):
    return math.sin(13 * math.pi / 2 * t) * math.pow(2, 10 * (t - 1))
def glitchy(t):
    return t + 0.1 * math.sin(40 * t)
def exponential_ease_out(t):
    return 1 - (1 - t) ** 4

def gaussian_blur(mask, blur_radius):
    if blur_radius > 0:
        kernel_size = int(blur_radius * 2) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure kernel size is odd
        sigma = blur_radius / 3
        x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32)
        x = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel1d = x / x.sum()
        kernel2d = kernel1d[:, None] * kernel1d[None, :]
        kernel2d = kernel2d.to(mask.device)
        kernel2d = kernel2d.expand(mask.shape[2], 1, kernel2d.shape[0], kernel2d.shape[1])
        mask = mask.permute(2, 0, 1).unsqueeze(0)  # Change to [C, H, W] and add batch dimension
        mask = F.conv2d(mask, kernel2d, padding=kernel_size // 2, groups=mask.shape[1])
        mask = mask.squeeze(0).permute(1, 2, 0)  # Change back to [H, W, C]
    return mask

easing_functions = {
    "linear": lambda t: t,
    "ease_in": ease_in,
    "ease_out": ease_out,
    "ease_in_out": ease_in_out,
    "bounce": bounce,
    "elastic": elastic,
    "glitchy": glitchy,
    "exponential_ease_out": exponential_ease_out,
}

class TransitionImagesMulti:
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "transition"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Creates transitions between images.
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "inputcount": ("INT", {"default": 2, "min": 2, "max": 1000, "step": 1}),
                 "image_1": ("IMAGE",),
                 "image_2": ("IMAGE",),
                 "interpolation": (["linear", "ease_in", "ease_out", "ease_in_out", "bounce", "elastic", "glitchy", "exponential_ease_out"],),
                 "transition_type": (["horizontal slide", "vertical slide", "box", "circle", "horizontal door", "vertical door", "fade"],),
                 "transitioning_frames": ("INT", {"default": 1,"min": 0, "max": 4096, "step": 1}),
                 "blur_radius": ("FLOAT", {"default": 0.0,"min": 0.0, "max": 100.0, "step": 0.1}),
                 "reverse": ("BOOLEAN", {"default": False}),
                 "device": (["CPU", "GPU"], {"default": "CPU"}),
        },
    } 

    def transition(self, inputcount, transitioning_frames, transition_type, interpolation, device, blur_radius, reverse, **kwargs):

        gpu = model_management.get_torch_device()

        image_1 = kwargs["image_1"]
        height = image_1.shape[1]
        width = image_1.shape[2]

        easing_function = easing_functions[interpolation]
    
        for c in range(1, inputcount):
            frames = []
            new_image = kwargs[f"image_{c + 1}"]
            new_image_height = new_image.shape[1]
            new_image_width = new_image.shape[2]

            if new_image_height != height or new_image_width != width:
                new_image = common_upscale(new_image.movedim(-1, 1), width, height, "lanczos", "disabled")
                new_image = new_image.movedim(1, -1)  # Move channels back to the last dimension

            last_frame_image_1 = image_1[-1]
            first_frame_image_2 = new_image[0]
            if device == "GPU":
                last_frame_image_1 = last_frame_image_1.to(gpu)
                first_frame_image_2 = first_frame_image_2.to(gpu)

            if reverse:
                last_frame_image_1, first_frame_image_2 = first_frame_image_2, last_frame_image_1

            for frame in range(transitioning_frames):
                t = frame / (transitioning_frames - 1)
                alpha = easing_function(t)
                alpha_tensor = torch.tensor(alpha, dtype=last_frame_image_1.dtype, device=last_frame_image_1.device)
                frame_image = transition_images(last_frame_image_1, first_frame_image_2, alpha_tensor, transition_type, blur_radius, reverse)
                frames.append(frame_image)
        
            frames = torch.stack(frames).cpu()
            image_1 = torch.cat((image_1, frames, new_image), dim=0)
        
        return image_1.cpu(),

class TransitionImagesInBatch:
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "transition"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Creates transitions between images in a batch.
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "images": ("IMAGE",),
                 "interpolation": (["linear", "ease_in", "ease_out", "ease_in_out", "bounce", "elastic", "glitchy", "exponential_ease_out"],),
                 "transition_type": (["horizontal slide", "vertical slide", "box", "circle", "horizontal door", "vertical door", "fade"],),
                 "transitioning_frames": ("INT", {"default": 1,"min": 0, "max": 4096, "step": 1}),
                 "blur_radius": ("FLOAT", {"default": 0.0,"min": 0.0, "max": 100.0, "step": 0.1}),
                 "reverse": ("BOOLEAN", {"default": False}),
                 "device": (["CPU", "GPU"], {"default": "CPU"}),
        },
    } 

    #transitions from matteo's essential nodes
    def transition(self, images, transitioning_frames, transition_type, interpolation, device, blur_radius, reverse):
        if images.shape[0] == 1:
            return images,

        gpu = model_management.get_torch_device()

        easing_function = easing_functions[interpolation]
        
        images_list = []
        pbar = ProgressBar(images.shape[0] - 1)
        for i in range(images.shape[0] - 1):
            frames = []
            image_1 = images[i]
            image_2 = images[i + 1]

            if device == "GPU":
                image_1 = image_1.to(gpu)
                image_2 = image_2.to(gpu)

            if reverse:
                image_1, image_2 = image_2, image_1
                
            for frame in range(transitioning_frames):
                t = frame / (transitioning_frames - 1)
                alpha = easing_function(t)
                alpha_tensor = torch.tensor(alpha, dtype=image_1.dtype, device=image_1.device)
                frame_image = transition_images(image_1, image_2, alpha_tensor, transition_type, blur_radius, reverse)
                frames.append(frame_image)
            pbar.update(1)
        
            frames = torch.stack(frames).cpu()
            images_list.append(frames)
        images = torch.cat(images_list, dim=0)
        
        return images.cpu(),

class ShuffleImageBatch:
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "shuffle"
    CATEGORY = "KJNodes/image"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "images": ("IMAGE",),
                 "seed": ("INT", {"default": 123,"min": 0, "max": 0xffffffffffffffff, "step": 1}),
        },
    } 

    def shuffle(self, images, seed):
        torch.manual_seed(seed)
        B, H, W, C = images.shape
        indices = torch.randperm(B)
        shuffled_images = images[indices]

        return shuffled_images,

class GetImageRangeFromBatch:
    
    RETURN_TYPES = ("IMAGE", "MASK", )
    FUNCTION = "imagesfrombatch"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Returns a range of images from a batch.
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "start_index": ("INT", {"default": 0,"min": -1, "max": 4096, "step": 1}),
                 "num_frames": ("INT", {"default": 1,"min": 1, "max": 4096, "step": 1}),
        },
        "optional": {
            "images": ("IMAGE",),
            "masks": ("MASK",),
        }
    } 
    
    def imagesfrombatch(self, start_index, num_frames, images=None, masks=None):
        chosen_images = None
        chosen_masks = None

        # Process images if provided
        if images is not None:
            if start_index == -1:
                start_index = max(0, len(images) - num_frames)
            if start_index < 0 or start_index >= len(images):
                raise ValueError("Start index is out of range")
            end_index = min(start_index + num_frames, len(images))
            chosen_images = images[start_index:end_index]

        # Process masks if provided
        if masks is not None:
            if start_index == -1:
                start_index = max(0, len(masks) - num_frames)
            if start_index < 0 or start_index >= len(masks):
                raise ValueError("Start index is out of range for masks")
            end_index = min(start_index + num_frames, len(masks))
            chosen_masks = masks[start_index:end_index]

        return (chosen_images, chosen_masks,)

class GetLatentRangeFromBatch:
    
    RETURN_TYPES = ("LATENT", )
    FUNCTION = "latentsfrombatch"
    CATEGORY = "KJNodes/latents"
    DESCRIPTION = """
Returns a range of latents from a batch.
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latents": ("LATENT",),
                "start_index": ("INT", {"default": 0,"min": -1, "max": 4096, "step": 1}),
                "num_frames": ("INT", {"default": 1,"min": -1, "max": 4096, "step": 1}),
        },
    } 
    
    def latentsfrombatch(self, latents, start_index, num_frames):
        chosen_latents = None
        samples = latents["samples"]
        if len(samples.shape) == 4:
            B, C, H, W = samples.shape
            num_latents = B
        elif len(samples.shape) == 5:
            B, C, T, H, W = samples.shape
            num_latents = T

        if start_index == -1:
            start_index = max(0, num_latents - num_frames)
        if start_index < 0 or start_index >= num_latents:
            raise ValueError("Start index is out of range")
        
        end_index = num_latents if num_frames == -1 else min(start_index + num_frames, num_latents)
        
        if len(samples.shape) == 4:
            chosen_latents = samples[start_index:end_index]
        elif len(samples.shape) == 5:
            chosen_latents = samples[:, :, start_index:end_index]

        return ({"samples": chosen_latents.contiguous(),},)
    
class InsertLatentToIndex:
    
    RETURN_TYPES = ("LATENT", )
    FUNCTION = "insert"
    CATEGORY = "KJNodes/latents"
    DESCRIPTION = """
Inserts a latent at the specified index into the original latent batch.
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source": ("LATENT",),
                "destination": ("LATENT",),
                "index": ("INT", {"default": 0,"min": -1, "max": 4096, "step": 1}),
        },
    } 
    
    def insert(self, source, destination, index):
        samples_destination = destination["samples"]
        samples_source = source["samples"].to(samples_destination)
        
        if len(samples_source.shape) == 4:
            B, C, H, W = samples_source.shape
            num_latents = B
        elif len(samples_source.shape) == 5:
            B, C, T, H, W = samples_source.shape
            num_latents = T
        
        if index >= num_latents or index < 0:
            raise ValueError(f"Index {index} out of bounds for tensor with {num_latents} latents")
        
        if len(samples_source.shape) == 4:
            joined_latents = torch.cat([
                samples_destination[:index],
                samples_source,
                samples_destination[index+1:]
            ], dim=0)
        else:
            joined_latents = torch.cat([
                samples_destination[:, :, :index],
                samples_source,
                samples_destination[:, :, index+1:]
            ], dim=2)

        return ({"samples": joined_latents,},)

class ImageBatchFilter:
    
    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("images", "removed_indices",)
    FUNCTION = "filter"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = "Removes empty images from a batch"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "images": ("IMAGE",),
                 "empty_color": ("STRING", {"default": "0, 0, 0"}),
                "empty_threshold": ("FLOAT", {"default": 0.01,"min": 0.0, "max": 1.0, "step": 0.01}),
        },
        "optional": {
            "replacement_image": ("IMAGE",),
        }
    } 
    
    def filter(self, images, empty_color, empty_threshold, replacement_image=None):
        B, H, W, C = images.shape

        input_images = images.clone()

        empty_color_list = [int(color.strip()) for color in empty_color.split(',')]
        empty_color_tensor = torch.tensor(empty_color_list, dtype=torch.float32).to(input_images.device)
        
        color_diff = torch.abs(input_images - empty_color_tensor)
        mean_diff = color_diff.mean(dim=(1, 2, 3))

        empty_indices = mean_diff <= empty_threshold
        empty_indices_string = ', '.join([str(i) for i in range(B) if empty_indices[i]])
        
        if replacement_image is not None:
            B_rep, H_rep, W_rep, C_rep = replacement_image.shape
            replacement = replacement_image.clone()
            if (H_rep != images.shape[1]) or (W_rep != images.shape[2]) or (C_rep != images.shape[3]):
                replacement = common_upscale(replacement.movedim(-1, 1), W, H, "lanczos", "center").movedim(1, -1)
            input_images[empty_indices] = replacement[0]

            return (input_images, empty_indices_string,) 
        else:
            non_empty_images = input_images[~empty_indices]
            return (non_empty_images, empty_indices_string,)
    
class GetImagesFromBatchIndexed:
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "indexedimagesfrombatch"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Selects and returns the images at the specified indices as an image batch.
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "images": ("IMAGE",),
                 "indexes": ("STRING", {"default": "0, 1, 2", "multiline": True}),
        },
    } 
    
    def indexedimagesfrombatch(self, images, indexes):
        
        # Parse the indexes string into a list of integers
        index_list = [int(index.strip()) for index in indexes.split(',')]
        
        # Convert list of indices to a PyTorch tensor
        indices_tensor = torch.tensor(index_list, dtype=torch.long)
        
        # Select the images at the specified indices
        chosen_images = images[indices_tensor]
        
        return (chosen_images,)

class InsertImagesToBatchIndexed:
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "insertimagesfrombatch"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Inserts images at the specified indices into the original image batch.
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original_images": ("IMAGE",),
                "images_to_insert": ("IMAGE",),
                "indexes": ("STRING", {"default": "0, 1, 2", "multiline": True}),
            },
            "optional": {
                "mode": (["replace", "insert"],),
            }
        }
    
    def insertimagesfrombatch(self, original_images, images_to_insert, indexes, mode="replace"):
        if indexes == "":
            return (original_images,)

        input_images = original_images.clone()
        
        # Parse the indexes string into a list of integers
        index_list = [int(index.strip()) for index in indexes.split(',')]
        
        # Convert list of indices to a PyTorch tensor
        indices_tensor = torch.tensor(index_list, dtype=torch.long)
        
        # Ensure the images_to_insert is a tensor
        if not isinstance(images_to_insert, torch.Tensor):
            images_to_insert = torch.tensor(images_to_insert)
        
        if mode == "replace":
            # Replace the images at the specified indices
            for index, image in zip(indices_tensor, images_to_insert):
                input_images[index] = image
        else:
            # Create a list to hold the new image sequence
            new_images = []
            insert_offset = 0
            
            for i in range(len(input_images) + len(indices_tensor)):
                if insert_offset < len(indices_tensor) and i == indices_tensor[insert_offset]:
                    # Use modulo to cycle through images_to_insert
                    new_images.append(images_to_insert[insert_offset % len(images_to_insert)])
                    insert_offset += 1
                else:
                    new_images.append(input_images[i - insert_offset])
            
            # Convert the list back to a tensor
            input_images = torch.stack(new_images, dim=0)
        
        return (input_images,)

class PadImageBatchInterleaved:
    
    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("images", "masks",)
    FUNCTION = "pad"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Inserts empty frames between the images in a batch.
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "empty_frames_per_image": ("INT", {"default": 1,"min": 0, "max": 4096, "step": 1}),
                "pad_frame_value": ("FLOAT", {"default": 0.0,"min": 0.0, "max": 1.0, "step": 0.01}),
                "add_after_last": ("BOOLEAN", {"default": False}),
            },
        }
    
    def pad(self, images, empty_frames_per_image, pad_frame_value, add_after_last):
        B, H, W, C = images.shape
        
        # Handle single frame case specifically
        if B == 1:
            total_frames = 1 + empty_frames_per_image if add_after_last else 1
        else:
            # Original B images + (B-1) sets of empty frames between them
            total_frames = B + (B-1) * empty_frames_per_image
            # Add additional empty frames after the last image if requested
            if add_after_last:
                total_frames += empty_frames_per_image
        
        # Create new tensor with zeros (empty frames)
        padded_batch = torch.ones((total_frames, H, W, C), 
                                dtype=images.dtype, 
                                device=images.device) * pad_frame_value
        # Create mask tensor (1 for original frames, 0 for empty frames)
        mask = torch.zeros((total_frames, H, W), 
                        dtype=images.dtype, 
                        device=images.device)
        
        # Fill in original images at their new positions
        for i in range(B):
            if B == 1:
                # For single frame, just place it at the beginning
                new_pos = 0
            else:
                # Each image is separated by empty_frames_per_image blank frames
                new_pos = i * (empty_frames_per_image + 1)
                
            padded_batch[new_pos] = images[i]
            mask[new_pos] = 1.0  # Mark this as an original frame
        
        return (padded_batch, mask)

class ReplaceImagesInBatch:
    
    RETURN_TYPES = ("IMAGE", "MASK",)
    FUNCTION = "replace"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Replaces the images in a batch, starting from the specified start index,  
with the replacement images.
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "start_index": ("INT", {"default": 1,"min": 0, "max": 4096, "step": 1}),
        },
        "optional": {
            "original_images": ("IMAGE",),
            "replacement_images": ("IMAGE",),
            "original_masks": ("MASK",),
            "replacement_masks": ("MASK",),
        }
    } 
    
    def replace(self, original_images=None, replacement_images=None, start_index=1, original_masks=None, replacement_masks=None):
        images = None
        masks = None
        
        if original_images is not None and replacement_images is not None:
            if start_index >= len(original_images):
                raise ValueError("ReplaceImagesInBatch: Start index is out of range")
            end_index = start_index + len(replacement_images)
            if end_index > len(original_images):
                raise ValueError("ReplaceImagesInBatch: End index is out of range")
            
            original_images_copy = original_images.clone()
            if original_images_copy.shape[2] != replacement_images.shape[2] or original_images_copy.shape[3] != replacement_images.shape[3]:
                replacement_images = common_upscale(replacement_images.movedim(-1, 1), original_images_copy.shape[1], original_images_copy.shape[2], "lanczos", "center").movedim(1, -1)
            
            original_images_copy[start_index:end_index] = replacement_images
            images = original_images_copy
        else:
            images = torch.zeros((1, 64, 64, 3))
        
        if original_masks is not None and replacement_masks is not None:
            if start_index >= len(original_masks):
                raise ValueError("ReplaceImagesInBatch: Start index is out of range")
            end_index = start_index + len(replacement_masks)
            if end_index > len(original_masks):
                raise ValueError("ReplaceImagesInBatch: End index is out of range")

            original_masks_copy = original_masks.clone()
            if original_masks_copy.shape[1] != replacement_masks.shape[1] or original_masks_copy.shape[2] != replacement_masks.shape[2]:
                replacement_masks = common_upscale(replacement_masks.unsqueeze(1), original_masks_copy.shape[1], original_masks_copy.shape[2], "nearest-exact", "center").squeeze(0)
                
            original_masks_copy[start_index:end_index] = replacement_masks
            masks = original_masks_copy
        else:
            masks = torch.zeros((1, 64, 64))
        
        return (images, masks)
    

class ReverseImageBatch:
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "reverseimagebatch"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Reverses the order of the images in a batch.
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "images": ("IMAGE",),
        },
    } 
    
    def reverseimagebatch(self, images):
        reversed_images = torch.flip(images, [0])
        return (reversed_images, )

class ImageBatchMulti:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "inputcount": ("INT", {"default": 2, "min": 2, "max": 1000, "step": 1}),
                "image_1": ("IMAGE", ),
                "image_2": ("IMAGE", ),
            },
    }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "combine"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Creates an image batch from multiple images.  
You can set how many inputs the node has,  
with the **inputcount** and clicking update.
"""

    def combine(self, inputcount, **kwargs):
        from nodes import ImageBatch
        image_batch_node = ImageBatch()
        image = kwargs["image_1"]
        for c in range(1, inputcount):
            new_image = kwargs[f"image_{c + 1}"]
            image, = image_batch_node.batch(image, new_image)
        return (image,)


class ImageTensorList:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image1": ("IMAGE",),
            "image2": ("IMAGE",),
        }}

    RETURN_TYPES = ("IMAGE",)
    #OUTPUT_IS_LIST = (True,)
    FUNCTION = "append"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Creates an image list from the input images.
"""

    def append(self, image1, image2):
        image_list = []
        if isinstance(image1, torch.Tensor) and isinstance(image2, torch.Tensor):
            image_list = [image1, image2]
        elif isinstance(image1, list) and isinstance(image2, torch.Tensor):
            image_list = image1 + [image2]
        elif isinstance(image1, torch.Tensor) and isinstance(image2, list):
            image_list = [image1] + image2
        elif isinstance(image1, list) and isinstance(image2, list):
            image_list = image1 + image2
        return image_list,

class ImageAddMulti:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "inputcount": ("INT", {"default": 2, "min": 2, "max": 1000, "step": 1}),
                "image_1": ("IMAGE", ),
                "image_2": ("IMAGE", ),
                "blending": (
                [   'add',
                    'subtract',
                    'multiply',
                    'difference',
                ],
                {
                "default": 'add'
                }),
                "blend_amount": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
            },
    }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "add"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Add blends multiple images together.    
You can set how many inputs the node has,  
with the **inputcount** and clicking update.
"""

    def add(self, inputcount, blending, blend_amount, **kwargs):
        image = kwargs["image_1"]
        for c in range(1, inputcount):
            new_image = kwargs[f"image_{c + 1}"]
            if blending == "add":
                image = torch.add(image * blend_amount, new_image * blend_amount)
            elif blending == "subtract":
                image = torch.sub(image * blend_amount, new_image * blend_amount)
            elif blending == "multiply":
                image = torch.mul(image * blend_amount, new_image * blend_amount)
            elif blending == "difference":
                image = torch.sub(image, new_image)
        return (image,)    

class ImageConcatMulti:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "inputcount": ("INT", {"default": 2, "min": 2, "max": 1000, "step": 1}),
                "image_1": ("IMAGE", ),
                "image_2": ("IMAGE", ),
                "direction": (
                [   'right',
                    'down',
                    'left',
                    'up',
                ],
            {
            "default": 'right'
             }),
            "match_image_size": ("BOOLEAN", {"default": False}),
            },
    }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "combine"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Creates an image from multiple images.  
You can set how many inputs the node has,  
with the **inputcount** and clicking update.
"""

    def combine(self, inputcount, direction, match_image_size, **kwargs):
        image = kwargs["image_1"]
        first_image_shape = None
        if first_image_shape is None:
            first_image_shape = image.shape
        for c in range(1, inputcount):
            new_image = kwargs[f"image_{c + 1}"]
            image, = ImageConcanate.concatenate(self, image, new_image, direction, match_image_size, first_image_shape=first_image_shape)
        first_image_shape = None
        return (image,)

class PreviewAnimation:
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 1

    methods = {"default": 4, "fastest": 0, "slowest": 6}
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                     "fps": ("FLOAT", {"default": 8.0, "min": 0.01, "max": 1000.0, "step": 0.01}),
                     },
                "optional": {
                    "images": ("IMAGE", ),
                    "masks": ("MASK", ),
                },
            }

    RETURN_TYPES = ()
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = "KJNodes/image"

    def preview(self, fps, images=None, masks=None):
        filename_prefix = "AnimPreview"
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        results = list()

        pil_images = []

        if images is not None and masks is not None:
            for image in images:
                i = 255. * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                pil_images.append(img)
            for mask in masks:
                if pil_images: 
                    mask_np = mask.cpu().numpy()
                    mask_np = np.clip(mask_np * 255, 0, 255).astype(np.uint8)  # Convert to values between 0 and 255
                    mask_img = Image.fromarray(mask_np, mode='L')
                    img = pil_images.pop(0)  # Remove and get the first image
                    img = img.convert("RGBA")  # Convert base image to RGBA

                    # Create a new RGBA image based on the grayscale mask
                    rgba_mask_img = Image.new("RGBA", img.size, (255, 255, 255, 255))
                    rgba_mask_img.putalpha(mask_img)  # Use the mask image as the alpha channel

                    # Composite the RGBA mask onto the base image
                    composited_img = Image.alpha_composite(img, rgba_mask_img)
                    pil_images.append(composited_img)  # Add the composited image back

        elif images is not None and masks is None:
            for image in images:
                i = 255. * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                pil_images.append(img)

        elif masks is not None and images is None:
            for mask in masks:
                mask_np = 255. * mask.cpu().numpy()
                mask_img = Image.fromarray(np.clip(mask_np, 0, 255).astype(np.uint8))
                pil_images.append(mask_img)
        else:
            print("PreviewAnimation: No images or masks provided")
            return { "ui": { "images": results, "animated": (None,), "text": "empty" }}

        num_frames = len(pil_images)

        c = len(pil_images)
        for i in range(0, c, num_frames):
            file = f"{filename}_{counter:05}_.webp"
            pil_images[i].save(os.path.join(full_output_folder, file), save_all=True, duration=int(1000.0/fps), append_images=pil_images[i + 1:i + num_frames], lossless=False, quality=50, method=0)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        animated = num_frames != 1
        return { "ui": { "images": results, "animated": (animated,), "text": [f"{num_frames}x{pil_images[0].size[0]}x{pil_images[0].size[1]}"] } }
    
class ImageResizeKJ:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                "height": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                "upscale_method": (s.upscale_methods,),
                "keep_proportion": ("BOOLEAN", { "default": False }),
                "divisible_by": ("INT", { "default": 2, "min": 0, "max": 512, "step": 1, }),
            },
            "optional" : {
                #"width_input": ("INT", { "forceInput": True}),
                #"height_input": ("INT", { "forceInput": True}),
                "get_image_size": ("IMAGE",),
                "crop": (["disabled","center", 0], { "tooltip": "0 will do the default center crop, this is a workaround for the widget order changing with the new frontend, as in old workflows the value of this widget becomes 0 automatically" }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "width", "height",)
    FUNCTION = "resize"
    CATEGORY = "KJNodes/image"
    DEPRECATED = True
    DESCRIPTION = """
DEPRECATED!

Due to ComfyUI frontend changes, this node should no longer be used, please check the   
v2 of the node. This node is only kept to not completely break older workflows.  

"""

    def resize(self, image, width, height, keep_proportion, upscale_method, divisible_by, 
               width_input=None, height_input=None, get_image_size=None, crop="disabled"):
        B, H, W, C = image.shape

        if width_input:
            width = width_input
        if height_input:
            height = height_input
        if get_image_size is not None:
            _, height, width, _ = get_image_size.shape
        
        if keep_proportion and get_image_size is None:
                # If one of the dimensions is zero, calculate it to maintain the aspect ratio
                if width == 0 and height != 0:
                    ratio = height / H
                    width = round(W * ratio)
                elif height == 0 and width != 0:
                    ratio = width / W
                    height = round(H * ratio)
                elif width != 0 and height != 0:
                    # Scale based on which dimension is smaller in proportion to the desired dimensions
                    ratio = min(width / W, height / H)
                    width = round(W * ratio)
                    height = round(H * ratio)
        else:
            if width == 0:
                width = W
            if height == 0:
                height = H
      
        if divisible_by > 1 and get_image_size is None:
            width = width - (width % divisible_by)
            height = height - (height % divisible_by)
        
        if crop == 0: #workaround for old workflows
            crop = "center"

        image = image.movedim(-1,1)
        image = common_upscale(image, width, height, upscale_method, crop)
        image = image.movedim(1,-1)

        return(image, image.shape[2], image.shape[1],)

class ImageResizeKJv2:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                "height": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                "upscale_method": (s.upscale_methods,),
                "keep_proportion": (["stretch", "resize", "pad", "pad_edge", "crop"], { "default": False }),
                "pad_color": ("STRING", { "default": "0, 0, 0", "tooltip": "Color to use for padding."}),
                "crop_position": (["center", "top", "bottom", "left", "right"], { "default": "center" }),
                "divisible_by": ("INT", { "default": 2, "min": 0, "max": 512, "step": 1, }),
            },
            "optional" : {
                "mask": ("MASK",),
                "device": (["cpu", "gpu"],),
            },
             "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "MASK",)
    RETURN_NAMES = ("IMAGE", "width", "height", "mask",)
    FUNCTION = "resize"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Resizes the image to the specified width and height.  
Size can be retrieved from the input.

Keep proportions keeps the aspect ratio of the image, by  
highest dimension.  
"""

    def resize(self, image, width, height, keep_proportion, upscale_method, divisible_by, pad_color, crop_position, unique_id, device="cpu", mask=None):
        B, H, W, C = image.shape

        if device == "gpu":
            if upscale_method == "lanczos":
                raise Exception("Lanczos is not supported on the GPU")
            device = model_management.get_torch_device()
        else:
            device = torch.device("cpu")

        if width == 0:
            width = W
        if height == 0:
            height = H
        
        if keep_proportion == "resize" or keep_proportion.startswith("pad"):
            # If one of the dimensions is zero, calculate it to maintain the aspect ratio
            if width == 0 and height != 0:
                ratio = height / H
                new_width = round(W * ratio)
            elif height == 0 and width != 0:
                ratio = width / W
                new_height = round(H * ratio)
            elif width != 0 and height != 0:
                # Scale based on which dimension is smaller in proportion to the desired dimensions
                ratio = min(width / W, height / H)
                new_width = round(W * ratio)
                new_height = round(H * ratio)

            if keep_proportion.startswith("pad"):
                # Calculate padding based on position
                if crop_position == "center":
                    pad_left = (width - new_width) // 2
                    pad_right = width - new_width - pad_left
                    pad_top = (height - new_height) // 2
                    pad_bottom = height - new_height - pad_top
                elif crop_position == "top":
                    pad_left = (width - new_width) // 2
                    pad_right = width - new_width - pad_left
                    pad_top = 0
                    pad_bottom = height - new_height
                elif crop_position == "bottom":
                    pad_left = (width - new_width) // 2
                    pad_right = width - new_width - pad_left
                    pad_top = height - new_height
                    pad_bottom = 0
                elif crop_position == "left":
                    pad_left = 0
                    pad_right = width - new_width
                    pad_top = (height - new_height) // 2
                    pad_bottom = height - new_height - pad_top
                elif crop_position == "right":
                    pad_left = width - new_width
                    pad_right = 0
                    pad_top = (height - new_height) // 2
                    pad_bottom = height - new_height - pad_top

            width = new_width
            height = new_height

        if divisible_by > 1:
            width = width - (width % divisible_by)
            height = height - (height % divisible_by)

        out_image = image.clone().to(device)

        if mask is not None:
            out_mask = mask.clone().to(device)
        
        if keep_proportion == "crop":
            old_width = W
            old_height = H
            old_aspect = old_width / old_height
            new_aspect = width / height
            
            # Calculate dimensions to keep
            if old_aspect > new_aspect:  # Image is wider than target
                crop_w = round(old_height * new_aspect)
                crop_h = old_height
            else:  # Image is taller than target
                crop_w = old_width
                crop_h = round(old_width / new_aspect)
            
            # Calculate crop position
            if crop_position == "center":
                x = (old_width - crop_w) // 2
                y = (old_height - crop_h) // 2
            elif crop_position == "top":
                x = (old_width - crop_w) // 2
                y = 0
            elif crop_position == "bottom":
                x = (old_width - crop_w) // 2
                y = old_height - crop_h
            elif crop_position == "left":
                x = 0
                y = (old_height - crop_h) // 2
            elif crop_position == "right":
                x = old_width - crop_w
                y = (old_height - crop_h) // 2
            
            # Apply crop
            out_image = out_image.narrow(-2, x, crop_w).narrow(-3, y, crop_h)
            if mask is not None:
                out_mask = out_mask.narrow(-1, x, crop_w).narrow(-2, y, crop_h)
        
        out_image = common_upscale(out_image.movedim(-1,1), width, height, upscale_method, crop="disabled").movedim(1,-1)

        if mask is not None:
            if upscale_method == "lanczos":
                out_mask = common_upscale(out_mask.unsqueeze(1).repeat(1, 3, 1, 1), width, height, upscale_method, crop="disabled").movedim(1,-1)[:, :, :, 0]
            else:
                out_mask = common_upscale(out_mask.unsqueeze(1), width, height, upscale_method, crop="disabled").squeeze(1)
            
        if keep_proportion.startswith("pad"):
            if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
                padded_width = width + pad_left + pad_right
                padded_height = height + pad_top + pad_bottom
                if divisible_by > 1:
                    width_remainder = padded_width % divisible_by
                    height_remainder = padded_height % divisible_by
                    if width_remainder > 0:
                        extra_width = divisible_by - width_remainder
                        pad_right += extra_width
                    if height_remainder > 0:
                        extra_height = divisible_by - height_remainder
                        pad_bottom += extra_height
                out_image, _ = ImagePadKJ.pad(self, out_image, pad_left, pad_right, pad_top, pad_bottom, 0, pad_color, "edge" if keep_proportion == "pad_edge" else "color")
                if mask is not None:
                    out_mask = out_mask.unsqueeze(1).repeat(1, 3, 1, 1).movedim(1,-1)
                    out_mask, _ = ImagePadKJ.pad(self, out_mask, pad_left, pad_right, pad_top, pad_bottom, 0, pad_color, "edge" if keep_proportion == "pad_edge" else "color")
                    out_mask = out_mask[:, :, :, 0]

        if unique_id and PromptServer is not None:
            try:
                num_elements = out_image.numel()
                element_size = out_image.element_size()
                memory_size_mb = (num_elements * element_size) / (1024 * 1024)
                
                PromptServer.instance.send_progress_text(
                    f"<tr><td>Output: </td><td><b>{out_image.shape[0]}</b> x <b>{out_image.shape[2]}</b> x <b>{out_image.shape[1]} | {memory_size_mb:.2f}MB</b></td></tr>",
                    unique_id
                )
            except:
                pass

        return(out_image.cpu(), out_image.shape[2], out_image.shape[1], out_mask.cpu() if mask is not None else torch.zeros(64,64, device=torch.device("cpu"), dtype=torch.float32))
    
import pathlib    
class LoadAndResizeImage:
    _color_channels = ["alpha", "red", "green", "blue"]
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f.name for f in pathlib.Path(input_dir).iterdir() if f.is_file()]
        return {"required":
                    {
                    "image": (sorted(files), {"image_upload": True}),
                    "resize": ("BOOLEAN", { "default": False }),
                    "width": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                    "height": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                    "repeat": ("INT", { "default": 1, "min": 1, "max": 4096, "step": 1, }),
                    "keep_proportion": ("BOOLEAN", { "default": False }),
                    "divisible_by": ("INT", { "default": 2, "min": 0, "max": 512, "step": 1, }),
                    "mask_channel": (s._color_channels, {"tooltip": "Channel to use for the mask output"}), 
                    "background_color": ("STRING", { "default": "", "tooltip": "Fills the alpha channel with the specified color."}),
                    },
                }

    CATEGORY = "KJNodes/image"
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "STRING",)
    RETURN_NAMES = ("image", "mask", "width", "height","image_path",)
    FUNCTION = "load_image"

    def load_image(self, image, resize, width, height, repeat, keep_proportion, divisible_by, mask_channel, background_color):
        from PIL import ImageColor, Image, ImageOps, ImageSequence
        import numpy as np
        import torch
        image_path = folder_paths.get_annotated_filepath(image)
        
        import node_helpers
        img = node_helpers.pillow(Image.open, image_path)

        # Process the background_color
        if background_color:
            try:
                # Try to parse as RGB tuple
                bg_color_rgba = tuple(int(x.strip()) for x in background_color.split(','))
            except ValueError:
                # If parsing fails, it might be a hex color or named color
                if background_color.startswith('#') or background_color.lower() in ImageColor.colormap:
                    bg_color_rgba = ImageColor.getrgb(background_color)
                else:
                    raise ValueError(f"Invalid background color: {background_color}")

            bg_color_rgba += (255,)  # Add alpha channel
        else:
            bg_color_rgba = None  # No background color specified
        
        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']

        W, H = img.size
        if resize:
            if keep_proportion:
                ratio = min(width / W, height / H)
                width = round(W * ratio)
                height = round(H * ratio)
            else:
                if width == 0:
                    width = W
                if height == 0:
                    height = H

            if divisible_by > 1:
                width = width - (width % divisible_by)
                height = height - (height % divisible_by)
        else:
            width, height = W, H

        for frame in ImageSequence.Iterator(img):
            frame = node_helpers.pillow(ImageOps.exif_transpose, frame)

            if frame.mode == 'I':
                frame = frame.point(lambda i: i * (1 / 255))
            
            if frame.mode == 'P':
                frame = frame.convert("RGBA")
            elif 'A' in frame.getbands():
                frame = frame.convert("RGBA")
            
            # Extract alpha channel if it exists
            if 'A' in frame.getbands() and bg_color_rgba:
                alpha_mask = np.array(frame.getchannel('A')).astype(np.float32) / 255.0
                alpha_mask = 1. - torch.from_numpy(alpha_mask)
                bg_image = Image.new("RGBA", frame.size, bg_color_rgba)
                # Composite the frame onto the background
                frame = Image.alpha_composite(bg_image, frame)
            else:
                alpha_mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            
            image = frame.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]
            
            if image.size[0] != w or image.size[1] != h:
                continue
            if resize:
                image = image.resize((width, height), Image.Resampling.BILINEAR)

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            
            c = mask_channel[0].upper()
            if c in frame.getbands():
                if resize:
                    frame = frame.resize((width, height), Image.Resampling.BILINEAR)
                mask = np.array(frame.getchannel(c)).astype(np.float32) / 255.0
                mask = torch.from_numpy(mask)
                if c == 'A' and bg_color_rgba:
                    mask = alpha_mask
                elif c == 'A':
                    mask = 1. - mask
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
            if repeat > 1:
                output_image = output_image.repeat(repeat, 1, 1, 1)
                output_mask = output_mask.repeat(repeat, 1, 1)

        return (output_image, output_mask, width, height, image_path)
        

    # @classmethod
    # def IS_CHANGED(s, image, **kwargs):
    #     image_path = folder_paths.get_annotated_filepath(image)
    #     m = hashlib.sha256()
    #     with open(image_path, 'rb') as f:
    #         m.update(f.read())
    #     return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True

import hashlib
class LoadImagesFromFolderKJ:
    # Dictionary to store folder hashes
    folder_hashes = {}

    @classmethod
    def IS_CHANGED(cls, folder, **kwargs):
        if not os.path.isdir(folder):
            return float("NaN")
        
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.tga']
        include_subfolders = kwargs.get('include_subfolders', False)
        
        file_data = []
        if include_subfolders:
            for root, _, files in os.walk(folder):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in valid_extensions):
                        path = os.path.join(root, file)
                        try:
                            mtime = os.path.getmtime(path)
                            file_data.append((path, mtime))
                        except OSError:
                            pass
        else:
            for file in os.listdir(folder):
                if any(file.lower().endswith(ext) for ext in valid_extensions):
                    path = os.path.join(folder, file)
                    try:
                        mtime = os.path.getmtime(path)
                        file_data.append((path, mtime))
                    except OSError:
                        pass
        
        file_data.sort()
        
        combined_hash = hashlib.md5()
        combined_hash.update(folder.encode('utf-8'))
        combined_hash.update(str(len(file_data)).encode('utf-8'))
        
        for path, mtime in file_data:
            combined_hash.update(f"{path}:{mtime}".encode('utf-8'))
        
        current_hash = combined_hash.hexdigest()
        
        old_hash = cls.folder_hashes.get(folder)
        cls.folder_hashes[folder] = current_hash
        
        if old_hash == current_hash:
            return old_hash
        
        return current_hash

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder": ("STRING", {"default": ""}),
                "width": ("INT", {"default": 1024, "min": -1, "step": 1}),
                "height": ("INT", {"default": 1024, "min": -1, "step": 1}),
                "keep_aspect_ratio": (["crop", "pad", "stretch",],), 
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                "start_index": ("INT", {"default": 0, "min": 0, "step": 1}),
                "include_subfolders": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "STRING",)
    RETURN_NAMES = ("image", "mask", "count", "image_path",)
    FUNCTION = "load_images"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """Loads images from a folder into a batch, images are resized and loaded into a batch."""

    def load_images(self, folder, width, height, image_load_cap, start_index, keep_aspect_ratio, include_subfolders=False):
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Folder '{folder} cannot be found.'")
        
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.tga']
        image_paths = []
        if include_subfolders:
            for root, _, files in os.walk(folder):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in valid_extensions):
                        image_paths.append(os.path.join(root, file))
        else:
            for file in os.listdir(folder):
                if any(file.lower().endswith(ext) for ext in valid_extensions):
                    image_paths.append(os.path.join(folder, file))

        dir_files = sorted(image_paths)

        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{folder}'.")

        # start at start_index
        dir_files = dir_files[start_index:]

        images = []
        masks = []
        image_path_list = []

        limit_images = False
        if image_load_cap > 0:
            limit_images = True
        image_count = 0

        for image_path in dir_files:
            if os.path.isdir(image_path):
                continue
            if limit_images and image_count >= image_load_cap:
                break
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            
            # Resize image to maximum dimensions
            if width == -1 and height == -1:
                width = i.size[0]
                height = i.size[1]
            if i.size != (width, height):
                i = self.resize_with_aspect_ratio(i, width, height, keep_aspect_ratio)
            
            
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
                if mask.shape != (height, width):
                    mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), 
                                                         size=(height, width), 
                                                         mode='bilinear', 
                                                         align_corners=False).squeeze()
            else:
                mask = torch.zeros((height, width), dtype=torch.float32, device="cpu")
            
            images.append(image)
            masks.append(mask)
            image_path_list.append(image_path)
            image_count += 1

        if len(images) == 1:
            return (images[0], masks[0], 1, image_path_list)
        
        elif len(images) > 1:
            image1 = images[0]
            mask1 = masks[0].unsqueeze(0)

            for image2 in images[1:]:
                image1 = torch.cat((image1, image2), dim=0)

            for mask2 in masks[1:]:
                mask1 = torch.cat((mask1, mask2.unsqueeze(0)), dim=0)

            return (image1, mask1, len(images), image_path_list)
    def resize_with_aspect_ratio(self, img, width, height, mode):
        if mode == "stretch":
            return img.resize((width, height), Image.Resampling.LANCZOS)
        
        img_width, img_height = img.size
        aspect_ratio = img_width / img_height
        target_ratio = width / height

        if mode == "crop":
            # Calculate dimensions for center crop
            if aspect_ratio > target_ratio:
                # Image is wider - crop width
                new_width = int(height * aspect_ratio)
                img = img.resize((new_width, height), Image.Resampling.LANCZOS)
                left = (new_width - width) // 2
                return img.crop((left, 0, left + width, height))
            else:
                # Image is taller - crop height
                new_height = int(width / aspect_ratio)
                img = img.resize((width, new_height), Image.Resampling.LANCZOS)
                top = (new_height - height) // 2
                return img.crop((0, top, width, top + height))

        elif mode == "pad":
            pad_color = self.get_edge_color(img)
            # Calculate dimensions for padding
            if aspect_ratio > target_ratio:
                # Image is wider - pad height
                new_height = int(width / aspect_ratio)
                img = img.resize((width, new_height), Image.Resampling.LANCZOS)
                padding = (height - new_height) // 2
                padded = Image.new('RGBA', (width, height), pad_color)
                padded.paste(img, (0, padding))
                return padded
            else:
                # Image is taller - pad width
                new_width = int(height * aspect_ratio)
                img = img.resize((new_width, height), Image.Resampling.LANCZOS)
                padding = (width - new_width) // 2
                padded = Image.new('RGBA', (width, height), pad_color)
                padded.paste(img, (padding, 0))
                return padded
    def get_edge_color(self, img):
        from PIL import ImageStat
        """Sample edges and return dominant color"""
        width, height = img.size
        img = img.convert('RGBA')
        
        # Create 1-pixel high/wide images from edges
        top = img.crop((0, 0, width, 1))
        bottom = img.crop((0, height-1, width, height))
        left = img.crop((0, 0, 1, height))
        right = img.crop((width-1, 0, width, height))
        
        # Combine edges into single image
        edges = Image.new('RGBA', (width*2 + height*2, 1))
        edges.paste(top, (0, 0))
        edges.paste(bottom, (width, 0))
        edges.paste(left.resize((height, 1)), (width*2, 0))
        edges.paste(right.resize((height, 1)), (width*2 + height, 0))
        
        # Get median color
        stat = ImageStat.Stat(edges)
        median = tuple(map(int, stat.median))
        return median
        

class ImageGridtoBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "image": ("IMAGE", ),
                    "columns": ("INT", {"default": 3, "min": 1, "max": 8, "tooltip": "The number of columns in the grid."}),
                    "rows": ("INT", {"default": 0, "min": 1, "max": 8, "tooltip": "The number of rows in the grid. Set to 0 for automatic calculation."}),
                  }
                }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decompose"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = "Converts a grid of images to a batch of images."
        
    def decompose(self, image, columns, rows):
        B, H, W, C = image.shape
        print("input size: ", image.shape)
        
        # Calculate cell width, rounding down
        cell_width = W // columns
        
        if rows == 0:
            # If rows is 0, calculate number of full rows
            rows = H // cell_height
        else:
            # If rows is specified, adjust cell_height
            cell_height = H // rows
        
        # Crop the image to fit full cells
        image = image[:, :rows*cell_height, :columns*cell_width, :]
        
        # Reshape and permute the image to get the grid
        image = image.view(B, rows, cell_height, columns, cell_width, C)
        image = image.permute(0, 1, 3, 2, 4, 5).contiguous()
        image = image.view(B, rows * columns, cell_height, cell_width, C)
        
        # Reshape to the final batch tensor
        img_tensor = image.view(-1, cell_height, cell_width, C)
        
        return (img_tensor,)

class SaveImageKJ:
    def __init__(self):
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."}),
                "output_folder": ("STRING", {"default": "output", "tooltip": "The folder to save the images to."}),
            },
            "optional": {
                "caption_file_extension": ("STRING", {"default": ".txt", "tooltip": "The extension for the caption file."}),
                "caption": ("STRING", {"forceInput": True, "tooltip": "string to save as .txt file"}), 
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename",)
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "KJNodes/image"
    DESCRIPTION = "Saves the input images to your ComfyUI output directory."

    def save_images(self, images, output_folder, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None, caption=None, caption_file_extension=".txt"):
        filename_prefix += self.prefix_append

        if os.path.isabs(output_folder):
            if not os.path.exists(output_folder):
                os.makedirs(output_folder, exist_ok=True)
            full_output_folder = output_folder
            _, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, output_folder, images[0].shape[1], images[0].shape[0])
        else:
            self.output_dir = folder_paths.get_output_directory()
            full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])

        results = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            base_file_name = f"{filename_with_batch_num}_{counter:05}_"
            file = f"{base_file_name}.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            if caption is not None:
                txt_file = base_file_name + caption_file_extension
                file_path = os.path.join(full_output_folder, txt_file)
                with open(file_path, 'w') as f:
                    f.write(caption)

            counter += 1

        return file, 
    
class SaveStringKJ:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "string": ("STRING", {"forceInput": True, "tooltip": "string to save as .txt file"}), 
                "filename_prefix": ("STRING", {"default": "text", "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."}),
                "output_folder": ("STRING", {"default": "output", "tooltip": "The folder to save the images to."}),
            },
            "optional": {
                "file_extension": ("STRING", {"default": ".txt", "tooltip": "The extension for the caption file."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename",)
    FUNCTION = "save_string"

    OUTPUT_NODE = True

    CATEGORY = "KJNodes/misc"
    DESCRIPTION = "Saves the input string to your ComfyUI output directory."

    def save_string(self, string, output_folder, filename_prefix="text", file_extension=".txt"):
        filename_prefix += self.prefix_append
        
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        if output_folder != "output":
            if not os.path.exists(output_folder):
                os.makedirs(output_folder, exist_ok=True)
            full_output_folder = output_folder

        base_file_name = f"{filename_prefix}_{counter:05}_"
        results = list()

        txt_file = base_file_name + file_extension
        file_path = os.path.join(full_output_folder, txt_file)
        with open(file_path, 'w') as f:
            f.write(string)

        return results,
    
to_pil_image = T.ToPILImage()

class FastPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", ),
                "format": (["JPEG", "PNG", "WEBP"], {"default": "JPEG"}),
                "quality" : ("INT", {"default": 75, "min": 1, "max": 100, "step": 1}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "preview"
    CATEGORY = "KJNodes/experimental"
    OUTPUT_NODE = True
    DESCRIPTION = "Experimental node for faster image previews by displaying through base64 it without saving to disk."

    def preview(self, image, format, quality):        
        pil_image = to_pil_image(image[0].permute(2, 0, 1))

        with io.BytesIO() as buffered:
            pil_image.save(buffered, format=format, quality=quality)
            img_bytes = buffered.getvalue()

        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    
        return {
            "ui": {"bg_image": [img_base64]}, 
            "result": ()
        }
    
class ImageCropByMaskAndResize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "mask": ("MASK", ),
                "base_resolution": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                "padding": ("INT", { "default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                "min_crop_resolution": ("INT", { "default": 128, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                "max_crop_resolution": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
           
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BBOX", )
    RETURN_NAMES = ("images", "masks", "bbox",)
    FUNCTION = "crop"
    CATEGORY = "KJNodes/image"

    def crop_by_mask(self, mask, padding=0, min_crop_resolution=None, max_crop_resolution=None):
        iy, ix = (mask == 1).nonzero(as_tuple=True)
        h0, w0 = mask.shape

        if iy.numel() == 0:
            x_c = w0 / 2.0
            y_c = h0 / 2.0
            width = 0
            height = 0
        else:
            x_min = ix.min().item()
            x_max = ix.max().item()
            y_min = iy.min().item()
            y_max = iy.max().item()

            width = x_max - x_min
            height = y_max - y_min

            if width > w0 or height > h0:
                raise Exception("Masked area out of bounds")

            x_c = (x_min + x_max) / 2.0
            y_c = (y_min + y_max) / 2.0

        if min_crop_resolution:
            width = max(width, min_crop_resolution)
            height = max(height, min_crop_resolution)

        if max_crop_resolution:
            width = min(width, max_crop_resolution)
            height = min(height, max_crop_resolution)

        if w0 <= width:
            x0 = 0
            w = w0
        else:
            x0 = max(0, x_c - width / 2 - padding)
            w = width + 2 * padding
            if x0 + w > w0:
                x0 = w0 - w

        if h0 <= height:
            y0 = 0
            h = h0
        else:
            y0 = max(0, y_c - height / 2 - padding)
            h = height + 2 * padding
            if y0 + h > h0:
                y0 = h0 - h

        return (int(x0), int(y0), int(w), int(h))

    def crop(self, image, mask, base_resolution, padding=0, min_crop_resolution=128, max_crop_resolution=512):
        mask = mask.round()
        image_list = []
        mask_list = []
        bbox_list = []

        # First, collect all bounding boxes
        bbox_params = []
        aspect_ratios = []
        for i in range(image.shape[0]):
            x0, y0, w, h = self.crop_by_mask(mask[i], padding, min_crop_resolution, max_crop_resolution)
            bbox_params.append((x0, y0, w, h))
            aspect_ratios.append(w / h)

        # Find maximum width and height
        max_w = max([w for x0, y0, w, h in bbox_params])
        max_h = max([h for x0, y0, w, h in bbox_params])
        max_aspect_ratio = max(aspect_ratios)

        # Ensure dimensions are divisible by 16
        max_w = (max_w + 15) // 16 * 16
        max_h = (max_h + 15) // 16 * 16
        # Calculate common target dimensions
        if max_aspect_ratio > 1:
            target_width = base_resolution
            target_height = int(base_resolution / max_aspect_ratio)
        else:
            target_height = base_resolution
            target_width = int(base_resolution * max_aspect_ratio)

        for i in range(image.shape[0]):
            x0, y0, w, h = bbox_params[i]

            # Adjust cropping to use maximum width and height
            x_center = x0 + w / 2
            y_center = y0 + h / 2

            x0_new = int(max(0, x_center - max_w / 2))
            y0_new = int(max(0, y_center - max_h / 2))
            x1_new = int(min(x0_new + max_w, image.shape[2]))
            y1_new = int(min(y0_new + max_h, image.shape[1]))
            x0_new = x1_new - max_w
            y0_new = y1_new - max_h

            cropped_image = image[i][y0_new:y1_new, x0_new:x1_new, :]
            cropped_mask = mask[i][y0_new:y1_new, x0_new:x1_new]
            
            # Ensure dimensions are divisible by 16
            target_width = (target_width + 15) // 16 * 16
            target_height = (target_height + 15) // 16 * 16

            cropped_image = cropped_image.unsqueeze(0).movedim(-1, 1)  # Move C to the second position (B, C, H, W)
            cropped_image = common_upscale(cropped_image, target_width, target_height, "lanczos", "disabled")
            cropped_image = cropped_image.movedim(1, -1).squeeze(0)

            cropped_mask = cropped_mask.unsqueeze(0).unsqueeze(0)
            cropped_mask = common_upscale(cropped_mask, target_width, target_height, 'bilinear', "disabled")
            cropped_mask = cropped_mask.squeeze(0).squeeze(0)

            image_list.append(cropped_image)
            mask_list.append(cropped_mask)
            bbox_list.append((x0_new, y0_new, x1_new, y1_new))


        return (torch.stack(image_list), torch.stack(mask_list), bbox_list)
    
class ImageCropByMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "mask": ("MASK", ),           
            },
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )
    FUNCTION = "crop"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = "Crops the input images based on the provided mask."

    def crop(self, image, mask):
        B, H, W, C = image.shape
        mask = mask.round()
        
        # Find bounding box for each batch
        crops = []
        
        for b in range(B):
            # Get coordinates of non-zero elements
            rows = torch.any(mask[min(b, mask.shape[0]-1)] > 0, dim=1)
            cols = torch.any(mask[min(b, mask.shape[0]-1)] > 0, dim=0)
            
            # Find boundaries
            y_min, y_max = torch.where(rows)[0][[0, -1]]
            x_min, x_max = torch.where(cols)[0][[0, -1]]
            
            # Crop image and mask
            crop = image[b:b+1, y_min:y_max+1, x_min:x_max+1, :]            
            crops.append(crop)
        
        # Stack results back together
        cropped_images = torch.cat(crops, dim=0)
        
        return (cropped_images, )

       
    
class ImageUncropByMask:

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {    
                        "destination": ("IMAGE",),
                        "source": ("IMAGE",),
                        "mask": ("MASK",),
                        "bbox": ("BBOX",),
                     },
                }

    CATEGORY = "KJNodes/image"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "uncrop"

    def uncrop(self, destination, source, mask, bbox=None):

        output_list = []

        B, H, W, C = destination.shape
       
        for i in range(source.shape[0]):
            x0, y0, x1, y1 = bbox[i]
            bbox_height = y1 - y0
            bbox_width = x1 - x0

            # Resize source image to match the bounding box dimensions
            #resized_source = F.interpolate(source[i].unsqueeze(0).movedim(-1, 1), size=(bbox_height, bbox_width), mode='bilinear', align_corners=False)
            resized_source = common_upscale(source[i].unsqueeze(0).movedim(-1, 1), bbox_width, bbox_height, "lanczos", "disabled")
            resized_source = resized_source.movedim(1, -1).squeeze(0)
    
            # Resize mask to match the bounding box dimensions
            resized_mask = common_upscale(mask[i].unsqueeze(0).unsqueeze(0), bbox_width, bbox_height, "bilinear", "disabled")
            resized_mask = resized_mask.squeeze(0).squeeze(0)

            # Calculate padding values
            pad_left = x0
            pad_right = W - x1
            pad_top = y0
            pad_bottom = H - y1

            # Pad the resized source image and mask to fit the destination dimensions
            padded_source = F.pad(resized_source, pad=(0, 0, pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
            padded_mask = F.pad(resized_mask, pad=(pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

            # Ensure the padded mask has the correct shape
            padded_mask = padded_mask.unsqueeze(2).expand(-1, -1, destination[i].shape[2])
            # Ensure the padded source has the correct shape
            padded_source = padded_source.unsqueeze(2).expand(-1, -1, -1, destination[i].shape[2]).squeeze(2)
            
            # Combine the destination and padded source images using the mask
            result = destination[i] * (1.0 - padded_mask) + padded_source * padded_mask

            output_list.append(result)


        return (torch.stack(output_list),)
    
class ImageCropByMaskBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "image": ("IMAGE", ),
                    "masks": ("MASK", ),
                    "width": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                    "height": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                    "padding": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1, }),
                    "preserve_size": ("BOOLEAN", {"default": False}),
                    "bg_color": ("STRING", {"default": "0, 0, 0", "tooltip": "Color as RGB values in range 0-255, separated by commas."}),
                  }
                }
    
    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("images", "masks",)
    FUNCTION = "crop"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = "Crops the input images based on the provided masks."
        
    def crop(self, image, masks, width, height, bg_color, padding, preserve_size):
        B, H, W, C = image.shape
        BM, HM, WM = masks.shape
        mask_count = BM
        if HM != H or WM != W:
            masks = F.interpolate(masks.unsqueeze(1), size=(H, W), mode='nearest-exact').squeeze(1)
            print(masks.shape)
        output_images = []
        output_masks = []

        bg_color = [int(x.strip())/255.0 for x in bg_color.split(",")]
        
        # For each mask
        for i in range(mask_count):
            curr_mask = masks[i]
            
            # Find bounds
            y_indices, x_indices = torch.nonzero(curr_mask, as_tuple=True)
            if len(y_indices) == 0 or len(x_indices) == 0:
                continue
                
            # Get exact bounds with padding
            min_y = max(0, y_indices.min().item() - padding)
            max_y = min(H, y_indices.max().item() + 1 + padding)
            min_x = max(0, x_indices.min().item() - padding)
            max_x = min(W, x_indices.max().item() + 1 + padding)
            
            # Ensure mask has correct shape for multiplication
            curr_mask = curr_mask.unsqueeze(-1).expand(-1, -1, C)
            
            # Crop image and mask together
            cropped_img = image[0, min_y:max_y, min_x:max_x, :]
            cropped_mask = curr_mask[min_y:max_y, min_x:max_x, :]

            crop_h, crop_w = cropped_img.shape[0:2]
            new_w = crop_w
            new_h = crop_h

            if not preserve_size or crop_w > width or crop_h > height:
                scale = min(width/crop_w, height/crop_h)
                new_w = int(crop_w * scale)
                new_h = int(crop_h * scale)
                
                # Resize RGB
                resized_img = common_upscale(cropped_img.permute(2,0,1).unsqueeze(0), new_w, new_h, "lanczos", "disabled").squeeze(0).permute(1,2,0)
                resized_mask = torch.nn.functional.interpolate(
                    cropped_mask.permute(2,0,1).unsqueeze(0),
                    size=(new_h, new_w),
                    mode='nearest'
                ).squeeze(0).permute(1,2,0)
            else:
                resized_img = cropped_img
                resized_mask = cropped_mask

            # Create empty tensors
            new_img = torch.zeros((height, width, 3), dtype=image.dtype)
            new_mask = torch.zeros((height, width), dtype=image.dtype)

            # Pad both
            pad_x = (width - new_w) // 2
            pad_y = (height - new_h) // 2
            new_img[pad_y:pad_y+new_h, pad_x:pad_x+new_w, :] = resized_img
            if len(resized_mask.shape) == 3:
                resized_mask = resized_mask[:,:,0]  # Take first channel if 3D
            new_mask[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized_mask

            output_images.append(new_img)
            output_masks.append(new_mask)

        if not output_images:
            return (torch.zeros((0, height, width, 3), dtype=image.dtype),)

        out_rgb = torch.stack(output_images, dim=0)
        out_masks = torch.stack(output_masks, dim=0)

        # Apply mask to RGB
        mask_expanded = out_masks.unsqueeze(-1).expand(-1, -1, -1, 3)
        background_color = torch.tensor(bg_color, dtype=torch.float32, device=image.device)
        out_rgb = out_rgb * mask_expanded + background_color * (1 - mask_expanded)

        return (out_rgb, out_masks)
    
class ImagePadKJ:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "image": ("IMAGE", ),
                    "left": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                    "right": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                    "top": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                    "bottom": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                    "extra_padding": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                    "pad_mode": (["edge", "color"],),
                    "color": ("STRING", {"default": "0, 0, 0", "tooltip": "Color as RGB values in range 0-255, separated by commas."}),
                  },
                "optional": {
                    "mask": ("MASK", ),
                    "target_width": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1, "forceInput": True}),
                    "target_height": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1, "forceInput": True}),
                }
                }
    
    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("images", "masks",)
    FUNCTION = "pad"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = "Pad the input image and optionally mask with the specified padding."
        
    def pad(self, image, left, right, top, bottom, extra_padding, color, pad_mode, mask=None, target_width=None, target_height=None):
        B, H, W, C = image.shape
        
        # Resize masks to image dimensions if necessary
        if mask is not None:
            BM, HM, WM = mask.shape
            if HM != H or WM != W:
                mask = F.interpolate(mask.unsqueeze(1), size=(H, W), mode='nearest-exact').squeeze(1)

        # Parse background color
        bg_color = [int(x.strip())/255.0 for x in color.split(",")]
        if len(bg_color) == 1:
            bg_color = bg_color * 3  # Grayscale to RGB
        bg_color = torch.tensor(bg_color, dtype=image.dtype, device=image.device)
        
        # Calculate padding sizes with extra padding
        if target_width is not None and target_height is not None:
            if extra_padding > 0:
                image = common_upscale(image.movedim(-1, 1), W - extra_padding, H - extra_padding, "lanczos", "disabled").movedim(1, -1)
                B, H, W, C = image.shape

            padded_width = target_width
            padded_height = target_height
            pad_left = (padded_width - W) // 2
            pad_right = padded_width - W - pad_left
            pad_top = (padded_height - H) // 2
            pad_bottom = padded_height - H - pad_top
        else:
            pad_left = left + extra_padding
            pad_right = right + extra_padding
            pad_top = top + extra_padding
            pad_bottom = bottom + extra_padding

            padded_width = W + pad_left + pad_right
            padded_height = H + pad_top + pad_bottom
        out_image = torch.zeros((B, padded_height, padded_width, C), dtype=image.dtype, device=image.device)
        
        # Fill padded areas
        for b in range(B):
            if pad_mode == "edge":
                # Pad with edge color
                # Define edge pixels
                top_edge = image[b, 0, :, :]
                bottom_edge = image[b, H-1, :, :]
                left_edge = image[b, :, 0, :]
                right_edge = image[b, :, W-1, :]

                # Fill borders with edge colors
                out_image[b, :pad_top, :, :] = top_edge.mean(dim=0)
                out_image[b, pad_top+H:, :, :] = bottom_edge.mean(dim=0)
                out_image[b, :, :pad_left, :] = left_edge.mean(dim=0)
                out_image[b, :, pad_left+W:, :] = right_edge.mean(dim=0)
                out_image[b, pad_top:pad_top+H, pad_left:pad_left+W, :] = image[b]
            else:
                # Pad with specified background color
                out_image[b, :, :, :] = bg_color.unsqueeze(0).unsqueeze(0)  # Expand for H and W dimensions
                out_image[b, pad_top:pad_top+H, pad_left:pad_left+W, :] = image[b]

        
        if mask is not None:
            out_masks = torch.nn.functional.pad(
                mask, 
                (pad_left, pad_right, pad_top, pad_bottom),
                mode='replicate'
            )
        else:
            out_masks = torch.ones((B, padded_height, padded_width), dtype=image.dtype, device=image.device)
            for m in range(B):
                out_masks[m, pad_top:pad_top+H, pad_left:pad_left+W] = 0.0

        return (out_image, out_masks)
