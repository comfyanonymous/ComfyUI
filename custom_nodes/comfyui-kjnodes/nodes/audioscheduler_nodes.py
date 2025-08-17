# to be used with https://github.com/a1lazydog/ComfyUI-AudioScheduler 
import torch
from torchvision.transforms import functional as TF
from PIL import Image, ImageDraw
import numpy as np
from ..utility.utility import pil2tensor
from nodes import MAX_RESOLUTION

class NormalizedAmplitudeToMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "normalized_amp": ("NORMALIZED_AMPLITUDE",),
                    "width": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                    "height": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                    "frame_offset": ("INT", {"default": 0,"min": -255, "max": 255, "step": 1}),
                    "location_x": ("INT", {"default": 256,"min": 0, "max": 4096, "step": 1}),
                    "location_y": ("INT", {"default": 256,"min": 0, "max": 4096, "step": 1}),
                    "size": ("INT", {"default": 128,"min": 8, "max": 4096, "step": 1}),
                    "shape": (
                        [   
                            'none',
                            'circle',
                            'square',
                            'triangle',
                        ],
                        {
                        "default": 'none'
                        }),
                    "color": (
                        [   
                            'white',
                            'amplitude',
                        ],
                        {
                        "default": 'amplitude'
                        }),
                     },}

    CATEGORY = "KJNodes/audio"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "convert"
    DESCRIPTION = """
Works as a bridge to the AudioScheduler -nodes:  
https://github.com/a1lazydog/ComfyUI-AudioScheduler  
Creates masks based on the normalized amplitude.
"""

    def convert(self, normalized_amp, width, height, frame_offset, shape, location_x, location_y, size, color):
        # Ensure normalized_amp is an array and within the range [0, 1]
        normalized_amp = np.clip(normalized_amp, 0.0, 1.0)

        # Offset the amplitude values by rolling the array
        normalized_amp = np.roll(normalized_amp, frame_offset)
        
        # Initialize an empty list to hold the image tensors
        out = []
        # Iterate over each amplitude value to create an image
        for amp in normalized_amp:
            # Scale the amplitude value to cover the full range of grayscale values
            if color == 'amplitude':
                grayscale_value = int(amp * 255)
            elif color == 'white':
                grayscale_value = 255
            # Convert the grayscale value to an RGB format
            gray_color = (grayscale_value, grayscale_value, grayscale_value)
            finalsize = size * amp
            
            if shape == 'none':
                shapeimage = Image.new("RGB", (width, height), gray_color)
            else:
                shapeimage = Image.new("RGB", (width, height), "black")

            draw = ImageDraw.Draw(shapeimage)
            if shape == 'circle' or shape == 'square':
                # Define the bounding box for the shape
                left_up_point = (location_x - finalsize, location_y - finalsize)
                right_down_point = (location_x + finalsize,location_y + finalsize)
                two_points = [left_up_point, right_down_point]

                if shape == 'circle':
                    draw.ellipse(two_points, fill=gray_color)
                elif shape == 'square':
                    draw.rectangle(two_points, fill=gray_color)
                    
            elif shape == 'triangle':
                # Define the points for the triangle
                left_up_point = (location_x - finalsize, location_y + finalsize) # bottom left
                right_down_point = (location_x + finalsize, location_y + finalsize) # bottom right
                top_point = (location_x, location_y) # top point
                draw.polygon([top_point, left_up_point, right_down_point], fill=gray_color)
            
            shapeimage = pil2tensor(shapeimage)
            mask = shapeimage[:, :, :, 0]
            out.append(mask)
        
        return (torch.cat(out, dim=0),)
    
class NormalizedAmplitudeToFloatList:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "normalized_amp": ("NORMALIZED_AMPLITUDE",),
                     },}

    CATEGORY = "KJNodes/audio"
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "convert"
    DESCRIPTION = """
Works as a bridge to the AudioScheduler -nodes:  
https://github.com/a1lazydog/ComfyUI-AudioScheduler  
Creates a list of floats from the normalized amplitude.
"""

    def convert(self, normalized_amp):
        # Ensure normalized_amp is an array and within the range [0, 1]
        normalized_amp = np.clip(normalized_amp, 0.0, 1.0)
        return (normalized_amp.tolist(),)

class OffsetMaskByNormalizedAmplitude:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "normalized_amp": ("NORMALIZED_AMPLITUDE",),
                "mask": ("MASK",),
                "x": ("INT", { "default": 0, "min": -4096, "max": MAX_RESOLUTION, "step": 1, "display": "number" }),
                "y": ("INT", { "default": 0, "min": -4096, "max": MAX_RESOLUTION, "step": 1, "display": "number" }),
                "rotate": ("BOOLEAN", { "default": False }),
                "angle_multiplier": ("FLOAT", { "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001, "display": "number" }),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "offset"
    CATEGORY = "KJNodes/audio"
    DESCRIPTION = """
Works as a bridge to the AudioScheduler -nodes:  
https://github.com/a1lazydog/ComfyUI-AudioScheduler  
Offsets masks based on the normalized amplitude.
"""

    def offset(self, mask, x, y, angle_multiplier, rotate, normalized_amp):

         # Ensure normalized_amp is an array and within the range [0, 1]
        offsetmask = mask.clone()
        normalized_amp = np.clip(normalized_amp, 0.0, 1.0)
       
        batch_size, height, width = mask.shape

        if rotate:
            for i in range(batch_size):
                rotation_amp = int(normalized_amp[i] * (360 * angle_multiplier))
                rotation_angle = rotation_amp
                offsetmask[i] = TF.rotate(offsetmask[i].unsqueeze(0), rotation_angle).squeeze(0)
        if x != 0 or y != 0:
            for i in range(batch_size):
                offset_amp = normalized_amp[i] * 10
                shift_x = min(x*offset_amp, width-1)
                shift_y = min(y*offset_amp, height-1)
                if shift_x != 0:
                    offsetmask[i] = torch.roll(offsetmask[i], shifts=int(shift_x), dims=1)
                if shift_y != 0:
                    offsetmask[i] = torch.roll(offsetmask[i], shifts=int(shift_y), dims=0)
        
        return offsetmask,

class ImageTransformByNormalizedAmplitude:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "normalized_amp": ("NORMALIZED_AMPLITUDE",),
            "zoom_scale": ("FLOAT", { "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001, "display": "number" }),
            "x_offset": ("INT", { "default": 0, "min": (1 -MAX_RESOLUTION), "max": MAX_RESOLUTION, "step": 1, "display": "number" }),
            "y_offset": ("INT", { "default": 0, "min": (1 -MAX_RESOLUTION), "max": MAX_RESOLUTION, "step": 1, "display": "number" }),
            "cumulative": ("BOOLEAN", { "default": False }),
            "image": ("IMAGE",),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "amptransform"
    CATEGORY = "KJNodes/audio"
    DESCRIPTION = """
Works as a bridge to the AudioScheduler -nodes:  
https://github.com/a1lazydog/ComfyUI-AudioScheduler  
Transforms image based on the normalized amplitude.
"""

    def amptransform(self, image, normalized_amp, zoom_scale, cumulative, x_offset, y_offset):
        # Ensure normalized_amp is an array and within the range [0, 1]
        normalized_amp = np.clip(normalized_amp, 0.0, 1.0)
        transformed_images = []

        # Initialize the cumulative zoom factor
        prev_amp = 0.0

        for i in range(image.shape[0]):
            img = image[i]  # Get the i-th image in the batch
            amp = normalized_amp[i]  # Get the corresponding amplitude value

            # Incrementally increase the cumulative zoom factor
            if cumulative:
                prev_amp += amp
                amp += prev_amp

            # Convert the image tensor from BxHxWxC to CxHxW format expected by torchvision
            img = img.permute(2, 0, 1)
            
            # Convert PyTorch tensor to PIL Image for processing
            pil_img = TF.to_pil_image(img)
            
            # Calculate the crop size based on the amplitude
            width, height = pil_img.size
            crop_size = int(min(width, height) * (1 - amp * zoom_scale))
            crop_size = max(crop_size, 1)
            
            # Calculate the crop box coordinates (centered crop)
            left = (width - crop_size) // 2
            top = (height - crop_size) // 2
            right = (width + crop_size) // 2
            bottom = (height + crop_size) // 2
            
            # Crop and resize back to original size
            cropped_img = TF.crop(pil_img, top, left, crop_size, crop_size)
            resized_img = TF.resize(cropped_img, (height, width))
            
            # Convert back to tensor in CxHxW format
            tensor_img = TF.to_tensor(resized_img)
            
            # Convert the tensor back to BxHxWxC format
            tensor_img = tensor_img.permute(1, 2, 0)
            
            # Offset the image based on the amplitude
            offset_amp = amp * 10  # Calculate the offset magnitude based on the amplitude
            shift_x = min(x_offset * offset_amp, img.shape[1] - 1)  # Calculate the shift in x direction
            shift_y = min(y_offset * offset_amp, img.shape[0] - 1)  # Calculate the shift in y direction

            # Apply the offset to the image tensor
            if shift_x != 0:
                tensor_img = torch.roll(tensor_img, shifts=int(shift_x), dims=1)
            if shift_y != 0:
                tensor_img = torch.roll(tensor_img, shifts=int(shift_y), dims=0)

            # Add to the list
            transformed_images.append(tensor_img)
        
        # Stack all transformed images into a batch
        transformed_batch = torch.stack(transformed_images)
        
        return (transformed_batch,)