import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import scipy.ndimage
import numpy as np
from contextlib import nullcontext
import os

from comfy import model_management
from comfy.utils import ProgressBar
from comfy.utils import common_upscale
from nodes import MAX_RESOLUTION

import folder_paths

from ..utility.utility import tensor2pil, pil2tensor

script_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class BatchCLIPSeg:

    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
       
        return {"required":
                    {
                        "images": ("IMAGE",),
                        "text": ("STRING", {"multiline": False}),
                        "threshold": ("FLOAT", {"default": 0.5,"min": 0.0, "max": 10.0, "step": 0.001}),
                        "binary_mask": ("BOOLEAN", {"default": True}),
                        "combine_mask": ("BOOLEAN", {"default": False}),
                        "use_cuda": ("BOOLEAN", {"default": True}),
                     },
                     "optional":
                    {
                        "blur_sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                        "opt_model": ("CLIPSEGMODEL", ),
                        "prev_mask": ("MASK", {"default": None}),
                        "image_bg_level": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                        "invert": ("BOOLEAN", {"default": False}),
                    }
                }

    CATEGORY = "KJNodes/masking"
    RETURN_TYPES = ("MASK", "IMAGE", )
    RETURN_NAMES = ("Mask", "Image", )
    FUNCTION = "segment_image"
    DESCRIPTION = """
Segments an image or batch of images using CLIPSeg.
"""

    def segment_image(self, images, text, threshold, binary_mask, combine_mask, use_cuda, blur_sigma=0.0, opt_model=None, prev_mask=None, invert= False, image_bg_level=0.5):
        from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
        import torchvision.transforms as transforms
        offload_device = model_management.unet_offload_device()
        device = model_management.get_torch_device()
        if not use_cuda:
            device = torch.device("cpu")
        dtype = model_management.unet_dtype()

        if opt_model is None:
            checkpoint_path = os.path.join(folder_paths.models_dir,'clip_seg', 'clipseg-rd64-refined-fp16')
            if not hasattr(self, "model"):
                try:
                    if not os.path.exists(checkpoint_path):
                        from huggingface_hub import snapshot_download
                        snapshot_download(repo_id="Kijai/clipseg-rd64-refined-fp16", local_dir=checkpoint_path, local_dir_use_symlinks=False)
                    self.model = CLIPSegForImageSegmentation.from_pretrained(checkpoint_path)
                except:
                    checkpoint_path = "CIDAS/clipseg-rd64-refined"
                    self.model = CLIPSegForImageSegmentation.from_pretrained(checkpoint_path)
            processor = CLIPSegProcessor.from_pretrained(checkpoint_path)

        else:
            self.model = opt_model['model']
            processor = opt_model['processor']

        self.model.to(dtype).to(device)

        B, H, W, C = images.shape
        images = images.to(device)
        
        autocast_condition = (dtype != torch.float32) and not model_management.is_device_mps(device)
        with torch.autocast(model_management.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():

            PIL_images = [Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)) for image in images ]
            prompt = [text] * len(images)
            input_prc = processor(text=prompt, images=PIL_images, return_tensors="pt")

            for key in input_prc:
                input_prc[key] = input_prc[key].to(device)
            outputs = self.model(**input_prc)

        mask_tensor = torch.sigmoid(outputs.logits)
        mask_tensor = (mask_tensor - mask_tensor.min()) / (mask_tensor.max() - mask_tensor.min())
        mask_tensor = torch.where(mask_tensor > (threshold), mask_tensor, torch.tensor(0, dtype=torch.float))
        print(mask_tensor.shape)
        if len(mask_tensor.shape) == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        mask_tensor = F.interpolate(mask_tensor.unsqueeze(1), size=(H, W), mode='nearest')
        mask_tensor = mask_tensor.squeeze(1)

        self.model.to(offload_device)
        
        if binary_mask:
            mask_tensor = (mask_tensor > 0).float()
        if blur_sigma > 0:
            kernel_size = int(6 * int(blur_sigma) + 1) 
            blur = transforms.GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=(blur_sigma, blur_sigma))
            mask_tensor = blur(mask_tensor)

        if combine_mask:
            mask_tensor = torch.max(mask_tensor, dim=0)[0]
            mask_tensor = mask_tensor.unsqueeze(0).repeat(len(images),1,1)

        del outputs
        model_management.soft_empty_cache()

        if prev_mask is not None:
            if prev_mask.shape != mask_tensor.shape:
                prev_mask = F.interpolate(prev_mask.unsqueeze(1), size=(H, W), mode='nearest')
            mask_tensor = mask_tensor + prev_mask.to(device)
            torch.clamp(mask_tensor, min=0.0, max=1.0)

        if invert:
            mask_tensor = 1 - mask_tensor

        image_tensor = images * mask_tensor.unsqueeze(-1) + (1 - mask_tensor.unsqueeze(-1)) * image_bg_level
        image_tensor = torch.clamp(image_tensor, min=0.0, max=1.0).cpu().float()

        mask_tensor = mask_tensor.cpu().float()
    
        return mask_tensor, image_tensor, 

class DownloadAndLoadCLIPSeg:

    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
       
        return {"required":
                    {     
                    "model": (
                    [   'Kijai/clipseg-rd64-refined-fp16',
                        'CIDAS/clipseg-rd64-refined',
                    ],
                    ),
                     },
                }

    CATEGORY = "KJNodes/masking"
    RETURN_TYPES = ("CLIPSEGMODEL",)
    RETURN_NAMES = ("clipseg_model",)
    FUNCTION = "segment_image"
    DESCRIPTION = """
Downloads and loads CLIPSeg model with huggingface_hub,  
to ComfyUI/models/clip_seg
"""

    def segment_image(self, model):
        from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
        checkpoint_path = os.path.join(folder_paths.models_dir,'clip_seg', os.path.basename(model))
        if not hasattr(self, "model"):
            if not os.path.exists(checkpoint_path):
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id=model, local_dir=checkpoint_path, local_dir_use_symlinks=False)
            self.model = CLIPSegForImageSegmentation.from_pretrained(checkpoint_path)

        processor = CLIPSegProcessor.from_pretrained(checkpoint_path)

        clipseg_model = {}
        clipseg_model['model'] = self.model
        clipseg_model['processor'] = processor

        return clipseg_model,

class CreateTextMask:

    RETURN_TYPES = ("IMAGE", "MASK",)
    FUNCTION = "createtextmask"
    CATEGORY = "KJNodes/text"
    DESCRIPTION = """
Creates a text image and mask.  
Looks for fonts from this folder:  
ComfyUI/custom_nodes/ComfyUI-KJNodes/fonts
  
If start_rotation and/or end_rotation are different values,  
creates animation between them.
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "invert": ("BOOLEAN", {"default": False}),
                 "frames": ("INT", {"default": 1,"min": 1, "max": 4096, "step": 1}),
                 "text_x": ("INT", {"default": 0,"min": 0, "max": 4096, "step": 1}),
                 "text_y": ("INT", {"default": 0,"min": 0, "max": 4096, "step": 1}),
                 "font_size": ("INT", {"default": 32,"min": 8, "max": 4096, "step": 1}),
                 "font_color": ("STRING", {"default": "white"}),
                 "text": ("STRING", {"default": "HELLO!", "multiline": True}),
                 "font": (folder_paths.get_filename_list("kjnodes_fonts"), ),
                 "width": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                 "height": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                 "start_rotation": ("INT", {"default": 0,"min": 0, "max": 359, "step": 1}),
                 "end_rotation": ("INT", {"default": 0,"min": -359, "max": 359, "step": 1}),
        },
    } 

    def createtextmask(self, frames, width, height, invert, text_x, text_y, text, font_size, font_color, font, start_rotation, end_rotation):
    # Define the number of images in the batch
        batch_size = frames
        out = []
        masks = []
        rotation = start_rotation
        if start_rotation != end_rotation:
            rotation_increment = (end_rotation - start_rotation) / (batch_size - 1)

        font_path = folder_paths.get_full_path("kjnodes_fonts", font)
        # Generate the text
        for i in range(batch_size):
            image = Image.new("RGB", (width, height), "black")
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype(font_path, font_size)
            
            # Split the text into words
            words = text.split()
            
            # Initialize variables for line creation
            lines = []
            current_line = []
            current_line_width = 0
            try: #new pillow  
                # Iterate through words to create lines
                for word in words:
                    word_width = font.getbbox(word)[2]
                    if current_line_width + word_width <= width - 2 * text_x:
                        current_line.append(word)
                        current_line_width += word_width + font.getbbox(" ")[2] # Add space width
                    else:
                        lines.append(" ".join(current_line))
                        current_line = [word]
                        current_line_width = word_width
            except: #old pillow             
                for word in words:
                    word_width = font.getsize(word)[0]
                    if current_line_width + word_width <= width - 2 * text_x:
                        current_line.append(word)
                        current_line_width += word_width + font.getsize(" ")[0] # Add space width
                    else:
                        lines.append(" ".join(current_line))
                        current_line = [word]
                        current_line_width = word_width
            
            # Add the last line if it's not empty
            if current_line:
                lines.append(" ".join(current_line))
            
            # Draw each line of text separately
            y_offset = text_y
            for line in lines:
                text_width = font.getlength(line)
                text_height = font_size
                text_center_x = text_x + text_width / 2
                text_center_y = y_offset + text_height / 2
                try:
                    draw.text((text_x, y_offset), line, font=font, fill=font_color, features=['-liga'])
                except:
                    draw.text((text_x, y_offset), line, font=font, fill=font_color)
                y_offset += text_height # Move to the next line
            
            if start_rotation != end_rotation:
                image = image.rotate(rotation, center=(text_center_x, text_center_y))
                rotation += rotation_increment
            
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            mask = image[:, :, :, 0] 
            masks.append(mask)
            out.append(image)
            
        if invert:
            return (1.0 - torch.cat(out, dim=0), 1.0 - torch.cat(masks, dim=0),)
        return (torch.cat(out, dim=0),torch.cat(masks, dim=0),)

class ColorToMask:
    
    RETURN_TYPES = ("MASK",)
    FUNCTION = "clip"
    CATEGORY = "KJNodes/masking"
    DESCRIPTION = """
Converts chosen RGB value to a mask.  
With batch inputs, the **per_batch**  
controls the number of images processed at once.
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "images": ("IMAGE",),
                 "invert": ("BOOLEAN", {"default": False}),
                 "red": ("INT", {"default": 0,"min": 0, "max": 255, "step": 1}),
                 "green": ("INT", {"default": 0,"min": 0, "max": 255, "step": 1}),
                 "blue": ("INT", {"default": 0,"min": 0, "max": 255, "step": 1}),
                 "threshold": ("INT", {"default": 10,"min": 0, "max": 255, "step": 1}),
                 "per_batch": ("INT", {"default": 16, "min": 1, "max": 4096, "step": 1}),
        },
    } 

    def clip(self, images, red, green, blue, threshold, invert, per_batch):

        color = torch.tensor([red, green, blue], dtype=torch.uint8)  
        black = torch.tensor([0, 0, 0], dtype=torch.uint8)
        white = torch.tensor([255, 255, 255], dtype=torch.uint8)
        
        if invert:
            black, white = white, black

        steps = images.shape[0]
        pbar = ProgressBar(steps)
        tensors_out = []
        
        for start_idx in range(0, images.shape[0], per_batch):

            # Calculate color distances
            color_distances = torch.norm(images[start_idx:start_idx+per_batch] * 255 - color, dim=-1)
            
            # Create a mask based on the threshold
            mask = color_distances <= threshold
            
            # Apply the mask to create new images
            mask_out = torch.where(mask.unsqueeze(-1), white, black).float()
            mask_out = mask_out.mean(dim=-1)

            tensors_out.append(mask_out.cpu())
            batch_count = mask_out.shape[0]
            pbar.update(batch_count)
       
        tensors_out = torch.cat(tensors_out, dim=0)
        tensors_out = torch.clamp(tensors_out, min=0.0, max=1.0)
        return tensors_out,
      
class CreateFluidMask:
    
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "createfluidmask"
    CATEGORY = "KJNodes/masking/generate"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "invert": ("BOOLEAN", {"default": False}),
                 "frames": ("INT", {"default": 1,"min": 1, "max": 4096, "step": 1}),
                 "width": ("INT", {"default": 256,"min": 16, "max": 4096, "step": 1}),
                 "height": ("INT", {"default": 256,"min": 16, "max": 4096, "step": 1}),
                 "inflow_count": ("INT", {"default": 3,"min": 0, "max": 255, "step": 1}),
                 "inflow_velocity": ("INT", {"default": 1,"min": 0, "max": 255, "step": 1}),
                 "inflow_radius": ("INT", {"default": 8,"min": 0, "max": 255, "step": 1}),
                 "inflow_padding": ("INT", {"default": 50,"min": 0, "max": 255, "step": 1}),
                 "inflow_duration": ("INT", {"default": 60,"min": 0, "max": 255, "step": 1}),
        },
    } 
    #using code from https://github.com/GregTJ/stable-fluids
    def createfluidmask(self, frames, width, height, invert, inflow_count, inflow_velocity, inflow_radius, inflow_padding, inflow_duration):
        from ..utility.fluid import Fluid
        try:
            from scipy.special import erf
        except:
            from scipy.spatial import erf
        out = []
        masks = []
        RESOLUTION = width, height
        DURATION = frames

        INFLOW_PADDING = inflow_padding
        INFLOW_DURATION = inflow_duration
        INFLOW_RADIUS = inflow_radius
        INFLOW_VELOCITY = inflow_velocity
        INFLOW_COUNT = inflow_count

        print('Generating fluid solver, this may take some time.')
        fluid = Fluid(RESOLUTION, 'dye')

        center = np.floor_divide(RESOLUTION, 2)
        r = np.min(center) - INFLOW_PADDING

        points = np.linspace(-np.pi, np.pi, INFLOW_COUNT, endpoint=False)
        points = tuple(np.array((np.cos(p), np.sin(p))) for p in points)
        normals = tuple(-p for p in points)
        points = tuple(r * p + center for p in points)

        inflow_velocity = np.zeros_like(fluid.velocity)
        inflow_dye = np.zeros(fluid.shape)
        for p, n in zip(points, normals):
            mask = np.linalg.norm(fluid.indices - p[:, None, None], axis=0) <= INFLOW_RADIUS
            inflow_velocity[:, mask] += n[:, None] * INFLOW_VELOCITY
            inflow_dye[mask] = 1

        
        for f in range(DURATION):
            print(f'Computing frame {f + 1} of {DURATION}.')
            if f <= INFLOW_DURATION:
                fluid.velocity += inflow_velocity
                fluid.dye += inflow_dye

            curl = fluid.step()[1]
            # Using the error function to make the contrast a bit higher. 
            # Any other sigmoid function e.g. smoothstep would work.
            curl = (erf(curl * 2) + 1) / 4

            color = np.dstack((curl, np.ones(fluid.shape), fluid.dye))
            color = (np.clip(color, 0, 1) * 255).astype('uint8')
            image = np.array(color).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            mask = image[:, :, :, 0] 
            masks.append(mask)
            out.append(image)
        
        if invert:
            return (1.0 - torch.cat(out, dim=0),1.0 - torch.cat(masks, dim=0),)
        return (torch.cat(out, dim=0),torch.cat(masks, dim=0),)

class CreateAudioMask:
       
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "createaudiomask"
    CATEGORY = "KJNodes/deprecated"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "invert": ("BOOLEAN", {"default": False}),
                 "frames": ("INT", {"default": 16,"min": 1, "max": 255, "step": 1}),
                 "scale": ("FLOAT", {"default": 0.5,"min": 0.0, "max": 2.0, "step": 0.01}),
                 "audio_path": ("STRING", {"default": "audio.wav"}),
                 "width": ("INT", {"default": 256,"min": 16, "max": 4096, "step": 1}),
                 "height": ("INT", {"default": 256,"min": 16, "max": 4096, "step": 1}),
        },
    } 

    def createaudiomask(self, frames, width, height, invert, audio_path, scale):
        try:
            import librosa
        except ImportError:
            raise Exception("Can not import librosa. Install it with 'pip install librosa'")
        batch_size = frames
        out = []
        masks = []
        if audio_path == "audio.wav": #I don't know why relative path won't work otherwise...
            audio_path = os.path.join(script_directory, audio_path)
        audio, sr = librosa.load(audio_path)
        spectrogram = np.abs(librosa.stft(audio))
        
        for i in range(batch_size):
           image = Image.new("RGB", (width, height), "black")
           draw = ImageDraw.Draw(image)
           frame = spectrogram[:, i]
           circle_radius = int(height * np.mean(frame))
           circle_radius *= scale
           circle_center = (width // 2, height // 2)  # Calculate the center of the image

           draw.ellipse([(circle_center[0] - circle_radius, circle_center[1] - circle_radius),
                      (circle_center[0] + circle_radius, circle_center[1] + circle_radius)],
                      fill='white')
             
           image = np.array(image).astype(np.float32) / 255.0
           image = torch.from_numpy(image)[None,]
           mask = image[:, :, :, 0] 
           masks.append(mask)
           out.append(image)

        if invert:
            return (1.0 - torch.cat(out, dim=0),)
        return (torch.cat(out, dim=0),torch.cat(masks, dim=0),)
       
class CreateGradientMask:
    
    RETURN_TYPES = ("MASK",)
    FUNCTION = "createmask"
    CATEGORY = "KJNodes/masking/generate"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "invert": ("BOOLEAN", {"default": False}),
                 "frames": ("INT", {"default": 0,"min": 0, "max": 255, "step": 1}),
                 "width": ("INT", {"default": 256,"min": 16, "max": 4096, "step": 1}),
                 "height": ("INT", {"default": 256,"min": 16, "max": 4096, "step": 1}),
        },
    } 
    def createmask(self, frames, width, height, invert):
        # Define the number of images in the batch
        batch_size = frames
        out = []
        # Create an empty array to store the image batch
        image_batch = np.zeros((batch_size, height, width), dtype=np.float32)
        # Generate the black to white gradient for each image
        for i in range(batch_size):
            gradient = np.linspace(1.0, 0.0, width, dtype=np.float32)
            time = i / frames  # Calculate the time variable
            offset_gradient = gradient - time  # Offset the gradient values based on time
            image_batch[i] = offset_gradient.reshape(1, -1)
        output = torch.from_numpy(image_batch)
        mask = output
        out.append(mask)
        if invert:
            return (1.0 - torch.cat(out, dim=0),)
        return (torch.cat(out, dim=0),)

class CreateFadeMask:
    
    RETURN_TYPES = ("MASK",)
    FUNCTION = "createfademask"
    CATEGORY = "KJNodes/deprecated"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "invert": ("BOOLEAN", {"default": False}),
                 "frames": ("INT", {"default": 2,"min": 2, "max": 10000, "step": 1}),
                 "width": ("INT", {"default": 256,"min": 16, "max": 4096, "step": 1}),
                 "height": ("INT", {"default": 256,"min": 16, "max": 4096, "step": 1}),
                 "interpolation": (["linear", "ease_in", "ease_out", "ease_in_out"],),
                 "start_level": ("FLOAT", {"default": 1.0,"min": 0.0, "max": 1.0, "step": 0.01}),
                 "midpoint_level": ("FLOAT", {"default": 0.5,"min": 0.0, "max": 1.0, "step": 0.01}),
                 "end_level": ("FLOAT", {"default": 0.0,"min": 0.0, "max": 1.0, "step": 0.01}),
                 "midpoint_frame": ("INT", {"default": 0,"min": 0, "max": 4096, "step": 1}),
        },
    } 
    
    def createfademask(self, frames, width, height, invert, interpolation, start_level, midpoint_level, end_level, midpoint_frame):
        def ease_in(t):
            return t * t

        def ease_out(t):
            return 1 - (1 - t) * (1 - t)

        def ease_in_out(t):
            return 3 * t * t - 2 * t * t * t

        batch_size = frames
        out = []
        image_batch = np.zeros((batch_size, height, width), dtype=np.float32)

        if midpoint_frame == 0:
            midpoint_frame = batch_size // 2

        for i in range(batch_size):
            if i <= midpoint_frame:
                t = i / midpoint_frame
                if interpolation == "ease_in":
                    t = ease_in(t)
                elif interpolation == "ease_out":
                    t = ease_out(t)
                elif interpolation == "ease_in_out":
                    t = ease_in_out(t)
                color = start_level - t * (start_level - midpoint_level)
            else:
                t = (i - midpoint_frame) / (batch_size - midpoint_frame)
                if interpolation == "ease_in":
                    t = ease_in(t)
                elif interpolation == "ease_out":
                    t = ease_out(t)
                elif interpolation == "ease_in_out":
                    t = ease_in_out(t)
                color = midpoint_level - t * (midpoint_level - end_level)

            color = np.clip(color, 0, 255)
            image = np.full((height, width), color, dtype=np.float32)
            image_batch[i] = image

        output = torch.from_numpy(image_batch)
        mask = output
        out.append(mask)

        if invert:
            return (1.0 - torch.cat(out, dim=0),)
        return (torch.cat(out, dim=0),)

class CreateFadeMaskAdvanced:
    
    RETURN_TYPES = ("MASK",)
    FUNCTION = "createfademask"
    CATEGORY = "KJNodes/masking/generate"
    DESCRIPTION = """
Create a batch of masks interpolated between given frames and values. 
Uses same syntax as Fizz' BatchValueSchedule.
First value is the frame index (not that this starts from 0, not 1) 
and the second value inside the brackets is the float value of the mask in range 0.0 - 1.0  

For example the default values:  
0:(0.0)  
7:(1.0)  
15:(0.0)  
  
Would create a mask batch fo 16 frames, starting from black, 
interpolating with the chosen curve to fully white at the 8th frame, 
and interpolating from that to fully black at the 16th frame.
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "points_string": ("STRING", {"default": "0:(0.0),\n7:(1.0),\n15:(0.0)\n", "multiline": True}),
                 "invert": ("BOOLEAN", {"default": False}),
                 "frames": ("INT", {"default": 16,"min": 2, "max": 10000, "step": 1}),
                 "width": ("INT", {"default": 512,"min": 1, "max": 4096, "step": 1}),
                 "height": ("INT", {"default": 512,"min": 1, "max": 4096, "step": 1}),
                 "interpolation": (["linear", "ease_in", "ease_out", "ease_in_out", "none", "default_to_black"],),
        },
    } 
    
    def createfademask(self, frames, width, height, invert, points_string, interpolation):
        def ease_in(t):
            return t * t
        
        def ease_out(t):
            return 1 - (1 - t) * (1 - t)

        def ease_in_out(t):
            return 3 * t * t - 2 * t * t * t
        
        # Parse the input string into a list of tuples
        points = []
        points_string = points_string.rstrip(',\n')
        for point_str in points_string.split(','):
            frame_str, color_str = point_str.split(':')
            frame = int(frame_str.strip())
            color = float(color_str.strip()[1:-1])  # Remove parentheses around color
            points.append((frame, color))

        # Check if the last frame is already in the points
        if (interpolation != "default_to_black") and (len(points) == 0 or points[-1][0] != frames - 1):
            # If not, add it with the color of the last specified frame
            points.append((frames - 1, points[-1][1] if points else 0))

        # Sort the points by frame number
        points.sort(key=lambda x: x[0])

        batch_size = frames
        out = []
        image_batch = np.zeros((batch_size, height, width), dtype=np.float32)

        # Index of the next point to interpolate towards
        next_point = 1

        for i in range(batch_size):
            while next_point < len(points) and i > points[next_point][0]:
                next_point += 1

            # Interpolate between the previous point and the next point
            prev_point = next_point - 1

            if interpolation == "none":
                exact_match = False
                for p in points:
                    if p[0] == i:  # Exact frame match
                        color = p[1]
                        exact_match = True
                        break
                if not exact_match:
                    color = points[prev_point][1]

            elif interpolation == "default_to_black":
                exact_match = False
                for p in points:
                    if p[0] == i:  # Exact frame match
                        color = p[1]
                        exact_match = True
                        break
                if not exact_match:
                    color = 0        
            else:
                t = (i - points[prev_point][0]) / (points[next_point][0] - points[prev_point][0])
                if interpolation == "ease_in":
                    t = ease_in(t)
                elif interpolation == "ease_out":
                    t = ease_out(t)
                elif interpolation == "ease_in_out":
                    t = ease_in_out(t)
                elif interpolation == "linear":
                    pass  # No need to modify `t` for linear interpolation

                color = points[prev_point][1] - t * (points[prev_point][1] - points[next_point][1])
                
            color = np.clip(color, 0, 255)
            image = np.full((height, width), color, dtype=np.float32)
            image_batch[i] = image

        output = torch.from_numpy(image_batch)
        mask = output
        out.append(mask)

        if invert:
            return (1.0 - torch.cat(out, dim=0),)
        return (torch.cat(out, dim=0),)

class CreateMagicMask:
    
    RETURN_TYPES = ("MASK", "MASK",)
    RETURN_NAMES = ("mask", "mask_inverted",)
    FUNCTION = "createmagicmask"
    CATEGORY = "KJNodes/masking/generate"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "frames": ("INT", {"default": 16,"min": 2, "max": 4096, "step": 1}),
                 "depth": ("INT", {"default": 12,"min": 1, "max": 500, "step": 1}),
                 "distortion": ("FLOAT", {"default": 1.5,"min": 0.0, "max": 100.0, "step": 0.01}),
                 "seed": ("INT", {"default": 123,"min": 0, "max": 99999999, "step": 1}),
                 "transitions": ("INT", {"default": 1,"min": 1, "max": 20, "step": 1}),
                 "frame_width": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                 "frame_height": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
        },
    } 

    def createmagicmask(self, frames, transitions, depth, distortion, seed, frame_width, frame_height):
        from ..utility.magictex import coordinate_grid, random_transform, magic
        import matplotlib.pyplot as plt
        rng = np.random.default_rng(seed)
        out = []
        coords = coordinate_grid((frame_width, frame_height))

        # Calculate the number of frames for each transition
        frames_per_transition = frames // transitions

        # Generate a base set of parameters
        base_params = {
            "coords": random_transform(coords, rng),
            "depth": depth,
            "distortion": distortion,
        }
        for t in range(transitions):
        # Generate a second set of parameters that is at most max_diff away from the base parameters
            params1 = base_params.copy()
            params2 = base_params.copy()

            params1['coords'] = random_transform(coords, rng)
            params2['coords'] = random_transform(coords, rng)

            for i in range(frames_per_transition):
                # Compute the interpolation factor
                alpha = i / frames_per_transition

                # Interpolate between the two sets of parameters
                params = params1.copy()
                params['coords'] = (1 - alpha) * params1['coords'] + alpha * params2['coords']

                tex = magic(**params)

                dpi = frame_width / 10
                fig = plt.figure(figsize=(10, 10), dpi=dpi)

                ax = fig.add_subplot(111)
                plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
                
                ax.get_yaxis().set_ticks([])
                ax.get_xaxis().set_ticks([])
                ax.imshow(tex, aspect='auto')
                
                fig.canvas.draw()
                img = np.array(fig.canvas.renderer._renderer)
                
                plt.close(fig)
                
                pil_img = Image.fromarray(img).convert("L")
                mask = torch.tensor(np.array(pil_img)) / 255.0
                
                out.append(mask)
        
        return (torch.stack(out, dim=0), 1.0 - torch.stack(out, dim=0),)
        
class CreateShapeMask:
    
    RETURN_TYPES = ("MASK", "MASK",)
    RETURN_NAMES = ("mask", "mask_inverted",)
    FUNCTION = "createshapemask"
    CATEGORY = "KJNodes/masking/generate"
    DESCRIPTION = """
Creates a mask or batch of masks with the specified shape.  
Locations are center locations.  
Grow value is the amount to grow the shape on each frame, creating animated masks.
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "shape": (
            [   'circle',
                'square',
                'triangle',
            ],
            {
            "default": 'circle'
             }),
                "frames": ("INT", {"default": 1,"min": 1, "max": 4096, "step": 1}),
                "location_x": ("INT", {"default": 256,"min": 0, "max": 4096, "step": 1}),
                "location_y": ("INT", {"default": 256,"min": 0, "max": 4096, "step": 1}),
                "grow": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                "frame_width": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                "frame_height": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                "shape_width": ("INT", {"default": 128,"min": 8, "max": 4096, "step": 1}),
                "shape_height": ("INT", {"default": 128,"min": 8, "max": 4096, "step": 1}),
        },
    } 

    def createshapemask(self, frames, frame_width, frame_height, location_x, location_y, shape_width, shape_height, grow, shape):
        # Define the number of images in the batch
        batch_size = frames
        out = []
        color = "white"
        for i in range(batch_size):
            image = Image.new("RGB", (frame_width, frame_height), "black")
            draw = ImageDraw.Draw(image)

            # Calculate the size for this frame and ensure it's not less than 0
            current_width = max(0, shape_width + i*grow)
            current_height = max(0, shape_height + i*grow)

            if shape == 'circle' or shape == 'square':
                # Define the bounding box for the shape
                left_up_point = (location_x - current_width // 2, location_y - current_height // 2)
                right_down_point = (location_x + current_width // 2, location_y + current_height // 2)
                two_points = [left_up_point, right_down_point]

                if shape == 'circle':
                    draw.ellipse(two_points, fill=color)
                elif shape == 'square':
                    draw.rectangle(two_points, fill=color)
                    
            elif shape == 'triangle':
                # Define the points for the triangle
                left_up_point = (location_x - current_width // 2, location_y + current_height // 2) # bottom left
                right_down_point = (location_x + current_width // 2, location_y + current_height // 2) # bottom right
                top_point = (location_x, location_y - current_height // 2) # top point
                draw.polygon([top_point, left_up_point, right_down_point], fill=color)

            image = pil2tensor(image)
            mask = image[:, :, :, 0]
            out.append(mask)
        outstack = torch.cat(out, dim=0)
        return (outstack, 1.0 - outstack,)
    
class CreateVoronoiMask:
    
    RETURN_TYPES = ("MASK", "MASK",)
    RETURN_NAMES = ("mask", "mask_inverted",)
    FUNCTION = "createvoronoi"
    CATEGORY = "KJNodes/masking/generate"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "frames": ("INT", {"default": 16,"min": 2, "max": 4096, "step": 1}),
                 "num_points": ("INT", {"default": 15,"min": 1, "max": 4096, "step": 1}),
                 "line_width": ("INT", {"default": 4,"min": 1, "max": 4096, "step": 1}),
                 "speed": ("FLOAT", {"default": 0.5,"min": 0.0, "max": 1.0, "step": 0.01}),
                 "frame_width": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                 "frame_height": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
        },
    } 

    def createvoronoi(self, frames, num_points, line_width, speed, frame_width, frame_height):
        from scipy.spatial import Voronoi
        # Define the number of images in the batch
        batch_size = frames
        out = []
          
        # Calculate aspect ratio
        aspect_ratio = frame_width / frame_height
        
        # Create start and end points for each point, considering the aspect ratio
        start_points = np.random.rand(num_points, 2)
        start_points[:, 0] *= aspect_ratio
        
        end_points = np.random.rand(num_points, 2)
        end_points[:, 0] *= aspect_ratio

        for i in range(batch_size):
            # Interpolate the points' positions based on the current frame
            t = (i * speed) / (batch_size - 1)  # normalize to [0, 1] over the frames
            t = np.clip(t, 0, 1)  # ensure t is in [0, 1]
            points = (1 - t) * start_points + t * end_points  # lerp

            # Adjust points for aspect ratio
            points[:, 0] *= aspect_ratio

            vor = Voronoi(points)

            # Create a blank image with a white background
            fig, ax = plt.subplots()
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
            ax.set_xlim([0, aspect_ratio]); ax.set_ylim([0, 1])  # adjust x limits
            ax.axis('off')
            ax.margins(0, 0)
            fig.set_size_inches(aspect_ratio * frame_height/100, frame_height/100)  # adjust figure size
            ax.fill_between([0, 1], [0, 1], color='white')

            # Plot each Voronoi ridge
            for simplex in vor.ridge_vertices:
                simplex = np.asarray(simplex)
                if np.all(simplex >= 0):
                    plt.plot(vor.vertices[simplex, 0], vor.vertices[simplex, 1], 'k-', linewidth=line_width)

            fig.canvas.draw()
            img = np.array(fig.canvas.renderer._renderer)

            plt.close(fig)

            pil_img = Image.fromarray(img).convert("L")
            mask = torch.tensor(np.array(pil_img)) / 255.0

            out.append(mask)

        return (torch.stack(out, dim=0), 1.0 - torch.stack(out, dim=0),)
    
class GetMaskSizeAndCount:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "mask": ("MASK",),
        }}

    RETURN_TYPES = ("MASK","INT", "INT", "INT",)
    RETURN_NAMES = ("mask", "width", "height", "count",)
    FUNCTION = "getsize"
    CATEGORY = "KJNodes/masking"
    DESCRIPTION = """
Returns the width, height and batch size of the mask,  
and passes it through unchanged.  

"""

    def getsize(self, mask):
        width = mask.shape[2]
        height = mask.shape[1]
        count = mask.shape[0]
        return {"ui": {
            "text": [f"{count}x{width}x{height}"]}, 
            "result": (mask, width, height, count) 
        }

class GrowMaskWithBlur:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "expand": ("INT", {"default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1}),
                "incremental_expandrate": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "tapered_corners": ("BOOLEAN", {"default": True}),
                "flip_input": ("BOOLEAN", {"default": False}),
                "blur_radius": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 100,
                    "step": 0.1
                }),
                "lerp_alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "decay_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "fill_holes": ("BOOLEAN", {"default": False}),
            },
        }

    CATEGORY = "KJNodes/masking"
    RETURN_TYPES = ("MASK", "MASK",)
    RETURN_NAMES = ("mask", "mask_inverted",)
    FUNCTION = "expand_mask"
    DESCRIPTION = """
# GrowMaskWithBlur
- mask: Input mask or mask batch
- expand: Expand or contract mask or mask batch by a given amount
- incremental_expandrate: increase expand rate by a given amount per frame
- tapered_corners: use tapered corners
- flip_input: flip input mask
- blur_radius: value higher than 0 will blur the mask
- lerp_alpha: alpha value for interpolation between frames
- decay_factor: decay value for interpolation between frames
- fill_holes: fill holes in the mask (slow)"""
    
    def expand_mask(self, mask, expand, tapered_corners, flip_input, blur_radius, incremental_expandrate, lerp_alpha, decay_factor, fill_holes=False):
        alpha = lerp_alpha
        decay = decay_factor
        if flip_input:
            mask = 1.0 - mask
        c = 0 if tapered_corners else 1
        kernel = np.array([[c, 1, c],
                           [1, 1, 1],
                           [c, 1, c]])
        growmask = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).cpu()
        out = []
        previous_output = None
        current_expand = expand
        for m in growmask:
            output = m.numpy().astype(np.float32)
            for _ in range(abs(round(current_expand))):
                if current_expand < 0:
                    output = scipy.ndimage.grey_erosion(output, footprint=kernel)
                else:
                    output = scipy.ndimage.grey_dilation(output, footprint=kernel)
            if current_expand < 0:
                current_expand -= abs(incremental_expandrate)
            else:
                current_expand += abs(incremental_expandrate)
            if fill_holes:
                binary_mask = output > 0
                output = scipy.ndimage.binary_fill_holes(binary_mask)
                output = output.astype(np.float32) * 255
            output = torch.from_numpy(output)
            if alpha < 1.0 and previous_output is not None:
                # Interpolate between the previous and current frame
                output = alpha * output + (1 - alpha) * previous_output
            if decay < 1.0 and previous_output is not None:
                # Add the decayed previous output to the current frame
                output += decay * previous_output
                output = output / output.max()
            previous_output = output
            out.append(output)

        if blur_radius != 0:
            # Convert the tensor list to PIL images, apply blur, and convert back
            for idx, tensor in enumerate(out):
                # Convert tensor to PIL image
                pil_image = tensor2pil(tensor.cpu().detach())[0]
                # Apply Gaussian blur
                pil_image = pil_image.filter(ImageFilter.GaussianBlur(blur_radius))
                # Convert back to tensor
                out[idx] = pil2tensor(pil_image)
            blurred = torch.cat(out, dim=0)
            return (blurred, 1.0 - blurred)
        else:
            return (torch.stack(out, dim=0), 1.0 - torch.stack(out, dim=0),)
        
class MaskBatchMulti:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "inputcount": ("INT", {"default": 2, "min": 2, "max": 1000, "step": 1}),
                "mask_1": ("MASK", ),
                "mask_2": ("MASK", ),
            },
    }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("masks",)
    FUNCTION = "combine"
    CATEGORY = "KJNodes/masking"
    DESCRIPTION = """
Creates an image batch from multiple masks.  
You can set how many inputs the node has,  
with the **inputcount** and clicking update.
"""

    def combine(self, inputcount, **kwargs):
        mask = kwargs["mask_1"]
        for c in range(1, inputcount):
            new_mask = kwargs[f"mask_{c + 1}"]
            if mask.shape[1:] != new_mask.shape[1:]:
                new_mask = F.interpolate(new_mask.unsqueeze(1), size=(mask.shape[1], mask.shape[2]), mode="bicubic").squeeze(1)
            mask = torch.cat((mask, new_mask), dim=0)
        return (mask,)

class OffsetMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "x": ("INT", { "default": 0, "min": -4096, "max": MAX_RESOLUTION, "step": 1, "display": "number" }),
                "y": ("INT", { "default": 0, "min": -4096, "max": MAX_RESOLUTION, "step": 1, "display": "number" }),
                "angle": ("INT", { "default": 0, "min": -360, "max": 360, "step": 1, "display": "number" }),
                "duplication_factor": ("INT", { "default": 1, "min": 1, "max": 1000, "step": 1, "display": "number" }),
                "roll": ("BOOLEAN", { "default": False }),
                "incremental": ("BOOLEAN", { "default": False }),
                "padding_mode": (
            [   
                'empty',
                'border',
                'reflection',
                
            ], {
               "default": 'empty'
            }),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "offset"
    CATEGORY = "KJNodes/masking"
    DESCRIPTION = """
Offsets the mask by the specified amount.  
 - mask: Input mask or mask batch
 - x: Horizontal offset
 - y: Vertical offset
 - angle: Angle in degrees
 - roll: roll edge wrapping
 - duplication_factor: Number of times to duplicate the mask to form a batch
 - border padding_mode: Padding mode for the mask
"""

    def offset(self, mask, x, y, angle, roll=False, incremental=False, duplication_factor=1, padding_mode="empty"):
        # Create duplicates of the mask batch
        mask = mask.repeat(duplication_factor, 1, 1).clone()

        batch_size, height, width = mask.shape

        if angle != 0 and incremental:
            for i in range(batch_size):
                rotation_angle = angle * (i+1)
                mask[i] = TF.rotate(mask[i].unsqueeze(0), rotation_angle).squeeze(0)
        elif angle > 0:
            for i in range(batch_size):
                mask[i] = TF.rotate(mask[i].unsqueeze(0), angle).squeeze(0)

        if roll:
            if incremental:
                for i in range(batch_size):
                    shift_x = min(x*(i+1), width-1)
                    shift_y = min(y*(i+1), height-1)
                    if shift_x != 0:
                        mask[i] = torch.roll(mask[i], shifts=shift_x, dims=1)
                    if shift_y != 0:
                        mask[i] = torch.roll(mask[i], shifts=shift_y, dims=0)
            else:
                shift_x = min(x, width-1)
                shift_y = min(y, height-1)
                if shift_x != 0:
                    mask = torch.roll(mask, shifts=shift_x, dims=2)
                if shift_y != 0:
                    mask = torch.roll(mask, shifts=shift_y, dims=1)
        else:
            
            for i in range(batch_size):
                if incremental:
                    temp_x = min(x * (i+1), width-1)
                    temp_y = min(y * (i+1), height-1)
                else:
                    temp_x = min(x, width-1)
                    temp_y = min(y, height-1)
                if temp_x > 0:
                    if padding_mode == 'empty':
                        mask[i] = torch.cat([torch.zeros((height, temp_x)), mask[i, :, :-temp_x]], dim=1)
                    elif padding_mode in ['replicate', 'reflect']:
                        mask[i] = F.pad(mask[i, :, :-temp_x], (0, temp_x), mode=padding_mode)
                elif temp_x < 0:
                    if padding_mode == 'empty':
                        mask[i] = torch.cat([mask[i, :, :temp_x], torch.zeros((height, -temp_x))], dim=1)
                    elif padding_mode in ['replicate', 'reflect']:
                        mask[i] = F.pad(mask[i, :, -temp_x:], (temp_x, 0), mode=padding_mode)

                if temp_y > 0:
                    if padding_mode == 'empty':
                        mask[i] = torch.cat([torch.zeros((temp_y, width)), mask[i, :-temp_y, :]], dim=0)
                    elif padding_mode in ['replicate', 'reflect']:
                        mask[i] = F.pad(mask[i, :-temp_y, :], (0, temp_y), mode=padding_mode)
                elif temp_y < 0:
                    if padding_mode == 'empty':
                        mask[i] = torch.cat([mask[i, :temp_y, :], torch.zeros((-temp_y, width))], dim=0)
                    elif padding_mode in ['replicate', 'reflect']:
                        mask[i] = F.pad(mask[i, -temp_y:, :], (temp_y, 0), mode=padding_mode)
           
        return mask,
        
class RoundMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "mask": ("MASK",),  
        }}

    RETURN_TYPES = ("MASK",)
    FUNCTION = "round"
    CATEGORY = "KJNodes/masking"
    DESCRIPTION = """
Rounds the mask or batch of masks to a binary mask.  
<img src="https://github.com/kijai/ComfyUI-KJNodes/assets/40791699/52c85202-f74e-4b96-9dac-c8bda5ddcc40" width="300" height="250" alt="RoundMask example">

"""

    def round(self, mask):
        mask = mask.round()
        return (mask,)
    
class ResizeMask:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "width": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1, "display": "number" }),
                "height": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1, "display": "number" }),
                "keep_proportions": ("BOOLEAN", { "default": False }),
                "upscale_method": (s.upscale_methods,),
                "crop": (["disabled","center"],),
            }
        }

    RETURN_TYPES = ("MASK", "INT", "INT",)
    RETURN_NAMES = ("mask", "width", "height",)
    FUNCTION = "resize"
    CATEGORY = "KJNodes/masking"
    DESCRIPTION = """
Resizes the mask or batch of masks to the specified width and height.
"""

    def resize(self, mask, width, height, keep_proportions, upscale_method,crop):
        if keep_proportions:
            _, oh, ow = mask.shape
            width = ow if width == 0 else width
            height = oh if height == 0 else height
            ratio = min(width / ow, height / oh)
            width = round(ow*ratio)
            height = round(oh*ratio)

        if upscale_method == "lanczos":
            out_mask = common_upscale(mask.unsqueeze(1).repeat(1, 3, 1, 1), width, height, upscale_method, crop=crop).movedim(1,-1)[:, :, :, 0]
        else:
            out_mask = common_upscale(mask.unsqueeze(1), width, height, upscale_method, crop=crop).squeeze(1)

        return(out_mask, out_mask.shape[2], out_mask.shape[1],)

class RemapMaskRange:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "min": ("FLOAT", {"default": 0.0,"min": -10.0, "max": 1.0, "step": 0.01}),
                "max": ("FLOAT", {"default": 1.0,"min": 0.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "remap"
    CATEGORY = "KJNodes/masking"
    DESCRIPTION = """
Sets new min and max values for the mask.
"""

    def remap(self, mask, min, max):

         # Find the maximum value in the mask
        mask_max = torch.max(mask)
        
        # If the maximum mask value is zero, avoid division by zero by setting it to 1
        mask_max = mask_max if mask_max > 0 else 1
        
        # Scale the mask values to the new range defined by min and max
        # The highest pixel value in the mask will be scaled to max
        scaled_mask = (mask / mask_max) * (max - min) + min
        
        # Clamp the values to ensure they are within [0.0, 1.0]
        scaled_mask = torch.clamp(scaled_mask, min=0.0, max=1.0)
        
        return (scaled_mask, )


def get_mask_polygon(self, mask_np):
    import cv2
    """Helper function to get polygon points from mask"""
    # Find contours
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Approximate polygon
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    return polygon.squeeze()

import cv2
class SeparateMasks:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", ),
                "size_threshold_width" : ("INT", {"default": 256, "min": 0.0, "max": 4096, "step": 1}),
                "size_threshold_height" : ("INT", {"default": 256, "min": 0.0, "max": 4096, "step": 1}),
                "mode": (["convex_polygons", "area", "box"],),
                "max_poly_points": ("INT", {"default": 8, "min": 3, "max": 32, "step": 1}),

            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "separate"
    CATEGORY = "KJNodes/masking"
    OUTPUT_NODE = True
    DESCRIPTION = "Separates a mask into multiple masks based on the size of the connected components."

    def polygon_to_mask(self, polygon, shape):
        mask = np.zeros((shape[0], shape[1]), dtype=np.uint8)  # Fixed shape handling

        if len(polygon.shape) == 2:  # Check if polygon points are valid
            polygon = polygon.astype(np.int32)
            cv2.fillPoly(mask, [polygon], 1)
        return mask

    def get_mask_polygon(self, mask_np, max_points):
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest_contour)
        
        # Initialize with smaller epsilon for more points
        perimeter = cv2.arcLength(hull, True)
        epsilon = perimeter * 0.01  # Start smaller
        
        min_eps = perimeter * 0.001  # Much smaller minimum
        max_eps = perimeter * 0.2   # Smaller maximum
        
        best_approx = None
        best_diff = float('inf')
        max_iterations = 20
        
        #print(f"Target points: {max_points}, Perimeter: {perimeter}")
        
        for i in range(max_iterations):
            curr_eps = (min_eps + max_eps) / 2
            approx = cv2.approxPolyDP(hull, curr_eps, True)
            points_diff = len(approx) - max_points
            
            #print(f"Iteration {i}: points={len(approx)}, eps={curr_eps:.4f}")
            
            if abs(points_diff) < best_diff:
                best_approx = approx
                best_diff = abs(points_diff)
            
            if len(approx) > max_points:
                min_eps = curr_eps * 1.1  # More gradual adjustment
            elif len(approx) < max_points:
                max_eps = curr_eps * 0.9  # More gradual adjustment
            else:
                return approx.squeeze()
            
            if abs(max_eps - min_eps) < perimeter * 0.0001:  # Relative tolerance
                break
        
        # If we didn't find exact match, return best approximation
        return best_approx.squeeze() if best_approx is not None else hull.squeeze()

    def separate(self, mask: torch.Tensor, size_threshold_width: int, size_threshold_height: int, max_poly_points: int, mode: str):
        from scipy.ndimage import label, center_of_mass
        import numpy as np
        
        B, H, W = mask.shape
        separated = []

        mask = mask.round()
        
        for b in range(B):
            mask_np = mask[b].cpu().numpy().astype(np.uint8)
            structure = np.ones((3, 3), dtype=np.int8)
            labeled, ncomponents = label(mask_np, structure=structure)
            pbar = ProgressBar(ncomponents)
            
            for component in range(1, ncomponents + 1):
                component_mask_np = (labeled == component).astype(np.uint8)
                
                rows = np.any(component_mask_np, axis=1)
                cols = np.any(component_mask_np, axis=0)
                y_min, y_max = np.where(rows)[0][[0, -1]]
                x_min, x_max = np.where(cols)[0][[0, -1]]
                
                width = x_max - x_min + 1
                height = y_max - y_min + 1
                centroid_x = (x_min + x_max) / 2  # Calculate x centroid
                print(f"Component {component}: width={width}, height={height}, x_pos={centroid_x}")
                
                if width >= size_threshold_width and height >= size_threshold_height:
                    if mode == "convex_polygons":
                        polygon = self.get_mask_polygon(component_mask_np, max_poly_points)
                        if polygon is not None:
                            poly_mask = self.polygon_to_mask(polygon, (H, W))
                            poly_mask = torch.tensor(poly_mask, device=mask.device)
                            separated.append((centroid_x, poly_mask))
                    elif mode == "box":
                        # Create bounding box mask
                        box_mask = np.zeros((H, W), dtype=np.uint8)
                        box_mask[y_min:y_max+1, x_min:x_max+1] = 1
                        box_mask = torch.tensor(box_mask, device=mask.device)
                        separated.append((centroid_x, box_mask))
                    else:
                        area_mask = torch.tensor(component_mask_np, device=mask.device)
                        separated.append((centroid_x, area_mask))
                pbar.update(1)
        
        if len(separated) > 0:
            # Sort by x position and extract only the masks
            separated.sort(key=lambda x: x[0])
            separated = [x[1] for x in separated]
            out_masks = torch.stack(separated, dim=0)
            return out_masks,
        else:
            return torch.empty((1, 64, 64), device=mask.device),
        