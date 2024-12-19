from io import BytesIO
import time
import folder_paths
from comfy.cli_args import args
import torch
from PIL import Image
import cairosvg
from lxml import etree

import numpy as np
import json
import os
import logging

# setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

WATERMARK = """
<svg width="256" height="256" viewBox="0 0 256 256" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M60.0859 196.8C65.9526 179.067 71.5526 161.667 76.8859 144.6C79.1526 137.4 81.4859 129.867 83.8859 122C86.2859 114.133 88.6859 106.333 91.0859 98.6C93.4859 90.8667 95.6859 83.4 97.6859 76.2C99.8193 69 101.686 62.3333 103.286 56.2C110.619 56.2 117.553 55.8 124.086 55C130.619 54.2 137.686 53.4667 145.286 52.8C144.886 55.7333 144.419 59.0667 143.886 62.8C143.486 66.4 142.953 70.2 142.286 74.2C141.753 78.2 141.153 82.3333 140.486 86.6C139.819 90.8667 139.019 96.3333 138.086 103C137.153 109.667 135.886 118 134.286 128H136.886C140.753 117.867 143.953 109.467 146.486 102.8C149.019 96 151.086 90.4667 152.686 86.2C154.286 81.9333 155.886 77.8 157.486 73.8C159.219 69.6667 160.819 65.8 162.286 62.2C163.886 58.4667 165.353 55.2 166.686 52.4C170.019 52.1333 173.153 51.8 176.086 51.4C179.019 51 181.953 50.6 184.886 50.2C187.819 49.6667 190.753 49.2 193.686 48.8C196.753 48.2667 200.086 47.6667 203.686 47C202.353 54.7333 201.086 62.6667 199.886 70.8C198.686 78.9333 197.619 87.0667 196.686 95.2C195.753 103.2 194.819 111.133 193.886 119C193.086 126.867 192.353 134.333 191.686 141.4C190.086 157.933 188.686 174.067 187.486 189.8L152.686 196C152.686 195.333 152.753 193.533 152.886 190.6C153.153 187.667 153.419 184.067 153.686 179.8C154.086 175.533 154.553 170.8 155.086 165.6C155.753 160.4 156.353 155.2 156.886 150C157.553 144.8 158.219 139.8 158.886 135C159.553 130.067 160.219 125.867 160.886 122.4H159.086C157.219 128 155.153 133.933 152.886 140.2C150.619 146.333 148.286 152.6 145.886 159C143.619 165.4 141.353 171.667 139.086 177.8C136.819 183.933 134.819 189.8 133.086 195.4C128.419 195.533 124.419 195.733 121.086 196C117.753 196.133 113.886 196.333 109.486 196.6L115.886 122.4H112.886C112.619 124.133 111.953 127.067 110.886 131.2C109.819 135.2 108.553 139.867 107.086 145.2C105.753 150.4 104.286 155.867 102.686 161.6C101.086 167.2 99.5526 172.467 98.0859 177.4C96.7526 182.2 95.6193 186.2 94.6859 189.4C93.7526 192.467 93.2193 194.2 93.0859 194.6L60.0859 196.8Z" fill="white"/>
</svg>
"""
WATERMARK_SIZE = 32

class MD_SaveAnimatedWEBP:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    methods = {"default": 4, "fastest": 0, "slowest": 6}
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "images": ("IMAGE", ),
                    "filename_prefix": ("STRING", {"default": "memedeck_video"}),
                    "fps": ("FLOAT", {"default": 24.0, "min": 0.01, "max": 1000.0, "step": 0.01}),
                    "lossless": ("BOOLEAN", {"default": False}),
                    "quality": ("INT", {"default": 90, "min": 0, "max": 100}),
                    "method": (list(s.methods.keys()),),
                    "crf": ("INT",),
                    "motion_prompt": ("STRING", ),
                    "negative_prompt": ("STRING", ),
                    "img2vid_metadata": ("STRING", ),
                    "sampler_metadata": ("STRING", ),
                },
                # "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "MemeDeck"
    
    def save_images(self, images, fps, filename_prefix, lossless, quality, method, crf=None, motion_prompt=None, negative_prompt=None, img2vid_metadata=None, sampler_metadata=None):
        start_time = time.time()
        method = self.methods.get(method)
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = []

        # Vectorized conversion to PIL images
        pil_images = [Image.fromarray(np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8)) for image in images]
        
        first_image = pil_images[0]
        padding = 12
        x = first_image.width - WATERMARK_SIZE - padding
        y = first_image.height - WATERMARK_SIZE - padding
        first_image_background_brightness = self.analyze_background_brightness(first_image, x, y, WATERMARK_SIZE)
        
        watermarked_images = [self.add_watermark_to_image(img, first_image_background_brightness) for img in pil_images]

        metadata = pil_images[0].getexif()
        num_frames = len(pil_images)

        json_metadata = {
            "crf": crf,
            "motion_prompt": motion_prompt,
            "negative_prompt": negative_prompt,
            "img2vid_metadata": json.loads(img2vid_metadata),
            "sampler_metadata": json.loads(sampler_metadata),
        }

        # Optimized saving logic
        if num_frames == 1:  # Single image, save once
            file = f"{filename}_{counter:05}_.webp"
            watermarked_images[0].save(os.path.join(full_output_folder, file), exif=metadata, lossless=lossless, quality=quality, method=method)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type,
            })
        else: # multiple images, save as animation
            file = f"{filename}_{counter:05}_.webp"
            watermarked_images[0].save(os.path.join(full_output_folder, file), save_all=True, duration=int(1000.0 / fps), append_images=watermarked_images[1:], exif=metadata, lossless=lossless, quality=quality, method=method)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type,
            })

        animated = num_frames != 1

        end_time = time.time()
        logger.info(f"Save images took: {end_time - start_time} seconds")

        return {
            "ui": {
                "images": results,
                "animated": (animated,),
                "metadata": (json.dumps(json_metadata),)
            },
        }
        
    def add_watermark_to_image(self, img, background_brightness=None):
        """
        Adds a watermark to a single PIL Image.

        Args:
        img: A PIL Image object.

        Returns:
        A PIL Image object with the watermark added.
        """

        padding = 12
        x = img.width - WATERMARK_SIZE - padding
        y = img.height - WATERMARK_SIZE - padding

        if background_brightness is None:
            background_brightness = self.analyze_background_brightness(img, x, y, WATERMARK_SIZE)

        # Generate watermark image (replace this with your actual watermark generation)
        watermark = self.generate_watermark(WATERMARK_SIZE, background_brightness)

        # Overlay the watermark
        img.paste(watermark, (x, y), watermark)

        return img


    def analyze_background_brightness(self, img, x, y, size):
        """
        Analyzes the average brightness of a region in the image.

        Args:
        img: A PIL Image object.
        x: The x-coordinate of the top-left corner of the region.
        y: The y-coordinate of the top-left corner of the region.
        size: The size of the region (square).

        Returns:
        The average brightness of the region as an integer.
        """
        region = img.crop((x, y, x + size, y + size))
        pixels = np.array(region)
        total_brightness = np.sum(
            0.299 * pixels[:, :, 0] + 0.587 * pixels[:, :, 1] + 0.114 * pixels[:, :, 2]
        ) / 1000
        print(f"total_brightness: {total_brightness}")
        return  max(0, min(255, total_brightness)) 

    def generate_watermark(self, size, background_brightness):
        """
        Generates a watermark image from an SVG string.

        Args:
        size: The size of the watermark (square).
        background_brightness: The background brightness at the watermark position.

        Returns:
        A PIL Image object representing the watermark.
        """

        # Determine watermark color based on background brightness
        watermark_color = (0, 0, 0, 165) if background_brightness > 128 else (255, 255, 255, 165)

        # Parse the SVG string
        svg_tree = etree.fromstring(WATERMARK)

        # Find the path element and set its fill attribute
        path_element = svg_tree.find(".//{http://www.w3.org/2000/svg}path")
        if path_element is not None:
            r, g, b, a = watermark_color
            fill_color = f"rgba({r},{g},{b},{a/255})"  # Convert to rgba string
            path_element.set("fill", fill_color)

        # Convert the modified SVG tree back to a string
        modified_svg = etree.tostring(svg_tree, encoding="unicode")

        # Render the modified SVG to a PNG image with a transparent background
        png_data = cairosvg.svg2png(
            bytestring=modified_svg,
            output_width=size,
            output_height=size,
            background_color="transparent"
        )
        watermark_img = Image.open(BytesIO(png_data))

        # Convert the watermark to RGBA to handle transparency
        watermark_img = watermark_img.convert("RGBA")

        return watermark_img
    
    # def save_images(self, images, fps, filename_prefix, lossless, quality, method, crf=None, motion_prompt=None, negative_prompt=None, img2vid_metadata=None, sampler_metadata=None):
    #     start_time = time.time()
    #     method = self.methods.get(method)
    #     filename_prefix += self.prefix_append
    #     full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
    #         filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
    #     )
    #     results = []

    #     # Prepare PIL images in one loop
    #     pil_images = [
    #         Image.fromarray(np.clip((255. * image.cpu().numpy()), 0, 255).astype(np.uint8))
    #         for image in images
    #     ]

    #     metadata = pil_images[0].getexif()
    #     num_frames = len(pil_images)

    #     # Pre-serialize JSON metadata
    #     json_metadata = json.dumps({
    #         "crf": crf,
    #         "motion_prompt": motion_prompt,
    #         "negative_prompt": negative_prompt,
    #         "img2vid_metadata": json.loads(img2vid_metadata),
    #         "sampler_metadata": json.loads(sampler_metadata),
    #     })

    #     # Save images directly
    #     duration = int(1000.0 / fps)
    #     for i in range(0, len(pil_images), num_frames):
    #         file = f"{filename}_{counter:05}_.webp"
    #         pil_images[i].save(
    #             os.path.join(full_output_folder, file),
    #             save_all=True,
    #             duration=duration,
    #             append_images=pil_images[i + 1:i + num_frames],
    #             exif=metadata,
    #             lossless=lossless,
    #             quality=quality,
    #             method=method
    #         )
    #         results.append({"filename": file, "subfolder": subfolder, "type": self.type})
    #         counter += 1
        
    #     end_time = time.time()
    #     logger.info(f"Save images took: {end_time - start_time} seconds")

    #     return {
    #         "ui": {
    #             "images": results,
    #             "animated": (num_frames != 1,),
    #             "metadata": (json_metadata,),
    #         },
    #     }

    # def save_images(self, images, fps, filename_prefix, lossless, quality, method, crf=None, motion_prompt=None, negative_prompt=None, img2vid_metadata=None, sampler_metadata=None):
    #     method = self.methods.get(method)
    #     filename_prefix += self.prefix_append
    #     full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
    #     results = list()

    #     pil_images = []
    #     for image in images:
    #         i = 255. * image.cpu().numpy()
    #         img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    #         pil_images.append(img)

    #     metadata = pil_images[0].getexif()
    #     num_frames = len(pil_images)
        
    #     json_metadata = {
    #       "crf": crf,
    #       "motion_prompt": motion_prompt,
    #       "negative_prompt": negative_prompt,
    #       "img2vid_metadata": json.loads(img2vid_metadata),
    #       "sampler_metadata": json.loads(sampler_metadata),
    #     }

    #     c = len(pil_images)
    #     for i in range(0, c, num_frames):
    #         file = f"{filename}_{counter:05}_.webp"
    #         pil_images[i].save(os.path.join(full_output_folder, file), save_all=True, duration=int(1000.0/fps), append_images=pil_images[i + 1:i + num_frames], exif=metadata, lossless=lossless, quality=quality, method=method)
    #         results.append({
    #             "filename": file,
    #             "subfolder": subfolder,
    #             "type": self.type,
    #         })
    #         counter += 1

    #     animated = num_frames != 1
    #     # properly serialize metadata
    #     return { 
    #         "ui": { 
    #             "images": results, 
    #             "animated": (animated,), 
    #             "metadata": (json.dumps(json_metadata),)
    #         },
    #     }
    
class MD_VAEDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "samples": ("LATENT", {"tooltip": "The latent to be decoded."}), 
                "vae": ("VAE", {"tooltip": "The VAE model used for decoding the latent."})
            }
        }
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_TOOLTIPS = ("The decoded image.",)
    FUNCTION = "decode"

    CATEGORY = "latent"
    DESCRIPTION = "Decodes latent images back into pixel space images."

    def decode(self, vae, samples):
        start_time = time.time()

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            images = vae.decode(samples["samples"])

        print(prof.key_averages().table(sort_by="cuda_time_total"))  # Print profiling results

        if len(images.shape) == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])

        end_time = time.time()
        print(f"VAE decoding time: {end_time - start_time:.4f} seconds")

        return (images,)
      
      
class MD_SaveMP4:
  def __init__(self):
      # Get absolute path of the output directory
      self.output_dir = os.path.abspath("output/video_gen")
      self.type = "output"
      self.prefix_append = ""

  methods = {"default": 4, "fastest": 0, "slowest": 6}

  @classmethod
  def INPUT_TYPES(s):
      return {"required":
                  {"images": ("IMAGE", ),
                    "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                    "fps": ("FLOAT", {"default": 24.0, "min": 0.01, "max": 1000.0, "step": 0.01}),
                    "quality": ("INT", {"default": 80, "min": 0, "max": 100}),
                    },
              "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
              }

  RETURN_TYPES = ()
  FUNCTION = "save_video"

  OUTPUT_NODE = True

  CATEGORY = "MemeDeck"

  def save_video(self, images, fps, filename_prefix, quality, prompt=None, extra_pnginfo=None):
      filename_prefix += self.prefix_append
      full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
          filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
      )
      results = list()
      video_path = os.path.join(full_output_folder, f"{filename}_{counter:05}.mp4")

      # Determine video resolution
      height, width = images[0].shape[1], images[0].shape[2]
      video_writer = cv2.VideoWriter(
          video_path,
          cv2.VideoWriter_fourcc(*'mp4v'),
          fps,
          (width, height)
      )

      # Write each frame to the video
      for image in images:
          i = 255. * image.cpu().numpy()
          frame = np.clip(i, 0, 255).astype(np.uint8)
          frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
          video_writer.write(frame)

      video_writer.release()

      results.append({
          "filename": os.path.basename(video_path),
          "subfolder": subfolder,
          "type": self.type
      })

      return {"ui": {"videos": results}}