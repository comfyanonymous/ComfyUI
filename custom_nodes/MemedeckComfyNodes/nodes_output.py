import folder_paths
from comfy.cli_args import args

from PIL import Image
from PIL.PngImagePlugin import PngInfo

import numpy as np
import json
import os

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
        method = self.methods.get(method)
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()

        pil_images = []
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            pil_images.append(img)

        metadata = pil_images[0].getexif()
        num_frames = len(pil_images)
        
        json_metadata = json.dumps({
          "crf": crf,
          "motion_prompt": motion_prompt,
          "negative_prompt": negative_prompt,
          "img2vid_metadata": img2vid_metadata,
          "sampler_metadata": sampler_metadata,
        }, indent=4)

        c = len(pil_images)
        for i in range(0, c, num_frames):
            file = f"{filename}_{counter:05}_.webp"
            pil_images[i].save(os.path.join(full_output_folder, file), save_all=True, duration=int(1000.0/fps), append_images=pil_images[i + 1:i + num_frames], exif=metadata, lossless=lossless, quality=quality, method=method)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type,
            })
            counter += 1

        animated = num_frames != 1
        
        return { "ui": { "images": results, "animated": (animated,), "metadata": json_metadata } }
      
      
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