"""A simpler SDXL Power Prompt that doesn't load Loras, like for negative."""
import os
import re
import folder_paths
from nodes import MAX_RESOLUTION, LoraLoader
from comfy_extras.nodes_clip_sdxl import CLIPTextEncodeSDXL
from .sdxl_power_prompt_postive import RgthreeSDXLPowerPromptPositive

from .log import log_node_warn, log_node_info, log_node_success

from .constants import get_category, get_name

NODE_NAME = get_name('SDXL Power Prompt - Simple / Negative')


class RgthreeSDXLPowerPromptSimple(RgthreeSDXLPowerPromptPositive):
  """A simpler SDXL Power Prompt that doesn't handle Loras."""

  NAME = NODE_NAME
  CATEGORY = get_category()

  @classmethod
  def INPUT_TYPES(cls):  # pylint: disable = invalid-name, missing-function-docstring
    # Removed Saved Prompts feature; No sure it worked any longer. UI should fail gracefully,
    # TODO: Rip out saved prompt input data
    SAVED_PROMPTS_FILES=[]
    SAVED_PROMPTS_CONTENT=[]
    return {
      'required': {
        'prompt_g': ('STRING', {
          'multiline': True,
          'dynamicPrompts': True
        }),
        'prompt_l': ('STRING', {
          'multiline': True,
          'dynamicPrompts': True
        }),
      },
      'optional': {
        "opt_clip": ("CLIP",),
        "opt_clip_width": ("INT", {
          "forceInput": True,
          "default": 1024.0,
          "min": 0,
          "max": MAX_RESOLUTION
        }),
        "opt_clip_height": ("INT", {
          "forceInput": True,
          "default": 1024.0,
          "min": 0,
          "max": MAX_RESOLUTION
        }),
        'insert_embedding': ([
          'CHOOSE',
        ] + [os.path.splitext(x)[0] for x in folder_paths.get_filename_list('embeddings')],),
        'insert_saved': ([
          'CHOOSE',
        ] + SAVED_PROMPTS_FILES,),
        # We'll hide these in the UI for now.
        "target_width": ("INT", {
          "default": -1,
          "min": -1,
          "max": MAX_RESOLUTION
        }),
        "target_height": ("INT", {
          "default": -1,
          "min": -1,
          "max": MAX_RESOLUTION
        }),
        "crop_width": ("INT", {
          "default": -1,
          "min": -1,
          "max": MAX_RESOLUTION
        }),
        "crop_height": ("INT", {
          "default": -1,
          "min": -1,
          "max": MAX_RESOLUTION
        }),
      },
      'hidden': {
        'values_insert_saved': (['CHOOSE'] + SAVED_PROMPTS_CONTENT,),
      }
    }

  RETURN_TYPES = ('CONDITIONING', 'STRING', 'STRING')
  RETURN_NAMES = ('CONDITIONING', 'TEXT_G', 'TEXT_L')
  FUNCTION = 'main'

  def main(self,
           prompt_g,
           prompt_l,
           opt_clip=None,
           opt_clip_width=None,
           opt_clip_height=None,
           insert_embedding=None,
           insert_saved=None,
           target_width=-1,
           target_height=-1,
           crop_width=-1,
           crop_height=-1,
           values_insert_saved=None):

    conditioning = self.get_conditioning(prompt_g, prompt_l, opt_clip, opt_clip_width,
                                         opt_clip_height, target_width, target_height, crop_width, crop_height)
    return (conditioning, prompt_g, prompt_l)
