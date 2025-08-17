import asyncio
import folder_paths

from typing import Union

from nodes import LoraLoader
from .constants import get_category, get_name
from .power_prompt_utils import get_lora_by_filename
from .utils import FlexibleOptionalInputType, any_type
from .server.utils_info import get_model_info
from .log import log_node_warn

NODE_NAME = get_name('Power Lora Loader')


class RgthreePowerLoraLoader:
  """ The Power Lora Loader is a powerful, flexible node to add multiple loras to a model/clip."""

  NAME = NODE_NAME
  CATEGORY = get_category()

  @classmethod
  def INPUT_TYPES(cls):  # pylint: disable = invalid-name, missing-function-docstring
    return {
      "required": {
      },
      # Since we will pass any number of loras in from the UI, this needs to always allow an
      "optional": FlexibleOptionalInputType(type=any_type, data={
        "model": ("MODEL",),
        "clip": ("CLIP",),
      }),
      "hidden": {},
    }

  RETURN_TYPES = ("MODEL", "CLIP")
  RETURN_NAMES = ("MODEL", "CLIP")
  FUNCTION = "load_loras"

  def load_loras(self, model=None, clip=None, **kwargs):
    """Loops over the provided loras in kwargs and applies valid ones."""
    for key, value in kwargs.items():
      key = key.upper()
      if key.startswith('LORA_') and 'on' in value and 'lora' in value and 'strength' in value:
        strength_model = value['strength']
        # If we just passed one strength value, then use it for both, if we passed a strengthTwo
        # as well, then our `strength` will be for the model, and `strengthTwo` for clip.
        strength_clip = value['strengthTwo'] if 'strengthTwo' in value else None
        if clip is None:
          if strength_clip is not None and strength_clip != 0:
            log_node_warn(NODE_NAME, 'Recieved clip strength eventhough no clip supplied!')
          strength_clip = 0
        else:
          strength_clip = strength_clip if strength_clip is not None else strength_model
        if value['on'] and (strength_model != 0 or strength_clip != 0):
          lora = get_lora_by_filename(value['lora'], log_node=self.NAME)
          if model is not None and lora is not None:
            model, clip = LoraLoader().load_lora(model, clip, lora, strength_model, strength_clip)

    return (model, clip)

  @classmethod
  def get_enabled_loras_from_prompt_node(cls,
                                         prompt_node: dict) -> list[dict[str, Union[str, float]]]:
    """Gets enabled loras of a node within a server prompt."""
    result = []
    for name, lora in prompt_node['inputs'].items():
      if name.startswith('lora_') and lora['on']:
        lora_file = get_lora_by_filename(lora['lora'], log_node=cls.NAME)
        if lora_file is not None:  # Add the same safety check
          lora_dict = {
            'name': lora['lora'],
            'strength': lora['strength'],
            'path': folder_paths.get_full_path("loras", lora_file)
          }
          if 'strengthTwo' in lora:
            lora_dict['strength_clip'] = lora['strengthTwo']
          result.append(lora_dict)
    return result

  @classmethod
  def get_enabled_triggers_from_prompt_node(cls, prompt_node: dict, max_each: int = 1):
    """Gets trigger words up to the max for enabled loras of a node within a server prompt."""
    loras = [l['name'] for l in cls.get_enabled_loras_from_prompt_node(prompt_node)]
    trained_words = []
    for lora in loras:
      info = asyncio.run(get_model_info(lora, 'loras'))
      if not info or not info.keys():
        log_node_warn(
          NODE_NAME,
          f'No info found for lora {lora} when grabbing triggers. Have you generated an info file'
          ' from the Power Lora Loader "Show Info" dialog?'
        )
        continue
      if 'trainedWords' not in info or not info['trainedWords']:
        log_node_warn(
          NODE_NAME,
          f'No trained words for lora {lora} when grabbing triggers. Have you fetched data from'
          'civitai or manually added words?'
        )
        continue
      trained_words += [w for wi in info['trainedWords'][:max_each] if (wi and (w := wi['word']))]
    return trained_words
