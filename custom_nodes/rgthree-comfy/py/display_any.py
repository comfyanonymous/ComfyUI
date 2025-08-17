import json
from .constants import get_category, get_name
from .utils import any_type, get_dict_value


class RgthreeDisplayAny:
  """Display any data node."""

  NAME = get_name('Display Any')
  CATEGORY = get_category()

  @classmethod
  def INPUT_TYPES(cls):  # pylint: disable = invalid-name, missing-function-docstring
    return {
      "required": {
        "source": (any_type, {}),
      },
      "hidden": {
        "unique_id": "UNIQUE_ID",
        "extra_pnginfo": "EXTRA_PNGINFO",
      },
    }

  RETURN_TYPES = ()
  FUNCTION = "main"
  OUTPUT_NODE = True

  def main(self, source=None, unique_id=None, extra_pnginfo=None):
    value = 'None'
    if isinstance(source, str):
      value = source
    elif isinstance(source, (int, float, bool)):
      value = str(source)
    elif source is not None:
      try:
        value = json.dumps(source)
      except Exception:
        try:
          value = str(source)
        except Exception:
          value = 'source exists, but could not be serialized.'

    # Save the output to the pnginfo so it's pre-filled when loading the data.
    if extra_pnginfo and unique_id:
      for node in get_dict_value(extra_pnginfo, 'workflow.nodes', []):
        if str(node['id']) == str(unique_id):
          node['widgets_values'] = [value]
          break

    return {"ui": {"text": (value,)}}


class RgthreeDisplayInt:
  """Old DisplayInt node.

  Can be ported over to DisplayAny if https://github.com/comfyanonymous/ComfyUI/issues/1527 fixed.
  """

  NAME = get_name('Display Int')
  CATEGORY = get_category()

  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "input": ("INT", {
          "forceInput": True
        }),
      },
    }

  RETURN_TYPES = ()
  FUNCTION = "main"
  OUTPUT_NODE = True

  def main(self, input=None):
    return {"ui": {"text": (input,)}}
