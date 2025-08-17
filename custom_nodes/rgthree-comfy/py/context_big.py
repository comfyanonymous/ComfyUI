"""The Conmtext big node."""
from .constants import get_category, get_name
from .context_utils import (ALL_CTX_OPTIONAL_INPUTS, ALL_CTX_RETURN_NAMES, ALL_CTX_RETURN_TYPES,
                            new_context, get_context_return_tuple)


class RgthreeBigContext:
  """The Context Big node.

  This context node will expose all context fields as inputs and outputs. It is backwards compatible
  with other context nodes and can be intertwined with them.
  """

  NAME = get_name("Context Big")
  CATEGORY = get_category()

  @classmethod
  def INPUT_TYPES(cls):  # pylint: disable = invalid-name,missing-function-docstring
    return {
      "required": {},
      "optional": ALL_CTX_OPTIONAL_INPUTS,
      "hidden": {},
    }

  RETURN_TYPES = ALL_CTX_RETURN_TYPES
  RETURN_NAMES = ALL_CTX_RETURN_NAMES
  FUNCTION = "convert"

  def convert(self, base_ctx=None, **kwargs):  # pylint: disable = missing-function-docstring
    ctx = new_context(base_ctx, **kwargs)
    return get_context_return_tuple(ctx)
