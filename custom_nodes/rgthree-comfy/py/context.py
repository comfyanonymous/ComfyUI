"""The Context node."""
from .context_utils import (ORIG_CTX_OPTIONAL_INPUTS, ORIG_CTX_RETURN_NAMES, ORIG_CTX_RETURN_TYPES,
                            get_orig_context_return_tuple, new_context)
from .constants import get_category, get_name


class RgthreeContext:
  """The initial Context node.

  For now, this nodes' outputs will remain as-is, as they are perfect for most 1.5 application, but
  is also backwards compatible with other Context nodes.
  """

  NAME = get_name("Context")
  CATEGORY = get_category()

  @classmethod
  def INPUT_TYPES(cls):  # pylint: disable = invalid-name, missing-function-docstring
    return {
      "required": {},
      "optional": ORIG_CTX_OPTIONAL_INPUTS,
      "hidden": {
        "version": "FLOAT"
      },
    }

  RETURN_TYPES = ORIG_CTX_RETURN_TYPES
  RETURN_NAMES = ORIG_CTX_RETURN_NAMES
  FUNCTION = "convert"

  def convert(self, base_ctx=None, **kwargs):  # pylint: disable = missing-function-docstring
    ctx = new_context(base_ctx, **kwargs)
    return get_orig_context_return_tuple(ctx)
