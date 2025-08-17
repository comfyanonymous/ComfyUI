"""The original Context Switch."""
from .constants import get_category, get_name
from .context_utils import (ORIG_CTX_RETURN_TYPES, ORIG_CTX_RETURN_NAMES, is_context_empty,
                            get_orig_context_return_tuple)
from .utils import FlexibleOptionalInputType


class RgthreeContextSwitch:
  """The (original) Context Switch node."""

  NAME = get_name("Context Switch")
  CATEGORY = get_category()

  @classmethod
  def INPUT_TYPES(cls):  # pylint: disable = invalid-name, missing-function-docstring
    return {
      "required": {},
      "optional": FlexibleOptionalInputType("RGTHREE_CONTEXT"),
    }

  RETURN_TYPES = ORIG_CTX_RETURN_TYPES
  RETURN_NAMES = ORIG_CTX_RETURN_NAMES
  FUNCTION = "switch"

  def get_return_tuple(self, ctx):
    """Returns the context data. Separated so it can be overridden."""
    return get_orig_context_return_tuple(ctx)

  def switch(self, **kwargs):
    """Chooses the first non-empty Context to output."""
    ctx = None
    for key, value in kwargs.items():
      if key.startswith('ctx_') and not is_context_empty(value):
        ctx = value
        break
    return self.get_return_tuple(ctx)
