"""The Context Switch (Big)."""
from .constants import get_category, get_name
from .context_utils import (ORIG_CTX_RETURN_TYPES, ORIG_CTX_RETURN_NAMES, merge_new_context,
                            get_orig_context_return_tuple, is_context_empty)
from .utils import FlexibleOptionalInputType


class RgthreeContextMerge:
  """The Context Merge node."""

  NAME = get_name("Context Merge")
  CATEGORY = get_category()

  @classmethod
  def INPUT_TYPES(cls):  # pylint: disable = invalid-name, missing-function-docstring
    return {
      "required": {},
      "optional": FlexibleOptionalInputType("RGTHREE_CONTEXT"),
    }

  RETURN_TYPES = ORIG_CTX_RETURN_TYPES
  RETURN_NAMES = ORIG_CTX_RETURN_NAMES
  FUNCTION = "merge"

  def get_return_tuple(self, ctx):
    """Returns the context data. Separated so it can be overridden."""
    return get_orig_context_return_tuple(ctx)

  def merge(self, **kwargs):
    """Merges any non-null passed contexts; later ones overriding earlier."""
    ctxs = [
      value for key, value in kwargs.items()
      if key.startswith('ctx_') and not is_context_empty(value)
    ]
    ctx = merge_new_context(*ctxs)

    return self.get_return_tuple(ctx)
