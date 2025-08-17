"""The Context Switch (Big)."""
from .constants import get_category, get_name
from .context_utils import (ALL_CTX_RETURN_TYPES, ALL_CTX_RETURN_NAMES, get_context_return_tuple)
from .context_merge import RgthreeContextMerge


class RgthreeContextMergeBig(RgthreeContextMerge):
  """The Context Merge Big node."""

  NAME = get_name("Context Merge Big")
  RETURN_TYPES = ALL_CTX_RETURN_TYPES
  RETURN_NAMES = ALL_CTX_RETURN_NAMES

  def get_return_tuple(self, ctx):
    """Returns the context data. Separated so it can be overridden."""
    return get_context_return_tuple(ctx)
