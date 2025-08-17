"""The Context Switch (Big)."""
from .constants import get_category, get_name
from .context_utils import (ALL_CTX_RETURN_TYPES, ALL_CTX_RETURN_NAMES, get_context_return_tuple)
from .context_switch import RgthreeContextSwitch


class RgthreeContextSwitchBig(RgthreeContextSwitch):
  """The Context Switch Big node."""

  NAME = get_name("Context Switch Big")
  RETURN_TYPES = ALL_CTX_RETURN_TYPES
  RETURN_NAMES = ALL_CTX_RETURN_NAMES

  def get_return_tuple(self, ctx):
    """Overrides the RgthreeContextSwitch `get_return_tuple` to return big context data."""
    return get_context_return_tuple(ctx)
