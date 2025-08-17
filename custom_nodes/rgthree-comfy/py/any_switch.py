from .context_utils import is_context_empty
from .constants import get_category, get_name
from .utils import FlexibleOptionalInputType, any_type


def is_none(value):
  """Checks if a value is none. Pulled out in case we want to expand what 'None' means."""
  if value is not None:
    if isinstance(value, dict) and 'model' in value and 'clip' in value:
      return is_context_empty(value)
  return value is None


class RgthreeAnySwitch:
  """The dynamic Any Switch. """

  NAME = get_name("Any Switch")
  CATEGORY = get_category()

  @classmethod
  def INPUT_TYPES(cls):  # pylint: disable = invalid-name, missing-function-docstring
    return {
      "required": {},
      "optional": FlexibleOptionalInputType(any_type),
    }

  RETURN_TYPES = (any_type,)
  RETURN_NAMES = ('*',)
  FUNCTION = "switch"

  def switch(self, **kwargs):
    """Chooses the first non-empty item to output."""
    any_value = None
    for key, value in kwargs.items():
      if key.startswith('any_') and not is_none(value):
        any_value = value
        break
    return (any_value,)
