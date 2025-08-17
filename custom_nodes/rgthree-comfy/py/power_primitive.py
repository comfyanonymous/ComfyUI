import re

from .utils import FlexibleOptionalInputType, any_type
from .constants import get_category, get_name


def cast_to_str(x):
  """Handles our cast to a string."""
  if x is None:
    return ''
  try:
    return str(x)
  except (ValueError, TypeError):
    return ''


def cast_to_float(x):
  """Handles our cast to a float."""
  try:
    return float(x)
  except (ValueError, TypeError):
    return 0.0


def cast_to_bool(x):
  """Handles our cast to a bool."""
  try:
    return bool(float(x))
  except (ValueError, TypeError):
    return str(x).lower() not in ['0', 'false', 'null', 'none', '']


output_to_type = {
  'STRING': {
    'cast': cast_to_str,
    'null': '',
  },
  'FLOAT': {
    'cast': cast_to_float,
    'null': 0.0,
  },
  'INT': {
    'cast': lambda x: int(cast_to_float(x)),
    'null': 0,
  },
  'BOOLEAN': {
    'cast': cast_to_bool,
    'null': False,
  },
  # This can be removed soon, there was a bug where this should have been BOOLEAN
  'BOOL': {
    'cast': cast_to_bool,
    'null': False,
  },
}


class RgthreePowerPrimitive:
  """The Power Primitive Node."""

  NAME = get_name('Power Primitive')
  CATEGORY = get_category()

  @classmethod
  def INPUT_TYPES(cls):  # pylint: disable = invalid-name, missing-function-docstring
    return {
      "required": {},
      "optional": FlexibleOptionalInputType(any_type),
    }

  RETURN_TYPES = (any_type,)
  RETURN_NAMES = ('*',)
  FUNCTION = "main"

  def main(self, **kwargs):
    """Outputs the expected type."""
    output = kwargs.get('value', None)
    output_type = re.sub(r'\s*\([^\)]*\)\s*$', '', kwargs.get('type', ''))
    output_type = output_to_type[output_type]
    cast = output_type['cast']
    output = cast(output)

    return (output,)
