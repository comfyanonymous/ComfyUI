"""The original Context Switch."""
from .constants import get_category, get_name
from .context_utils import is_context_empty
from .utils import ByPassTypeTuple, FlexibleOptionalInputType


class RgthreeDynamicContextSwitch:
  """The initial Context Switch node."""

  NAME = get_name("Dynamic Context Switch")
  CATEGORY = get_category()

  @classmethod
  def INPUT_TYPES(cls):  # pylint: disable = invalid-name, missing-function-docstring
    return {
      "required": {},
      "optional": FlexibleOptionalInputType("RGTHREE_DYNAMIC_CONTEXT"),
    }

  RETURN_TYPES = ByPassTypeTuple(("RGTHREE_DYNAMIC_CONTEXT",))
  RETURN_NAMES = ByPassTypeTuple(("CONTEXT",))
  FUNCTION = "switch"

  def switch(self, **kwargs):
    """Chooses the first non-empty Context to output."""

    output_keys = kwargs.get('output_keys', None)

    ctx = None
    for key, value in kwargs.items():
      if key.startswith('ctx_') and not is_context_empty(value):
        ctx = value
        break

    res = [ctx]
    output_keys = output_keys.split(',') if output_keys is not None else []
    for key in output_keys:
      res.append(ctx[key] if ctx is not None and key in ctx else None)
    return tuple(res)
