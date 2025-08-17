"""The Dynamic Context node."""
from mimetypes import add_type
from .constants import get_category, get_name
from .utils import ByPassTypeTuple, FlexibleOptionalInputType


class RgthreeDynamicContext:
  """The Dynamic Context node.

  Similar to the static Context and Context Big nodes, this allows users to add any number and
  variety of inputs to a Dynamic Context node, and return the outputs by key name.
  """

  NAME = get_name("Dynamic Context")
  CATEGORY = get_category()

  @classmethod
  def INPUT_TYPES(cls):  # pylint: disable = invalid-name,missing-function-docstring
    return {
      "required": {},
      "optional": FlexibleOptionalInputType(add_type),
      "hidden": {},
    }

  RETURN_TYPES = ByPassTypeTuple(("RGTHREE_DYNAMIC_CONTEXT",))
  RETURN_NAMES = ByPassTypeTuple(("CONTEXT",))
  FUNCTION = "main"

  def main(self, **kwargs):
    """Creates a new context from the provided data, with an optional base ctx to start.

    This node takes a list of named inputs that are the named keys (with an optional "+ " prefix)
    which are to be stored within the ctx dict as well as a list of keys contained in `output_keys`
    to determine the list of output data.
    """

    base_ctx = kwargs.get('base_ctx', None)
    output_keys = kwargs.get('output_keys', None)

    new_ctx = base_ctx.copy() if base_ctx is not None else {}

    for key_raw, value in kwargs.items():
      if key_raw in ['base_ctx', 'output_keys']:
        continue
      key = key_raw.upper()
      if key.startswith('+ '):
        key = key[2:]
      new_ctx[key] = value

    print(new_ctx)

    res = [new_ctx]
    output_keys = output_keys.split(',') if output_keys is not None else []
    for key in output_keys:
      res.append(new_ctx[key] if key in new_ctx else None)
    return tuple(res)
