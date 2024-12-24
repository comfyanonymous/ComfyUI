from comfy_types import IO, ComfyNodeABC, InputTypeDict
from inspect import cleandoc


class ExampleNode(ComfyNodeABC):
    """An example node that just adds 1 to an input integer.

    * Requires an IDE configured with analysis paths etc to be worth looking at.
    * Not intended for use in ComfyUI.
    """

    DESCRIPTION = cleandoc(__doc__)
    CATEGORY = "examples"

    @classmethod
    def INPUT_TYPES(s) -> InputTypeDict:
        return {
            "required": {
                "input_int": (IO.INT, {"defaultInput": True}),
            }
        }

    RETURN_TYPES = (IO.INT,)
    RETURN_NAMES = ("input_plus_one",)
    FUNCTION = "execute"

    def execute(self, input_int: int):
        return (input_int + 1,)
