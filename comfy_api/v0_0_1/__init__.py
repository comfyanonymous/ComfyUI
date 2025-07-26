from comfy_api.v0_0_2 import (
    ComfyAPIAdapter_v0_0_2,
    Input as Input_v0_0_2,
    InputImpl as InputImpl_v0_0_2,
    Types as Types_v0_0_2,
)
from typing import Type, TYPE_CHECKING
from comfy_api.internal.async_to_sync import create_sync_class


# This version only exists to serve as a template for future version adapters.
# There is no reason anyone should ever use it.
class ComfyAPIAdapter_v0_0_1(ComfyAPIAdapter_v0_0_2):
    VERSION = "0.0.1"
    STABLE = True

class Input(Input_v0_0_2):
    pass

class InputImpl(InputImpl_v0_0_2):
    pass

class Types(Types_v0_0_2):
    pass

ComfyAPI = ComfyAPIAdapter_v0_0_1

# Create a synchronous version of the API
if TYPE_CHECKING:
    from comfy_api.v0_0_1.generated.ComfyAPISyncStub import ComfyAPISyncStub  # type: ignore

    ComfyAPISync: Type[ComfyAPISyncStub]

ComfyAPISync = create_sync_class(ComfyAPIAdapter_v0_0_1)

__all__ = [
    "ComfyAPI",
    "ComfyAPISync",
    "Input",
    "InputImpl",
    "Types",
]
