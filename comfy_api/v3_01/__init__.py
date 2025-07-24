from comfy_api.v3._io import _IO
from comfy_api.v3._ui import _UI
from comfy_api.v3._resources import _RESOURCES
import logging

class Int(_IO.Int):
    class Input(_IO.Int.Input):
        def as_dict(self):
            logging.info("I am in V3_01 def of Int 😎")
            return super().as_dict()


class IO_01(_IO):
    Int = Int


io = IO_01
ui = _UI
resources = _RESOURCES

__all__ = ["io", "ui", "resources"]
