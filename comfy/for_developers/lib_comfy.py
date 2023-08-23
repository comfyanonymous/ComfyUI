from asyncio import AbstractEventLoop

from ..cmd.execution import PromptQueue
from ..cmd.server import PromptServer


class Comfy:
    loop: AbstractEventLoop
    server: PromptServer
    queue: PromptQueue

    def __init__(self):
        pass
