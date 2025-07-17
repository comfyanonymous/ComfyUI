from abc import ABC, abstractmethod

class ComfyNodeInternal(ABC):
    """Class that all V3-based APIs inherit from for ComfyNode.

    This is intended to only be referenced within execution.py, as it has to handle all V3 APIs going forward."""
    @classmethod
    @abstractmethod
    def GET_NODE_INFO_V1(cls):
        ...
