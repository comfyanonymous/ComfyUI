from __future__ import annotations

from abc import ABC, abstractmethod

from .basic_types import ImageInput


class ImageStreamInput(ABC):
    """Abstract base class for pull-based image stream inputs.

    Consumers request up to ``max_frames`` frames at a time. Producers must not
    over-return; a batch with fewer than ``max_frames`` frames signals EOF.
    """

    def __init__(self):
        #Subclasses must call this init for future core ComfyUI change compatibilty
        pass

    def reset(self) -> None:
        #This API is final. Subclasses must NOT override this for future core ComfyUI
        #change compatability. Override do_reset instead.
        return self.do_reset()

    def pull(self, max_frames: int) -> ImageInput:
        #This API is final. Subclasses must NOT override this for future core ComfyUI
        #change compatability. Override do_pull instead.
        return self.do_pull(max_frames)

    @abstractmethod
    def get_dimensions(self) -> tuple[int, int]:
        """Return the stream frame dimensions as ``(width, height)``."""
        pass

    @abstractmethod
    def do_reset(self) -> None:
        """Reset the stream so the next pull starts from frame 0."""
        pass

    @abstractmethod
    def do_pull(self, max_frames: int) -> ImageInput:
        """Return up to ``max_frames`` images.

        The returned tensor uses the normal ``IMAGE`` batch shape. A short
        return, where the batch dimension is less than ``max_frames``, is the
        EOF signal. Sources are expected to short-return at least once before
        exhaustion, including returning an empty batch.
        """
        pass
