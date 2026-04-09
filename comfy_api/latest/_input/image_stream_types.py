from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext

from comfy_execution.utils import CurrentNodeContext, get_executing_context
from comfy_execution.progress import get_progress_state
from .basic_types import ImageInput


class ImageStreamInput(ABC):
    """Abstract base class for pull-based image stream inputs.

    Consumers request up to ``max_frames`` frames at a time. Producers must not
    over-return; a batch with fewer than ``max_frames`` frames signals EOF.
    """

    def __init__(self):
        #Subclasses must call this init for future core ComfyUI change compatibilty
        self._ctx = get_executing_context()

    def reset(self) -> None:
        #This API is final. Subclasses must NOT override this for future core ComfyUI
        #change compatability. Override do_reset instead.
        with (nullcontext() if self._ctx is None else
              CurrentNodeContext(self._ctx.prompt_id, self._ctx.node_id, self._ctx.list_index)):
            self.do_reset()

        if self._ctx is not None:
            get_progress_state().finish_progress(self._ctx.node_id)

    def pull(self, max_frames: int) -> ImageInput:
        #This API is final. Subclasses must NOT override this for future core ComfyUI
        #change compatability. Override do_pull instead.
        with (nullcontext() if self._ctx is None else
              CurrentNodeContext(self._ctx.prompt_id, self._ctx.node_id, self._ctx.list_index)):
            result = self.do_pull(max_frames)

        if self._ctx is not None:
            registry = get_progress_state()
            entry = registry.nodes.get(self._ctx.node_id)
            if (int(result.shape[0]) < max_frames or
                (entry is not None and entry["max"] > 0 and entry["value"] >= entry["max"])):
                registry.finish_progress(self._ctx.node_id)

        return result

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
