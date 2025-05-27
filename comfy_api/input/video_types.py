from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
from comfy_api.util import VideoContainer, VideoCodec, VideoComponents

class VideoInput(ABC):
    """
    Abstract base class for video input types.
    """

    @abstractmethod
    def get_components(self) -> VideoComponents:
        """
        Abstract method to get the video components (images, audio, and frame rate).

        Returns:
            VideoComponents containing images, audio, and frame rate
        """
        pass

    @abstractmethod
    def save_to(
        self,
        path: str,
        format: VideoContainer = VideoContainer.AUTO,
        codec: VideoCodec = VideoCodec.AUTO,
        metadata: Optional[dict] = None
    ):
        """
        Abstract method to save the video input to a file.
        """
        pass

    # Provide a default implementation, but subclasses can provide optimized versions
    # if possible.
    def get_dimensions(self) -> tuple[int, int]:
        """
        Returns the dimensions of the video input.

        Returns:
            Tuple of (width, height)
        """
        components = self.get_components()
        return components.images.shape[2], components.images.shape[1]

    def get_duration(self) -> float:
        """
        Returns the duration of the video in seconds.

        Returns:
            Duration in seconds
        """
        components = self.get_components()
        frame_count = components.images.shape[0]
        return float(frame_count / components.frame_rate)
