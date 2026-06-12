from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Optional
from .._input import ImageInput, AudioInput, MaskInput

class VideoCodec(str, Enum):
    AUTO = "auto"
    H264 = "h264"

    @classmethod
    def as_input(cls) -> list[str]:
        """
        Returns a list of codec names that can be used as node input.
        """
        return [member.value for member in cls]


class VideoBitDepth(str, Enum):
    AUTO = "auto"
    BIT_8 = "8-bit"
    BIT_10 = "10-bit"

    @classmethod
    def as_input(cls) -> list[str]:
        """Returns a list of bit depth names that can be used as node input."""
        return [member.value for member in cls]

    def bits(self) -> Optional[int]:
        """Returns the numeric bit depth, or None for AUTO."""
        if self == VideoBitDepth.AUTO:
            return None
        return int(self.value.split("-")[0])

class VideoContainer(str, Enum):
    AUTO = "auto"
    MP4 = "mp4"

    @classmethod
    def as_input(cls) -> list[str]:
        """
        Returns a list of container names that can be used as node input.
        """
        return [member.value for member in cls]

    @classmethod
    def get_extension(cls, value) -> str:
        """
        Returns the file extension for the container.
        """
        if isinstance(value, str):
            value = cls(value)
        if value == VideoContainer.MP4 or value == VideoContainer.AUTO:
            return "mp4"
        return ""

@dataclass
class VideoComponents:
    """
    Dataclass representing the components of a video.
    """

    images: ImageInput
    frame_rate: Fraction
    audio: Optional[AudioInput] = None
    metadata: Optional[dict] = None
    alpha: Optional[MaskInput] = None
