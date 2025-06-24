import dataclasses
from typing import List

from typing_extensions import TypedDict, Literal, NotRequired


class FileOutput(TypedDict, total=False):
    filename: str
    subfolder: str
    type: Literal["output", "input", "temp"]
    abs_path: str
    name: NotRequired[str]


class Output(TypedDict, total=False):
    latents: NotRequired[list[FileOutput]]
    images: NotRequired[list[FileOutput]]
    videos: NotRequired[list[FileOutput]]


@dataclasses.dataclass
class V1QueuePromptResponse:
    urls: List[str]
    outputs: dict[str, Output]
