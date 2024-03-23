import dataclasses
from typing import List

from typing_extensions import TypedDict, Literal, NotRequired, Dict


class FileOutput(TypedDict, total=False):
    filename: str
    subfolder: str
    type: Literal["output", "input", "temp"]
    abs_path: str
    name: NotRequired[str]


class Output(TypedDict, total=False):
    latents: NotRequired[List[FileOutput]]
    images: NotRequired[List[FileOutput]]


@dataclasses.dataclass
class V1QueuePromptResponse:
    urls: List[str]
    outputs: Dict[str, Output]
