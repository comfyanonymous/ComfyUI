from __future__ import annotations

from dataclasses import dataclass, field
from typing_extensions import TypedDict, NotRequired, Generic
from typing import Union, Optional, Sequence, Dict, ClassVar, Protocol, Tuple, TypeVar, Any, Literal, \
    Callable, List, Type

T = TypeVar('T')


class IntSpecOptions(TypedDict, total=True):
    default: int
    min: int
    max: int
    step: NotRequired[int]
    display: NotRequired[Literal["number", "slider"]]


class FloatSpecOptions(TypedDict, total=True):
    default: float
    min: float
    max: float
    step: NotRequired[float]
    round: NotRequired[float]
    display: NotRequired[Literal["number", "slider"]]


class StringSpecOptions(TypedDict, total=True):
    multiline: NotRequired[bool]
    default: NotRequired[str]
    dynamicPrompts: NotRequired[bool]


class BoolSpecOptions(TypedDict):
    default: NotRequired[bool]


class DefaultSpecOptions(TypedDict):
    default: NotRequired[Any]


# todo: analyze the base_nodes for these types
CommonReturnTypes = Union[
    Literal["IMAGE", "STRING", "INT", "BOOLEAN", "FLOAT", "CONDITIONING", "LATENT", "MASK", "MODEL", "VAE", "CLIP"], str, List]

IntSpec = Tuple[Literal["INT"], IntSpecOptions]

FloatSpec = Tuple[Literal["FLOAT"], FloatSpecOptions]

StringSpec = Tuple[Literal["STRING"], StringSpecOptions]

BooleanSpec = Tuple[Literal["BOOLEAN"], BoolSpecOptions]

ChoiceSpec = Tuple[Union[Sequence[str], Sequence[float], Sequence[int]]]

NonPrimitiveTypeSpec = Tuple[CommonReturnTypes, Any]

InputTypeSpec = Union[IntSpec, FloatSpec, StringSpec, BooleanSpec, ChoiceSpec, NonPrimitiveTypeSpec]


class HiddenSpec(TypedDict, total=True):
    prompt: Literal["PROMPT"]
    extra_pnginfo: Literal["EXTRA_PNGINFO"]


class InputTypes(TypedDict, total=True):
    required: Dict[str, InputTypeSpec]
    optional: NotRequired[Dict[str, InputTypeSpec]]
    hidden: NotRequired[HiddenSpec]


ValidateInputsMethod = Optional[Callable[..., Union[bool, str]]]

IsChangedMethod = Callable[[Type[Any], ...], str]

class FunctionReturnsUIVariables(TypedDict):
    ui: dict
    result: NotRequired[Sequence[Any]]


class SaveNodeResult(TypedDict, total=True):
    abs_path: NotRequired[str]
    filename: str
    subfolder: str
    type: Literal["output", "input", "temp"]


class UIImagesImagesResult(TypedDict, total=True):
    images: List[SaveNodeResult]


class UIImagesResult(TypedDict, total=True):
    ui: UIImagesImagesResult
    result: NotRequired[Sequence[Any]]


class UILatentsLatentsResult(TypedDict, total=True):
    latents: List[SaveNodeResult]


class UILatentsResult(TypedDict, total=True):
    ui: UILatentsLatentsResult
    result: NotRequired[Sequence[Any]]


ValidatedNodeResult = Union[Tuple, UIImagesResult, UILatentsResult, FunctionReturnsUIVariables]


class CustomNode(Protocol):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes: ...

    # Optional method signature for VALIDATE_INPUTS
    VALIDATE_INPUTS: Optional[ClassVar[ValidateInputsMethod]]

    RETURN_TYPES: ClassVar[Tuple[CommonReturnTypes, ...]]
    RETURN_NAMES: Optional[ClassVar[Tuple[str, ...]]]
    OUTPUT_IS_LIST: Optional[ClassVar[Tuple[bool, ...]]]
    INPUT_IS_LIST: Optional[ClassVar[bool]]
    FUNCTION: ClassVar[str]
    CATEGORY: ClassVar[str]
    OUTPUT_NODE: Optional[ClassVar[bool]]

    IS_CHANGED: Optional[ClassVar[IsChangedMethod]]


@dataclass
class ExportedNodes:
    NODE_CLASS_MAPPINGS: Dict[str, CustomNode] = field(default_factory=dict)
    NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = field(default_factory=dict)
    EXTENSION_WEB_DIRS: Dict[str, str] = field(default_factory=dict)

    def update(self, exported_nodes: ExportedNodes) -> ExportedNodes:
        self.NODE_CLASS_MAPPINGS.update(exported_nodes.NODE_CLASS_MAPPINGS)
        self.NODE_DISPLAY_NAME_MAPPINGS.update(exported_nodes.NODE_DISPLAY_NAME_MAPPINGS)
        self.EXTENSION_WEB_DIRS.update(exported_nodes.EXTENSION_WEB_DIRS)
        return self

    def __len__(self):
        return len(self.NODE_CLASS_MAPPINGS)

    def __sub__(self, other: ExportedNodes):
        exported_nodes = ExportedNodes().update(self)
        for self_key in frozenset(exported_nodes.NODE_CLASS_MAPPINGS):
            if self_key in other.NODE_CLASS_MAPPINGS:
                exported_nodes.NODE_CLASS_MAPPINGS.pop(self_key)
            if self_key in other.NODE_DISPLAY_NAME_MAPPINGS:
                exported_nodes.NODE_DISPLAY_NAME_MAPPINGS.pop(self_key)
        for self_key in frozenset(exported_nodes.EXTENSION_WEB_DIRS):
            if self_key in other.EXTENSION_WEB_DIRS:
                exported_nodes.EXTENSION_WEB_DIRS.pop(self_key)
        return exported_nodes

    def __add__(self, other):
        exported_nodes = ExportedNodes().update(self)
        return exported_nodes.update(other)
