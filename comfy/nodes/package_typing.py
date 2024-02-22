from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict, Union, Optional, Sequence, Dict, ClassVar, Protocol, Tuple, TypeVar, Any, Literal, \
    Callable

T = TypeVar('T')


class NumberSpecOptions(TypedDict, total=False):
    default: Union[int, float]
    min: Union[int, float]
    max: Union[int, float]
    step: Union[int, float]
    round: int


IntSpec = Dict[str, Union[
    Literal["INT"],
    Tuple[Literal["INT"], Dict[str, Union[int, float, str]]]
]]
FloatSpec = Dict[str, Union[
    Literal["FLOAT"],
    Tuple[Literal["FLOAT"], Dict[str, Union[int, float, str]]]
]]
StringSpec = Dict[str, Union[
    Literal["STRING"],
    Tuple[Literal["STRING"], Dict[str, str]]
]]
ChoiceSpec = Dict[str, Union[
    Sequence[str],  # Directly a list of choices
    Tuple[Sequence[str], Dict[str, Any]]  # Choices with additional specifications
]]

ComplexInputSpec = Dict[str, Any]
InputTypeSpec = Union[IntSpec, FloatSpec, StringSpec, ChoiceSpec, ComplexInputSpec]


class InputTypes(Protocol):
    required: Dict[str, InputTypeSpec]
    optional: Optional[Dict[str, InputTypeSpec]]
    hidden: Optional[Dict[str, InputTypeSpec]]


ValidateInputsMethod = Optional[Callable[..., Union[bool, str]]]


class CustomNode(Protocol):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes: ...

    # Optional method signature for VALIDATE_INPUTS
    VALIDATE_INPUTS: ClassVar[ValidateInputsMethod] = None

    RETURN_TYPES: ClassVar[Sequence[str]]
    RETURN_NAMES: Optional[ClassVar[Tuple[str]]]
    OUTPUT_IS_LIST: Optional[ClassVar[Sequence[bool]]]
    INPUT_IS_LIST: Optional[ClassVar[bool]]
    FUNCTION: ClassVar[str]
    CATEGORY: ClassVar[str]
    OUTPUT_NODE: Optional[ClassVar[bool]]

    def __call__(self) -> T:
        ...


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
        for self_key in exported_nodes.NODE_CLASS_MAPPINGS:
            if self_key in other.NODE_CLASS_MAPPINGS:
                exported_nodes.NODE_CLASS_MAPPINGS.pop(self_key)
            if self_key in other.NODE_DISPLAY_NAME_MAPPINGS:
                exported_nodes.NODE_DISPLAY_NAME_MAPPINGS.pop(self_key)
        for self_key in exported_nodes.EXTENSION_WEB_DIRS:
            if self_key in other.EXTENSION_WEB_DIRS:
                exported_nodes.EXTENSION_WEB_DIRS.pop(self_key)
        return exported_nodes

    def __add__(self, other):
        exported_nodes = ExportedNodes().update(self)
        return exported_nodes.update(other)
