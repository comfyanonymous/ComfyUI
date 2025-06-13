from __future__ import annotations
from typing import Any, Literal, TYPE_CHECKING, TypeVar
from enum import Enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from comfy.comfy_types.node_typing import IO

# if TYPE_CHECKING:
import torch


class InputBehavior(str, Enum):
    '''Likely deprecated; required/optional can be a bool, unlikely to be more categories that fit.'''
    required = "required"
    optional = "optional"


class FolderType(str, Enum):
    input = "input"
    output = "output"
    temp = "temp"

class RemoteOptions:
    def __init__(self, route: str, refresh_button: bool, control_after_refresh: Literal["first", "last"]="first",
                 timeout: int=None, max_retries: int=None, refresh: int=None):
        self.route = route
        """The route to the remote source."""
        self.refresh_button = refresh_button
        """Specifies whether to show a refresh button in the UI below the widget."""
        self.control_after_refresh = control_after_refresh
        """Specifies the control after the refresh button is clicked. If "first", the first item will be automatically selected, and so on."""
        self.timeout = timeout
        """The maximum amount of time to wait for a response from the remote source in milliseconds."""
        self.max_retries = max_retries
        """The maximum number of retries before aborting the request."""
        self.refresh = refresh
        """The TTL of the remote input's value in milliseconds. Specifies the interval at which the remote input's value is refreshed."""

    def as_dict(self):
        return prune_dict({
            "route": self.route,
            "refresh_button": self.refresh_button,
            "control_after_refresh": self.control_after_refresh,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "refresh": self.refresh,
        })


def is_class(obj):
    '''
    Returns True if is a class type.
    Returns False if is a class instance.
    '''
    return isinstance(obj, type)


class NumberDisplay(str, Enum):
    number = "number"
    slider = "slider"


class IO_V3:
    '''
    Base class for V3 Inputs and Outputs.
    '''
    Type = Any

    def __init__(self):
        pass

    def __init_subclass__(cls, io_type: IO | str, **kwargs):
        # TODO: do we need __ne__ trick for io_type? (see IO.__ne__ for details)
        cls.io_type = io_type
        super().__init_subclass__(**kwargs)

class InputV3(IO_V3, io_type=None):
    '''
    Base class for a V3 Input.
    '''
    def __init__(self, id: str, display_name: str=None, optional=False, tooltip: str=None, lazy: bool=None):
        super().__init__()
        self.id = id
        self.display_name = display_name
        self.optional = optional
        self.tooltip = tooltip
        self.lazy = lazy
    
    def as_dict_V1(self):
        return prune_dict({
            "display_name": self.display_name,
            "tooltip": self.tooltip,
            "lazy": self.lazy
        })
    
    def get_io_type_V1(self):
        return self.io_type

class WidgetInputV3(InputV3, io_type=None):
    '''
    Base class for a V3 Input with widget.
    '''
    def __init__(self, id: str, display_name: str=None, optional=False, tooltip: str=None, lazy: bool=None,
                 default: Any=None,
                 socketless: bool=None, widgetType: str=None):
        super().__init__(id, display_name, optional, tooltip, lazy)
        self.default = default
        self.socketless = socketless
        self.widgetType = widgetType
    
    def as_dict_V1(self):
        return super().as_dict_V1() | prune_dict({
            "default": self.default,
            "socketless": self.socketless,
            "widgetType": self.widgetType,
        })

class OutputV3(IO_V3, io_type=None):
    def __init__(self, id: str, display_name: str=None, tooltip: str=None,
                 is_output_list=False):
        self.id = id
        self.display_name = display_name
        self.tooltip = tooltip
        self.is_output_list = is_output_list

def CustomType(io_type: IO | str) -> type[IO_V3]:
    name = f"{io_type}_IO_V3"
    return type(name, (IO_V3,), {}, io_type=io_type)

def CustomInput(id: str, io_type: IO | str, display_name: str=None, optional=False, tooltip: str=None, lazy: bool=None) -> InputV3:
    '''
    Defines input for 'io_type'. Can be used to stand in for non-core types.
    '''
    input_kwargs = {
        "id": id,
        "display_name": display_name,
        "optional": optional,
        "tooltip": tooltip,
        "lazy": lazy,
    }
    return type(f"{io_type}Input", (InputV3,), {}, io_type=io_type)(**input_kwargs)

def CustomOutput(id: str, io_type: IO | str, display_name: str=None, tooltip: str=None) -> OutputV3:
    '''
    Defines output for 'io_type'. Can be used to stand in for non-core types.
    '''
    input_kwargs = {
        "id": id,
        "display_name": display_name,
        "tooltip": tooltip,
    }
    return type(f"{io_type}Output", (OutputV3,), {}, io_type=io_type)(**input_kwargs)


class BooleanInput(WidgetInputV3, io_type=IO.BOOLEAN):
    '''
    Boolean input.
    '''
    Type = bool
    def __init__(self, id: str, display_name: str=None, optional=False, tooltip: str=None, lazy: bool=None,
                 default: bool=None, label_on: str=None, label_off: str=None,
                 socketless: bool=None, widgetType: str=None):
        super().__init__(id, display_name, optional, tooltip, lazy, default, socketless, widgetType)
        self.label_on = label_on
        self.label_off = label_off
        self.default: bool
    
    def as_dict_V1(self):
        return super().as_dict_V1() | prune_dict({
            "label_on": self.label_on,
            "label_off": self.label_off,
        })

class BooleanOutput(OutputV3, io_type=IO.BOOLEAN):
    ...

class IntegerInput(WidgetInputV3, io_type=IO.INT):
    '''
    Integer input.
    '''
    Type = int
    def __init__(self, id: str, display_name: str=None, optional=False, tooltip: str=None, lazy: bool=None,
                 default: int=None, min: int=None, max: int=None, step: int=None, control_after_generate: bool=None,
                 display_mode: NumberDisplay=None, socketless: bool=None, widgetType: str=None):
        super().__init__(id, display_name, optional, tooltip, lazy, default, socketless, widgetType)
        self.min = min
        self.max = max
        self.step = step
        self.control_after_generate = control_after_generate
        self.display_mode = display_mode
        self.default: int
    
    def as_dict_V1(self):
        return super().as_dict_V1() | prune_dict({
            "min": self.min,
            "max": self.max,
            "step": self.step,
            "control_after_generate": self.control_after_generate,
            "display": self.display_mode, # NOTE: in frontend, the parameter is called "display"
        })

class IntegerOutput(OutputV3, io_type=IO.INT):
    ...

class FloatInput(WidgetInputV3, io_type=IO.FLOAT):
    '''
    Float input.
    '''
    Type = float
    def __init__(self, id: str, display_name: str=None, optional=False, tooltip: str=None, lazy: bool=None,
                 default: float=None, min: float=None, max: float=None, step: float=None, round: float=None,
                 display_mode: NumberDisplay=None, socketless: bool=None, widgetType: str=None):
        super().__init__(id, display_name, optional, tooltip, lazy, default, socketless, widgetType)
        self.default = default
        self.min = min
        self.max = max
        self.step = step
        self.round = round
        self.display_mode = display_mode
        self.default: float
    
    def as_dict_V1(self):
        return super().as_dict_V1() | prune_dict({
            "min": self.min,
            "max": self.max,
            "step": self.step,
            "round": self.round,
            "display": self.display_mode, # NOTE: in frontend, the parameter is called "display"
        })
    
class FloatOutput(OutputV3, io_type=IO.FLOAT):
    ...

class StringInput(WidgetInputV3, io_type=IO.STRING):
    '''
    String input.
    '''
    Type = str
    def __init__(self, id: str, display_name: str=None, optional=False, tooltip: str=None, lazy: bool=None,
                 multiline=False, placeholder: str=None, default: int=None,
                 socketless: bool=None, widgetType: str=None):
        super().__init__(id, display_name, optional, tooltip, lazy, default, socketless, widgetType)
        self.multiline = multiline
        self.placeholder = placeholder
        self.default: str
    
    def as_dict_V1(self):
        return super().as_dict_V1() | prune_dict({
            "multiline": self.multiline,
            "placeholder": self.placeholder,
        })

class StringOutput(OutputV3, io_type=IO.STRING):
    ...

class ComboInput(WidgetInputV3, io_type=IO.COMBO):
    '''Combo input (dropdown).'''
    Type = str
    def __init__(self, id: str, options: list[str]=None, display_name: str=None, optional=False, tooltip: str=None, lazy: bool=None,
                 default: str=None, control_after_generate: bool=None,
                 image_upload: bool=None, image_folder: FolderType=None,
                 remote: RemoteOptions=None,
                 socketless: bool=None, widgetType: str=None):
        super().__init__(id, display_name, optional, tooltip, lazy, default, socketless, widgetType)
        self.multiselect = False
        self.options = options
        self.control_after_generate = control_after_generate
        self.image_upload = image_upload
        self.image_folder = image_folder
        self.remote = remote
        self.default: str
    
    def as_dict_V1(self):
        return super().as_dict_V1() | prune_dict({
            "multiselect": self.multiselect,
            "options": self.options,
            "control_after_generate": self.control_after_generate,
            "image_upload": self.image_upload,
            "image_folder": self.image_folder.value if self.image_folder else None,
            "remote": self.remote.as_dict() if self.remote else None,
        })

class MultiselectComboWidget(ComboInput, io_type=IO.COMBO):
    '''Multiselect Combo input (dropdown for selecting potentially more than one value).'''
    def __init__(self, id: str, options: list[str], display_name: str=None, optional=False, tooltip: str=None, lazy: bool=None,
                 default: list[str]=None, placeholder: str=None, chip: bool=None, control_after_generate: bool=None,
                 socketless: bool=None, widgetType: str=None):
        super().__init__(id, options, display_name, optional, tooltip, lazy, default, control_after_generate, socketless, widgetType)
        self.multiselect = True
        self.placeholder = placeholder
        self.chip = chip
        self.default: list[str]
    
    def as_dict_V1(self):
        return super().as_dict_V1() | prune_dict({
            "multiselect": self.multiselect,
            "placeholder": self.placeholder,
            "chip": self.chip,
        })

class ImageInput(InputV3, io_type=IO.IMAGE):
    '''Image input.'''
    Type = torch.Tensor

class ImageOutput(OutputV3, io_type=IO.IMAGE):
    '''Image output.'''
    Type = torch.Tensor

class MaskInput(InputV3, io_type=IO.MASK):
    '''Mask input.'''
    Type = torch.Tensor

class MaskOutput(OutputV3, io_type=IO.MASK):
    '''Mask output.'''
    Type = torch.Tensor

class LatentInput(InputV3, io_type=IO.LATENT):
    '''Latent input.'''
    # TODO: make Type a TypedDict
    ...

class LatentOutput(OutputV3, io_type=IO.LATENT):
    '''Latent output.'''
    # TODO: make Type a TypedDict
    ...

class ConditioningInput(InputV3, io_type=IO.CONDITIONING):
    '''Conditioning input.'''
    # TODO: make Type a TypedDict
    ...

class ConditioningOutput(OutputV3, io_type=IO.CONDITIONING):
    '''Conditioning output.'''
    # TODO: make Type a TypedDict
    ...

class SamplerInput(InputV3, io_type=IO.SAMPLER):
    '''Sampler input.'''
    ...

class SamplerOutput(OutputV3, io_type=IO.SAMPLER):
    '''Sampler output.'''
    ...

class SigmasInput(InputV3, io_type=IO.SIGMAS):
    '''Sigmas input.'''
    ...

class SigmasOutput(OutputV3, io_type=IO.SIGMAS):
    '''Sigmas output.'''
    ...

class GuiderInput(InputV3, io_type=IO.GUIDER):
    '''Guider input.'''
    ...

class GuiderOutput(OutputV3, io_type=IO.GUIDER):
    '''Guider output.'''
    ...

class NoiseInput(InputV3, io_type=IO.NOISE):
    '''Noise input.'''
    ...

class NoiseOutput(OutputV3, io_type=IO.NOISE):
    '''Noise output.'''
    ...

class ClipInput(InputV3, io_type=IO.CLIP):
    '''Clip input.'''
    ...

class ClipOutput(OutputV3, io_type=IO.CLIP):
    '''Clip output.'''
    ...

class ControlNetInput(InputV3, io_type=IO.CONTROL_NET):
    '''ControlNet input.'''
    ...

class ControlNetOutput(OutputV3, io_type=IO.CONTROL_NET):
    '''ControlNet output.'''
    ...

class VaeInput(InputV3, io_type=IO.VAE):
    '''Vae input.'''
    ...

class VaeOutput(OutputV3, io_type=IO.VAE):
    '''Vae output.'''
    ...

class ModelInput(InputV3, io_type=IO.MODEL):
    '''Model input.'''
    ...

class ModelOutput(OutputV3, io_type=IO.MODEL):
    '''Model output.'''
    ...

class ClipVisionInput(InputV3, io_type=IO.CLIP_VISION):
    '''ClipVision input.'''
    ...

class ClipVisionOutput(OutputV3, io_type=IO.CLIP_VISION):
    '''ClipVision output.'''
    ...

class ClipVisionOutputInput(InputV3, io_type=IO.CLIP_VISION_OUTPUT):
    '''CLipVisionOutput input.'''
    ...

class ClipVisionOutputOutput(OutputV3, io_type=IO.CLIP_VISION_OUTPUT):
    '''CLipVisionOutput output.'''
    ...

class StyleModelInput(InputV3, io_type=IO.STYLE_MODEL):
    '''StyleModel input.'''
    ...

class StyleModelOutput(OutputV3, io_type=IO.STYLE_MODEL):
    '''StyleModel output.'''
    ...

class GligenInput(InputV3, io_type=IO.GLIGEN):
    '''Gligen input.'''
    ...

class GligenOutput(OutputV3, io_type=IO.GLIGEN):
    '''Gligen output.'''
    ...

class UpscaleModelInput(InputV3, io_type=IO.UPSCALE_MODEL):
    '''UpscaleModel input.'''
    ...

class UpscaleModelOutput(OutputV3, io_type=IO.UPSCALE_MODEL):
    '''UpscaleModel output.'''
    ...

class AudioInput(InputV3, io_type=IO.AUDIO):
    '''Audio input.'''
    ...

class AudioOutput(OutputV3, io_type=IO.AUDIO):
    '''Audio output.'''
    ...

class PointInput(InputV3, io_type=IO.POINT):
    '''Point input.'''
    ...

class PointOutput(OutputV3, io_type=IO.POINT):
    '''Point output.'''
    ...

class FaceAnalysisInput(InputV3, io_type=IO.FACE_ANALYSIS):
    '''FaceAnalysis input.'''
    ...

class FaceAnalysisOutput(OutputV3, io_type=IO.FACE_ANALYSIS):
    '''FaceAnalysis output.'''
    ...

class BBOXInput(InputV3, io_type=IO.BBOX):
    '''Bbox input.'''
    ...

class BBOXOutput(OutputV3, io_type=IO.BBOX):
    '''Bbox output.'''
    ...

class SEGSInput(InputV3, io_type=IO.SEGS):
    '''SEGS input.'''
    ...

class SEGSOutput(OutputV3, io_type=IO.SEGS):
    '''SEGS output.'''
    ...

class VideoInput(InputV3, io_type=IO.VIDEO):
    '''Video input.'''
    ...

class VideoOutput(OutputV3, io_type=IO.VIDEO):
    '''Video output.'''
    ...


class MultitypedInput(InputV3, io_type="COMFY_MULTITYPED_V3"):
    '''
    Input that permits more than one input type.
    '''
    def __init__(self, id: str, io_types: list[type[IO_V3] | InputV3 | IO |str], display_name: str=None, optional=False, tooltip: str=None,):
        super().__init__(id, display_name, optional, tooltip)
        self._io_types = io_types
    
    @property
    def io_types(self) -> list[type[InputV3]]:
        '''
        Returns list of InputV3 class types permitted.
        '''
        io_types = []
        for x in self._io_types:
            if not is_class(x):
                io_types.append(type(x))
            else:
                io_types.append(x)
        return io_types
    
    def get_io_type_V1(self):
        return ",".join(x.io_type for x in self.io_types)


class DynamicInput(InputV3, io_type=None):
    '''
    Abstract class for dynamic input registration.
    '''
    def __init__(self, io_type: str, id: str, display_name: str=None):
        super().__init__(io_type, id, display_name)

class DynamicOutput(OutputV3, io_type=None):
    '''
    Abstract class for dynamic output registration.
    '''
    def __init__(self, io_type: str, id: str, display_name: str=None):
        super().__init__(io_type, id, display_name)

class AutoGrowDynamicInput(DynamicInput, io_type="COMFY_MULTIGROW_V3"):
    '''
    Dynamic Input that adds another template_input each time one is provided.

    Additional inputs are forced to have 'InputBehavior.optional'.
    '''
    def __init__(self, id: str, template_input: InputV3, min: int=1, max: int=None):
        super().__init__("AutoGrowDynamicInput", id)
        self.template_input = template_input
        if min is not None:
            assert(min >= 1)
        if max is not None:
            assert(max >= 1)
        self.min = min
        self.max = max

class ComboDynamicInput(DynamicInput, io_type="COMFY_COMBODYNAMIC_V3"):
    def __init__(self, id: str):
        pass

AutoGrowDynamicInput(id="dynamic", template_input=ImageInput(id="image"))


class Hidden(str, Enum):
    '''
    Enumerator for requesting hidden variables in nodes.
    '''
    
    unique_id = "UNIQUE_ID"
    """UNIQUE_ID is the unique identifier of the node, and matches the id property of the node on the client side. It is commonly used in client-server communications (see messages)."""
    prompt = "PROMPT"
    """PROMPT is the complete prompt sent by the client to the server. See the prompt object for a full description."""
    extra_pnginfo = "EXTRA_PNGINFO"
    """EXTRA_PNGINFO is a dictionary that will be copied into the metadata of any .png files saved. Custom nodes can store additional information in this dictionary for saving (or as a way to communicate with a downstream node)."""
    dynprompt = "DYNPROMPT"
    """DYNPROMPT is an instance of comfy_execution.graph.DynamicPrompt. It differs from PROMPT in that it may mutate during the course of execution in response to Node Expansion."""
    auth_token_comfy_org = "AUTH_TOKEN_COMFY_ORG"
    """AUTH_TOKEN_COMFY_ORG is a token acquired from signing into a ComfyOrg account on frontend."""
    api_key_comfy_org = "API_KEY_COMFY_ORG"
    """API_KEY_COMFY_ORG is an API Key generated by ComfyOrg that allows skipping signing into a ComfyOrg account on frontend."""


@dataclass
class NodeInfoV1:
    input: dict=None
    input_order: dict[str, list[str]]=None
    output: list[str]=None
    output_is_list: list[bool]=None
    output_name: list[str]=None
    output_tooltips: list[str]=None
    name: str=None
    display_name: str=None
    description: str=None
    python_module: Any=None
    category: str=None
    output_node: bool=None
    deprecated: bool=None
    experimental: bool=None
    api_node: bool=None


def as_pruned_dict(dataclass_obj):
    '''Return dict of dataclass object with pruned None values.'''
    return prune_dict(asdict(dataclass_obj))

def prune_dict(d: dict):
    return {k: v for k,v in d.items() if v is not None}


@dataclass
class SchemaV3:
    """Definition of V3 node properties."""

    node_id: str
    """ID of node - should be globally unique. If this is a custom node, add a prefix or postfix to avoid name clashes."""
    display_name: str = None
    """Display name of node."""
    category: str = "sd"
    """The category of the node, as per the "Add Node" menu."""
    inputs: list[InputV3]=None
    outputs: list[OutputV3]=None
    hidden: list[Hidden]=None
    description: str=""
    """Node description, shown as a tooltip when hovering over the node."""
    is_input_list: bool = False
    """A flag indicating if this node implements the additional code necessary to deal with OUTPUT_IS_LIST nodes.

    All inputs of ``type`` will become ``list[type]``, regardless of how many items are passed in.  This also affects ``check_lazy_status``.

    From the docs:

    A node can also override the default input behaviour and receive the whole list in a single call. This is done by setting a class attribute `INPUT_IS_LIST` to ``True``.

    Comfy Docs: https://docs.comfy.org/custom-nodes/backend/lists#list-processing
    """
    is_output_node: bool=False
    """Flags this node as an output node, causing any inputs it requires to be executed.

    If a node is not connected to any output nodes, that node will not be executed.  Usage::

        OUTPUT_NODE = True

    From the docs:

    By default, a node is not considered an output. Set ``OUTPUT_NODE = True`` to specify that it is.

    Comfy Docs: https://docs.comfy.org/custom-nodes/backend/server_overview#output-node
    """
    is_deprecated: bool=False
    """Flags a node as deprecated, indicating to users that they should find alternatives to this node."""
    is_experimental: bool=False
    """Flags a node as experimental, informing users that it may change or not work as expected."""
    is_api_node: bool=False
    """Flags a node as an API node. See: https://docs.comfy.org/tutorials/api-nodes/overview."""

# class SchemaV3Class:
#     def __init__(self,
#             node_id: str,
#             node_name: str,
#             category: str,
#             inputs: list[InputV3],
#             outputs: list[OutputV3]=None,
#             hidden: list[Hidden]=None,
#             description: str="",
#             is_input_list: bool = False,
#             is_output_node: bool=False,
#             is_deprecated: bool=False,
#             is_experimental: bool=False,
#             is_api_node: bool=False,
#     ):
#         self.node_id = node_id
#         """ID of node - should be globally unique. If this is a custom node, add a prefix or postfix to avoid name clashes."""
#         self.node_name = node_name
#         """Display name of node."""
#         self.category = category
#         """The category of the node, as per the "Add Node" menu."""
#         self.inputs = inputs
#         self.outputs = outputs
#         self.hidden = hidden
#         self.description = description
#         """Node description, shown as a tooltip when hovering over the node."""
#         self.is_input_list = is_input_list
#         """A flag indicating if this node implements the additional code necessary to deal with OUTPUT_IS_LIST nodes.

#     All inputs of ``type`` will become ``list[type]``, regardless of how many items are passed in.  This also affects ``check_lazy_status``.

#     From the docs:

#     A node can also override the default input behaviour and receive the whole list in a single call. This is done by setting a class attribute `INPUT_IS_LIST` to ``True``.

#     Comfy Docs: https://docs.comfy.org/custom-nodes/backend/lists#list-processing
#     """
#         self.is_output_node = is_output_node
#         """Flags this node as an output node, causing any inputs it requires to be executed.

#     If a node is not connected to any output nodes, that node will not be executed.  Usage::

#         OUTPUT_NODE = True

#     From the docs:

#     By default, a node is not considered an output. Set ``OUTPUT_NODE = True`` to specify that it is.

#     Comfy Docs: https://docs.comfy.org/custom-nodes/backend/server_overview#output-node
#     """
#         self.is_deprecated = is_deprecated
#         """Flags a node as deprecated, indicating to users that they should find alternatives to this node."""
#         self.is_experimental = is_experimental
#         """Flags a node as experimental, informing users that it may change or not work as expected."""
#         self.is_api_node = is_api_node
#         """Flags a node as an API node. See: https://docs.comfy.org/tutorials/api-nodes/overview."""


class Serializer:
    def __init_subclass__(cls, io_type: IO | str, **kwargs):
        cls.io_type = io_type
        super().__init_subclass__(**kwargs)

    @classmethod
    def serialize(cls, o: Any) -> str:
        pass

    @classmethod
    def deserialize(cls, s: str) -> Any:
        pass


def prepare_class_clone(c: ComfyNodeV3 | type[ComfyNodeV3]) -> type[ComfyNodeV3]:
    """Creates clone of real node class to prevent monkey-patching."""
    c_type: type[ComfyNodeV3] = c if is_class(c) else type(c)
    type_clone: type[ComfyNodeV3] = type(f"CLEAN_{c_type.__name__}", c_type.__bases__, {})
    # TODO: what parameters should be carried over?
    type_clone.SCHEMA = c_type.SCHEMA
    # TODO: add anything we would want to expose inside node's execute function
    return type_clone


class classproperty(object):
    def __init__(self, f):
        self.f = f
    def __get__(self, obj, owner):
        return self.f(owner)


class ComfyNodeV3(ABC):
    """Common base class for all V3 nodes."""

    RELATIVE_PYTHON_MODULE = None
    SCHEMA = None

    @classmethod
    def GET_NODE_INFO_V3(cls) -> dict[str, Any]:
        schema = cls.GET_SCHEMA()
        # TODO: finish
        return None

    @classmethod
    @abstractmethod
    def DEFINE_SCHEMA(cls) -> SchemaV3:
        """
        Override this function with one that returns a SchemaV3 instance.
        """
        return None
    DEFINE_SCHEMA = None

    @classmethod
    @abstractmethod
    def execute(cls, **kwargs) -> NodeOutput:
        pass
    execute = None

    @classmethod
    def GET_SERIALIZERS(cls) -> list[Serializer]:
        return []

    def __init__(self):
        self.__class__.VALIDATE_CLASS()

    @classmethod
    def VALIDATE_CLASS(cls):
        if not callable(cls.DEFINE_SCHEMA):
            raise Exception(f"No DEFINE_SCHEMA function was defined for node class {cls.__name__}.")
        if not callable(cls.execute):
            raise Exception(f"No execute function was defined for node class {cls.__name__}.")

    @classmethod
    def prepare_class_clone(cls) -> type[ComfyNodeV3]:
        """Creates clone of real node class to prevent monkey-patching."""
        c_type: type[ComfyNodeV3] = cls if is_class(cls) else type(cls)
        type_clone: type[ComfyNodeV3] = type(f"CLEAN_{c_type.__name__}", c_type.__bases__, {})
        # TODO: what parameters should be carried over?
        type_clone.SCHEMA = c_type.SCHEMA
        # TODO: add anything we would want to expose inside node's execute function
        return type_clone

    #############################################
    # V1 Backwards Compatibility code
    #--------------------------------------------
    _DESCRIPTION = None
    @classproperty
    def DESCRIPTION(cls):
        if cls._DESCRIPTION is None:
            cls.GET_SCHEMA()
        return cls._DESCRIPTION

    _CATEGORY = None
    @classproperty
    def CATEGORY(cls):
        if cls._CATEGORY is None:
            cls.GET_SCHEMA()
        return cls._CATEGORY

    _EXPERIMENTAL = None
    @classproperty
    def EXPERIMENTAL(cls):
        if cls._EXPERIMENTAL is None:
            cls.GET_SCHEMA()
        return cls._EXPERIMENTAL

    _DEPRECATED = None
    @classproperty
    def DEPRECATED(cls):
        if cls._DEPRECATED is None:
            cls.GET_SCHEMA()
        return cls._DEPRECATED

    _API_NODE = None
    @classproperty
    def API_NODE(cls):
        if cls._API_NODE is None:
            cls.GET_SCHEMA()
        return cls._API_NODE

    _OUTPUT_NODE = None
    @classproperty
    def OUTPUT_NODE(cls):
        if cls._OUTPUT_NODE is None:
            cls.GET_SCHEMA()
        return cls._OUTPUT_NODE

    _INPUT_IS_LIST = None
    @classproperty
    def INPUT_IS_LIST(cls):
        if cls._INPUT_IS_LIST is None:
            cls.GET_SCHEMA()
        return cls._INPUT_IS_LIST
    _OUTPUT_IS_LIST = None

    @classproperty
    def OUTPUT_IS_LIST(cls):
        if cls._OUTPUT_IS_LIST is None:
            cls.GET_SCHEMA()
        return cls._OUTPUT_IS_LIST

    _RETURN_TYPES = None
    @classproperty
    def RETURN_TYPES(cls):
        if cls._RETURN_TYPES is None:
            cls.GET_SCHEMA()
        return cls._RETURN_TYPES

    _RETURN_NAMES = None
    @classproperty
    def RETURN_NAMES(cls):
        if cls._RETURN_NAMES is None:
            cls.GET_SCHEMA()
        return cls._RETURN_NAMES

    _OUTPUT_TOOLTIPS = None
    @classproperty
    def OUTPUT_TOOLTIPS(cls):
        if cls._OUTPUT_TOOLTIPS is None:
            cls.GET_SCHEMA()
        return cls._OUTPUT_TOOLTIPS

    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict]:
        schema = cls.DEFINE_SCHEMA()
        # for V1, make inputs be a dict with potential keys {required, optional, hidden}
        input = {
            "required": {}
        }
        if schema.inputs:
            for i in schema.inputs:
                key = "optional" if i.optional else "required"
                input.setdefault(key, {})[i.id] = (i.get_io_type_V1(), i.as_dict_V1())
        if schema.hidden:
            for hidden in schema.hidden:
                input.setdefault("hidden", {})[hidden.name] = (hidden.value,)
        return input

    @classmethod
    def GET_SCHEMA(cls) -> SchemaV3:
        cls.VALIDATE_CLASS()
        schema = cls.DEFINE_SCHEMA()
        if cls._DESCRIPTION is None:
            cls._DESCRIPTION = schema.description
        if cls._CATEGORY is None:
            cls._CATEGORY = schema.category
        if cls._EXPERIMENTAL is None:
            cls._EXPERIMENTAL = schema.is_experimental
        if cls._DEPRECATED is None:
            cls._DEPRECATED = schema.is_deprecated
        if cls._API_NODE is None:
            cls._API_NODE = schema.is_api_node
        if cls._OUTPUT_NODE is None:
            cls._OUTPUT_NODE = schema.is_output_node
        if cls._INPUT_IS_LIST is None:
            cls._INPUT_IS_LIST = schema.is_input_list
        
        if cls._RETURN_TYPES is None:
            output = []
            output_name = []
            output_is_list = []
            output_tooltips = []
            if schema.outputs:
                for o in schema.outputs:
                    output.append(o.io_type)
                    output_name.append(o.display_name if o.display_name else o.io_type)
                    output_is_list.append(o.is_output_list)
                    output_tooltips.append(o.tooltip if o.tooltip else None)
            
            cls._RETURN_TYPES = output
            cls._RETURN_NAMES = output_name
            cls._OUTPUT_IS_LIST = output_is_list
            cls._OUTPUT_TOOLTIPS = output_tooltips
        cls.SCHEMA = schema
        return schema

    @classmethod
    def GET_NODE_INFO_V1(cls) -> dict[str, Any]:
        schema = cls.GET_SCHEMA()
        # get V1 inputs
        input = cls.INPUT_TYPES()

        # create separate lists from output fields
        output = []
        output_is_list = []
        output_name = []
        output_tooltips = []
        if schema.outputs:
            for o in schema.outputs:
                output.append(o.io_type)
                output_is_list.append(o.is_output_list)
                output_name.append(o.display_name if o.display_name else o.io_type)
                output_tooltips.append(o.tooltip if o.tooltip else None)

        info = NodeInfoV1(
            input=input,
            input_order={key: list(value.keys()) for (key, value) in input.items()},
            output=output,
            output_is_list=output_is_list,
            output_name=output_name,
            output_tooltips=output_tooltips,
            name=schema.node_id,
            display_name=schema.display_name,
            category=schema.category,
            description=schema.description,
            output_node=schema.is_output_node,
            deprecated=schema.is_deprecated,
            experimental=schema.is_experimental,
            api_node=schema.is_api_node,
            python_module=getattr(cls, "RELATIVE_PYTHON_MODULE", "nodes")
        )
        return asdict(info)
    #--------------------------------------------
    #############################################

# class ReturnedInputs:
#     def __init__(self):
#         pass

# class ReturnedOutputs:
#     def __init__(self):
#         pass


class NodeOutput:
    '''
    Standardized output of a node; can pass in any number of args and/or a UIOutput into 'ui' kwarg.
    '''
    def __init__(self, *args: Any, ui: UIOutput | dict=None, expand: dict=None, block_execution: str=None, **kwargs):
        self.args = args
        self.ui = ui
        self.expand = expand
        self.block_execution = block_execution

    @property
    def result(self):
        return self.args if len(self.args) > 0 else None


class SavedResult:
    def __init__(self, filename: str, subfolder: str, type: FolderType):
        self.filename = filename
        self.subfolder = subfolder
        self.type = type
    
    def as_dict(self):
        return {
            "filename": self.filename,
            "subfolder": self.subfolder,
            "type": self.type.value
        }

class UIOutput(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def as_dict(self) -> dict:
        ... # TODO: finish

class UIImages(UIOutput):
    def __init__(self, values: list[SavedResult | dict], animated=False, **kwargs):
        self.values = values
        self.animated = animated
    
    def as_dict(self):
        values = [x.as_dict() if isinstance(x, SavedResult) else x for x in self.values]
        return {
            "images": values,
            "animated": (self.animated,)
        }

class UILatents(UIOutput):
    def __init__(self, values: list[SavedResult | dict], **kwargs):
        self.values = values
    
    def as_dict(self):
        values = [x.as_dict() if isinstance(x, SavedResult) else x for x in self.values]
        return {
            "latents": values,
        }

class UIAudio(UIOutput):
    def __init__(self, values: list[SavedResult | dict], **kwargs):
        self.values = values
    
    def as_dict(self):
        values = [x.as_dict() if isinstance(x, SavedResult) else x for x in self.values]
        return {
            "audio": values,
        }

class UI3D(UIOutput):
    def __init__(self, values: list[SavedResult | dict], **kwargs):
        self.values = values
    
    def as_dict(self):
        values = [x.as_dict() if isinstance(x, SavedResult) else x for x in self.values]
        return {
            "3d": values,
        }

class UIText(UIOutput):
    def __init__(self, value: str, **kwargs):
        self.value = value
    
    def as_dict(self):
        return {"text": (self.value,)}


class TestNode(ComfyNodeV3):
    @classmethod
    def DEFINE_SCHEMA(cls):
        return SchemaV3(
        node_id="TestNode_v3",
        display_name="Test Node (V3)",
        category="v3_test",
        inputs=[IntegerInput("my_int"),
                #AutoGrowDynamicInput("growing", ImageInput),
                MaskInput("thing"),
                ],
        outputs=[ImageOutput("image_output")],
        hidden=[Hidden.api_key_comfy_org, Hidden.auth_token_comfy_org, Hidden.unique_id]
    )

    @classmethod
    def execute(cls, **kwargs):
        pass


if __name__ == "__main__":
    print("hello there")
    inputs: list[InputV3] = [
        IntegerInput("tessfes", widgetType=IO.STRING),
        IntegerInput("my_int"),
        CustomInput("xyz", "XYZ"),
        CustomInput("model1", "MODEL_M"),
        ImageInput("my_image"),
        FloatInput("my_float"),
        MultitypedInput("my_inputs", [StringInput, CustomType("MODEL_M"), CustomType("XYZ")]),
    ]

    outputs: list[OutputV3] = [
        ImageOutput("image"),
        CustomOutput("xyz", "XYZ")
    ]

    for c in inputs:
        if isinstance(c, MultitypedInput):
            print(f"{c}, {type(c)}, {type(c).io_type}, {c.id}, {[x.io_type for x in c.io_types]}")
            print(c.get_io_type_V1())
        else:
            print(f"{c}, {type(c)}, {type(c).io_type}, {c.id}")

    for c in outputs:
        print(f"{c}, {type(c)}, {type(c).io_type}, {c.id}")

    zz = TestNode()
    print(zz.GET_NODE_INFO_V1())

    # aa = NodeInfoV1()
    # print(asdict(aa))
    # print(as_pruned_dict(aa))
