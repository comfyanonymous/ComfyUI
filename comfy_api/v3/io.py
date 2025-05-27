from __future__ import annotations
from typing import Union, Any
from enum import Enum
from abc import ABC, abstractmethod

class InputBehavior(str, Enum):
    required = "required"
    optional = "optional"
# TODO: handle hidden inputs


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
    def __init__(self):
        pass

    def __init_subclass__(cls, io_type, **kwargs):
        cls.io_type = io_type
        super().__init_subclass__(**kwargs)

class InputV3(IO_V3, io_type=None):
    '''
    Base class for a V3 Input.
    '''
    def __init__(self, id: str, display_name: str=None, behavior=InputBehavior.required, tooltip: str=None, lazy: bool=None):
        super().__init__()
        self.id = id
        self.display_name = display_name
        self.behavior = behavior
        self.tooltip = tooltip
        self.lazy = lazy

class WidgetInputV3(InputV3, io_type=None):
    '''
    Base class for a V3 Input with widget.
    '''
    def __init__(self, id: str, display_name: str=None, behavior=InputBehavior.required, tooltip: str=None, lazy: bool=None,
                 default: Any=None,
                 socketless: bool=None, widgetType: str=None):
        super().__init__(id, display_name, behavior, tooltip, lazy)
        self.default = default
        self.socketless = socketless
        self.widgetType = widgetType

def CustomType(io_type: str) -> type[IO_V3]:
    name = f"{io_type}_IO_V3"
    return type(name, (IO_V3,), {}, io_type=io_type)

def CustomInput(id: str, io_type: str, display_name: str=None, behavior=InputBehavior.required, tooltip: str=None, lazy: bool=None) -> InputV3:
    '''
    Defines input for 'io_type'. Can be used to stand in for non-core types.
    '''
    input_kwargs = {
        "id": id,
        "display_name": display_name,
        "behavior": behavior,
        "tooltip": tooltip,
        "lazy": lazy,
    }
    return type(f"{io_type}Input", (InputV3,), {}, io_type=io_type)(**input_kwargs)

def CustomOutput(id: str, io_type: str, display_name: str=None, tooltip: str=None) -> OutputV3:
    '''
    Defines output for 'io_type'. Can be used to stand in for non-core types.
    '''
    input_kwargs = {
        "id": id,
        "display_name": display_name,
        "tooltip": tooltip,
    }
    return type(f"{io_type}Output", (OutputV3,), {}, io_type=io_type)(**input_kwargs)


class BooleanInput(WidgetInputV3, io_type="BOOLEAN"):
    '''
    Boolean input.
    '''
    def __init__(self, id: str, display_name: str=None, behavior=InputBehavior.required, tooltip: str=None, lazy: bool=None,
                 default: bool=None, label_on: str=None, label_off: str=None,
                 socketless: bool=None, widgetType: str=None):
        super().__init__(id, display_name, behavior, tooltip, lazy, default, socketless, widgetType)
        self.label_on = label_on
        self.label_off = label_off
        self.default: bool

class IntegerInput(WidgetInputV3, io_type="INT"):
    '''
    Integer input.
    '''
    def __init__(self, id: str, display_name: str=None, behavior=InputBehavior.required, tooltip: str=None, lazy: bool=None,
                 default: int=None, min: int=None, max: int=None, step: int=None, control_after_generate: bool=None,
                 display_mode: NumberDisplay=None, socketless: bool=None, widgetType: str=None):
        super().__init__(id, display_name, behavior, tooltip, lazy, default, socketless, widgetType)
        self.min = min
        self.max = max
        self.step = step
        self.control_after_generate = control_after_generate
        self.display_mode = display_mode
        self.default: int

class FloatInput(WidgetInputV3, io_type="FLOAT"):
    '''
    Float input.
    '''
    def __init__(self, id: str, display_name: str=None, behavior=InputBehavior.required, tooltip: str=None, lazy: bool=None,
                 default: float=None, min: float=None, max: float=None, step: float=None, round: float=None,
                 display_mode: NumberDisplay=None, socketless: bool=None, widgetType: str=None):
        super().__init__(id, display_name, behavior, tooltip, lazy, default, socketless, widgetType)
        self.default = default
        self.min = min
        self.max = max
        self.step = step
        self.round = round
        self.display_mode = display_mode
        self.default: float

class StringInput(WidgetInputV3, io_type="STRING"):
    '''
    String input.
    '''
    def __init__(self, id: str, display_name: str=None, behavior=InputBehavior.required, tooltip: str=None, lazy: bool=None,
                 multiline=False, placeholder: str=None, default: int=None,
                 socketless: bool=None, widgetType: str=None):
        super().__init__(id, display_name, behavior, tooltip, lazy, default, socketless, widgetType)
        self.multiline = multiline
        self.placeholder = placeholder
        self.default: str

class ComboInput(WidgetInputV3, io_type="COMBO"):
    '''Combo input (dropdown).'''
    def __init__(self, id: str, options: list[str], display_name: str=None, behavior=InputBehavior.required, tooltip: str=None, lazy: bool=None,
                 default: str=None, control_after_generate: bool=None,
                 socketless: bool=None, widgetType: str=None):
        super().__init__(id, display_name, behavior, tooltip, lazy, default, socketless, widgetType)
        self.multiselect = False
        self.options = options
        self.control_after_generate = control_after_generate
        self.default: str

class MultiselectComboWidget(ComboInput, io_type="COMBO"):
    '''Multiselect Combo input (dropdown for selecting potentially more than one value).'''
    def __init__(self, id: str, options: list[str], display_name: str=None, behavior=InputBehavior.required, tooltip: str=None, lazy: bool=None,
                 default: list[str]=None, placeholder: str=None, chip: bool=None, control_after_generate: bool=None,
                 socketless: bool=None, widgetType: str=None):
        super().__init__(id, options, display_name, behavior, tooltip, lazy, default, control_after_generate, socketless, widgetType)
        self.multiselect = True
        self.placeholder = placeholder
        self.chip = chip
        self.default: list[str]

class ImageInput(InputV3, io_type="IMAGE"):
    '''
    Image input.
    '''
    def __init__(self, id: str, display_name: str=None, behavior=InputBehavior.required, tooltip: str=None):
        super().__init__(id, display_name, behavior, tooltip)

class MaskInput(InputV3, io_type="MASK"):
    '''
    Mask input.
    '''
    def __init__(self, id: str, display_name: str=None, behavior=InputBehavior.required, tooltip: str=None):
        super().__init__(id, display_name, behavior, tooltip)

class LatentInput(InputV3, io_type="LATENT"):
    '''
    Latent input.
    '''
    def __init__(self, id: str, display_name: str=None, behavior=InputBehavior.required, tooltip: str=None):
        super().__init__(id, display_name, behavior, tooltip)

class MultitypedInput(InputV3, io_type="COMFY_MULTITYPED_V3"):
    '''
    Input that permits more than one input type.
    '''
    def __init__(self, id: str, io_types: list[Union[type[IO_V3], InputV3, str]], display_name: str=None, behavior=InputBehavior.required, tooltip: str=None,):
        super().__init__(id, display_name, behavior, tooltip)
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


class OutputV3:
    def __init__(self, id: str, display_name: str=None, tooltip: str=None,
                 is_output_list=False):
        self.id = id
        self.display_name = display_name
        self.tooltip = tooltip
        self.is_output_list = is_output_list
    
    def __init_subclass__(cls, io_type, **kwargs):
        cls.io_type = io_type
        super().__init_subclass__(**kwargs)

class IntegerOutput(OutputV3, io_type="INT"):
    pass

class FloatOutput(OutputV3, io_type="FLOAT"):
    pass

class StringOutput(OutputV3, io_type="STRING"):
    pass
    # def __init__(self, id: str, display_name: str=None, tooltip: str=None):
    #     super().__init__(id, display_name, tooltip)

class ImageOutput(OutputV3, io_type="IMAGE"):
    pass

class MaskOutput(OutputV3, io_type="MASK"):
    pass

class LatentOutput(OutputV3, io_type="LATENT"):
    pass


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

    # '''
    # Request hidden value based on hidden_var key.
    # '''
    # def __init__(self, hidden_var: str):
    #     self.hidden_var = hidden_var

# NOTE: does this exist?
# class HiddenNodeId(Hidden):
#     """UNIQUE_ID is the unique identifier of the node, and matches the id property of the node on the client side. It is commonly used in client-server communications (see messages)."""
#     def __init__(self):
#         super().__init__("NODE_ID")

# class HiddenUniqueId(Hidden):
#     """UNIQUE_ID is the unique identifier of the node, and matches the id property of the node on the client side. It is commonly used in client-server communications (see messages)."""
#     def __init__(self):
#         super().__init__("UNIQUE_ID")

# class HiddenPrompt(Hidden):
#     """PROMPT is the complete prompt sent by the client to the server. See the prompt object for a full description."""
#     def __init__(self):
#         super().__init__("PROMPT")

# class HiddenExtraPngInfo(Hidden):
#     """EXTRA_PNGINFO is a dictionary that will be copied into the metadata of any .png files saved. Custom nodes can store additional information in this dictionary for saving (or as a way to communicate with a downstream node)."""
#     def __init__(self):
#         super().__init__("EXTRA_PNGINFO")

# class HiddenDynPrompt(Hidden):
#     """DYNPROMPT is an instance of comfy_execution.graph.DynamicPrompt. It differs from PROMPT in that it may mutate during the course of execution in response to Node Expansion."""
#     def __init__(self):
#         super().__init__("DYNPROMPT")

# class HiddenAuthTokenComfyOrg(Hidden):
#     """Token acquired from signing into a ComfyOrg account on frontend."""
#     def __init__(self):
#         super().__init__("AUTH_TOKEN_COMFY_ORG")

# class HiddenApiKeyComfyOrg(Hidden):
#     """API Key generated by ComfyOrg that allows skipping signing into a ComfyOrg account on frontend."""
#     def __init__(self):
#         super().__init__("API_KEY_COMFY_ORG")


# class HiddenParam:
#     def __init__(self):
#         pass

#     def __init_subclass__(cls, hidden_var, **kwargs):
#         cls.hidden_var = hidden_var
#         super().__init_subclass__(**kwargs)

# def Hidden(hidden_var: str) -> type[HiddenParam]:
#     return type(f"{hidden_var}_HIDDEN", (HiddenParam,), {}, hidden_var=hidden_var)


class SchemaV3:
    def __init__(self,
            category: str,
            inputs: list[InputV3],
            outputs: list[OutputV3]=None,
            hidden: list[Hidden]=None,
            description: str="",
            is_input_list: bool = False,
            is_output_node: bool=False,
            is_deprecated: bool=False,
            is_experimental: bool=False,
            is_api_node: bool=False,
    ):
        self.category = category
        """The category of the node, as per the "Add Node" menu."""
        self.inputs = inputs
        self.outputs = outputs
        self.hidden = hidden
        self.description = description
        """Node description, shown as a tooltip when hovering over the node."""
        self.is_input_list = is_input_list
        """A flag indicating if this node implements the additional code necessary to deal with OUTPUT_IS_LIST nodes.

    All inputs of ``type`` will become ``list[type]``, regardless of how many items are passed in.  This also affects ``check_lazy_status``.

    From the docs:

    A node can also override the default input behaviour and receive the whole list in a single call. This is done by setting a class attribute `INPUT_IS_LIST` to ``True``.

    Comfy Docs: https://docs.comfy.org/custom-nodes/backend/lists#list-processing
    """
        self.is_output_node = is_output_node
        """Flags this node as an output node, causing any inputs it requires to be executed.

    If a node is not connected to any output nodes, that node will not be executed.  Usage::

        OUTPUT_NODE = True

    From the docs:

    By default, a node is not considered an output. Set ``OUTPUT_NODE = True`` to specify that it is.

    Comfy Docs: https://docs.comfy.org/custom-nodes/backend/server_overview#output-node
    """
        self.is_deprecated = is_deprecated
        """Flags a node as deprecated, indicating to users that they should find alternatives to this node."""
        self.is_experimental = is_experimental
        """Flags a node as experimental, informing users that it may change or not work as expected."""
        self.is_api_node = is_api_node
        """Flags a node as an API node. See: https://docs.comfy.org/tutorials/api-nodes/overview."""


class ComfyNodeV3(ABC):

    @classmethod
    def GET_SCHEMA(cls) -> SchemaV3:
        """
        Override this function with one that returns a SchemaV3 instance.
        """
        return None
    GET_SCHEMA = None

    def __init__(self):
        if self.GET_SCHEMA is None:
            raise Exception("No GET_SCHEMA function was defined for this node.")

    @abstractmethod
    def execute(self, inputs, outputs, hidden, **kwargs):
        pass

    # @classmethod
    # @abstractmethod
    # def INPUTS(cls) -> list[InputV3]:
    #     pass

    # @classmethod
    # @abstractmethod
    # def OUTPUTS(cls) -> list[OutputV3]:
    #     pass

    # @abstractmethod
    # def execute(self, inputs, outputs, hidden):
    #     pass


# class ComfyNodeV3:
#     INPUTS = [
#         ImageInput("image"),
#         IntegerInput("count", min=1, max=6),
#     ]

#     OUTPUTS = [

#     ]


    # OUTPUTS = [
    #     ImageOutput(),
    # ]


# class CustomInput(InputV3):
#     def __init__(self, id: str, io_type: str):
#         super().__init__(id)
#         IO_TYPE = IO_TYPE


# class AnimateDiffModelInput(InputV3, io_type="MODEL_M"):
#     def __init__(self):
#         pass

#     def execute(inputs, outputs, hidden):
#         pass


class ReturnedInputs:
    def __init__(self):
        pass

class ReturnedOutputs:
    def __init__(self):
        pass


class NodeOutputV3:
    def __init__(self):
        pass

class UINodeOutput:
    def __init__(self):
        pass


class TestNode(ComfyNodeV3):
    SCHEMA = SchemaV3(
        category="v3_test",
        inputs=[],
    )

    # @classmethod
    # def GET_SCHEMA(cls):
    #     return cls.SCHEMA

    @classmethod
    def GET_SCHEMA(cls):
        return cls.SCHEMA

    def execute(**kwargs):
        pass

if __name__ == "__main__":
    print("hello there")
    inputs: list[InputV3] = [
        IntegerInput("my_int"),
        CustomInput("xyz", "XYZ"),
        CustomInput("model1", "MODEL_M"),
        ImageInput("my_image"),
        FloatInput("my_float"),
        MultitypedInput("my_inputs", [CustomType("MODEL_M"), CustomType("XYZ")]),
    ]

    outputs: list[OutputV3] = [
        ImageOutput("image"),
        CustomOutput("xyz", "XYZ")
    ]

    for c in inputs:
        if isinstance(c, MultitypedInput):
            print(f"{c}, {type(c)}, {type(c).io_type}, {c.id}, {[x.io_type for x in c.io_types]}")
        else:
            print(f"{c}, {type(c)}, {type(c).io_type}, {c.id}")

    for c in outputs:
        print(f"{c}, {type(c)}, {type(c).io_type}, {c.id}")

    zzz = TestNode()
