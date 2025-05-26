from __future__ import annotations
from typing import Union
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


class IO_V3:
    def __init__(self):
        pass

    def __init_subclass__(cls, io_type, **kwargs):
        cls.io_type = io_type
        super().__init_subclass__(**kwargs)

class InputV3(IO_V3, io_type=None):
    def __init__(self, id: str, display_name: str=None, behavior=InputBehavior.required, tooltip: str=None):
        super().__init__()
        self.id = id
        self.display_name = display_name
        self.behavior = behavior
        self.tooltip = tooltip

def CustomType(io_type: str) -> type[IO_V3]:
    name = f"{io_type}_IO_V3"
    return type(name, (IO_V3,), {}, io_type=io_type)

def CustomInput(io_type: str, id: str, display_name: str=None, behavior=InputBehavior.required, tooltip: str=None) -> InputV3:
    '''
    Defines input for 'io_type'. Can be used to stand in for non-core types.
    '''
    input_kwargs = {
        "id": id,
        "display_name": display_name,
        "behavior": behavior,
        "tooltip": tooltip,
    }
    return type(f"{io_type}Input", (InputV3,), {}, io_type=io_type)(**input_kwargs)

def CustomOutput(io_type: str, id: str, display_name: str=None, tooltip: str=None) -> OutputV3:
    '''
    Defines output for 'io_type'. Can be used to stand in for non-core types.
    '''
    input_kwargs = {
        "id": id,
        "display_name": display_name,
        "tooltip": tooltip,
    }
    return type(f"{io_type}Output", (OutputV3,), {}, io_type=io_type)(**input_kwargs)


class IntegerInput(InputV3, io_type="INT"):
    '''
    Integer input.
    '''
    def __init__(self, id: str, display_name: str=None, behavior=InputBehavior.required, tooltip: str=None,
                 min: int=None, max: int=None, step: int=None):
        super().__init__(id, display_name, behavior, tooltip)
        self.min = min
        self.max = max
        self.step = step
        self.tooltip = tooltip

class FloatInput(IntegerInput, io_type="FLOAT"):
    '''
    Float input.
    '''
    pass

class StringInput(InputV3, io_type="STRING"):
    '''
    String input.
    '''
    def __init__(self, id: str, display_name: str=None, behavior=InputBehavior.required, tooltip: str=None,
                 multiline=False):
        super().__init__(id, display_name, behavior, tooltip)
        self.multiline = multiline

class ComboInput(InputV3, io_type="COMBO"):
    '''Combo input (dropdown).'''
    def __init__(self, id: str, combo_list: list[str], display_name: str=None, behavior=InputBehavior.required, tooltip: str=None):
        super().__init__(id, display_name, behavior, tooltip)
        self.combo_list = combo_list

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
    def __init__(self, id: str, io_types: list[Union[type[IO_V3], InputV3]], display_name: str=None, behavior=InputBehavior.required, tooltip: str=None,):
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
    def __init__(self, id: str, display_name: str, tooltip: str=None):
        self.id = id
        self.display_name = display_name
        self.tooltip = tooltip
    
    def __init_subclass__(cls, io_type, **kwargs):
        cls.io_type = io_type
        super().__init_subclass__(**kwargs)

class IntegerOutput(OutputV3, io_type="INT"):
    def __init__(self, id: str, display_name: str=None, tooltip: str=None):
        super().__init__(id, display_name, tooltip)

class FloatOutput(OutputV3, io_type="FLOAT"):
    pass

class StringOutput(OutputV3, io_type="STRING"):
    def __init__(self, id: str, display_name: str=None, tooltip: str=None):
        super().__init__(id, display_name, tooltip)



class ImageOutput(OutputV3, io_type="IMAGE"):
    def __init__(self, id: str, display_name: str=None, tooltip: str=None):
        super().__init__(id, display_name, tooltip)

class MaskOutput(OutputV3, io_type="MASK"):
    def __init__(self, id: str, display_name: str=None, tooltip: str=None):
        super().__init__(id, display_name, tooltip)

class LatentOutput(OutputV3, io_type="LATENT"):
    def __init__(self, id: str, display_name: str=None, tooltip: str=None):
        super().__init__(id, display_name, tooltip)


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
    
AutoGrowDynamicInput(id="dynamic", template_input=ImageInput(id="image"))


class Hidden:
    def __init__(self, hidden_var: str):
        self.hidden_var = hidden_var



class ComboDynamicInput(DynamicInput, io_type="COMFY_COMBODYNAMIC_V3"):
    def __init__(self, id: str):
        pass


# class HiddenParam:
#     def __init__(self):
#         pass

#     def __init_subclass__(cls, hidden_var, **kwargs):
#         cls.hidden_var = hidden_var
#         super().__init_subclass__(**kwargs)

# def Hidden(hidden_var: str) -> type[HiddenParam]:
#     return type(f"{hidden_var}_HIDDEN", (HiddenParam,), {}, hidden_var=hidden_var)


Hidden("AUTH_KEY")


class SchemaV3:
    def __init__(self,
            inputs: list[InputV3],
            category: str,
            outputs: list[OutputV3]=None,
            hidden: list[Hidden]=None, 
            is_output_node: bool=False,
            is_deprecated: bool=False,
            is_experimental: bool=False,
    ):
        self.inputs = inputs
        self.category = category
        self.outputs = outputs
        self.hidden = hidden
        self.is_output_node = is_output_node
        self.is_deprecated = is_deprecated
        self.is_experimental = is_experimental


class ComfyNodeV3(ABC):
    SCHEMA = None

    def __init__(self, schema=SCHEMA):
        if schema is None: 
            raise Exception("No schema was defined for this node!")

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
    

class TestNode(ComfyNodeV3):
    pass


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


class ReturnedInputs:
    def __init__(self):
        pass

class ReturnedOutputs:
    def __init__(self):
        pass


class AnimateDiffModelInput(InputV3, io_type="MODEL_M"):
    def __init__(self):
        pass

    def execute(inputs, outputs, hidden):
        pass

if __name__ == "__main__":
    print("hello there")
    inputs: list[InputV3] = [
        IntegerInput("my_int"),
        CustomInput("XYZ", id="xyz"),
        CustomInput("MODEL_M", id="model1"),
        ImageInput("my_image"),
        FloatInput("my_float"),
        MultitypedInput("my_inputs", [CustomType("MODEL_M")]),
    ]

    outputs: list[OutputV3] = [
        ImageOutput("image"),
        CustomOutput("XYZ", "xyz")
    ]

    for c in inputs:
        if isinstance(c, MultitypedInput):
            print(f"{c}, {type(c)}, {type(c).io_type}, {c.id}, {[x.io_type for x in c.io_types]}")
        else:
            print(f"{c}, {type(c)}, {type(c).io_type}, {c.id}")

    for c in outputs:
        print(f"{c}, {type(c)}, {type(c).io_type}, {c.id}")

    zzz = TestNode()