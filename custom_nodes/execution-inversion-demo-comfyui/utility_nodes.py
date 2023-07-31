from comfy.graph_utils import GraphBuilder
import torch

class AccumulateNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "to_add": ("*",),
            },
            "optional": {
                "accumulation": ("ACCUMULATION",),
            },
        }

    RETURN_TYPES = ("ACCUMULATION",)
    FUNCTION = "accumulate"

    CATEGORY = "InversionDemo Nodes/Lists"

    def accumulate(self, to_add, accumulation = None):
        if accumulation is None:
            value = [to_add]
        else:
            value = accumulation["accum"] + [to_add]
        return ({"accum": value},)

class AccumulationHeadNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "accumulation": ("ACCUMULATION",),
            },
        }

    RETURN_TYPES = ("ACCUMULATION", "*",)
    FUNCTION = "accumulation_head"

    CATEGORY = "InversionDemo Nodes/Lists"

    def accumulation_head(self, accumulation):
        accum = accumulation["accum"]
        if len(accum) == 0:
            return (accumulation, None)
        else:
            return ({"accum": accum[1:]}, accum[0])

class AccumulationTailNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "accumulation": ("ACCUMULATION",),
            },
        }

    RETURN_TYPES = ("ACCUMULATION", "*",)
    FUNCTION = "accumulation_tail"

    CATEGORY = "InversionDemo Nodes/Lists"

    def accumulation_tail(self, accumulation):
        accum = accumulation["accum"]
        if len(accum) == 0:
            return (None, accumulation)
        else:
            return ({"accum": accum[:-1]}, accum[-1])

class AccumulationToListNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "accumulation": ("ACCUMULATION",),
            },
        }

    RETURN_TYPES = ("*",)
    OUTPUT_IS_LIST = (True,)

    FUNCTION = "accumulation_to_list"

    CATEGORY = "InversionDemo Nodes/Lists"

    def accumulation_to_list(self, accumulation):
        return (accumulation["accum"],)

class ListToAccumulationNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list": ("*",),
            },
        }

    RETURN_TYPES = ("ACCUMULATION",)
    INPUT_IS_LIST = (True,)

    FUNCTION = "list_to_accumulation"

    CATEGORY = "InversionDemo Nodes/Lists"

    def list_to_accumulation(self, list):
        return ({"accum": list},)

class AccumulationGetLengthNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "accumulation": ("ACCUMULATION",),
            },
        }

    RETURN_TYPES = ("INT",)

    FUNCTION = "accumlength"

    CATEGORY = "InversionDemo Nodes/Lists"

    def accumlength(self, accumulation):
        return (len(accumulation['accum']),)
        
class AccumulationGetItemNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "accumulation": ("ACCUMULATION",),
                "index": ("INT", {"default":0, "step":1})
            },
        }

    RETURN_TYPES = ("*",)

    FUNCTION = "get_item"

    CATEGORY = "InversionDemo Nodes/Lists"

    def get_item(self, accumulation, index):
        return (accumulation['accum'][index],)
        
class AccumulationSetItemNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "accumulation": ("ACCUMULATION",),
                "index": ("INT", {"default":0, "step":1}),
                "value": ("*",),
            },
        }

    RETURN_TYPES = ("ACCUMULATION",)

    FUNCTION = "set_item"

    CATEGORY = "InversionDemo Nodes/Lists"

    def set_item(self, accumulation, index, value):
        new_accum = accumulation['accum'][:]
        new_accum[index] = value
        return ({"accum": new_accum},)

class IntMathOperation:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("INT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "step": 1}),
                "b": ("INT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "step": 1}),
                "operation": (["add", "subtract", "multiply", "divide", "modulo", "power"],),
            },
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "int_math_operation"

    CATEGORY = "InversionDemo Nodes/Logic"

    def int_math_operation(self, a, b, operation):
        if operation == "add":
            return (a + b,)
        elif operation == "subtract":
            return (a - b,)
        elif operation == "multiply":
            return (a * b,)
        elif operation == "divide":
            return (a // b,)
        elif operation == "modulo":
            return (a % b,)
        elif operation == "power":
            return (a ** b,)


from .flow_control import NUM_FLOW_SOCKETS
class ForLoopOpen:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "remaining": ("INT", {"default": 1, "min": 0, "max": 100000, "step": 1}),
            },
            "optional": {
                "initial_value%d" % i: ("*",) for i in range(1, NUM_FLOW_SOCKETS)
            },
            "hidden": {
                "initial_value0": ("*",)
            }
        }

    RETURN_TYPES = tuple(["FLOW_CONTROL", "INT",] + ["*"] * (NUM_FLOW_SOCKETS-1))
    RETURN_NAMES = tuple(["flow_control", "remaining"] + ["value%d" % i for i in range(1, NUM_FLOW_SOCKETS)])
    FUNCTION = "for_loop_open"

    CATEGORY = "InversionDemo Nodes/Flow"

    def for_loop_open(self, remaining, **kwargs):
        graph = GraphBuilder()
        if "initial_value0" in kwargs:
            remaining = kwargs["initial_value0"]
        while_open = graph.node("WhileLoopOpen", condition=remaining, initial_value0=remaining, **{("initial_value%d" % i): kwargs.get("initial_value%d" % i, None) for i in range(1, NUM_FLOW_SOCKETS)})
        outputs = [kwargs.get("initial_value%d" % i, None) for i in range(1, NUM_FLOW_SOCKETS)]
        return {
            "result": tuple(["stub", remaining] + outputs),
            "expand": graph.finalize(),
        }

class ForLoopClose:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow_control": ("FLOW_CONTROL", {"rawLink": True}),
                "old_remaining": ("INT", {"default": 1, "min": 0, "max": 100000, "step": 1, "forceInput": True}),
            },
            "optional": {
                "initial_value%d" % i: ("*",{"rawLink": True}) for i in range(1, NUM_FLOW_SOCKETS)
            },
        }

    RETURN_TYPES = tuple(["*"] * (NUM_FLOW_SOCKETS-1))
    RETURN_NAMES = tuple(["value%d" % i for i in range(1, NUM_FLOW_SOCKETS)])
    FUNCTION = "for_loop_close"

    CATEGORY = "InversionDemo Nodes/Flow"

    def for_loop_close(self, flow_control, old_remaining, **kwargs):
        graph = GraphBuilder()
        while_open = flow_control[0]
        # TODO - Requires WAS-ns. Will definitely want to solve before merging
        sub = graph.node("IntMathOperation", operation="subtract", a=[while_open,1], b=1)
        cond = graph.node("ToBoolNode", value=sub.out(0))
        input_values = {("initial_value%d" % i): kwargs.get("initial_value%d" % i, None) for i in range(1, NUM_FLOW_SOCKETS)}
        while_close = graph.node("WhileLoopClose",
                flow_control=flow_control,
                condition=cond.out(0),
                initial_value0=sub.out(0),
                **input_values)
        return {
            "result": tuple([while_close.out(i) for i in range(1, NUM_FLOW_SOCKETS)]),
            "expand": graph.finalize(),
        }

class DebugPrint:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("*",),
                "label": ("STRING", {"multiline": False}),
            },
        }

    RETURN_TYPES = ("*",)
    FUNCTION = "debug_print"

    CATEGORY = "InversionDemo Nodes/Debug"

    def debugtype(self, value):
        if isinstance(value, list):
            result = "["
            for i, v in enumerate(value):
                result += (self.debugtype(v) + ",")
            result += "]"
        elif isinstance(value, tuple):
            result = "("
            for i, v in enumerate(value):
                result += (self.debugtype(v) + ",")
            result += ")"
        elif isinstance(value, dict):
            result = "{"
            for k, v in value.items():
                result += ("%s: %s," % (self.debugtype(k), self.debugtype(v)))
            result += "}"
        elif isinstance(value, str):
            result = "'%s'" % value
        elif isinstance(value, bool) or isinstance(value, int) or isinstance(value, float):
            result = str(value)
        elif isinstance(value, torch.Tensor):
            result = "Tensor[%s]" % str(value.shape)
        else:
            result = type(value).__name__
        return result

    def debug_print(self, value, label):
        print("[%s]: %s" % (label, self.debugtype(value)))
        return (value,)

NUM_LIST_SOCKETS = 10
class MakeListNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value1": ("*",),
            },
            "optional": {
                "value%d" % i: ("*",) for i in range(1, NUM_LIST_SOCKETS)
            },
        }

    RETURN_TYPES = ("*",)
    FUNCTION = "make_list"
    OUTPUT_IS_LIST = (True,)

    CATEGORY = "InversionDemo Nodes/Lists"

    def make_list(self, **kwargs):
        result = []
        for i in range(NUM_LIST_SOCKETS):
            if "value%d" % i in kwargs:
                result.append(kwargs["value%d" % i])
        return (result,)

UTILITY_NODE_CLASS_MAPPINGS = {
    "AccumulateNode": AccumulateNode,
    "AccumulationHeadNode": AccumulationHeadNode,
    "AccumulationTailNode": AccumulationTailNode,
    "AccumulationToListNode": AccumulationToListNode,
    "ListToAccumulationNode": ListToAccumulationNode,
    "AccumulationGetLengthNode": AccumulationGetLengthNode,
    "AccumulationGetItemNode": AccumulationGetItemNode,
    "AccumulationSetItemNode": AccumulationSetItemNode,
    "ForLoopOpen": ForLoopOpen,
    "ForLoopClose": ForLoopClose,
    "IntMathOperation": IntMathOperation,
    "DebugPrint": DebugPrint,
    "MakeListNode": MakeListNode,
}
UTILITY_NODE_DISPLAY_NAME_MAPPINGS = {
    "AccumulateNode": "Accumulate",
    "AccumulationHeadNode": "Accumulation Head",
    "AccumulationTailNode": "Accumulation Tail",
    "AccumulationToListNode": "Accumulation to List",
    "ListToAccumulationNode": "List to Accumulation",
    "AccumulationGetLengthNode": "Accumulation Get Length",
    "AccumulationGetItemNode": "Accumulation Get Item",
    "AccumulationSetItemNode": "Accumulation Set Item",
    "ForLoopOpen": "For Loop Open",
    "ForLoopClose": "For Loop Close",
    "IntMathOperation": "Int Math Operation",
    "DebugPrint": "Debug Print",
    "MakeListNode": "Make List",
}
