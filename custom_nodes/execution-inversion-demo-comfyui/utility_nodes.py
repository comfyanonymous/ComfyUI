import torch
from comfy.graph_utils import GraphBuilder

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

    CATEGORY = "InversionDemo Nodes"

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

    CATEGORY = "InversionDemo Nodes"

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

    CATEGORY = "InversionDemo Nodes"

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

    CATEGORY = "InversionDemo Nodes"

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

    RETURN_TYPES = ("*",)
    INPUT_IS_LIST = (True,)

    FUNCTION = "list_to_accumulation"

    CATEGORY = "InversionDemo Nodes"

    def accumulation_to_list(self, list):
        return ({"accum": list},)

class IsTruthyNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("*",),
            },
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "is_truthy"

    CATEGORY = "InversionDemo Nodes"

    def is_truthy(self, value):
        if isinstance(value, torch.Tensor):
            if value.max().item() == 0 and value.min().item() == 0:
                return (0,)
            else:
                return (1,)
        try:
            return (int(bool(value)),)
        except:
            # Can't convert it? Well then it's something or other. I dunno, I'm not a Python programmer.
            return (1,)

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

    CATEGORY = "Flow Control"

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
                "flow_control": ("FLOW_CONTROL", {"raw_link": True}),
                "old_remaining": ("INT", {"default": 1, "min": 0, "max": 100000, "step": 1}),
            },
            "optional": {
                "initial_value%d" % i: ("*",{"raw_link": True}) for i in range(1, NUM_FLOW_SOCKETS)
            },
        }

    RETURN_TYPES = tuple(["*"] * (NUM_FLOW_SOCKETS-1))
    RETURN_NAMES = tuple(["value%d" % i for i in range(1, NUM_FLOW_SOCKETS)])
    FUNCTION = "for_loop_close"

    CATEGORY = "Flow Control"

    def for_loop_close(self, flow_control, old_remaining, **kwargs):
        graph = GraphBuilder()
        while_open = flow_control[0]
        # TODO - Requires WAS-ns. Will definitely want to solve before merging
        sub = graph.node("Number Operation", operation="subtraction", number_a=[while_open,1], number_b=1)
        input_values = {("initial_value%d" % i): kwargs.get("initial_value%d" % i, None) for i in range(1, NUM_FLOW_SOCKETS)}
        while_close = graph.node("WhileLoopClose",
                flow_control=flow_control,
                condition=sub.out(0),
                initial_value0=sub.out(0),
                **input_values)
        return {
            "result": tuple([while_close.out(i) for i in range(1, NUM_FLOW_SOCKETS)]),
            "expand": graph.finalize(),
        }


UTILITY_NODE_CLASS_MAPPINGS = {
    "AccumulateNode": AccumulateNode,
    "AccumulationHeadNode": AccumulationHeadNode,
    "AccumulationTailNode": AccumulationTailNode,
    "AccumulationToListNode": AccumulationToListNode,
    "ListToAccumulationNode": ListToAccumulationNode,
    "IsTruthyNode": IsTruthyNode,
    "ForLoopOpen": ForLoopOpen,
    "ForLoopClose": ForLoopClose,
}
UTILITY_NODE_DISPLAY_NAME_MAPPINGS = {
    "AccumulateNode": "Accumulate",
    "AccumulationHeadNode": "Accumulation Head",
    "AccumulationTailNode": "Accumulation Tail",
    "AccumulationToListNode": "Accumulation to List",
    "ListToAccumulationNode": "List to Accumulation",
    "IsTruthyNode": "Is Truthy",
    "ForLoopOpen": "For Loop Open",
    "ForLoopClose": "For Loop Close",
}
