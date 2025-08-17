from typing import Iterator, List, Tuple, Dict, Any, Union, Optional
from _decimal import Context, getcontext
from decimal import Decimal
from nodes import PreviewImage, SaveImage, NODE_CLASS_MAPPINGS as ALL_NODE_CLASS_MAPPINGS
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from PIL.PngImagePlugin import PngInfo
from ..libs.utils import AlwaysEqualProxy, ByPassTypeTuple, cleanGPUUsedForce, compare_revision
from ..libs.cache import cache, update_cache, remove_cache
from ..libs.log import log_node_info, log_node_warn
import numpy as np
import time
import os
import re
import csv
import json
import torch
import comfy.utils
import folder_paths

DEFAULT_FLOW_NUM = 2
MAX_FLOW_NUM = 20
lazy_options = {"lazy": True} if compare_revision(2543) else {}

any_type = AlwaysEqualProxy("*")

def validate_list_args(args: Dict[str, List[Any]]) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Checks that if there are multiple arguments, they are all the same length or 1
    :param args:
    :return: Tuple (Status, mismatched_key_1, mismatched_key_2)
    """
    # Only have 1 arg
    if len(args) == 1:
        return True, None, None

    len_to_match = None
    matched_arg_name = None
    for arg_name, arg in args.items():
        if arg_name == 'self':
            # self is in locals()
            continue

        if len(arg) != 1:
            if len_to_match is None:
                len_to_match = len(arg)
                matched_arg_name = arg_name
            elif len(arg) != len_to_match:
                return False, arg_name, matched_arg_name

    return True, None, None


def error_if_mismatched_list_args(args: Dict[str, List[Any]]) -> None:
    is_valid, failed_key1, failed_key2 = validate_list_args(args)
    if not is_valid:
        assert failed_key1 is not None
        assert failed_key2 is not None
        raise ValueError(
            f"Mismatched list inputs received. {failed_key1}({len(args[failed_key1])}) !== {failed_key2}({len(args[failed_key2])})"
        )


def zip_with_fill(*lists: Union[List[Any], None]) -> Iterator[Tuple[Any, ...]]:
    """
    Zips lists together, but if a list has 1 element, it will be repeated for each element in the other lists.
    If a list is None, None will be used for that element.
    (Not intended for use with lists of different lengths)
    :param lists:
    :return: Iterator of tuples of length len(lists)
    """
    max_len = max(len(lst) if lst is not None else 0 for lst in lists)
    for i in range(max_len):
        yield tuple(None if lst is None else (lst[0] if len(lst) == 1 else lst[i]) for lst in lists)


# ---------------------------------------------------------------类型 开始----------------------------------------------------------------------#

# 字符串
class String:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"value": ("STRING", {"default": ""})},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "execute"
    CATEGORY = "EasyUse/Logic/Type"

    def execute(self, value):
        return (value,)


# 整数
class Int:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"value": ("INT", {"default": 0, "min": -999999, "max": 999999, })},
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)
    FUNCTION = "execute"
    CATEGORY = "EasyUse/Logic/Type"

    def execute(self, value):
        return (value,)


# 整数范围
class RangeInt:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "range_mode": (["step", "num_steps"], {"default": "step"}),
                "start": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "stop": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "step": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "num_steps": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "end_mode": (["Inclusive", "Exclusive"], {"default": "Inclusive"}),
            },
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("range", "range_sizes")
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "build_range"

    CATEGORY = "EasyUse/Logic/Type"

    def build_range(
            self, range_mode, start, stop, step, num_steps, end_mode
    ) -> Tuple[List[int], List[int]]:
        error_if_mismatched_list_args(locals())

        ranges = []
        range_sizes = []
        for range_mode, e_start, e_stop, e_num_steps, e_step, e_end_mode in zip_with_fill(
                range_mode, start, stop, num_steps, step, end_mode
        ):
            if range_mode == 'step':
                if e_end_mode == "Inclusive":
                    e_stop += 1
                vals = list(range(e_start, e_stop, e_step))
                ranges.extend(vals)
                range_sizes.append(len(vals))
            elif range_mode == 'num_steps':
                direction = 1 if e_stop > e_start else -1
                if e_end_mode == "Exclusive":
                    e_stop -= direction
                vals = (np.rint(np.linspace(e_start, e_stop, e_num_steps)).astype(int).tolist())
                ranges.extend(vals)
                range_sizes.append(len(vals))
        return ranges, range_sizes


# 浮点数
class Float:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"value": ("FLOAT", {"default": 0, "step": 0.01, "min":-0xffffffffffffffff, "max":  0xffffffffffffffff, })},
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float",)
    FUNCTION = "execute"
    CATEGORY = "EasyUse/Logic/Type"

    def execute(self, value):
        return (round(value, 3),)


# 浮点数范围
class RangeFloat:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "range_mode": (["step", "num_steps"], {"default": "step"}),
                "start": ("FLOAT", {"default": 0, "min": -4096, "max": 4096, "step": 0.1}),
                "stop": ("FLOAT", {"default": 0, "min": -4096, "max": 4096, "step": 0.1}),
                "step": ("FLOAT", {"default": 0, "min": -4096, "max": 4096, "step": 0.1}),
                "num_steps": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "end_mode": (["Inclusive", "Exclusive"], {"default": "Inclusive"}),
            },
        }

    RETURN_TYPES = ("FLOAT", "INT")
    RETURN_NAMES = ("range", "range_sizes")
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "build_range"

    CATEGORY = "EasyUse/Logic/Type"

    @staticmethod
    def _decimal_range(
            range_mode: String, start: Decimal, stop: Decimal, step: Decimal, num_steps: Int, inclusive: bool
    ) -> Iterator[float]:
        if range_mode == 'step':
            ret_val = start
            if inclusive:
                stop = stop + step
            direction = 1 if step > 0 else -1
            while (ret_val - stop) * direction < 0:
                yield float(ret_val)
                ret_val += step
        elif range_mode == 'num_steps':
            step = (stop - start) / (num_steps - 1)
            direction = 1 if step > 0 else -1

            ret_val = start
            for _ in range(num_steps):
                if (ret_val - stop) * direction > 0:  # Ensure we don't exceed the 'stop' value
                    break
                yield float(ret_val)
                ret_val += step

    def build_range(
            self,
            range_mode,
            start,
            stop,
            step,
            num_steps,
            end_mode,
    ) -> Tuple[List[float], List[int]]:
        error_if_mismatched_list_args(locals())
        getcontext().prec = 12

        start = [round(Decimal(s),2) for s in start]
        stop = [round(Decimal(s),2) for s in stop]
        step = [round(Decimal(s),2) for s in step]

        ranges = []
        range_sizes = []
        for range_mode, e_start, e_stop, e_step, e_num_steps, e_end_mode in zip_with_fill(
                range_mode, start, stop, step, num_steps, end_mode
        ):
            vals = list(
                self._decimal_range(range_mode, e_start, e_stop, e_step, e_num_steps, e_end_mode == 'Inclusive')
            )
            ranges.extend(vals)
            range_sizes.append(len(vals))

        return ranges, range_sizes


# 布尔
class Boolean:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"value": ("BOOLEAN", {"default": False})},
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "execute"
    CATEGORY = "EasyUse/Logic/Type"

    def execute(self, value):
        return (value,)


# ---------------------------------------------------------------开关 开始----------------------------------------------------------------------#
class imageSwitch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "boolean": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_switch"

    CATEGORY = "EasyUse/Logic/Switch"

    def image_switch(self, image_a, image_b, boolean):

        if boolean:
            return (image_a,)
        else:
            return (image_b,)


class textSwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("INT", {"default": 1, "min": 1, "max": 2}),
            },
            "optional": {
                "text1": ("STRING", {"forceInput": True}),
                "text2": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    CATEGORY = "EasyUse/Logic/Switch"
    FUNCTION = "switch"

    def switch(self, input, text1=None, text2=None, ):
        if input == 1:
            return (text1,)
        else:
            return (text2,)


# ---------------------------------------------------------------Index Switch----------------------------------------------------------------------#

class ab:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "A or B": ("BOOLEAN", {"default": True, "label_on": "A", "label_off": "B"}),
            "in": (any_type,),
        },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = (any_type, any_type,)
    RETURN_NAMES = ("A", "B",)
    FUNCTION = "switch"

    CATEGORY = "EasyUse/Logic"

    def blocker(self, value, block=False):
        from comfy_execution.graph import ExecutionBlocker
        return ExecutionBlocker(None) if block else value

    def switch(self, unique_id, **kwargs):
        is_a = kwargs['A or B']
        a = self.blocker(kwargs['in'], not is_a)
        b = self.blocker(kwargs['in'], is_a)
        return (a, b)


class anythingInversedSwitch:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "index": ("INT", {"default": 0, "min": 0, "max": 9, "step": 1}),
            "in": (any_type,),
        },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ByPassTypeTuple(tuple([any_type]))
    RETURN_NAMES = ByPassTypeTuple(tuple(["out0"]))
    FUNCTION = "switch"

    CATEGORY = "EasyUse/Logic"

    def switch(self, index, unique_id, **kwargs):
        from comfy_execution.graph import ExecutionBlocker
        res = []

        for i in range(0, MAX_FLOW_NUM):
            if index == i:
                res.append(kwargs['in'])
            else:
                res.append(ExecutionBlocker(None))
        return res


class anythingIndexSwitch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "index": ("INT", {"default": 0, "min": 0, "max": 9, "step": 1}),
            },
            "optional": {
            }
        }
        for i in range(MAX_FLOW_NUM):
            inputs["optional"]["value%d" % i] = (any_type, lazy_options)
        return inputs

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("value",)
    FUNCTION = "index_switch"

    CATEGORY = "EasyUse/Logic/Index Switch"

    def check_lazy_status(self, index, **kwargs):
        key = "value%d" % index
        if kwargs.get(key, None) is None:
            return [key]

    def index_switch(self, index, **kwargs):
        key = "value%d" % index
        return (kwargs[key],)


class imageIndexSwitch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "index": ("INT", {"default": 0, "min": 0, "max": 9, "step": 1}),
            },
            "optional": {
            }
        }
        for i in range(MAX_FLOW_NUM):
            inputs["optional"]["image%d" % i] = ("IMAGE", lazy_options)
        return inputs

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "index_switch"

    CATEGORY = "EasyUse/Logic/Index Switch"

    def check_lazy_status(self, index, **kwargs):
        key = "image%d" % index
        if kwargs.get(key, None) is None:
            return [key]

    def index_switch(self, index, **kwargs):
        key = "image%d" % index
        return (kwargs[key],)


class textIndexSwitch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "index": ("INT", {"default": 0, "min": 0, "max": 9, "step": 1}),
            },
            "optional": {
            }
        }
        for i in range(MAX_FLOW_NUM):
            inputs["optional"]["text%d" % i] = ("STRING", {**lazy_options, "forceInput": True})
        return inputs

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "index_switch"

    CATEGORY = "EasyUse/Logic/Index Switch"

    def check_lazy_status(self, index, **kwargs):
        key = "text%d" % index
        if kwargs.get(key, None) is None:
            return [key]

    def index_switch(self, index, **kwargs):
        key = "text%d" % index
        return (kwargs[key],)


class conditioningIndexSwitch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "index": ("INT", {"default": 0, "min": 0, "max": 9, "step": 1}),
            },
            "optional": {
            }
        }
        for i in range(MAX_FLOW_NUM):
            inputs["optional"]["cond%d" % i] = ("CONDITIONING", lazy_options)
        return inputs

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "index_switch"

    CATEGORY = "EasyUse/Logic/Index Switch"

    def check_lazy_status(self, index, **kwargs):
        key = "cond%d" % index
        if kwargs.get(key, None) is None:
            return [key]

    def index_switch(self, index, **kwargs):
        key = "cond%d" % index
        return (kwargs[key],)


# ---------------------------------------------------------------Math----------------------------------------------------------------------#
class mathIntOperation:
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

    CATEGORY = "EasyUse/Logic/Math"

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


class mathFloatOperation:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("FLOAT", {"default": 0, "min": -999999999999.0, "max": 999999999999.0, "step": 0.01}),
                "b": ("FLOAT", {"default": 0, "min": -999999999999.0, "max": 999999999999.0, "step": 0.01}),
                "operation": (["add", "subtract", "multiply", "divide", "modulo", "power"],),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "float_math_operation"

    CATEGORY = "EasyUse/Logic/Math"

    def float_math_operation(self, a, b, operation):
        if operation == "add":
            return (round(a + b,3),)
        elif operation == "subtract":
            return (round(a - b,3),)
        elif operation == "multiply":
            return (round(a * b,3),)
        elif operation == "divide":
            return (round(a / b,3),)
        elif operation == "modulo":
            return (round(a % b,3),)
        elif operation == "power":
            return (round(a ** b,3),)


class mathStringOperation:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("STRING", {"multiline": False}),
                "b": ("STRING", {"multiline": False}),
                "operation": (["a == b", "a != b", "a IN b", "a MATCH REGEX(b)", "a BEGINSWITH b", "a ENDSWITH b"],),
                "case_sensitive": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "string_math_operation"

    CATEGORY = "EasyUse/Logic/Math"

    def string_math_operation(self, a, b, operation, case_sensitive):
        if not case_sensitive:
            a = a.lower()
            b = b.lower()

        if operation == "a == b":
            return (a == b,)
        elif operation == "a != b":
            return (a != b,)
        elif operation == "a IN b":
            return (a in b,)
        elif operation == "a MATCH REGEX(b)":
            try:
                return (re.match(b, a) is not None,)
            except:
                return (False,)
        elif operation == "a BEGINSWITH b":
            return (a.startswith(b),)
        elif operation == "a ENDSWITH b":
            return (a.endswith(b),)


# ---------------------------------------------------------------Flow----------------------------------------------------------------------#
try:
    from comfy_execution.graph_utils import GraphBuilder, is_link
except:
    GraphBuilder = None


class whileLoopStart:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "condition": ("BOOLEAN", {"default": True}),
            },
            "optional": {
            },
        }
        for i in range(MAX_FLOW_NUM):
            inputs["optional"]["initial_value%d" % i] = (any_type,)
        return inputs

    RETURN_TYPES = ByPassTypeTuple(tuple(["FLOW_CONTROL"] + [any_type] * MAX_FLOW_NUM))
    RETURN_NAMES = ByPassTypeTuple(tuple(["flow"] + ["value%d" % i for i in range(MAX_FLOW_NUM)]))
    FUNCTION = "while_loop_open"

    CATEGORY = "EasyUse/Logic/While Loop"

    def while_loop_open(self, condition, **kwargs):
        values = []
        for i in range(MAX_FLOW_NUM):
            values.append(kwargs.get("initial_value%d" % i, None))
        return tuple(["stub"] + values)


class whileLoopEnd:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "flow": ("FLOW_CONTROL", {"rawLink": True}),
                "condition": ("BOOLEAN", {}),
            },
            "optional": {
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }
        for i in range(MAX_FLOW_NUM):
            inputs["optional"]["initial_value%d" % i] = (any_type,)
        return inputs

    RETURN_TYPES = ByPassTypeTuple(tuple([any_type] * MAX_FLOW_NUM))
    RETURN_NAMES = ByPassTypeTuple(tuple(["value%d" % i for i in range(MAX_FLOW_NUM)]))
    FUNCTION = "while_loop_close"

    CATEGORY = "EasyUse/Logic/While Loop"

    def explore_dependencies(self, node_id, dynprompt, upstream, parent_ids):
        node_info = dynprompt.get_node(node_id)
        if "inputs" not in node_info:
            return

        for k, v in node_info["inputs"].items():
            if is_link(v):
                parent_id = v[0]
                display_id = dynprompt.get_display_node_id(parent_id)
                display_node = dynprompt.get_node(display_id)
                class_type = display_node["class_type"]
                if class_type not in ['easy forLoopEnd', 'easy whileLoopEnd']:
                    parent_ids.append(display_id)
                if parent_id not in upstream:
                    upstream[parent_id] = []
                    self.explore_dependencies(parent_id, dynprompt, upstream, parent_ids)

                upstream[parent_id].append(node_id)

    def explore_output_nodes(self, dynprompt, upstream, output_nodes, parent_ids):
        for parent_id in upstream:
            display_id = dynprompt.get_display_node_id(parent_id)
            for output_id in output_nodes:
                id = output_nodes[output_id][0]
                if id in parent_ids and display_id == id and output_id not in upstream[parent_id]:
                    if '.' in parent_id:
                        arr = parent_id.split('.')
                        arr[len(arr)-1] = output_id
                        upstream[parent_id].append('.'.join(arr))
                    else:
                        upstream[parent_id].append(output_id)

    def collect_contained(self, node_id, upstream, contained):
        if node_id not in upstream:
            return
        for child_id in upstream[node_id]:
            if child_id not in contained:
                contained[child_id] = True
                self.collect_contained(child_id, upstream, contained)

    def while_loop_close(self, flow, condition, dynprompt=None, unique_id=None,**kwargs):
        if not condition:
            # We're done with the loop
            values = []
            for i in range(MAX_FLOW_NUM):
                values.append(kwargs.get("initial_value%d" % i, None))
            return tuple(values)

        # We want to loop
        this_node = dynprompt.get_node(unique_id)
        upstream = {}
        # Get the list of all nodes between the open and close nodes
        parent_ids = []
        self.explore_dependencies(unique_id, dynprompt, upstream, parent_ids)
        parent_ids = list(set(parent_ids))
        # Get the list of all output nodes between the open and close nodes
        prompts = dynprompt.get_original_prompt()
        output_nodes = {}
        for id in prompts:
            node = prompts[id]
            if "inputs" not in node:
                continue
            class_type = node["class_type"]
            class_def = ALL_NODE_CLASS_MAPPINGS[class_type]
            if hasattr(class_def, 'OUTPUT_NODE') and class_def.OUTPUT_NODE == True:
                for k, v in node['inputs'].items():
                    if is_link(v):
                        output_nodes[id] = v

        graph = GraphBuilder()
        self.explore_output_nodes(dynprompt, upstream, output_nodes, parent_ids)
        contained = {}
        open_node = flow[0]
        self.collect_contained(open_node, upstream, contained)
        contained[unique_id] = True
        contained[open_node] = True

        for node_id in contained:
            original_node = dynprompt.get_node(node_id)
            node = graph.node(original_node["class_type"], "Recurse" if node_id == unique_id else node_id)
            node.set_override_display_id(node_id)
        for node_id in contained:
            original_node = dynprompt.get_node(node_id)
            node = graph.lookup_node("Recurse" if node_id == unique_id else node_id)
            for k, v in original_node["inputs"].items():
                if is_link(v) and v[0] in contained:
                    parent = graph.lookup_node(v[0])
                    node.set_input(k, parent.out(v[1]))
                else:
                    node.set_input(k, v)

        new_open = graph.lookup_node(open_node)
        for i in range(MAX_FLOW_NUM):
            key = "initial_value%d" % i
            new_open.set_input(key, kwargs.get(key, None))
        my_clone = graph.lookup_node("Recurse")
        result = map(lambda x: my_clone.out(x), range(MAX_FLOW_NUM))
        return {
            "result": tuple(result),
            "expand": graph.finalize(),
        }


class forLoopStart:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "total": ("INT", {"default": 1, "min": 1, "max": 100000, "step": 1}),
            },
            "optional": {
                "initial_value%d" % i: (any_type,) for i in range(1, MAX_FLOW_NUM)
            },
            "hidden": {
                "initial_value0": (any_type,),
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID"
            }
        }

    RETURN_TYPES = ByPassTypeTuple(tuple(["FLOW_CONTROL", "INT"] + [any_type] * (MAX_FLOW_NUM - 1)))
    RETURN_NAMES = ByPassTypeTuple(tuple(["flow", "index"] + ["value%d" % i for i in range(1, MAX_FLOW_NUM)]))
    FUNCTION = "for_loop_start"

    CATEGORY = "EasyUse/Logic/For Loop"

    def for_loop_start(self, total, prompt=None, extra_pnginfo=None, unique_id=None, **kwargs):
        graph = GraphBuilder()
        i = 0
        if "initial_value0" in kwargs:
            i = kwargs["initial_value0"]

        initial_values = {("initial_value%d" % num): kwargs.get("initial_value%d" % num, None) for num in
                          range(1, MAX_FLOW_NUM)}
        while_open = graph.node("easy whileLoopStart", condition=total, initial_value0=i, **initial_values)
        outputs = [kwargs.get("initial_value%d" % num, None) for num in range(1, MAX_FLOW_NUM)]
        return {
            "result": tuple(["stub", i] + outputs),
            "expand": graph.finalize(),
        }


class forLoopEnd:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow": ("FLOW_CONTROL", {"rawLink": True}),
            },
            "optional": {
                "initial_value%d" % i: (any_type, {"rawLink": True}) for i in range(1, MAX_FLOW_NUM)
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID"
            },
        }

    RETURN_TYPES = ByPassTypeTuple(tuple([any_type] * (MAX_FLOW_NUM - 1)))
    RETURN_NAMES = ByPassTypeTuple(tuple(["value%d" % i for i in range(1, MAX_FLOW_NUM)]))
    FUNCTION = "for_loop_end"

    CATEGORY = "EasyUse/Logic/For Loop"



    def for_loop_end(self, flow, dynprompt=None, extra_pnginfo=None, unique_id=None, **kwargs):
        graph = GraphBuilder()
        while_open = flow[0]
        total = None

        # Using dynprompt to get the original node
        forstart_node = dynprompt.get_node(while_open)
        if forstart_node['class_type'] == 'easy forLoopStart':
            inputs = forstart_node['inputs']
            total = inputs['total']
        elif forstart_node['class_type'] == 'easy loadImagesForLoop':
            inputs = forstart_node['inputs']
            limit = inputs['limit']
            start_index = inputs['start_index']
            # Filter files by extension
            directory = inputs['directory']
            total = graph.node('easy imagesCountInDirectory', directory=directory, limit=limit, start_index=start_index, extension='*').out(0)

        sub = graph.node("easy mathInt", operation="add", a=[while_open, 1], b=1)
        cond = graph.node("easy compare", a=sub.out(0), b=total, comparison='a < b')
        input_values = {("initial_value%d" % i): kwargs.get("initial_value%d" % i, None) for i in
                        range(1, MAX_FLOW_NUM)}
        while_close = graph.node("easy whileLoopEnd",
                                 flow=flow,
                                 condition=cond.out(0),
                                 initial_value0=sub.out(0),
                                 **input_values)
        return {
            "result": tuple([while_close.out(i) for i in range(1, MAX_FLOW_NUM)]),
            "expand": graph.finalize(),
        }


COMPARE_FUNCTIONS = {
    "a == b": lambda a, b: a == b,
    "a != b": lambda a, b: a != b,
    "a < b": lambda a, b: a < b,
    "a > b": lambda a, b: a > b,
    "a <= b": lambda a, b: a <= b,
    "a >= b": lambda a, b: a >= b,
}


# 比较
class Compare:
    @classmethod
    def INPUT_TYPES(s):
        compare_functions = list(COMPARE_FUNCTIONS.keys())
        return {
            "required": {
                "a": (any_type, {"default": 0}),
                "b": (any_type, {"default": 0}),
                "comparison": (compare_functions, {"default": "a == b"}),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "compare"
    CATEGORY = "EasyUse/Logic/Math"

    def compare(self, a, b, comparison):
        return (COMPARE_FUNCTIONS[comparison](a, b),)


# 判断
class IfElse:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "boolean": ("BOOLEAN",),
                "on_true": (any_type, lazy_options),
                "on_false": (any_type, lazy_options),
            },
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("*",)
    FUNCTION = "execute"
    CATEGORY = "EasyUse/Logic"

    def check_lazy_status(self, boolean, on_true=None, on_false=None):
        if boolean and on_true is None:
            return ["on_true"]
        if not boolean and on_false is None:
            return ["on_false"]

    def execute(self, *args, **kwargs):
        return (kwargs['on_true'] if kwargs['boolean'] else kwargs['on_false'],)


class Blocker:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "continue": ("BOOLEAN", {"default": False}),
                "in": (any_type, {"default": None}),
            },
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("out",)
    CATEGORY = "EasyUse/Logic"
    FUNCTION = "execute"

    def execute(self, **kwargs):
        from comfy_execution.graph import ExecutionBlocker
        return (kwargs['in'] if kwargs['continue'] else ExecutionBlocker(None),)


# 是否为SDXL
from comfy.sdxl_clip import SDXLClipModel, SDXLRefinerClipModel, SDXLClipG

class isMaskEmpty:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "execute"
    CATEGORY = "EasyUse/Logic"

    def execute(self, mask):
        if mask is None:
            return (True,)
        if torch.all(mask == 0):
            return (True,)
        return (False,)


class isNone:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": (any_type,)
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "execute"
    CATEGORY = "EasyUse/Logic"

    def execute(self, any):
        return (True if any is None else False,)


class isSDXL:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "optional_pipe": ("PIPE_LINE",),
                "optional_clip": ("CLIP",),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "execute"
    CATEGORY = "EasyUse/Logic"

    def execute(self, optional_pipe=None, optional_clip=None):
        if optional_pipe is None and optional_clip is None:
            raise Exception(f"[ERROR] optional_pipe or optional_clip is missing")
        clip = optional_clip if optional_clip is not None else optional_pipe['clip']
        if isinstance(clip.cond_stage_model, (SDXLClipModel, SDXLRefinerClipModel, SDXLClipG)):
            return (True,)
        else:
            return (False,)


class isFileExist:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "file_path": ("STRING", {"default": ""}),
                "file_name": ("STRING", {"default": ""}),
                "file_extension": ("STRING", {"default": ""}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "execute"
    CATEGORY = "EasyUse/Logic"

    def execute(self, file_path, file_name, file_extension):
        if not file_path:
            raise Exception("file_path is missing")

        if file_name:
            file_path = os.path.join(file_path, file_name)
        if file_extension:
            file_path = file_path + "." + file_extension

        if os.path.exists(file_path) and os.path.isfile(file_path):
            return (True,)
        else:
            return (False,)


from nodes import MAX_RESOLUTION
from ..config import BASE_RESOLUTIONS


class pixels:

    @classmethod
    def INPUT_TYPES(s):
        resolution_strings = [
            f"{width} x {height} (custom)" if width == 'width' and height == 'height' else f"{width} x {height}" for
            width, height in BASE_RESOLUTIONS]
        return {
            "required": {
                "resolution": (resolution_strings,),
                "width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "scale": ("FLOAT", {"default": 2.000, "min": 0.001, "max": 10, "step": 0.001}),
                "flip_w/h": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("INT", "INT", any_type, any_type, any_type)
    RETURN_NAMES = ("width_norm", "height_norm", "width", "height", "scale_factor")
    CATEGORY = "EasyUse/Logic"
    FUNCTION = "create"

    def create(self, resolution, width, height, scale, **kwargs):
        if resolution not in ["自定义 x 自定义", 'width x height (custom)']:
            try:
                _width, _height = map(int, resolution.split(' x '))
                width = _width
                height = _height
            except ValueError:
                raise ValueError("Invalid base_resolution format.")

        width = width * scale
        height = height * scale
        width_norm = int(width - width % 8)
        height_norm = int(height - height % 8)
        flip_wh = kwargs['flip_w/h']
        if flip_wh:
            width, height = height, width
            width_norm, height_norm = height_norm, width_norm

        return (width_norm, height_norm, width, height, scale)


# xy矩阵
class xyAny:

    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "X": (any_type, {}),
                "Y": (any_type, {}),
                "direction": (["horizontal", "vertical"], {"default": "horizontal"})
            }
        }

    RETURN_TYPES = (any_type, any_type)
    RETURN_NAMES = ("X", "Y")
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, True)
    CATEGORY = "EasyUse/Logic"
    FUNCTION = "to_xy"

    def to_xy(self, X, Y, direction):
        new_x = list()
        new_y = list()
        if direction[0] == "horizontal":
            for y in Y:
                for x in X:
                    new_x.append(x)
                    new_y.append(y)
        else:
            for x in X:
                for y in Y:
                    new_x.append(x)
                    new_y.append(y)

        return (new_x, new_y)


class lengthAnything:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": (any_type, {}),
            },
            "hidden":{
                "prompt": "PROMPT",
                "my_unique_id": "UNIQUE_ID"
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("length",)

    INPUT_IS_LIST = True

    FUNCTION = "getLength"
    CATEGORY = "EasyUse/Logic"

    def getLength(self, any, prompt=None, my_unique_id=None):
        prompt = prompt[0]
        my_unique_id = my_unique_id[0]
        my_unique_id = my_unique_id.split('.')[len(my_unique_id.split('.')) - 1] if "." in my_unique_id else my_unique_id
        id, slot = prompt[my_unique_id]['inputs']['any']
        class_type = prompt[id]['class_type']
        node_class = ALL_NODE_CLASS_MAPPINGS[class_type]
        output_is_list = node_class.OUTPUT_IS_LIST[slot] if hasattr(node_class, 'OUTPUT_IS_LIST') else False

        return (len(any) if output_is_list or len(any) > 1 else len(any[0]),)

class indexAnything:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": (any_type, {}),
                "index": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
            },
            "hidden":{
                "prompt": "PROMPT",
                "my_unique_id": "UNIQUE_ID"
            }
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("out",)

    INPUT_IS_LIST = True

    FUNCTION = "getIndex"
    CATEGORY = "EasyUse/Logic"

    def getIndex(self, any, index, prompt=None, my_unique_id=None):
        index = index[0]
        prompt = prompt[0]
        my_unique_id = my_unique_id[0]
        my_unique_id = my_unique_id.split('.')[len(my_unique_id.split('.')) - 1] if "." in my_unique_id else my_unique_id
        id, slot = prompt[my_unique_id]['inputs']['any']
        class_type = prompt[id]['class_type']
        node_class = ALL_NODE_CLASS_MAPPINGS[class_type]
        output_is_list = node_class.OUTPUT_IS_LIST[slot] if hasattr(node_class, 'OUTPUT_IS_LIST') else False

        if output_is_list or len(any) > 1:
            return (any[index],)
        elif isinstance(any[0], torch.Tensor):
            batch_index = min(any[0].shape[0] - 1, index)
            s = any[0][index:index + 1].clone()
            return (s,)
        else:
            return (any[0][index],)


class batchAnything:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any_1": (any_type, {}),
                "any_2": (any_type, {})
            }
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("batch",)

    FUNCTION = "batch"
    CATEGORY = "EasyUse/Logic"

    def latentBatch(self, any_1, any_2):
        samples_out = any_1.copy()
        s1 = any_1["samples"]
        s2 = any_2["samples"]

        if s1.shape[1:] != s2.shape[1:]:
            s2 = comfy.utils.common_upscale(s2, s1.shape[3], s1.shape[2], "bilinear", "center")
        s = torch.cat((s1, s2), dim=0)
        samples_out["samples"] = s
        samples_out["batch_index"] = any_1.get("batch_index",
                                               [x for x in range(0, s1.shape[0])]) + any_2.get(
            "batch_index", [x for x in range(0, s2.shape[0])])

        return samples_out

    def batch(self, any_1, any_2):
        if isinstance(any_1, torch.Tensor) or isinstance(any_2, torch.Tensor):
            if any_1 is None:
                return (any_2,)
            elif any_2 is None:
                return (any_1,)
            if any_1.shape[1:] != any_2.shape[1:]:
                any_2 = comfy.utils.common_upscale(any_2.movedim(-1, 1), any_1.shape[2], any_1.shape[1], "bilinear",
                                                   "center").movedim(1, -1)
            return (torch.cat((any_1, any_2), 0),)
        elif isinstance(any_1, (str, float, int)):
            if any_2 is None:
                return (any_1,)
            elif isinstance(any_2, tuple):
                return (any_2 + (any_1,),)
            elif isinstance(any_2, list):
                return (any_2 + [any_1],)
            return ([any_1, any_2],)
        elif isinstance(any_2, (str, float, int)):
            if any_1 is None:
                return (any_2,)
            elif isinstance(any_1, tuple):
                return (any_1 + (any_2,),)
            elif isinstance(any_1, list):
                return (any_1 + [any_2],)
            return ([any_2, any_1],)
        elif isinstance(any_1, dict) and 'samples' in any_1:
            if any_2 is None:
                return (any_1,)
            elif isinstance(any_2, dict) and 'samples' in any_2:
                return (self.latentBatch(any_1, any_2),)
        elif isinstance(any_2, dict) and 'samples' in any_2:
            if any_1 is None:
                return (any_2,)
            elif isinstance(any_1, dict) and 'samples' in any_1:
                return (self.latentBatch(any_2, any_1),)
        else:
            if any_1 is None:
                return (any_2,)
            elif any_2 is None:
                return (any_1,)
            return (any_1 + any_2,)


# 转换所有类型
class convertAnything:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "*": (any_type,),
            "output_type": (["string", "int", "float", "boolean"], {"default": "string"}),
        }}

    RETURN_TYPES = ByPassTypeTuple((any_type,))
    OUTPUT_NODE = True
    FUNCTION = "convert"
    CATEGORY = "EasyUse/Logic"

    def convert(self, *args, **kwargs):
        anything = kwargs['*']
        output_type = kwargs['output_type']
        params = None
        if output_type == 'string':
            params = str(anything)
        elif output_type == 'int':
            params = int(anything)
        elif output_type == 'float':
            params = float(anything)
        elif output_type == 'boolean':
            params = bool(anything)
        return (params,)

# 将所有类型的内容都转成字符串输出
class showAnything:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}, "optional": {"anything": (any_type, {}), },
                "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO",
                           }}

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ('output',)
    INPUT_IS_LIST = True
    OUTPUT_NODE = True
    FUNCTION = "log_input"
    CATEGORY = "EasyUse/Logic"

    def log_input(self, unique_id=None, extra_pnginfo=None, **kwargs):

        values = []
        if "anything" in kwargs:
            for val in kwargs['anything']:
                try:
                    if isinstance(val, str):
                        values.append(val)
                    elif isinstance(val, list):
                        values = val
                    elif isinstance(val, (int, float, bool)):
                        values.append(str(val))
                    else:
                        val = json.dumps(val)
                        values.append(str(val))
                except Exception:
                    values.append(str(val))
                    pass

        if not extra_pnginfo:
            pass
        elif (not isinstance(extra_pnginfo[0], dict) or "workflow" not in extra_pnginfo[0]):
            pass
        else:
            workflow = extra_pnginfo[0]["workflow"]
            node = next((x for x in workflow["nodes"] if str(x["id"]) == unique_id[0]), None)
            if node:
                node["widgets_values"] = [values]
        if isinstance(values, list) and len(values) == 1:
            return {"ui": {"text": values}, "result": (values[0],), }
        else:
            return {"ui": {"text": values}, "result": (values,), }

class showTensorShape:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"tensor": (any_type,)}, "optional": {},
                "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO"
                           }}

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    OUTPUT_NODE = True
    FUNCTION = "log_input"
    CATEGORY = "EasyUse/Logic"

    def log_input(self, tensor, unique_id=None, extra_pnginfo=None):
        shapes = []

        def tensorShape(tensor):
            if isinstance(tensor, dict):
                for k in tensor:
                    tensorShape(tensor[k])
            elif isinstance(tensor, list):
                for i in range(len(tensor)):
                    tensorShape(tensor[i])
            elif hasattr(tensor, 'shape'):
                shapes.append(list(tensor.shape))

        tensorShape(tensor)

        return {"ui": {"text": shapes}}


class outputToList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tuple": (any_type, {}),
            }, "optional": {},
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("list",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "output_to_List"
    CATEGORY = "EasyUse/Logic"

    def output_to_List(self, tuple):
        return (tuple,)


# cleanGpuUsed
class cleanGPUUsed:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"anything": (any_type, {})}, "optional": {},
                "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO",
                           }}

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("output",)
    OUTPUT_NODE = True
    FUNCTION = "empty_cache"
    CATEGORY = "EasyUse/Logic"

    def empty_cache(self, anything, unique_id=None, extra_pnginfo=None):
        cleanGPUUsedForce()
        remove_cache('*')
        return (anything,)


class clearCacheKey:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "anything": (any_type, {}),
            "cache_key": ("STRING", {"default": "*"}),
        }, "optional": {},
            "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO", }
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ('output',)
    OUTPUT_NODE = True
    FUNCTION = "empty_cache"
    CATEGORY = "EasyUse/Logic"

    def empty_cache(self, anything, cache_name, unique_id=None, extra_pnginfo=None):
        remove_cache(cache_name)
        return (anything,)


class clearCacheAll:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "anything": (any_type, {}),
        }, "optional": {},
            "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO", }
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("output",)
    OUTPUT_NODE = True
    FUNCTION = "empty_cache"
    CATEGORY = "EasyUse/Logic"

    def empty_cache(self, anything, unique_id=None, extra_pnginfo=None):
        remove_cache('*')
        return (anything,)


class saveText:

    def __init__(self):
        self.output_dir = folder_paths.output_directory
        self.type = 'output'

    @classmethod
    def INPUT_TYPES(s):
        input_types = {}
        input_types['required'] = {
            "text": ("STRING", {"default": "", "forceInput": True}),
            "output_file_path": ("STRING", {"multiline": False, "default": ""}),
            "file_name": ("STRING", {"multiline": False, "default": ""}),
            "file_extension": (["txt", "csv"],),
            "overwrite": ("BOOLEAN", {"default": True}),
        }
        input_types['optional'] = {
            "image": ("IMAGE",),
        }
        return input_types

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("text", 'image',)

    FUNCTION = "save_text"
    OUTPUT_NODE = True
    CATEGORY = "EasyUse/Logic"

    def save_image(self, images, filename_prefix='', extension='png',quality=100, prompt=None,
                   extra_pnginfo=None, delimiter='_', filename_number_start='true', number_padding=4,
                   overwrite_mode='prefix_as_filename', output_path='', show_history='true', show_previews='true',
                   embed_workflow='true', lossless_webp=False, ):
        results = list()
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            # Delegate metadata/pnginfo
            if extension == 'webp':
                img_exif = img.getexif()
                workflow_metadata = ''
                prompt_str = ''
                if prompt is not None:
                    prompt_str = json.dumps(prompt)
                    img_exif[0x010f] = "Prompt:" + prompt_str
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        workflow_metadata += json.dumps(extra_pnginfo[x])
                img_exif[0x010e] = "Workflow:" + workflow_metadata
                exif_data = img_exif.tobytes()
            else:
                metadata = PngInfo()
                if embed_workflow == 'true':
                    if prompt is not None:
                        metadata.add_text("prompt", json.dumps(prompt))
                    if extra_pnginfo is not None:
                        for x in extra_pnginfo:
                            metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                exif_data = metadata

            file = f"{filename_prefix}.{extension}"

            # Save the images
            try:
                output_file = os.path.abspath(os.path.join(output_path, file))
                if extension in ["jpg", "jpeg"]:
                    img.save(output_file,
                             quality=quality, optimize=True)
                elif extension == 'webp':
                    img.save(output_file,
                             quality=quality, lossless=lossless_webp, exif=exif_data)
                elif extension == 'png':
                    img.save(output_file,
                             pnginfo=exif_data, optimize=True)
                elif extension == 'bmp':
                    img.save(output_file)
                elif extension == 'tiff':
                    img.save(output_file,
                             quality=quality, optimize=True)
                else:
                    img.save(output_file,
                             pnginfo=exif_data, optimize=True)

            except OSError as e:
                print(e)
            except Exception as e:
                print(e)

    def save_text(self, text, output_file_path, file_name, file_extension, overwrite, filename_number_start='true', image=None, prompt=None,
                  extra_pnginfo=None):
        if isinstance(file_name, list):
            file_name = file_name[0]
        filepath = str(os.path.join(output_file_path, file_name)) + "." + file_extension
        index = 1

        if (output_file_path == "" or file_name == ""):
            log_node_warn("Save Text", "No file details found. No file output.")
            return ()

        if not os.path.exists(output_file_path):
            os.makedirs(output_file_path)

        if overwrite:
            file_mode = "w"
        else:
            file_mode = "a"

        log_node_info("Save Text", f"Saving to {filepath}")

        if file_extension == "csv":
            text_list = []
            for i in text.split("\n"):
                text_list.append(i.strip())

            with open(filepath, file_mode, newline="", encoding='utf-8') as csv_file:
                csv_writer = csv.writer(csv_file)
                # Write each line as a separate row in the CSV file
                for line in text_list:
                    csv_writer.writerow([line])
        else:
            with open(filepath, file_mode, newline="", encoding='utf-8') as text_file:
                for line in text:
                    text_file.write(line)

        result = {"result": (text, None)}

        if image is not None:
            imagepath = os.path.join(output_file_path, file_name)
            image_index = 1
            if not overwrite:
                while os.path.exists(filepath):
                    if os.path.exists(filepath):
                        imagepath = str(os.path.join(output_file_path, file_name)) + "_" + str(index)
                        index = index + 1
                    else:
                        break
            # result = self.save_images(image, imagepath, prompt, extra_pnginfo)

            delimiter = '_'
            number_padding = 4
            lossless_webp = (False,)

            original_output = self.output_dir

            # Setup output path
            if output_file_path in [None, '', "none", "."]:
                output_path = self.output_dir
            else:
                output_path = ''
            if not os.path.isabs(output_file_path):
                output_path = os.path.join(self.output_dir, output_path)
            base_output = os.path.basename(output_path)
            if output_path.endswith("ComfyUI/output") or output_path.endswith(r"ComfyUI\output"):
                base_output = ""

            # Check output destination
            if output_path.strip() != '':
                if not os.path.isabs(output_path):
                    output_path = os.path.join(folder_paths.output_directory, output_path)
                if not os.path.exists(output_path.strip()):
                    print(
                        f'The path `{output_path.strip()}` specified doesn\'t exist! Creating directory.')
                    os.makedirs(output_path, exist_ok=True)

            images = []
            images.append(image)
            images = torch.cat(images, dim=0)
            self.save_image(images, imagepath, 'png', 100, prompt, extra_pnginfo, filename_number_start=filename_number_start, output_path=output_path, delimiter=delimiter,
                             number_padding=number_padding, lossless_webp=lossless_webp)

            log_node_info("Save Text", f"Saving Image to {imagepath}")
            result['result'] = (text, image)

        return result


class sleep:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": (any_type, {}),
                "delay": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000000, "step": 0.1}),
            },
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("out",)
    FUNCTION = "execute"
    CATEGORY = "EasyUse/Logic"

    def execute(self, any, delay):
        time.sleep(delay)
        return (any,)

NODE_CLASS_MAPPINGS = {
    "easy string": String,
    "easy int": Int,
    "easy rangeInt": RangeInt,
    "easy float": Float,
    "easy rangeFloat": RangeFloat,
    "easy boolean": Boolean,
    "easy mathString": mathStringOperation,
    "easy mathInt": mathIntOperation,
    "easy mathFloat": mathFloatOperation,
    "easy compare": Compare,
    "easy imageSwitch": imageSwitch,
    "easy textSwitch": textSwitch,
    "easy imageIndexSwitch": imageIndexSwitch,
    "easy textIndexSwitch": textIndexSwitch,
    "easy conditioningIndexSwitch": conditioningIndexSwitch,
    "easy anythingIndexSwitch": anythingIndexSwitch,
    "easy ab": ab,
    "easy anythingInversedSwitch": anythingInversedSwitch,
    "easy whileLoopStart": whileLoopStart,
    "easy whileLoopEnd": whileLoopEnd,
    "easy forLoopStart": forLoopStart,
    "easy forLoopEnd": forLoopEnd,
    "easy blocker": Blocker,
    "easy ifElse": IfElse,
    "easy isMaskEmpty": isMaskEmpty,
    "easy isNone": isNone,
    "easy isSDXL": isSDXL,
    "easy isFileExist": isFileExist,
    "easy outputToList": outputToList,
    "easy pixels": pixels,
    "easy xyAny": xyAny,
    "easy lengthAnything": lengthAnything,
    "easy indexAnything": indexAnything,
    "easy batchAnything": batchAnything,
    "easy convertAnything": convertAnything,
    "easy showAnything": showAnything,
    "easy showTensorShape": showTensorShape,
    "easy clearCacheKey": clearCacheKey,
    "easy clearCacheAll": clearCacheAll,
    "easy cleanGpuUsed": cleanGPUUsed,
    "easy saveText": saveText,
    "easy sleep": sleep
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "easy string": "String",
    "easy int": "Int",
    "easy rangeInt": "Range(Int)",
    "easy float": "Float",
    "easy rangeFloat": "Range(Float)",
    "easy boolean": "Boolean",
    "easy compare": "Compare",
    "easy mathString": "Math String",
    "easy mathInt": "Math Int",
    "easy mathFloat": "Math Float",
    "easy imageSwitch": "Image Switch",
    "easy textSwitch": "Text Switch",
    "easy imageIndexSwitch": "Image Index Switch",
    "easy textIndexSwitch": "Text Index Switch",
    "easy conditioningIndexSwitch": "Conditioning Index Switch",
    "easy anythingIndexSwitch": "Any Index Switch",
    "easy ab": "A or B",
    "easy anythingInversedSwitch": "Any Inversed Switch",
    "easy whileLoopStart": "While Loop Start",
    "easy whileLoopEnd": "While Loop End",
    "easy forLoopStart": "For Loop Start",
    "easy forLoopEnd": "For Loop End",
    "easy ifElse": "If else",
    "easy blocker": "Blocker",
    "easy isMaskEmpty": "Is Mask Empty",
    "easy isNone": "Is None",
    "easy isSDXL": "Is SDXL",
    "easy isFileExist": "Is File Exist",
    "easy outputToList": "Output to List",
    "easy pixels": "Pixels W/H Norm",
    "easy xyAny": "XY Any",
    "easy lengthAnything": "Length Any",
    "easy indexAnything": "Index Any",
    "easy batchAnything": "Batch Any",
    "easy convertAnything": "Convert Any",
    "easy showAnything": "Show Any",
    "easy showTensorShape": "Show Tensor Shape",
    "easy clearCacheKey": "Clear Cache Key",
    "easy clearCacheAll": "Clear Cache All",
    "easy cleanGpuUsed": "Clean VRAM Used",
    "easy saveText": "Save Text",
    "easy sleep": "Sleep",
}
