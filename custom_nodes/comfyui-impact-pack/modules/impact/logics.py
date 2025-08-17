import sys
import time

import execution
import impact.impact_server
from server import PromptServer
from impact.utils import any_typ
import impact.core as core
import re
import nodes
import logging


class ImpactCompare:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cmp": (['a = b', 'a <> b', 'a > b', 'a < b', 'a >= b', 'a <= b', 'tt', 'ff'],),
                "a": (any_typ, ),
                "b": (any_typ, ),
            },
        }

    FUNCTION = "doit"
    CATEGORY = "ImpactPack/Logic"

    RETURN_TYPES = ("BOOLEAN", )

    def doit(self, cmp, a, b):
        if cmp == "a = b":
            return (a == b, )
        elif cmp == "a <> b":
            return (a != b, )
        elif cmp == "a > b":
            return (a > b, )
        elif cmp == "a < b":
            return (a < b, )
        elif cmp == "a >= b":
            return (a >= b, )
        elif cmp == "a <= b":
            return (a <= b, )
        elif cmp == 'tt':
            return (True, )
        else:
            return (False, )


class ImpactNotEmptySEGS:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"segs": ("SEGS",)}}

    FUNCTION = "doit"
    CATEGORY = "ImpactPack/Logic"

    RETURN_TYPES = ("BOOLEAN", )

    def doit(self, segs):
        return (segs[1] != [], )


class ImpactConditionalBranch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cond": ("BOOLEAN",),
                "tt_value": (any_typ,{"lazy": True}),
                "ff_value": (any_typ,{"lazy": True}),
            },
        }

    FUNCTION = "doit"
    CATEGORY = "ImpactPack/Logic"

    RETURN_TYPES = (any_typ, )

    def check_lazy_status(self, cond, tt_value=None, ff_value=None):
        if cond and tt_value is None:
            return ["tt_value"]
        if not cond and ff_value is None:
            return ["ff_value"]

    def doit(self, cond, tt_value=None, ff_value=None):
        if cond:
            return (tt_value,)
        else:
            return (ff_value,)


class ImpactConditionalBranchSelMode:
    @classmethod
    def INPUT_TYPES(cls):
        if not core.is_execution_model_version_supported():
            required_inputs = {
                "cond": ("BOOLEAN",),
                "sel_mode": ("BOOLEAN", {"default": True, "label_on": "select_on_prompt", "label_off": "select_on_execution"}),
                }
        else:
            required_inputs = {
                "cond": ("BOOLEAN",),
                }

        return {
            "required": required_inputs,
            "optional": {
                "tt_value": (any_typ,),
                "ff_value": (any_typ,),
            },
        }

    FUNCTION = "doit"
    CATEGORY = "ImpactPack/Logic"

    RETURN_TYPES = (any_typ, )

    def doit(self, cond, tt_value=None, ff_value=None, **kwargs):
        if cond:
            return (tt_value,)
        else:
            return (ff_value,)


class ImpactConvertDataType:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": (any_typ,)}}

    RETURN_TYPES = ("STRING", "FLOAT", "INT", "BOOLEAN")
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Logic"

    @staticmethod
    def is_number(string):
        pattern = re.compile(r'^[-+]?[0-9]*\.?[0-9]+$')
        return bool(pattern.match(string))

    def doit(self, value):
        if self.is_number(str(value)):
            num = value
        else:
            if str.lower(str(value)) != "false":
                num = 1
            else:
                num = 0
        return (str(value), float(num), int(float(num)), bool(float(num)), )


class ImpactIfNone:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {"signal": (any_typ,), "any_input": (any_typ,), }
        }

    RETURN_TYPES = (any_typ, "BOOLEAN")
    RETURN_NAMES = ("signal_opt", "bool")
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Logic"

    def doit(self, signal=None, any_input=None):
        if any_input is None:
            return (signal, False, )
        else:
            return (signal, True, )


class ImpactLogicalOperators:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "operator": (['and', 'or', 'xor'],),
                "bool_a": ("BOOLEAN", {"forceInput": True}),
                "bool_b": ("BOOLEAN", {"forceInput": True}),
            },
        }

    FUNCTION = "doit"
    CATEGORY = "ImpactPack/Logic"

    RETURN_TYPES = ("BOOLEAN", )

    def doit(self, operator, bool_a, bool_b):
        if operator == "and":
            return (bool_a and bool_b, )
        elif operator == "or":
            return (bool_a or bool_b, )
        else:
            return (bool_a != bool_b, )


class ImpactConditionalStopIteration:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { "cond": ("BOOLEAN", {"forceInput": True}), },
        }

    FUNCTION = "doit"
    CATEGORY = "ImpactPack/Logic"

    RETURN_TYPES = ()

    OUTPUT_NODE = True

    def doit(self, cond):
        if cond:
            PromptServer.instance.send_sync("stop-iteration", {})
        return {}


class ImpactNeg:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { "value": ("BOOLEAN", {"forceInput": True}), },
        }

    FUNCTION = "doit"
    CATEGORY = "ImpactPack/Logic"

    RETURN_TYPES = ("BOOLEAN", )

    def doit(self, value):
        return (not value, )


class ImpactInt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
            },
        }

    FUNCTION = "doit"
    CATEGORY = "ImpactPack/Logic"

    RETURN_TYPES = ("INT", )

    def doit(self, value):
        return (value, )


class ImpactFloat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {"default": 1.0, "min": -3.402823466e+38, "max": 3.402823466e+38}),
            },
        }

    FUNCTION = "doit"
    CATEGORY = "ImpactPack/Logic"

    RETURN_TYPES = ("FLOAT", )

    def doit(self, value):
        return (value, )


class ImpactBoolean:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("BOOLEAN", {"default": False}),
            },
        }

    FUNCTION = "doit"
    CATEGORY = "ImpactPack/Logic"

    RETURN_TYPES = ("BOOLEAN", )

    def doit(self, value):
        return (value, )


class ImpactValueSender:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "value": (any_typ, ),
                    "link_id": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
                    },
                "optional": {
                        "signal_opt": (any_typ,),
                    }
                }

    OUTPUT_NODE = True

    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Logic"

    RETURN_TYPES = (any_typ, )
    RETURN_NAMES = ("signal", )

    def doit(self, value, link_id=0, signal_opt=None):
        PromptServer.instance.send_sync("value-send", {"link_id": link_id, "value": value})
        return (signal_opt, )


class ImpactIntConstSender:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "signal": (any_typ, ),
                    "value": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
                    "link_id": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
                    },
                }

    OUTPUT_NODE = True

    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Logic"

    RETURN_TYPES = ()

    def doit(self, signal, value, link_id=0):
        PromptServer.instance.send_sync("value-send", {"link_id": link_id, "value": value})
        return {}


class ImpactValueReceiver:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "typ": (["STRING", "INT", "FLOAT", "BOOLEAN"], ),
                    "value": ("STRING", {"default": ""}),
                    "link_id": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
                    },
                }

    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Logic"

    RETURN_TYPES = (any_typ, )

    def doit(self, typ, value, link_id=0):
        if typ == "INT":
            return (int(value), )
        elif typ == "FLOAT":
            return (float(value), )
        elif typ == "BOOLEAN":
            return (value.lower() == "true", )
        else:
            return (value, )


class ImpactImageInfo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "value": ("IMAGE", ),
                    },
                }

    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Logic/_for_test"

    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = ("batch", "height", "width", "channel")

    def doit(self, value):
        return (value.shape[0], value.shape[1], value.shape[2], value.shape[3])


class ImpactLatentInfo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "value": ("LATENT", ),
                    },
                }

    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Logic/_for_test"

    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = ("batch", "height", "width", "channel")

    def doit(self, value):
        shape = value['samples'].shape
        return (shape[0], shape[2] * 8, shape[3] * 8, shape[1])


class ImpactMinMax:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "mode": ("BOOLEAN", {"default": True, "label_on": "max", "label_off": "min"}),
                    "a": (any_typ,),
                    "b": (any_typ,),
                    },
                }

    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Logic/_for_test"

    RETURN_TYPES = ("INT", )

    def doit(self, mode, a, b):
        if mode:
            return (max(a, b), )
        else:
            return (min(a, b),)


class ImpactQueueTrigger:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "signal": (any_typ,),
                    "mode": ("BOOLEAN", {"default": True, "label_on": "Trigger", "label_off": "Don't trigger"}),
                    }
                }

    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Logic/_for_test"
    RETURN_TYPES = (any_typ,)
    RETURN_NAMES = ("signal_opt",)
    OUTPUT_NODE = True

    def doit(self, signal, mode):
        if(mode):
            PromptServer.instance.send_sync("impact-add-queue", {})

        return (signal,)


class ImpactQueueTriggerCountdown:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "count": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "total": ("INT", {"default": 10, "min": 1, "max": 0xffffffffffffffff}),
                    "mode": ("BOOLEAN", {"default": True, "label_on": "Trigger", "label_off": "Don't trigger"}),
                    },
                "optional": {"signal": (any_typ,),},
                "hidden": {"unique_id": "UNIQUE_ID"}
                }

    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Logic/_for_test"
    RETURN_TYPES = (any_typ, "INT", "INT")
    RETURN_NAMES = ("signal_opt", "count", "total")
    OUTPUT_NODE = True

    def doit(self, count, total, mode, unique_id, signal=None):
        if (mode):
            if count < total - 1:
                PromptServer.instance.send_sync("impact-node-feedback",
                                                {"node_id": unique_id, "widget_name": "count", "type": "int", "value": count+1})
                PromptServer.instance.send_sync("impact-add-queue", {})
            if count >= total - 1:
                PromptServer.instance.send_sync("impact-node-feedback",
                                                {"node_id": unique_id, "widget_name": "count", "type": "int", "value": 0})

        return (signal, count, total)



class ImpactSetWidgetValue:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "signal": (any_typ,),
                    "node_id": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "widget_name": ("STRING", {"multiline": False}),
                    },
                "optional": {
                    "boolean_value": ("BOOLEAN", {"forceInput": True}),
                    "int_value": ("INT", {"forceInput": True}),
                    "float_value": ("FLOAT", {"forceInput": True}),
                    "string_value": ("STRING", {"forceInput": True}),
                    }
                }

    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Logic/_for_test"
    RETURN_TYPES = (any_typ,)
    RETURN_NAMES = ("signal_opt",)
    OUTPUT_NODE = True

    def doit(self, signal, node_id, widget_name, boolean_value=None, int_value=None, float_value=None, string_value=None, ):
        kind = None
        if boolean_value is not None:
            value = boolean_value
            kind = "BOOLEAN"
        elif int_value is not None:
            value = int_value
            kind = "INT"
        elif float_value is not None:
            value = float_value
            kind = "FLOAT"
        elif string_value is not None:
            value = string_value
            kind = "STRING"
        else:
            value = None

        if value is not None:
            PromptServer.instance.send_sync("impact-node-feedback",
                                            {"node_id": node_id, "widget_name": widget_name, "type": kind, "value": value})

        return (signal,)


class ImpactNodeSetMuteState:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "signal": (any_typ,),
                    "node_id": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "set_state": ("BOOLEAN", {"default": True, "label_on": "active", "label_off": "mute"}),
                    }
                }

    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Logic/_for_test"
    RETURN_TYPES = (any_typ,)
    RETURN_NAMES = ("signal_opt",)
    OUTPUT_NODE = True

    def doit(self, signal, node_id, set_state):
        PromptServer.instance.send_sync("impact-node-mute-state", {"node_id": node_id, "is_active": set_state})
        return (signal,)


class ImpactSleep:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "signal": (any_typ,),
                    "seconds": ("FLOAT", {"default": 0.5, "min": 0, "max": 3600}),
                    }
                }

    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Logic/_for_test"
    RETURN_TYPES = (any_typ,)
    RETURN_NAMES = ("signal_opt",)
    OUTPUT_NODE = True

    def doit(self, signal, seconds):
        time.sleep(seconds)
        return (signal,)


def workflow_to_map(workflow):
    nodes = {}
    links = {}
    for link in workflow['links']:
        links[link[0]] = link[1:]
    for node in workflow['nodes']:
        nodes[str(node['id'])] = node

    return nodes, links


class ImpactRemoteBoolean:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "node_id": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "widget_name": ("STRING", {"multiline": False}),
                    "value": ("BOOLEAN", {"default": True, "label_on": "True", "label_off": "False"}),
                    }}

    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Logic/_for_test"
    RETURN_TYPES = ()
    OUTPUT_NODE = True

    def doit(self, **kwargs):
        return {}


class ImpactRemoteInt:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "node_id": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "widget_name": ("STRING", {"multiline": False}),
                    "value": ("INT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff}),
                    }}

    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Logic/_for_test"
    RETURN_TYPES = ()
    OUTPUT_NODE = True

    def doit(self, **kwargs):
        return {}

class ImpactControlBridge:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                      "value": (any_typ,),
                      "mode": ("BOOLEAN", {"default": True, "label_on": "Active", "label_off": "Stop/Mute/Bypass"}),
                      "behavior": (["Stop", "Mute", "Bypass"], ),
                    },
                "hidden": {"unique_id": "UNIQUE_ID", "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}
                }

    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Logic"
    RETURN_TYPES = (any_typ,)
    RETURN_NAMES = ("value",)
    OUTPUT_NODE = True

    DESCRIPTION = ("When behavior is Stop and mode is active, the input value is passed directly to the output.\n"
                   "When behavior is Mute/Bypass and mode is active, the node connected to the output is changed to active state.\n"
                   "When behavior is Stop and mode is Stop/Mute/Bypass, the workflow execution of the current node is halted.\n"
                   "When behavior is Mute/Bypass and mode is Stop/Mute/Bypass, the node connected to the output is changed to Mute/Bypass state.")

    @classmethod
    def IS_CHANGED(self, value, mode, behavior="Stop", unique_id=None, prompt=None, extra_pnginfo=None):
        if behavior == "Stop":
            return value, mode, behavior
        else:
            # NOTE: extra_pnginfo is not populated for IS_CHANGED.
            #       so extra_pnginfo is useless in here
            try:
                workflow = core.current_prompt['extra_data']['extra_pnginfo']['workflow']
            except Exception:
                logging.info("[Impact Pack] core.current_prompt['extra_data']['extra_pnginfo']['workflow']")
                return 0

            nodes, links = workflow_to_map(workflow)
            next_nodes = []

            for link in nodes[unique_id]['outputs'][0]['links']:
                node_id = str(links[link][2])
                impact.utils.collect_non_reroute_nodes(nodes, links, next_nodes, node_id)

        return next_nodes

    def doit(self, value, mode, behavior="Stop", unique_id=None, prompt=None, extra_pnginfo=None):
        global error_skip_flag

        if core.is_execution_model_version_supported():
            from comfy_execution.graph import ExecutionBlocker
        else:
            logging.info("[Impact Pack] ImpactControlBridge: ComfyUI is outdated. The 'Stop' behavior cannot function properly.")

        if behavior == "Stop":
            if mode:
                return (value, )
            else:
                return (ExecutionBlocker(None), )
        elif extra_pnginfo is None:
            logging.warning(f"[Impact Pack] limitation: '{behavior}' behavior cannot be used in API execution.")
            return (value,)
        else:
            workflow_nodes, links = workflow_to_map(extra_pnginfo['workflow'])

            active_nodes = []
            mute_nodes = []
            bypass_nodes = []

            for link in workflow_nodes[unique_id]['outputs'][0]['links']:
                node_id = str(links[link][2])

                next_nodes = []
                impact.utils.collect_non_reroute_nodes(workflow_nodes, links, next_nodes, node_id)

                for next_node_id in next_nodes:
                    node_mode = workflow_nodes[next_node_id]['mode']

                    if node_mode == 0:
                        active_nodes.append(next_node_id)
                    elif node_mode == 2:
                        mute_nodes.append(next_node_id)
                    elif node_mode == 4:
                        bypass_nodes.append(next_node_id)

            if mode:
                # active
                should_be_active_nodes = mute_nodes + bypass_nodes
                if len(should_be_active_nodes) > 0:
                    PromptServer.instance.send_sync("impact-bridge-continue", {"node_id": unique_id, 'actives': list(should_be_active_nodes)})
                    nodes.interrupt_processing()

            elif behavior == "Mute" or behavior == True:  # noqa: E712
                # mute
                should_be_mute_nodes = active_nodes + bypass_nodes
                if len(should_be_mute_nodes) > 0:
                    PromptServer.instance.send_sync("impact-bridge-continue", {"node_id": unique_id, 'mutes': list(should_be_mute_nodes)})
                    nodes.interrupt_processing()

            else:
                # bypass
                should_be_bypass_nodes = active_nodes + mute_nodes
                if len(should_be_bypass_nodes) > 0:
                    PromptServer.instance.send_sync("impact-bridge-continue", {"node_id": unique_id, 'bypasses': list(should_be_bypass_nodes)})
                    nodes.interrupt_processing()

            return (value, )


class ImpactExecutionOrderController:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "signal": (any_typ,),
                    "value": (any_typ,),
                    }}

    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"
    RETURN_TYPES = (any_typ, any_typ)
    RETURN_NAMES = ("signal", "value")

    def doit(self, signal, value):
        return signal, value


class ImpactListBridge:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "list_input": (any_typ,),
                    }}

    FUNCTION = "doit"

    DESCRIPTION = "When passing the list output through this node, it collects and organizes the data before forwarding it, which ensures that the previous stage's sub-workflow has been completed."

    CATEGORY = "ImpactPack/Util"
    RETURN_TYPES = (any_typ, )
    RETURN_NAMES = ("list_output", )

    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, )

    @staticmethod
    def doit(list_input):
        return (list_input,)


original_handle_execution = execution.PromptExecutor.handle_execution_error


def handle_execution_error(**kwargs):
    execution.PromptExecutor.handle_execution_error(**kwargs)

