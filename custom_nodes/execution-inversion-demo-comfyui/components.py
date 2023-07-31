import os
import shutil
import folder_paths
import json
import copy
import comfy.graph_utils

comfy_path = os.path.dirname(folder_paths.__file__)
js_path = os.path.join(comfy_path, "web", "extensions")
inversion_demo_path = os.path.dirname(__file__)

def setup_js():
    # setup js
    js_dest_path = os.path.join(js_path, "inversion-demo-components")
    if not os.path.exists(js_dest_path):
        os.makedirs(js_dest_path)
    js_src_path = os.path.join(inversion_demo_path, "js", "inversion-demo-components.js")
    shutil.copy(js_src_path, js_dest_path)

class ComponentInput:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"multiline": False}),
                "data_type": ("STRING", {"multiline": False, "default": "IMAGE"}),
                "extra_args": ("STRING", {"multiline": False}),
                "explicit_input_order": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "optional": ([False, True],),
            },
            "optional": {
                "default_value": ("*",),
            },
        }

    RETURN_TYPES = ("*",)
    FUNCTION = "component_input"

    CATEGORY = "InversionDemo Nodes/Component Creation"

    def component_input(self, name, data_type, extra_args, explicit_input_order, optional, default_value = None):
        return (default_value,)

class ComponentOutput:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "data_type": ("STRING", {"multiline": False, "default": "IMAGE"}),
                "name": ("STRING", {"multiline": False}),
                "value": ("*",),
            },
        }

    RETURN_TYPES = ("*",)
    FUNCTION = "component_output"

    CATEGORY = "InversionDemo Nodes/Component Creation"

    def component_output(self, index, data_type, name, value):
        return (value,)

class ComponentMetadata:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"multiline": False}),
                "always_output": ([False, True],),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "nop"

    CATEGORY = "InversionDemo Nodes/Component Creation"

    def nop(self, name):
        return {}

COMPONENT_NODE_CLASS_MAPPINGS = {
    "ComponentInput": ComponentInput,
    "ComponentOutput": ComponentOutput,
    "ComponentMetadata": ComponentMetadata,
}
COMPONENT_NODE_DISPLAY_NAME_MAPPINGS = {
    "ComponentInput": "Component Input",
    "ComponentOutput": "Component Output",
    "ComponentMetadata": "Component Metadata",
}

DEFAULT_EXTRA_DATA = {
    "STRING": {"multiline": False},
    "INT": {"default": 0, "min": 0, "max": 1000, "step": 1},
    "FLOAT": {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.1},
}

def default_extra_data(data_type, extra_args):
    if data_type == "STRING":
        args = {"multiline": False}
    elif data_type == "INT":
        args = {"default": 0, "min": -1000000, "max": 1000000, "step": 1}
    elif data_type == "FLOAT":
        args = {"default": 0.0, "min": -1000000.0, "max": 1000000.0, "step": 0.1}
    else:
        args = {}
    args.update(extra_args)
    return args

def LoadComponent(component_file):
    try:
        with open(component_file, "r") as f:
            component_data = f.read()
            graph = json.loads(component_data)["output"]

        component_raw_name = os.path.basename(component_file).split(".")[0]
        component_display_name = component_raw_name
        component_inputs = []
        component_outputs = []
        is_output_component = False
        for node_id, data in graph.items():
            if data["class_type"] == "ComponentMetadata":
                component_display_name = data["inputs"].get("name", component_raw_name)
                is_output_component = data["inputs"].get("always_output", False)
            elif data["class_type"] == "ComponentInput":
                data_type = data["inputs"]["data_type"]
                if len(data_type) > 0 and data_type[0] == "[":
                    try:
                        data_type = json.loads(data_type)
                    except:
                        pass
                try:
                    extra_args = json.loads(data["inputs"]["extra_args"])
                except:
                    extra_args = {}
                component_inputs.append({
                    "node_id": node_id,
                    "name": data["inputs"]["name"],
                    "data_type": data_type,
                    "extra_args": extra_args,
                    "explicit_input_order": data["inputs"]["explicit_input_order"],
                    "optional": data["inputs"]["optional"],
                })
            elif data["class_type"] == "ComponentOutput":
                component_outputs.append({
                    "node_id": node_id,
                    "name": data["inputs"]["name"] or data["inputs"]["data_type"],
                    "index": data["inputs"]["index"],
                    "data_type": data["inputs"]["data_type"],
                })
        component_inputs.sort(key=lambda x: (x["explicit_input_order"], x["name"]))
        component_outputs.sort(key=lambda x: x["index"])
        for i in range(1, len(component_inputs)):
            if component_inputs[i]["name"] == component_inputs[i-1]["name"]:
                raise Exception("Component input name is not unique: {}".format(component_inputs[i]["name"]))
        for i in range(1, len(component_outputs)):
            if component_outputs[i]["index"] == component_outputs[i-1]["index"]:
                raise Exception("Component output index is not unique: {}".format(component_outputs[i]["index"]))
    except Exception as e:
        print("Error loading component file: {}: {}".format(component_file, e))
        return None
    
    class ComponentNode:
        def __init__(self):
            pass

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {node["name"]: (node["data_type"], default_extra_data(node["data_type"], node["extra_args"])) for node in component_inputs if not node["optional"]},
                "optional": {node["name"]: (node["data_type"], default_extra_data(node["data_type"], node["extra_args"])) for node in component_inputs if node["optional"]},
            }

        RETURN_TYPES = tuple([node["data_type"] for node in component_outputs])
        RETURN_NAMES = tuple([node["name"] for node in component_outputs])
        FUNCTION = "expand_component"

        CATEGORY = "Custom Components"
        OUTPUT_NODE = is_output_component

        def expand_component(self, **kwargs):
            new_graph = copy.deepcopy(graph)
            for input_node in component_inputs:
                if input_node["name"] in kwargs:
                    new_graph[input_node["node_id"]]["inputs"]["default_value"] = kwargs[input_node["name"]]
            outputs = tuple([[node["node_id"], 0] for node in component_outputs])
            new_graph, outputs = comfy.graph_utils.add_graph_prefix(new_graph, outputs, comfy.graph_utils.GraphBuilder.alloc_prefix())
            return {
                "result": outputs,
                "expand": new_graph,
            }
    ComponentNode.__name__ = component_raw_name
    COMPONENT_NODE_CLASS_MAPPINGS[component_raw_name] = ComponentNode
    COMPONENT_NODE_DISPLAY_NAME_MAPPINGS[component_raw_name] = component_display_name
    print("Loaded component: {}".format(component_display_name))

def load_components():
    component_dir = os.path.join(comfy_path, "components")
    if not os.path.exists(component_dir):
        return
    files = [f for f in os.listdir(component_dir) if os.path.isfile(os.path.join(component_dir, f)) and f.endswith(".json")]
    for f in files:
        print("Loading component file %s" % f)
        LoadComponent(os.path.join(component_dir, f))

load_components()
