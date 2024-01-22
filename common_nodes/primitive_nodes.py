



# ====================================================================
# String
# ====================================================================

class ConcatenateString:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "text0": ("STRING", {"forceInput": True}),
            "text1": ("STRING", {"default": "", "multiline": True}),
            "concatenate_by": ("STRING", {"default": ","})
                }}
    
    RETURN_TYPES = ("STRING", )
    FUNCTION = "execute"

    CATEGORY = "aiyoh"

    def execute(self, text0, text1, concatenate_by):
        cur_str = text0 if text0 is not None else ''
        concate = concatenate_by if concatenate_by is not None else ''
        if text1 is not None and text1 != '':
            cur_str = f"{cur_str}{concate}{text1}"
        
        return (cur_str, )
        


# ====================================================================
# List
# ====================================================================


class ListNode:
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "list": ("LIST", {"default": []})}}
    
    RETURN_TYPES = ("LIST", )
    FUNCTION = "execute"

    CATEGORY = "flow"
    
    
    def __init__(self) -> None:
        self.list_data = []
    
    
    def execute(self, list):
        self.list_data = list
        
        return (self.list_data, )    
        
        

class GetListItem:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "list": ("LIST",), "index": ("INT", {"default": 0})
                }}
    
    RETURN_TYPES = ("ANY_DATA", )
    FUNCTION = "execute"

    CATEGORY = "flow"
    
    
    def execute(self, list, index):
        print(f"Get list time: {list}")
        return (list[index], )
         

        
class SetListItem:
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "list": ("LIST",), "index": ("INT", {"default": 0}),
                    "element": ("ANY_DATA", )
                }}
    
    RETURN_TYPES = ("LIST", )
    FUNCTION = "execute"

    CATEGORY = "flow"
    
    
    def execute(self, list, index, element):
        list[index] = element
        print(f"Set list time: {list}")
        return (list, )
    
    
# ==========================================================
# Dict
# ==========================================================
    
class DictNode:
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "dict_data": ("DICT", {"default": {}})}}
    
    RETURN_TYPES = ("DICT", )
    FUNCTION = "execute"

    CATEGORY = "flow"
    
    
    def __init__(self) -> None:
        self.dict_data = {}
    
    
    def execute(self, dict_data):
        self.dict_data = dict_data
        
        return (self.dict_data, )
    
    
class SetDictItem:
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "dict_data": ("DICT",), "name": ("STRING", {"default": ""}),
                    "value": ("ANY_DATA", )
                }}
    
    RETURN_TYPES = ("DICT", )
    FUNCTION = "execute"

    CATEGORY = "flow"
    
    
    def execute(self, dict_data, name, value):
        dict_data[name] = value
        print(f"Set dict time: {dict_data}")
        return (dict_data, )   
    
    
    

class GetDictItem:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "dict_data": ("DICT",), "name": ("STRING", {"default": ""})
                }}
    
    RETURN_TYPES = ("ANY_DATA", )
    FUNCTION = "execute"

    CATEGORY = "flow"
    
    def execute(self, dict_data, name):
        print(f"Get dict time: {dict_data}, {name}")
        return (dict_data[name], )
    
    
    
class GraphInputs:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "default_inputs": ("DICT", {"default": {}})}}
    
    RETURN_TYPES = ("DICT", )
    FUNCTION = "execute"
    
    INIT_GRAPH_INPUTS = "init_graph_inputs"

    CATEGORY = "flow"
    
    
    def __init__(self) -> None:
        self.graph_inputs = None
    
    
    def init_graph_inputs(self, graph_inputs):
        self.graph_inputs = graph_inputs
    
    
    def execute(self, default_inputs):
        if self.graph_inputs is None:
            self.graph_inputs = default_inputs
        
        return (self.graph_inputs, )
    


class GraphOutputs:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "graph_outputs": ("DICT",)}}
    
    RETURN_TYPES = ()
    FUNCTION = "execute"
    
    GET_GRAPH_OUTPUTS = "get_graph_outputs"

    CATEGORY = "flow"
    
    
    def __init__(self) -> None:
        self.graph_outputs = None
    
    
    def get_graph_outputs(self):
        return self.graph_outputs
    
    
    def execute(self, graph_outputs):
        self.graph_outputs = graph_outputs
        return (self.graph_outputs, )
    
    
    
NODE_CLASS_MAPPINGS = {
    "ConcatenateString": ConcatenateString,
    
    "ListNode": ListNode,
    "SetListItem": SetListItem,
    "GetListItem": GetListItem,
    
    "DictNode": DictNode,
    "SetDictItem": SetDictItem,
    "GetDictItem": GetDictItem,
    
    "GraphInputs": GraphInputs,
    "GraphOutputs": GraphOutputs
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConcatenateString": "Concatenate String",
    
    "ListNode": "List",
    "SetListItem": "Set List Item",
    "GetListItem": "Get List Item",
    
    "DictNode": "Dict",
    "SetDictItem": "Set Dict Item",
    "GetDictItem": "Get Dict Item",
    
    "GraphInputs": "Graph Inputs",
    "GraphOutputs": "Graph Outputs"
} 