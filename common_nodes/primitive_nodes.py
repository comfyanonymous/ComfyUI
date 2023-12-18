



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
        print(f"Set list time: {dict_data}")
        return (dict_data, )    
    
    
    
NODE_CLASS_MAPPINGS = {
    
    "ListNode": ListNode,
    "SetListItem": SetListItem,
    "DictNode": DictNode,
    "SetDictItem": SetDictItem
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ListNode": "List",
    "SetListItem": "Set List Item",
    "DictNode": "Dict",
    "SetDictItem": "Set Dict Item"
} 