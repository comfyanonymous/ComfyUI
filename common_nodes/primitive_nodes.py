



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
    
    
    
NODE_CLASS_MAPPINGS = {
    
    "ListNode": ListNode,
    "SetListItem": SetListItem
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ListNode": "List",
    "SetListItem": "Set List Item"
} 