
from comfy.comfy_types.node_typing import IO

class Base():
    FUNCTION = "func"
    CATEGORY = "everywhere"
    RETURN_TYPES = ()

class SimpleString(Base):
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{ "string": ("STRING", {"default": ""}) }}
    RETURN_TYPES = ("STRING",)

    def func(self,string):
        return (string,)

class SeedEverywhere(Base):
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{ "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}) },
                 "hidden": {"id":"UNIQUE_ID"} }

    RETURN_TYPES = ("INT",)

    def func(self, seed, id):
        return (seed,)

class AnythingEverywhere(Base):
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{}, 
                "optional": { "anything" : (IO.ANY, {}), },
                 "hidden": {"id":"UNIQUE_ID"} }

    def func(self, **kwargs):
        return ()

class AnythingEverywherePrompts(Base):
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{}, 
                "optional": { "+ve" : (IO.ANY, {}), "-ve" : (IO.ANY, {}), } }
    
    def func(self, **kwargs):
        return ()
        
class AnythingEverywhereTriplet(Base):
    CATEGORY = "everywhere/deprecated"
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{}, 
                "optional": { "anything" : (IO.ANY, {}), "anything2" : (IO.ANY, {}), "anything3" : (IO.ANY, {}),} }
    
    def func(self, **kwargs):
        return ()
    
class AnythingSomewhere(Base):
    CATEGORY = "everywhere/deprecated"
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{}, 
                "optional": { 
                    "anything" : (IO.ANY, {}), 
                    "title_regex" : ("STRING", {"default":".*"}),
                    "input_regex" : ("STRING", {"default":".*"}),
                    "group_regex" : ("STRING", {"default":".*"}),
                    },
                 "hidden": {"id":"UNIQUE_ID"} }

    def func(self, **kwargs):
        return ()
