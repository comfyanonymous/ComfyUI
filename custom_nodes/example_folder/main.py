from utils import waste_cpu_resource
class ExampleFolder:
    """
    A example node

    Class methods
    -------------
    INPUT_TYPES (dict): 
        Tell the main program input parameters of nodes.

    Attributes
    ----------
    RETURN_TYPES (`tuple`): 
        The type of each element in the output tulple.
    FUNCTION (`str`):
        The name of the entry-point method which will return a tuple. For example, if `FUNCTION = "execute"` then it will run Example().execute()
    OUTPUT_NODE ([`bool`]):
        WIP
    CATEGORY (`str`):
        WIP
    execute(s) -> tuple || None:
        The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
        For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        """
            Return a dictionary which contains config for all input fields.
            The type can be a string indicate a type or a list indicate selection.
            Prebuilt types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
            Input in type "INT", "STRING" or "FLOAT" will be converted automatically from a string to the corresponse Python type before passing and have special config
            Argument: s (`None`): Useless ig
            Returns: `dict`:
                - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                - Value input_fields (`dict`): Contains input fields config:
                    * Key field_name (`string`): Name of a entry-point method's argument
                    * Value field_config (`tuple`):
                        + First value is a string indicate the type of field or a list for selection.
                        + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {
            "required": {
                "string_field": ("STRING", {
                    "multiline": True, #Allow the input to be multilined
                    "default": "Hello World!"
                }),
                "int_field": ("INT", {
                    "default": 0, 
                    "min": 0, #Minimum value
                    "max": 4096, #Maximum value
                    "step": 64 #Slider's step
                }),
                #Like INT
                "print_to_screen": (["Enable", "Disable"], {"default": "Enable"})
            },
            #"hidden": {
            #    "prompt": "PROMPT", 
            #    "extra_pnginfo": "EXTRA_PNGINFO"
            #},
        }

    RETURN_TYPES = ("STRING", "INT", "FLOAT", "STRING")
    FUNCTION = "test"

    #OUTPUT_NODE = True

    CATEGORY = "Example"

    def test(self, string_field, int_field, print_to_screen):
        rand_float = waste_cpu_resource()
        if print_to_screen == "Enable":
            print(f"""Your input contains:
                string_field aka input text: {string_field}
                int_field: {int_field}
                A random float number: {rand_float}
            """)
        return (string_field, int_field, rand_float, print_to_screen)

NODE_CLASS_MAPPINGS = {
    "ExampleFolder": ExampleFolder
}
"""
NODE_CLASS_MAPPINGS (dict): A dictionary contains all nodes you want to export
"""