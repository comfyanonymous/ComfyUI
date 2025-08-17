import os
import folder_paths
from nodes import CLIPTextEncode
from .constants import get_category, get_name
from .power_prompt import RgthreePowerPrompt

class RgthreePowerPromptSimple(RgthreePowerPrompt):

    NAME=get_name('Power Prompt - Simple')
    CATEGORY = get_category()

    @classmethod
    def INPUT_TYPES(cls):  # pylint: disable = invalid-name, missing-function-docstring
        # Removed Saved Prompts feature; No sure it worked any longer. UI should fail gracefully,
        # TODO: Rip out saved prompt input data
        SAVED_PROMPTS_FILES=[]
        SAVED_PROMPTS_CONTENT=[]
        return {
            'required': {
                'prompt': ('STRING', {'multiline': True, 'dynamicPrompts': True}),
            },
            'optional': {
                "opt_clip": ("CLIP", ),
                'insert_embedding': (['CHOOSE',] + [os.path.splitext(x)[0] for x in folder_paths.get_filename_list('embeddings')],),
                'insert_saved': (['CHOOSE',] + SAVED_PROMPTS_FILES,),
            },
            'hidden': {
                'values_insert_saved': (['CHOOSE'] + SAVED_PROMPTS_CONTENT,),
            }
        }

    RETURN_TYPES = ('CONDITIONING', 'STRING',)
    RETURN_NAMES = ('CONDITIONING', 'TEXT',)
    FUNCTION = 'main'

    def main(self, prompt, opt_clip=None, insert_embedding=None, insert_saved=None, values_insert_saved=None):
        conditioning=None
        if opt_clip != None:
            conditioning = CLIPTextEncode().encode(opt_clip, prompt)[0]

        return (conditioning, prompt)

