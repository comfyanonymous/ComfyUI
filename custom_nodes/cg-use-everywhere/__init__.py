from .use_everywhere import SeedEverywhere, AnythingEverywherePrompts

UE_VERSION = "7.0.1"

NODE_CLASS_MAPPINGS = { "Seed Everywhere": SeedEverywhere }

from .use_everywhere import AnythingEverywhere, AnythingSomewhere, AnythingEverywhereTriplet, SimpleString
NODE_CLASS_MAPPINGS["Anything Everywhere"] = AnythingEverywhere
NODE_CLASS_MAPPINGS["Anything Everywhere3"] = AnythingEverywhereTriplet
NODE_CLASS_MAPPINGS["Anything Everywhere?"] = AnythingSomewhere
NODE_CLASS_MAPPINGS["Prompts Everywhere"] = AnythingEverywherePrompts
NODE_CLASS_MAPPINGS["Simple String"] = SimpleString

import os, shutil
import folder_paths

# temporary code to remove old javascript installs
module_js_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), "js")
application_root_directory = os.path.dirname(folder_paths.__file__)
old_code_location = os.path.join(application_root_directory, "web", "extensions", "use_everywhere")
if os.path.exists(old_code_location):
    shutil.rmtree(old_code_location)

old_code_location = os.path.join(application_root_directory, "web", "extensions", "cg-nodes", "use_everywhere.js")
if os.path.exists(old_code_location):
    os.remove(old_code_location)
# end of temporary code

WEB_DIRECTORY = "./js"
__all__ = ["NODE_CLASS_MAPPINGS", "WEB_DIRECTORY"]
