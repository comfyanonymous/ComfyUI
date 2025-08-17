import os
import sys
import importlib.util

repositories_path = os.path.dirname(os.path.realpath(__file__))

# Import the script
script_name = os.path.join("scripts", "ultimate-upscale")
repo_name = "ultimate_sd_upscale"
script_path = os.path.join(repositories_path, repo_name, f"{script_name}.py")
spec = importlib.util.spec_from_file_location(script_name, script_path)
ultimate_upscale = importlib.util.module_from_spec(spec)
sys.modules[script_name] = ultimate_upscale
spec.loader.exec_module(ultimate_upscale)
