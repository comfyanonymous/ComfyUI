class AlwaysEqualProxy(str):
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False

class TautologyStr(str):
    def __ne__(self, other):
        return False

class ByPassTypeTuple(tuple):
    def __getitem__(self, index):
        if index>0:
            index=0
        item = super().__getitem__(index)
        if isinstance(item, str):
            return TautologyStr(item)
        return item

comfy_ui_revision = None
def get_comfyui_revision():
    try:
        import git
        import os
        import folder_paths
        repo = git.Repo(os.path.dirname(folder_paths.__file__))
        comfy_ui_revision = len(list(repo.iter_commits('HEAD')))
    except:
        comfy_ui_revision = "Unknown"
    return comfy_ui_revision


import sys
import importlib.util
import importlib.metadata
import comfy.model_management as mm
import gc
from packaging import version
from server import PromptServer
def is_package_installed(package):
    try:
        module = importlib.util.find_spec(package)
        return module is not None
    except ImportError as e:
        print(e)
        return False

def install_package(package, v=None, compare=True, compare_version=None):
    run_install = True
    if is_package_installed(package):
        try:
            installed_version = importlib.metadata.version(package)
            if v is not None:
                if compare_version is None:
                    compare_version = v
                if not compare or version.parse(installed_version) >= version.parse(compare_version):
                    run_install = False
            else:
                run_install = False
        except:
            run_install = False

    if run_install:
        import subprocess
        package_command = package + '==' + v if v is not None else package
        PromptServer.instance.send_sync("easyuse-toast", {'content': f"Installing {package_command}...", 'duration': 5000})
        result = subprocess.run([sys.executable, '-s', '-m', 'pip', 'install', package_command], capture_output=True, text=True)
        if result.returncode == 0:
            PromptServer.instance.send_sync("easyuse-toast", {'content': f"{package} installed successfully", 'type': 'success', 'duration': 5000})
            print(f"Package {package} installed successfully")
            return True
        else:
            PromptServer.instance.send_sync("easyuse-toast", {'content': f"{package} installed failed", 'type': 'error', 'duration': 5000})
            print(f"Package {package} installed failed")
            return False
    else:
        return False

def compare_revision(num):
    global comfy_ui_revision
    if not comfy_ui_revision:
        comfy_ui_revision = get_comfyui_revision()
    return True if comfy_ui_revision == 'Unknown' or int(comfy_ui_revision) >= num else False

def find_tags(string: str, sep="/") -> list[str]:
    """
    find tags from string use the sep for split
    Note: string may contain the \\ or / for path separator
    """
    if not string:
        return []
    string = string.replace("\\", "/")
    while "//" in string:
        string = string.replace("//", "/")
    if string and sep in string:
        return string.split(sep)[:-1]
    return []


from comfy.model_base import BaseModel
import comfy.supported_models
import comfy.supported_models_base
def get_sd_version(model):
    base: BaseModel = model.model
    model_config: comfy.supported_models.supported_models_base.BASE = base.model_config
    if isinstance(model_config, comfy.supported_models.SDXL):
        return 'sdxl'
    elif isinstance(model_config, comfy.supported_models.SDXLRefiner):
        return 'sdxl_refiner'
    elif isinstance(
            model_config, (comfy.supported_models.SD15, comfy.supported_models.SD20)
    ):
        return 'sd1'
    elif isinstance(
            model_config, (comfy.supported_models.SVD_img2vid)
    ):
        return 'svd'
    elif isinstance(model_config, comfy.supported_models.SD3):
        return 'sd3'
    elif isinstance(model_config, comfy.supported_models.HunyuanDiT):
        return 'hydit'
    elif isinstance(model_config, comfy.supported_models.Flux):
        return 'flux'
    elif isinstance(model_config, comfy.supported_models.GenmoMochi):
        return 'mochi'
    else:
        return 'unknown'

def find_nearest_steps(clip_id, prompt):
    """Find the nearest KSampler or preSampling node that references the given id."""
    def check_link_to_clip(node_id, clip_id, visited=None, node=None):
        """Check if a given node links directly or indirectly to a loader node."""
        if visited is None:
            visited = set()

        if node_id in visited:
            return False
        visited.add(node_id)
        if "pipe" in node["inputs"]:
            link_ids = node["inputs"]["pipe"]
            for id in link_ids:
                if id != 0 and id == str(clip_id):
                    return True
        return False

    for id in prompt:
        node = prompt[id]
        if "Sampler" in node["class_type"] or "sampler" in node["class_type"] or "Sampling" in node["class_type"]:
            # Check if this KSampler node directly or indirectly references the given CLIPTextEncode node
            if check_link_to_clip(id, clip_id, None, node):
                steps = node["inputs"]["steps"] if "steps" in node["inputs"] else 1
                return steps
    return 1

def find_wildcards_seed(clip_id, text, prompt):
    """ Find easy wildcards seed value"""
    def find_link_clip_id(id, seed, wildcard_id):
        node = prompt[id]
        if "positive" in node['inputs']:
            link_ids = node["inputs"]["positive"]
            if type(link_ids) == list:
                for id in link_ids:
                    if id != 0:
                        if id == wildcard_id:
                            wildcard_node = prompt[wildcard_id]
                            seed = wildcard_node["inputs"]["seed"] if "seed" in wildcard_node["inputs"] else None
                            if seed is None:
                                seed = wildcard_node["inputs"]["seed_num"] if "seed_num" in wildcard_node["inputs"] else None
                            return seed
                        else:
                            return find_link_clip_id(id, seed, wildcard_id)
            else:
                return None
        else:
            return None
    if "__" in text:
        seed = None
        for id in prompt:
            node = prompt[id]
            if "wildcards" in node["class_type"]:
                wildcard_id = id
                return find_link_clip_id(str(clip_id), seed, wildcard_id)
        return seed
    else:
        return None

def is_linked_styles_selector(prompt, unique_id, prompt_type='positive'):
    unique_id = unique_id.split('.')[len(unique_id.split('.')) - 1] if "." in unique_id else unique_id
    inputs_values = prompt[unique_id]['inputs'][prompt_type] if prompt_type in prompt[unique_id][
        'inputs'] else None
    if type(inputs_values) == list and inputs_values != 'undefined' and inputs_values[0]:
        return True if prompt[inputs_values[0]] and prompt[inputs_values[0]]['class_type'] == 'easy stylesSelector' else False
    else:
        return False

use_mirror = False
def get_local_filepath(url, dirname, local_file_name=None):
    """Get local file path when is already downloaded or download it"""
    import os
    from server import PromptServer
    from urllib.parse import urlparse
    from torch.hub import download_url_to_file
    global use_mirror
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    if not local_file_name:
        parsed_url = urlparse(url)
        local_file_name = os.path.basename(parsed_url.path)
    destination = os.path.join(dirname, local_file_name)
    if not os.path.exists(destination):
        try:
            if use_mirror:
                url = url.replace('huggingface.co', 'hf-mirror.com')
            print(f'downloading {url} to {destination}')
            PromptServer.instance.send_sync("easyuse-toast", {'content': f'Downloading model to {destination}, please wait...', 'duration': 10000})
            download_url_to_file(url, destination)
        except Exception as e:
            use_mirror = True
            url = url.replace('huggingface.co', 'hf-mirror.com')
            print(f'Unable to download from huggingface, trying mirror: {url}')
            PromptServer.instance.send_sync("easyuse-toast", {'content': f'Unable to connect to huggingface, trying mirror: {url}', 'duration': 10000})
            try:
                download_url_to_file(url, destination)
            except Exception as err:
                error_msg = str(err.args[0]) if err.args else str(err)
                PromptServer.instance.send_sync("easyuse-toast",
                                                {'content': f'Unable to download model from {url}', 'type':'error'})
                raise Exception(f'Download failed. Original URL and mirror both failed.\nError: {error_msg}')
    return destination

def to_lora_patch_dict(state_dict: dict) -> dict:
    """ Convert raw lora state_dict to patch_dict that can be applied on
    modelpatcher."""
    patch_dict = {}
    for k, w in state_dict.items():
        model_key, patch_type, weight_index = k.split('::')
        if model_key not in patch_dict:
            patch_dict[model_key] = {}
        if patch_type not in patch_dict[model_key]:
            patch_dict[model_key][patch_type] = [None] * 16
        patch_dict[model_key][patch_type][int(weight_index)] = w

    patch_flat = {}
    for model_key, v in patch_dict.items():
        for patch_type, weight_list in v.items():
            patch_flat[model_key] = (patch_type, weight_list)

    return patch_flat

def easySave(images, filename_prefix, output_type, prompt=None, extra_pnginfo=None):
    """Save or Preview Image"""
    from nodes import PreviewImage, SaveImage
    if output_type in ["Hide", "None"]:
        return list()
    elif output_type in ["Preview", "Preview&Choose"]:
        filename_prefix = 'easyPreview'
        results = PreviewImage().save_images(images, filename_prefix, prompt, extra_pnginfo)
        return results['ui']['images']
    else:
        results = SaveImage().save_images(images, filename_prefix, prompt, extra_pnginfo)
        return results['ui']['images']

def getMetadata(filepath):
    with open(filepath, "rb") as file:
        # https://github.com/huggingface/safetensors#format
        # 8 bytes: N, an unsigned little-endian 64-bit integer, containing the size of the header
        header_size = int.from_bytes(file.read(8), "little", signed=False)

        if header_size <= 0:
            raise BufferError("Invalid header size")

        header = file.read(header_size)
        if header_size <= 0:
            raise BufferError("Invalid header")

        return header

def cleanGPUUsedForce():
    gc.collect()
    mm.unload_all_models()
    mm.soft_empty_cache()
