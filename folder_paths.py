import os
from collections import namedtuple

supported_ckpt_extensions = set(['.ckpt', '.pth'])
supported_pt_extensions = set(['.ckpt', '.pt', '.bin', '.pth'])
try:
    import safetensors.torch
    supported_ckpt_extensions.add('.safetensors')
    supported_pt_extensions.add('.safetensors')
except:
    print("Could not import safetensors, safetensors support disabled.")


SearchLocation = namedtuple('SearchLocation', ['paths', 'extensions'])
model_search_locations = {}


def register_model_type(model_tag, supported_extensions):
    if model_tag not in model_search_locations:
        model_search_locations[model_tag] = SearchLocation([], supported_extensions)


def add_model_search_path(model_tag, full_folder_path):
    global model_search_locations
    if model_tag in model_search_locations:
        model_search_locations[model_tag].paths.append(full_folder_path)


def get_folder_paths(model_tag):
    return model_search_locations[model_tag][0][:]


def recursive_search(directory):
    result = []
    for root, subdir, file in os.walk(directory, followlinks=True):
        for filepath in file:
            #we os.path,join directory with a blank string to generate a path separator at the end.
            result.append(os.path.join(root, filepath).replace(os.path.join(directory,''),''))
    return result


def filter_files_extensions(files, extensions):
    return sorted(list(filter(lambda a: os.path.splitext(a)[-1].lower() in extensions, files)))


def get_full_path(model_tag, filename):
    global model_search_locations
    folders = model_search_locations[model_tag]
    for x in folders[0]:
        full_path = os.path.join(x, filename)
        if os.path.isfile(full_path):
            return full_path


def get_filename_list(model_tag):
    global model_search_locations
    output_list = set()
    folders = model_search_locations[model_tag]
    for x in folders[0]:
        output_list.update(filter_files_extensions(recursive_search(x), folders[1]))
    return sorted(list(output_list))


# default model types
register_model_type('ckpt', supported_ckpt_extensions)
register_model_type('config', ['.yaml'])
register_model_type('lora', supported_pt_extensions)
register_model_type('vae', supported_pt_extensions)
register_model_type('clip', supported_pt_extensions)
register_model_type('clip_vision', supported_pt_extensions)
register_model_type('style_model', supported_pt_extensions)
register_model_type('embedding', supported_pt_extensions)
register_model_type('controlnet', supported_pt_extensions)
register_model_type('upscale_model', supported_pt_extensions)

# default search paths
models_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")
add_model_search_path('ckpt', os.path.join(models_dir, "checkpoints"))
add_model_search_path('config', os.path.join(models_dir, "configs"))
add_model_search_path('lora', os.path.join(models_dir, "loras"))
add_model_search_path('vae', os.path.join(models_dir, "vae"))
add_model_search_path('clip', os.path.join(models_dir, "clip"))
add_model_search_path('clip_vision', os.path.join(models_dir, "clip_vision"))
add_model_search_path('style_model', os.path.join(models_dir, "style_models"))
add_model_search_path('embedding', os.path.join(models_dir, "embeddings"))
add_model_search_path('controlnet', os.path.join(models_dir, "controlnet"))
add_model_search_path('upscale_model', os.path.join(models_dir, "upscale_models"))
