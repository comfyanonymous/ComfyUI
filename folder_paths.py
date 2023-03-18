import os

supported_ckpt_extensions = set(['.ckpt', '.pth'])
supported_pt_extensions = set(['.ckpt', '.pt', '.bin', '.pth'])
try:
    import safetensors.torch
    supported_ckpt_extensions.add('.safetensors')
    supported_pt_extensions.add('.safetensors')
except:
    print("Could not import safetensors, safetensors support disabled.")


folder_names_and_paths = {}


models_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")
folder_names_and_paths["checkpoints"] = ([os.path.join(models_dir, "checkpoints")], supported_ckpt_extensions)
folder_names_and_paths["configs"] = ([os.path.join(models_dir, "configs")], [".yaml"])

folder_names_and_paths["loras"] = ([os.path.join(models_dir, "loras")], supported_pt_extensions)
folder_names_and_paths["vae"] = ([os.path.join(models_dir, "vae")], supported_pt_extensions)
folder_names_and_paths["clip"] = ([os.path.join(models_dir, "clip")], supported_pt_extensions)
folder_names_and_paths["clip_vision"] = ([os.path.join(models_dir, "clip_vision")], supported_pt_extensions)
folder_names_and_paths["style_models"] = ([os.path.join(models_dir, "style_models")], supported_pt_extensions)
# folder_names_and_paths["embeddings"] = ([os.path.join(models_dir, "embeddings")], supported_pt_extensions)

folder_names_and_paths["controlnet"] = ([os.path.join(models_dir, "controlnet"), os.path.join(models_dir, "t2i_adapter")], supported_pt_extensions)
folder_names_and_paths["upscale_models"] = ([os.path.join(models_dir, "upscale_models")], supported_pt_extensions)


def add_model_folder_path(folder_name, full_folder_path):
    global folder_names_and_paths
    if folder_name in folder_names_and_paths:
        folder_names_and_paths[folder_name][0].append(full_folder_path)


def recursive_search(directory):
    result = []
    for root, subdir, file in os.walk(directory, followlinks=True):
        for filepath in file:
            #we os.path,join directory with a blank string to generate a path separator at the end.
            result.append(os.path.join(root, filepath).replace(os.path.join(directory,''),''))
    return result

def filter_files_extensions(files, extensions):
    return sorted(list(filter(lambda a: os.path.splitext(a)[-1].lower() in extensions, files)))



def get_full_path(folder_name, filename):
    global folder_names_and_paths
    folders = folder_names_and_paths[folder_name]
    for x in folders[0]:
        full_path = os.path.join(x, filename)
        if os.path.isfile(full_path):
            return full_path


def get_filename_list(folder_name):
    global folder_names_and_paths
    output_list = set()
    folders = folder_names_and_paths[folder_name]
    for x in folders[0]:
        output_list.update(filter_files_extensions(recursive_search(x), folders[1]))
    return sorted(list(output_list))


