import os.path
import yaml

supported_ckpt_extensions = ['.ckpt']
supported_pt_extensions = ['.ckpt', '.pt', '.bin']
try:
    import safetensors.torch
    supported_ckpt_extensions += ['.safetensors']
    supported_pt_extensions += ['.safetensors']
except:
    print("Could not import safetensors, safetensors support disabled.")


model_kinds = {
    "configs": [".yml"],
    "checkpoints": supported_ckpt_extensions,
    "vae": supported_pt_extensions,
    "clip": supported_pt_extensions,
    "embeddings": supported_pt_extensions,
    "loras": supported_pt_extensions,
}


def recursive_search(directory):
    result = []
    for root, subdir, file in os.walk(directory, followlinks=True):
        for filepath in file:
            #we os.path,join directory with a blank string to generate a path separator at the end.
            result.append(os.path.join(root, filepath).replace(os.path.join(directory,''),''))
    return result

def filter_files_extensions(files, extensions):
    return sorted(list(filter(lambda a: os.path.splitext(a)[-1].lower() in extensions, files)))

def get_files(directories, extensions):
    files = []
    for dir in directories:
        files.extend(recursive_search(dir))
    return filter_files_extensions(files, extensions)

def get_model_paths(kind):
    models_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")
    model_dir = os.path.join(models_dir, kind)
    return [model_dir] + config["paths"][kind]

def get_model_files(kind):
    exts = model_kinds[kind]
    paths = get_model_paths(kind)
    return get_files(paths, exts)

def find_model_file(kind, basename):
    # TODO: find by model hash instead of filename
    for path in get_model_paths(kind):
        file = os.path.join(path, basename)
        if os.path.isfile(file):
            return file
    raise FileNotFoundError("Model not found: " + basename)


config = {}
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)["config"]
