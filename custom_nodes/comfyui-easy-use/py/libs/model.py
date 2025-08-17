import json
import os
import folder_paths
import server
from .utils import find_tags

class easyModelManager:

    def __init__(self):
        self.img_suffixes = [".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".svg", ".tif", ".tiff"]
        self.default_suffixes = [".ckpt", ".pt", ".bin", ".pth", ".safetensors"]
        self.models_config = {
            "checkpoints": {"suffix": self.default_suffixes},
            "loras": {"suffix": self.default_suffixes},
            "unet": {"suffix": self.default_suffixes},
        }
        self.model_lists = {}

    def find_thumbnail(self, model_type, name):
        file_no_ext = os.path.splitext(name)[0]
        for ext in self.img_suffixes:
            full_path = folder_paths.get_full_path(model_type, file_no_ext + ext)
            if os.path.isfile(str(full_path)):
                return full_path
        return None

    def get_model_lists(self, model_type):
        if model_type not in self.models_config:
            return []
        filenames = folder_paths.get_filename_list(model_type)
        model_lists = []
        for name in filenames:
            model_suffix = os.path.splitext(name)[-1]
            if model_suffix not in self.models_config[model_type]["suffix"]:
                continue
            else:
                cfg = {
                    "name": os.path.basename(os.path.splitext(name)[0]),
                    "full_name": name,
                    "remark": '',
                    "file_path": folder_paths.get_full_path(model_type, name),
                    "type": model_type,
                    "suffix": model_suffix,
                    "dir_tags": find_tags(name),
                    "cover": self.find_thumbnail(model_type, name),
                    "metadata": None,
                    "sha256": None
                }
                model_lists.append(cfg)

        return model_lists

    def get_model_info(self, model_type, model_name):
        pass

# if __name__ == "__main__":
#     manager = easyModelManager()
#     print(manager.get_model_lists("checkpoints"))