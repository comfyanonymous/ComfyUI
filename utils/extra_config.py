import os
import yaml
import folder_paths
import logging

def load_extra_path_config(yaml_path):
    with open(yaml_path, 'r') as stream:
        config = yaml.safe_load(stream)
    for c in config:
        conf = config[c]
        if conf is None:
            continue
        base_path = None
        if "base_path" in conf:
            base_path = conf.pop("base_path")
            base_path = os.path.expandvars(os.path.expanduser(base_path))
        is_default = False
        if "is_default" in conf:
            is_default = conf.pop("is_default")
        for x in conf:
            for y in conf[x].split("\n"):
                if len(y) == 0:
                    continue
                full_path = y
                if base_path is not None:
                    full_path = os.path.join(base_path, full_path)
                logging.info("Adding extra search path {} {}".format(x, full_path))
                folder_paths.add_model_folder_path(x, full_path, is_default)
