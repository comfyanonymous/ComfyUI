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
        base_paths = []
        if "base_path" in conf:
            base_path = conf.pop("base_path").strip()
            base_path = os.path.expandvars(os.path.expanduser(base_path))
            if len(base_path) > 0:
                base_paths.append(base_path)
        if "base_paths" in conf:
            base_paths_str = conf.pop("base_paths").strip()
            for bp in base_paths_str.split("\n"):
                bp = os.path.expanduser(os.path.expandvars(bp))
                if bp.find("$") >= 0:
                    logging.warning(f"Skipping base path {bp} as it contains an undefined variable")
                    continue
                if len(bp) > 0:
                    base_paths.append(bp)

        # simplify the logic below. os.path.join("", foo) == foo
        if len(base_paths) == 0:
            base_paths.append("")

        # allow overriding paths that are not normally overriden.
        # these can only be overriden to just one path (not a set).
        # the first one that exists wins.
        override_list = [ "models_dir", "output_directory", "temp_directory", "input_directory", "user_directory" ]
        for okey in override_list:
            if okey in conf:
                opath = conf.pop(okey).strip()
                is_absolute = False
                if opath[0] == '@':
                    opath = opath[1:]
                    is_absolute = True
                opath = os.path.expanduser(os.path.expandvars(opath))
                if is_absolute:
                    setattr(folder_paths, okey, opath)
                    logging.info(f"Set {okey} tp {opath}")
                    continue
                for bp in base_paths:
                    path = os.path.join(bp, opath)
                    if os.path.isdir(path):
                        setattr(folder_paths, okey, path)
                        logging.info(f"Set {okey} tp {path}")
                        break

        is_default = False
        if "is_default" in conf:
            is_default = conf.pop("is_default")

        for x in conf:
            for y in conf[x].split("\n"):
                if len(y.strip()) == 0:
                    continue
                is_absolute = False
                if y[0] == '@':
                    y = y[1:]
                    is_absolute = True
                path = os.path.expanduser(os.path.expandvars(y))
                if path.find("$") >= 0:
                    logging.warning(f"Skipping path {path} for {x} as it contains an undefined variable")
                    continue
                if is_absolute:
                    all_paths = [path]
                else:
                    all_paths = [os.path.join(bp, path) for bp in base_paths]

                # pull out only unique elements in order. Do this so that we don't add the same dir twice,
                # if for example two undefined variables are used, e.g. "$X/foo" and "$Y/foo",
                # both will evaluate to "/foo" if neither is set
                all_paths = [x for i, x in enumerate(all_paths) if x not in all_paths[:i]]

                # custom_nodes is special; it's the root directory of custom nodes, and Comfy calls listdir
                # on it assuming it exists which throws otherwise
                if x == "custom_nodes":
                    all_paths = [x for x in all_paths if os.path.isdir(x)]

                if is_default:
                    # if we're using is_default, the add_model_folder_path inserts at 0.
                    # so reverse this list, so that the (original) first element that the user
                    # specifies in the extra paths file actually ends up at 0
                    all_paths.reverse()
                for p in all_paths:
                    logging.info("Adding extra search path {} {}".format(x, p))
                    folder_paths.add_model_folder_path(x, p, is_default)

