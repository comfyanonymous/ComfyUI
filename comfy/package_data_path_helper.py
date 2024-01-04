from importlib.resources import path
import os


def get_editable_resource_path(caller_file, *package_path):
    filename = os.path.join(os.path.dirname(os.path.realpath(caller_file)), package_path[-1])
    if not os.path.exists(filename):
        filename = path(*package_path)
    return filename
