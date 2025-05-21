import torch
import torchvision
import torchaudio
from dataclasses import dataclass

import importlib
if importlib.util.find_spec("torch_directml"):
    from pip._vendor import pkg_resources


class VEnvException(Exception):
    pass


@dataclass
class TorchVersionInfo:
    name: str = None
    version: str = None
    extension: str = None
    is_nightly: bool = False 
    is_cpu: bool = False
    is_cuda: bool = False
    is_xpu: bool = False
    is_rocm: bool = False
    is_directml: bool = False


def get_bootstrap_requirements_string():
    '''
    Get string to insert into a 'pip install' command to get the same torch dependencies as current venv.
    '''
    torch_info = get_torch_info(torch)
    packages = [torchvision, torchaudio]
    infos = [torch_info] + [get_torch_info(x) for x in packages]
    # directml should be first dependency, if exists
    directml_info = get_torch_directml_info()
    if directml_info is not None:
        infos = [directml_info] + infos
    # create list of strings to combine into install string
    install_str_list = []
    for info in infos:
        info_string = f"{info.name}=={info.version}"
        if not info.is_cpu and not info.is_directml:
            info_string = f"{info_string}+{info.extension}"
        install_str_list.append(info_string)
    # handle extra_index_url, if needed
    extra_index_url = get_index_url(torch_info)
    if extra_index_url:
        install_str_list.append(extra_index_url)
    # format nightly install properly
    if torch_info.is_nightly:
        install_str_list = ["--pre"] + install_str_list

    install_str  = " ".join(install_str_list)
    return install_str

def get_index_url(info: TorchVersionInfo=None):
    '''
    Get --extra-index-url (or --index-url) for torch install.
    '''
    if info is None:
        info = get_torch_info()
    # for cpu, don't need any index_url
    if info.is_cpu and not info.is_nightly:
        return None
    # otherwise, format index_url
    base_url = "https://download.pytorch.org/whl/"
    if info.is_nightly:
        base_url = f"--index-url {base_url}nightly/"
    else:
        base_url = f"--extra-index-url {base_url}"
    base_url = f"{base_url}{info.extension}"
    return base_url

def get_torch_info(package=None):
    '''
    Get info about an installed torch-related package.
    '''
    if package is None:
        package = torch
    info = TorchVersionInfo(name=package.__name__)
    info.version = package.__version__
    info.extension = None
    info.is_nightly = False
    # get extension, separate from version
    info.version, info.extension = info.version.split('+', 1)
    if info.extension.startswith('cpu'):
        info.is_cpu = True
    elif info.extension.startswith('cu'):
        info.is_cuda = True
    elif info.extension.startswith('rocm'):
        info.is_rocm = True
    elif info.extension.startswith('xpu'):
        info.is_xpu = True
    # TODO: add checks for some odd pytorch versions, if possible

    # check if nightly install
    if 'dev' in info.version:
        info.is_nightly = True

    return info

def get_torch_directml_info():
    '''
    Get info specifically about torch-directml package.

    Returns None if torch-directml is not installed.
    '''
    # the import string and the pip string are different
    pip_name = "torch-directml"
    # if no torch_directml, do nothing
    if not importlib.util.find_spec("torch_directml"):
        return None
    info = TorchVersionInfo(name=pip_name)
    info.is_directml = True
    for p in pkg_resources.working_set:
        if p.project_name.lower() == pip_name:
            info.version = p.version
    if p.version is None:
        return None
    return info


if __name__ == '__main__':
    print(get_bootstrap_requirements_string())
