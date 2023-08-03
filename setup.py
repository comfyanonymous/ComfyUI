#!/usr/bin/env python3
# this script does a little housekeeping for your platform
import os.path
import platform
import subprocess

from pip._internal.index.collector import LinkCollector
from pip._internal.index.package_finder import PackageFinder
from pip._internal.models.search_scope import SearchScope
from pip._internal.models.selection_prefs import SelectionPreferences
from pip._internal.network.session import PipSession
from pip._internal.req import InstallRequirement
from pip._vendor.packaging.requirements import Requirement
from setuptools import setup, find_packages, find_namespace_packages

"""
The name of the package.
"""
package_name = "comfyui"

"""
The current version.
"""
version = '0.0.1'

"""
The package index to the torch built with AMD ROCm.
"""
amd_torch_index = "https://download.pytorch.org/whl/rocm5.4.2"

"""
The package index to torch built with CUDA.
Observe the CUDA version is in this URL.
"""
nvidia_torch_index = "https://download.pytorch.org/whl/cu118"

"""
The package index to torch built against CPU features.
This includes macOS MPS support.
"""
cpu_torch_index_nightlies = "https://download.pytorch.org/whl/nightly/cpu"

# xformers not required for new torch

"""
Packages that should have a specific option set when a GPU accelerator is present
"""
gpu_accelerated_packages = {"rembg": "rembg[gpu]"}


def _is_nvidia() -> bool:
    system = platform.system().lower()
    nvidia_smi_paths = []

    if system == "windows":
        nvidia_smi_paths.append(os.path.join(os.environ.get("SystemRoot", ""), "System32", "nvidia-smi.exe"))
    elif system == "linux":
        nvidia_smi_paths.extend(["/usr/bin/nvidia-smi", "/opt/nvidia/bin/nvidia-smi"])

    for nvidia_smi_path in nvidia_smi_paths:
        try:
            output = subprocess.check_output([nvidia_smi_path, "-L"]).decode("utf-8")

            if "GPU" in output:
                return True
        except:
            pass

    return False


def _is_amd() -> bool:
    system = platform.system().lower()
    rocminfo_paths = []

    # todo: torch windows doesn't support amd
    if system == "windows":
        rocminfo_paths.append(os.path.join(os.environ.get("ProgramFiles", ""), "AMD", "ROCm", "bin", "rocminfo.exe"))
    elif system == "linux":
        rocminfo_paths.extend(["/opt/rocm/bin/rocminfo", "/usr/bin/rocminfo"])

    for rocminfo_path in rocminfo_paths:
        try:
            output = subprocess.check_output([rocminfo_path]).decode("utf-8")

            if "Device" in output:
                return True
        except:
            pass

    return False


def dependencies() -> [str]:
    _dependencies = open(os.path.join(os.path.dirname(__file__), "requirements.txt")).readlines()
    # todo: also add all plugin dependencies
    _alternative_indices = [amd_torch_index, nvidia_torch_index, cpu_torch_index_nightlies]
    session = PipSession()

    gpu_accelerated = False
    index_urls = ['https://pypi.org/simple']
    # prefer nvidia over AMD because AM5/iGPU systems will have a valid ROCm device
    if _is_nvidia():
        index_urls += [nvidia_torch_index]
        gpu_accelerated = True
    elif _is_amd():
        index_urls += [amd_torch_index]
        gpu_accelerated = True
    else:
        index_urls += [cpu_torch_index_nightlies]

    if len(index_urls) == 1:
        return _dependencies

    try:
        # pip 23
        finder = PackageFinder.create(LinkCollector(session, SearchScope([], index_urls, no_index=False)),
                                      SelectionPreferences(allow_yanked=False, prefer_binary=False,
                                                           allow_all_prereleases=True))
    except:
        try:
            # pip 22
            finder = PackageFinder.create(LinkCollector(session, SearchScope([], index_urls)),
                                          SelectionPreferences(allow_yanked=False, prefer_binary=False,
                                                               allow_all_prereleases=True)
                                          , use_deprecated_html5lib=False)
        except:
            raise Exception("upgrade pip with\npip install -U pip")
    for i, package in enumerate(_dependencies[:]):
        requirement = InstallRequirement(Requirement(package), comes_from=f"{package_name}=={version}")
        candidate = finder.find_best_candidate(requirement.name, requirement.specifier)
        if candidate.best_candidate is not None:
            if gpu_accelerated and requirement.name in gpu_accelerated_packages:
                _dependencies[i] = gpu_accelerated_packages[requirement.name]
            if any([url in candidate.best_candidate.link.url for url in _alternative_indices]):
                _dependencies[i] = f"{requirement.name} @ {candidate.best_candidate.link.url}"
    return _dependencies


setup(
    # "comfyui"
    name=package_name,
    description="",
    author="",
    version=version,
    python_requires=">=3.9,<3.12",
    # todo: figure out how to include the web directory to eventually let main live inside the package
    # todo: see https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/ for more about adding plugins
    packages=find_packages(where="comfy") + find_packages(where="comfy_extras"),
    install_requires=dependencies(),
    setup_requires=["pip", "wheel"],
    entry_points={
        'console_scripts': [
            'comfyui-openapi-gen = comfy.cmd.openapi_gen:main',
            'comfyui = comfy.cmd.main:main'
        ],
    },
)
