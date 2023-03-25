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
from setuptools import setup, find_packages

package_name = "comfyui"
version = '0.0.1'


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


_amd_torch_index = "https://download.pytorch.org/whl/rocm5.4.2"
_nvidia_torch_index = "https://download.pytorch.org/whl/cu117"
_alternative_indices = [_amd_torch_index, _nvidia_torch_index]


def dependencies() -> [str]:
    _dependencies = open(os.path.join(os.path.dirname(__file__), "requirements.txt")).readlines()

    session = PipSession()

    index_urls = ['https://pypi.org/simple']
    # prefer nvidia over AMD because AM5/iGPU systems will have a valid ROCm device
    if _is_nvidia():
        index_urls += [_nvidia_torch_index]
        _dependencies += ["xformers==0.0.16"]
    elif _is_amd():
        index_urls += [_amd_torch_index]

    if len(index_urls) == 1:
        return _dependencies
    finder = PackageFinder.create(LinkCollector(session, SearchScope([], index_urls, no_index=False)),
                                  SelectionPreferences(allow_yanked=False, prefer_binary=False))

    for i, package in enumerate(_dependencies[:]):
        requirement = InstallRequirement(Requirement(package), comes_from=f"{package_name}=={version}")
        candidate = finder.find_best_candidate(requirement.name, requirement.specifier)
        if any([url in candidate.best_candidate.link.url for url in _alternative_indices]):
            _dependencies[i] = f"{requirement.name} @ {candidate.best_candidate.link.url}"
    return _dependencies


setup(
    # "comfyui"
    name=package_name,
    description="",
    author="",
    version=version,
    python_requires=">=3.9,<3.11",
    packages=find_packages(include=['comfy', 'comfy_extras']),
    install_requires=dependencies(),
)
