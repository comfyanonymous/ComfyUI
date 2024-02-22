#!/usr/bin/env python3
# this script does a little housekeeping for your platform
import os.path
import platform
import subprocess
import sys
from typing import List, Literal, Union, Optional

from pip._internal.index.collector import LinkCollector
from pip._internal.index.package_finder import PackageFinder
from pip._internal.models.search_scope import SearchScope
from pip._internal.models.selection_prefs import SelectionPreferences
from pip._internal.network.session import PipSession
from pip._internal.req import InstallRequirement
from pip._vendor.packaging.requirements import Requirement
from setuptools import setup, find_packages

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
amd_torch_index = ("https://download.pytorch.org/whl/rocm5.7", "https://download.pytorch.org/whl/nightly/rocm6.0")

"""
The package index to torch built with CUDA.
Observe the CUDA version is in this URL.
"""
nvidia_torch_index = ("https://download.pytorch.org/whl/cu121", "https://download.pytorch.org/whl/nightly/cu121")

"""
The package index to torch built against CPU features.
"""
cpu_torch_index = ("https://download.pytorch.org/whl/cpu", "https://download.pytorch.org/whl/nightly/cpu")

# xformers not required for new torch

"""
Indicates if this is installing an editable (develop) mode package
"""
is_editable = '--editable' in sys.argv or '-e' in sys.argv or (
        'python' in sys.argv and 'setup.py' in sys.argv and 'develop' in sys.argv)

# If we're installing with no build isolation, we can check if torch is already installed in the environment, and if so,
# go ahead and use the version that is already installed.
is_build_isolated_and_torch_version: Optional[str]
try:
    import torch
    print(f"comfyui setup.py: torch version was {torch.__version__} and built without build isolation, using this torch instead of upgrading", file=sys.stderr)
    is_build_isolated_and_torch_version = torch.__version__
except Exception as e:
    is_build_isolated_and_torch_version = None

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


def _is_linux_arm64():
    os_name = platform.system()
    architecture = platform.machine()

    return os_name == 'Linux' and architecture == 'aarch64'


def dependencies() -> List[str]:
    _dependencies = open(os.path.join(os.path.dirname(__file__), "requirements.txt")).readlines()
    # torch is already installed, and we could have only known this if the user specifically requested a
    # no-build-isolation build, so the user knows what is going on
    if is_build_isolated_and_torch_version is not None:
        for i, dep in enumerate(_dependencies):
            stripped = dep.strip()
            if stripped == "torch":
                _dependencies[i] = f"{stripped}=={is_build_isolated_and_torch_version}"
                break
        return _dependencies
    _alternative_indices = [amd_torch_index, nvidia_torch_index]
    session = PipSession()

    # (stable, nightly) tuple
    index_urls = [('https://pypi.org/simple', 'https://pypi.org/simple')]
    # prefer nvidia over AMD because AM5/iGPU systems will have a valid ROCm device
    if _is_nvidia():
        index_urls += [nvidia_torch_index]
    elif _is_amd():
        index_urls += [amd_torch_index]
    else:
        index_urls += [cpu_torch_index]

    if len(index_urls) == 1:
        return _dependencies

    if sys.version_info >= (3, 12):
        # use the nightlies
        index_urls_selected = [nightly for (_, nightly) in index_urls]
        _alternative_indices_selected = [nightly for (_, nightly) in _alternative_indices]
    else:
        index_urls_selected = [stable for (stable, _) in index_urls]
        _alternative_indices_selected = [stable for (stable, _) in _alternative_indices]
    try:
        # pip 23
        finder = PackageFinder.create(LinkCollector(session, SearchScope([], index_urls_selected, no_index=False)),
                                      SelectionPreferences(allow_yanked=False, prefer_binary=False,
                                                           allow_all_prereleases=True))
    except:
        try:
            # pip 22
            finder = PackageFinder.create(LinkCollector(session, SearchScope([], index_urls)),  # type: ignore
                                          SelectionPreferences(allow_yanked=False, prefer_binary=False,
                                                               allow_all_prereleases=True)
                                          , use_deprecated_html5lib=False)
        except:
            raise Exception("upgrade pip with\npython -m pip install -U pip")
    for i, package in enumerate(_dependencies[:]):
        requirement = InstallRequirement(Requirement(package), comes_from=f"{package_name}=={version}")
        candidate = finder.find_best_candidate(requirement.name, requirement.specifier)
        if candidate.best_candidate is not None:
            if any([url in candidate.best_candidate.link.url for url in _alternative_indices_selected]):
                _dependencies[i] = f"{requirement.name} @ {candidate.best_candidate.link.url}"
    return _dependencies


package_data = ['sd1_tokenizer/*', '**/*.json', '**/*.yaml']
if not is_editable:
    package_data.append('comfy/web/**/*')
dev_dependencies = open(os.path.join(os.path.dirname(__file__), "requirements-dev.txt")).readlines()
setup(
    name=package_name,
    description="",
    author="",
    version=version,
    python_requires=">=3.9,<3.13",
    packages=find_packages(exclude=["tests"] + [] if is_editable else ['custom_nodes']),
    package_dir={'': ''},
    include_package_data=True,
    install_requires=dependencies(),
    setup_requires=["pip", "wheel"],
    entry_points={
        'console_scripts': [
            'comfyui-openapi-gen = comfy.cmd.openapi_gen:main',
            'comfyui = comfy.cmd.main:entrypoint',
            'comfyui-worker = comfy.cmd.worker:entrypoint'
        ],
    },
    package_data={
        'comfy': package_data
    },
    tests_require=dev_dependencies,
    extras_require={
        'dev': dev_dependencies
    },
)
