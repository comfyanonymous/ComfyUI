#!/usr/bin/env python3
# this script does a little housekeeping for your platform
import os.path
import platform
import subprocess
import sys
from typing import List, Optional

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
version = "0.3.15"

"""
The package index to the torch built with AMD ROCm.
"""
amd_torch_index = ("https://download.pytorch.org/whl/rocm6.2", "https://download.pytorch.org/whl/nightly/rocm6.2.4")

"""
The package index to torch built with CUDA.
Observe the CUDA version is in this URL.
"""
nvidia_torch_index = ("https://download.pytorch.org/whl/cu124", "https://download.pytorch.org/whl/nightly/cu126")

"""
The package index to torch built against CPU features.
"""
cpu_torch_index = ("https://download.pytorch.org/whl/cpu", "https://download.pytorch.org/whl/nightly/cpu")

"""
Indicates if this is installing an editable (develop) mode package
"""
is_editable = '--editable' in sys.argv or '-e' in sys.argv or (
        'python' in sys.argv and 'setup.py' in sys.argv and 'develop' in sys.argv)


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
        output = None
        try:
            output = subprocess.check_output([rocminfo_path]).decode("utf-8")
        except:
            pass

        if output is None:
            return False
        elif "Device" in output:
            return True
        elif "Permission Denied" in output:
            msg = f"""
{output}

To resolve this issue on AMD:

sudo -i
usermod -a -G video $LOGNAME
usermod -a -G render $LOGNAME

You will need to reboot. Save your work, then:

reboot

"""
            print(msg, file=sys.stderr)
            raise RuntimeError(msg)
    return False


def _is_linux_arm64():
    os_name = platform.system()
    architecture = platform.machine()

    return os_name == 'Linux' and architecture == 'aarch64'


def dependencies(install_torch_for_system=False, force_nightly: bool = False) -> List[str]:
    _dependencies = open(os.path.join(os.path.dirname(__file__), "requirements.txt")).readlines()
    if not install_torch_for_system:
        return [dep for dep in _dependencies if "@" not in dep]
    # If we're installing with no build isolation, we can check if torch is already installed in the environment, and if
    # so, go ahead and use the version that is already installed.
    existing_torch: Optional[str]
    try:
        import torch
        print(f"comfyui setup.py: torch version was {torch.__version__} and built without build isolation, using this torch instead of upgrading", file=sys.stderr)
        existing_torch = torch.__version__
    except Exception:
        existing_torch = None

    if existing_torch is not None:
        for i, dep in enumerate(_dependencies):
            stripped = dep.strip()
            if stripped == "torch":
                _dependencies[i] = f"{stripped}=={existing_torch}"
                break
        return _dependencies
    # some torch packages redirect to https://download.pytorch.org/whl/
    _alternative_indices = [amd_torch_index, nvidia_torch_index, ("https://download.pytorch.org/whl/", "https://download.pytorch.org/whl/")]
    session = PipSession()

    # (stable, nightly) tuple
    index_urls = [('https://pypi.org/simple', 'https://pypi.org/simple')]
    # prefer nvidia over AMD because AM5/iGPU systems will have a valid ROCm device
    if _is_nvidia():
        index_urls = [nvidia_torch_index] + index_urls
    elif _is_amd():
        index_urls = [amd_torch_index] + index_urls
        _dependencies += ["pytorch-triton-rocm"]
    else:
        index_urls += [cpu_torch_index]

    if len(index_urls) == 1:
        return _dependencies

    if sys.version_info >= (3, 13) or force_nightly:
        # use the nightlies for python 3.13
        print("Using nightlies for Python 3.13 or higher. PyTorch may not yet build for it", file=sys.stderr)
        index_urls_selected = [nightly for (_, nightly) in index_urls]
        _alternative_indices_selected = [nightly for (_, nightly) in _alternative_indices]
    else:
        index_urls_selected = [stable for (stable, _) in index_urls]
        _alternative_indices_selected = [stable for (stable, _) in _alternative_indices]
    try:
        # pip 23, 24
        finder = PackageFinder.create(LinkCollector(session, SearchScope([], index_urls_selected, no_index=False)),
                                      SelectionPreferences(allow_yanked=False, prefer_binary=False,
                                                           allow_all_prereleases=True))
    except:
        try:
            # pip 22
            finder = PackageFinder.create(LinkCollector(session, SearchScope([], index_urls_selected)),  # type: ignore
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


package_data = [
    '**/*'
]
dev_dependencies = open(os.path.join(os.path.dirname(__file__), "requirements-dev.txt")).readlines()
triton_dependencies = open(os.path.join(os.path.dirname(__file__), "requirements-triton.txt")).readlines()
setup(
    name=package_name,
    description="An installable version of ComfyUI",
    author="Contributors_of_ComfyUI",
    version=version,
    python_requires=">=3.10,<3.13",
    packages=find_packages(exclude=["tests"] + [] if is_editable else ['custom_nodes']),
    install_requires=dependencies(install_torch_for_system=False),
    setup_requires=["pip", "wheel"],
    entry_points={
        'console_scripts': [
            'comfyui = comfy.cmd.main:entrypoint',
            'comfyui-worker = comfy.cmd.worker:entrypoint'
        ],
    },
    package_data={
        '': package_data
    },
    tests_require=dev_dependencies,
    extras_require={
        'withtorch': dependencies(install_torch_for_system=True),
        'withtorchnightly': dependencies(install_torch_for_system=True, force_nightly=True),
        'withtriton': dependencies(install_torch_for_system=True) + triton_dependencies,
        'dev': dev_dependencies
    },
)
