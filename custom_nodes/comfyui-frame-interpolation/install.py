import os
from pathlib import Path
import sys
import platform

def get_cuda_ver_from_dir(cuda_home):
    nvrtc = filter(lambda lib_file: "nvrtc-builtins" in lib_file, os.listdir(cuda_home))
    nvrtc = list(nvrtc)
    if len(nvrtc) == 0:
        return
    nvrtc = nvrtc[0]
    if ('102' in nvrtc) or ('10.2' in nvrtc):
        return '102'
    if '110' in nvrtc or ('11.0' in nvrtc):
        return '110'
    if '111' in nvrtc or ('11.1' in nvrtc):
        return '111'
    if '11' in nvrtc:
        return '11x'
    if '12' in nvrtc:
        return '12x'

s_param = '-s' if "python_embeded" in sys.executable else '' 

def get_cuda_home_path():
    if "CUDA_HOME" in os.environ:
        return os.environ["CUDA_HOME"]
    import torch
    torch_lib_path = Path(torch.__file__).parent / "lib"
    torch_lib_path = str(torch_lib_path.resolve())
    if os.path.exists(torch_lib_path):
        nvrtc = filter(lambda lib_file: "nvrtc-builtins" in lib_file, os.listdir(torch_lib_path))
        nvrtc = list(nvrtc)
        return torch_lib_path if len(nvrtc) > 0 else None

def install_cupy():
    cuda_home = get_cuda_home_path()
    try:
        if cuda_home is not None:
            os.environ["CUDA_HOME"] = cuda_home
            os.environ["CUDA_PATH"] = cuda_home
        import cupy
        print("CuPy is already installed.")
    except:
        print("Uninstall cupy if existed...")
        os.system(f'"{sys.executable}" {s_param} -m pip uninstall -y cupy-wheel cupy-cuda102 cupy-cuda110 cupy-cuda111 cupy-cuda11x cupy-cuda12x')
        print("Installing cupy...")
        cuda_ver = get_cuda_ver_from_dir(cuda_home)
        cupy_package = f"cupy-cuda{cuda_ver}" if cuda_ver is not None else "cupy-wheel"
        os.system(f'"{sys.executable}" {s_param} -m pip install {cupy_package}')

with open(Path(__file__).parent / "requirements-no-cupy.txt", 'r') as f:
    for package in f.readlines():
        package = package.strip()
        print(f"Installing {package}...")
        os.system(f'"{sys.executable}" {s_param} -m pip install {package}')

print("Checking cupy...")
install_cupy()