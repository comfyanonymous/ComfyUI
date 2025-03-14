# hide rocm and hip
import os
os.environ.pop("ROCM_HOME",None)
os.environ.pop("HIP_HOME",None)
os.environ.pop("ROCM_VERSION",None)
paths=os.environ["PATH"].split(";")
paths_no_rocm=[]
for path_ in paths:
    if "rocm" not in path_.lower():
        paths_no_rocm.append(path_)
os.environ["PATH"]=";".join(paths_no_rocm)
# hide rocm and hip end

# fix cublast errors for newer zluda versions "CUDA error: CUBLAS_STATUS_NOT_SUPPORTED when calling `cublasLtMatmulAlgoGetHeuristic" , comment it out if you have a working hipblast setup.
os.environ['DISABLE_ADDMM_CUDA_LT'] = '1'
        
import torch

# get package version using importlib.metadata
def get_package_version(package_name):
    try:
        # Try using importlib.metadata (Python 3.8+)
        from importlib.metadata import version
        return version(package_name)
    except ImportError:
        # Fallback to importlib_metadata for older Python versions
        from importlib_metadata import version
        return version(package_name)

# Check and install comfyui-frontend-package if not installed or if the version is lower than required
required_version = "1.12.11"
package_name = "comfyui-frontend-package"

try:
    installed_version = get_package_version(package_name)
    print(f"Installed version of {package_name}: {installed_version}")
    
    # Compare versions
    from packaging import version
    if version.parse(installed_version) < version.parse(required_version):
        import subprocess
        import sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', f'{package_name}=={required_version}', '--quiet', '--upgrade'])
        print(" ")
        print(f"Comfyui Frontend Package version {installed_version} is outdated, updating to latest recommended version {required_version}.")
except Exception as e:
    # If the package is not installed or version check fails, install it
    import subprocess
    import sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', f'{package_name}=={required_version}', '--quiet'])
    print(" ")
    print("Comfyui Frontend Package missing, it is installed. (one time only) ")
#audio patch
import torch._dynamo

if torch.cuda.is_available() and torch.cuda.get_device_name().endswith("[ZLUDA]"):
    _torch_stft = torch.stft
    _torch_istft = torch.istft

    def z_stft(input: torch.Tensor, window: torch.Tensor, *args, **kwargs):
        return _torch_stft(input=input.cpu(), window=window.cpu(), *args, **kwargs).to(input.device)

    def z_istft(input: torch.Tensor, window: torch.Tensor, *args, **kwargs):
        return _torch_istft(input=input.cpu(), window=window.cpu(), *args, **kwargs).to(input.device)

    def z_jit(f, *_, **__):
        f.graph = torch._C.Graph()
        return f
    
    torch._dynamo.config.suppress_errors = True
    torch.stft = z_stft
    torch.istft = z_istft
    torch.jit.script = z_jit
#audio patch end
    
# zluda
if torch.cuda.get_device_name().endswith("[ZLUDA]"):
    print(" ")
    print("***----------------------ZLUDA-----------------------------***")
    print("  ::  ZLUDA detected, disabling non-supported functions.      ")
    torch.backends.cudnn.enabled = False
    print("  ::  CuDNN, flash_sdp, mem_efficient_sdp disabled).          ")
    torch.backends.cuda.enable_flash_sdp(False) # enable if using 6.2 with latest nightly zluda
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    print("***--------------------------------------------------------***")
    print("  ::  Device : ", torch.cuda.get_device_name())
    print(" ")
else:
    print("  ::  ZLUDA isn't detected, please try patching it.")
