# ------------------- Hide ROCm/HIP -------------------
import os

os.environ.pop("ROCM_HOME", None)
os.environ.pop("HIP_HOME", None)
os.environ.pop("ROCM_VERSION", None)

paths = os.environ["PATH"].split(";")
paths_no_rocm = [p for p in paths if "rocm" not in p.lower()]
os.environ["PATH"] = ";".join(paths_no_rocm)
# ------------------- End ROCm/HIP Hiding -------------

# Fix for cublasLt errors on newer ZLUDA (if no hipblaslt)
os.environ['DISABLE_ADDMM_CUDA_LT'] = '1'

import torch

# ------------------- ComfyUI Frontend Version Check -------------------
def get_package_version(package_name):
    try:
        from importlib.metadata import version
        return version(package_name)
    except ImportError:
        from importlib_metadata import version
        return version(package_name)

# Check and install/update comfyui-frontend-package
required_frontend_version = "1.16.8"
frontend_package_name = "comfyui-frontend-package"

try:
    installed_version = get_package_version(frontend_package_name)
    print(f"Installed version of {frontend_package_name}: {installed_version}")
    
    from packaging import version
    if version.parse(installed_version) < version.parse(required_frontend_version):
        import subprocess
        import sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', f'{frontend_package_name}=={required_frontend_version}', '--quiet', '--upgrade'])
        print(f"\nComfyUI Frontend Package outdated. Upgraded to {required_frontend_version}.")
except Exception:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', f'{frontend_package_name}=={required_frontend_version}', '--quiet'])
    print("\nComfyUI Frontend Package was missing. Installed it.")

# Check and install/update comfyui-workflow-templates
required_templates_version = "0.1.1"
templates_package_name = "comfyui-workflow-templates"

try:
    installed_version = get_package_version(templates_package_name)
    print(f"Installed version of {templates_package_name}: {installed_version}")
    
    from packaging import version
    if version.parse(installed_version) < version.parse(required_templates_version):
        import subprocess
        import sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', f'{templates_package_name}=={required_templates_version}', '--quiet', '--upgrade'])
        print(f"\nComfyUI Workflow Templates outdated. Upgraded to {required_templates_version}.")
except Exception:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', f'{templates_package_name}=={required_templates_version}', '--quiet'])
    print("\nComfyUI Workflow Templates was missing. Installed it.")
# ------------------- End Frontend Version Check -------------------

# ------------------- ZLUDA Detection -------------------
zluda_device_name = torch.cuda.get_device_name() if torch.cuda.is_available() else ""
is_zluda = zluda_device_name.endswith("[ZLUDA]")
# ------------------- End Detection --------------------

# ------------------- Audio Ops Patch -------------------
if is_zluda:
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
# ------------------- End Audio Patch -------------------

# ------------------- Top-K Fallback Patch -------------------
if is_zluda:
    _topk = torch.topk

    def safe_topk(input: torch.Tensor, *args, **kwargs):
        device = input.device
        values, indices = _topk(input.cpu(), *args, **kwargs)
        return torch.return_types.topk((values.to(device), indices.to(device),))

    torch.topk = safe_topk
# ------------------- End Top-K Patch -------------------

# ------------------- ONNX Runtime Patch -------------------
try:
    import onnxruntime as ort

    if is_zluda:
        print("\n***----------------------ZLUDA-----------------------------***")
        print("  ::  Patching ONNX Runtime for ZLUDA — disabling CUDA EP.")

        # Store original get_available_providers
        original_get_available_providers = ort.get_available_providers

        def filtered_providers():
            return [ep for ep in original_get_available_providers() if ep != "CUDAExecutionProvider"]

        # Patch ONLY the _pybind_state version (used during session creation)
        ort.capi._pybind_state.get_available_providers = filtered_providers

        # Wrap InferenceSession to force CPU provider when CUDA is explicitly requested
        OriginalSession = ort.InferenceSession

        class SafeInferenceSession(OriginalSession):
            def __init__(self, *args, providers=None, **kwargs):
                if providers and "CUDAExecutionProvider" in providers:
                    print("  ::  Forcing ONNX to use CPUExecutionProvider instead of CUDA.")
                    providers = ["CPUExecutionProvider"]
                super().__init__(*args, providers=providers, **kwargs)

        ort.InferenceSession = SafeInferenceSession
except ImportError:
    print("  ::  ONNX Runtime not installed — skipping patch.")
except Exception as e:
    print("  ::  Failed to patch ONNX Runtime:", e)
# ------------------- End ONNX Patch -------------------

# ------------------- ZLUDA Backend Patch -------------------
if is_zluda:
    print("  ::  ZLUDA detected, disabling non-supported functions.      ")
    torch.backends.cudnn.enabled = False

    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    print("  ::  CuDNN, flash_sdp, mem_efficient_sdp disabled.           ")
 
if is_zluda:
    print(f"  ::  Using ZLUDA with device: {zluda_device_name}")
    print("***--------------------------------------------------------***\n")
else:
    print(f"  ::  CUDA device detected: {zluda_device_name or 'None'}")
    print("***--------------------------------------------------------***\n")
# ------------------- End Zluda detection -------------------
