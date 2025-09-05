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

# ------------------- ComfyUI Package Version Check -------------------
def get_package_version(package_name):
    try:
        from importlib.metadata import version
        return version(package_name)
    except ImportError:
        from importlib_metadata import version
        return version(package_name)

def parse_requirements_file(requirements_path):
    """Parse requirements.txt file and extract package versions."""
    requirements = {}
    try:
        with open(requirements_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Handle different version specifiers
                    if '==' in line:
                        pkg, version = line.split('==', 1)
                        requirements[pkg] = version.strip()
                    elif '>=' in line:
                        pkg, version = line.split('>=', 1)
                        requirements[pkg] = version.strip()
                    elif '~=' in line:  # Compatible release operator
                        pkg, version = line.split('~=', 1)
                        requirements[pkg] = version.strip()
                    # You can add more operators if needed (>, <, <=, !=)
    except FileNotFoundError:
        print(f"  ::  Warning: requirements.txt not found at {requirements_path}")
    return requirements

def is_compatible_version(installed_version, required_version, operator='>='):
    """Check if installed version meets requirement based on operator."""
    try:
        from packaging import version
        installed_v = version.parse(installed_version)
        required_v = version.parse(required_version)
        
        if operator == '>=':
            return installed_v >= required_v
        elif operator == '==':
            return installed_v == required_v
        elif operator == '~=':
            # Compatible release: ~=2.0 means >=2.0, <3.0
            # ~=2.1 means >=2.1, <2.2
            required_parts = required_v.release
            if len(required_parts) == 1:
                # ~=2 means >=2.0, <3.0
                return (installed_v >= required_v and 
                        installed_v.release[0] == required_parts[0])
            else:
                # ~=2.1 means >=2.1, <2.2
                return (installed_v >= required_v and 
                        installed_v.release[:len(required_parts)-1] == required_parts[:-1] and
                        installed_v.release[len(required_parts)-1] >= required_parts[-1])
        else:
            # Default to >= for unknown operators
            return installed_v >= required_v
    except Exception as e:
        print(f"  ::  Version comparison error for {installed_version} vs {required_version}: {e}")
        return False

def uninstall_package(package_name):
    """Uninstall a package quietly"""
    import subprocess
    import sys
    try:
        args = [sys.executable, '-m', 'pip', 'uninstall', package_name, '-y', '--quiet']
        subprocess.check_call(args)
        return True
    except subprocess.CalledProcessError:
        return False

def check_pydantic_compatibility():
    """Check if current pydantic packages are compatible, return True if they need reinstalling"""
    try:
        # Try to import the problematic class that causes the error
        from pydantic_settings import TomlConfigSettingsSource
        # If we get here, the packages are compatible
        return False
    except ImportError:
        # Import failed, packages are incompatible
        return True
    except Exception:
        # Any other error, assume incompatible
        return True

def handle_pydantic_packages(required_packages):
    """Special handling for pydantic packages to ensure compatibility"""
    import subprocess
    import sys
    
    pydantic_packages = ['pydantic', 'pydantic-settings']
    packages_in_requirements = [pkg for pkg in pydantic_packages if pkg in required_packages]
    
    if not packages_in_requirements:
        return  # No pydantic packages to handle
    
    # Check if both packages are available and what versions
    pydantic_installed = None
    pydantic_settings_installed = None
    
    try:
        pydantic_installed = get_package_version('pydantic')
    except:
        pass
    
    try:
        pydantic_settings_installed = get_package_version('pydantic-settings')
    except:
        pass
    
    # If both are installed, check compatibility
    if pydantic_installed and pydantic_settings_installed:
        print(f"Found pydantic: {pydantic_installed}, pydantic-settings: {pydantic_settings_installed}")
        
        # Check if they're compatible by testing the import
        if not check_pydantic_compatibility():
            print("  ::  Pydantic packages are compatible, skipping reinstall")
            return
        else:
            print("  ::  Pydantic packages are incompatible, need to reinstall")
    
    # If we get here, we need to install/reinstall pydantic packages
    print("  ::  Setting up pydantic packages for compatibility...")
    
    # Uninstall existing versions to avoid conflicts
    if pydantic_installed:
        print(f"  ::  Uninstalling existing pydantic {pydantic_installed}")
        uninstall_package('pydantic')
    
    if pydantic_settings_installed:
        print(f"  ::  Uninstalling existing pydantic-settings {pydantic_settings_installed}")
        uninstall_package('pydantic-settings')
    
    # Install both packages together
    try:
        print("  ::  Installing compatible pydantic packages...")
        combined_args = [sys.executable, '-m', 'pip', 'install', 
                       'pydantic~=2.0', 
                       'pydantic-settings~=2.0',
                       '--quiet', 
                       '--disable-pip-version-check']
        
        subprocess.check_call(combined_args)
        
        # Verify installation
        new_pydantic = get_package_version('pydantic')
        new_pydantic_settings = get_package_version('pydantic-settings')
        print(f"  ::  Successfully installed pydantic: {new_pydantic}, pydantic-settings: {new_pydantic_settings}")
        
    except subprocess.CalledProcessError as e:
        print(f"  ::  Failed to install pydantic packages: {e}")

def install_package(package_name, version_spec, upgrade=False):
    import subprocess
    import sys
    
    # For ~= operator, install with the compatible release syntax
    if '~=' in version_spec:
        package_spec = f'{package_name}~={version_spec}'
    else:
        package_spec = f'{package_name}=={version_spec}'
    
    args = [sys.executable, '-m', 'pip', 'install', 
            package_spec, 
            '--quiet', 
            '--disable-pip-version-check']
    if upgrade:
        args.append('--upgrade')
    
    try:
        subprocess.check_call(args)
    except subprocess.CalledProcessError as e:
        print(f"  ::  Failed to install {package_name}: {e}")
        # Try installing without version constraint as fallback
        if upgrade and '~=' in package_spec:
            try:
                print(f"  ::  Retrying {package_name} installation without version constraint...")
                fallback_args = [sys.executable, '-m', 'pip', 'install', 
                               package_name, 
                               '--upgrade',
                               '--quiet', 
                               '--disable-pip-version-check']
                subprocess.check_call(fallback_args)
                print(f"  ::  {package_name} installed successfully without version constraint")
            except subprocess.CalledProcessError as e2:
                print(f"  ::  Fallback installation also failed: {e2}")

def ensure_package(package_name, required_version, operator='>='):
    # Skip individual pydantic package handling - they're handled together
    if package_name in ['pydantic', 'pydantic-settings']:
        return
        
    try:
        installed_version = get_package_version(package_name)
        print(f"Installed version of {package_name}: {installed_version}")
        
        if not is_compatible_version(installed_version, required_version, operator):
            install_package(package_name, required_version, upgrade=True)
            print(f"\n{package_name} outdated. Upgraded to {required_version}.")
    except Exception as e:
        print(f"  ::  {package_name} not found or error checking version: {e}")
        install_package(package_name, required_version)
        print(f"\n{package_name} was missing. Installed it.")

# Determine operator from requirements.txt
def get_version_operator(requirements_path, package_name):
    """Extract the version operator used for a package in requirements.txt"""
    try:
        with open(requirements_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and line.startswith(package_name):
                    if '~=' in line:
                        return '~='
                    elif '==' in line:
                        return '=='
                    elif '>=' in line:
                        return '>='
    except FileNotFoundError:
        pass
    return '>='  # Default

import os
requirements_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'requirements.txt')
required_packages = parse_requirements_file(requirements_path)

packages_to_monitor = [
    "comfyui-frontend-package",
    "comfyui-workflow-templates",
    "av",
    "comfyui-embedded-docs",
    "pydantic",
    "pydantic-settings",
]

print("\n  ::  Checking package versions...")

# Handle pydantic packages first with special logic
handle_pydantic_packages(required_packages)

# Handle other packages
for package_name in packages_to_monitor:
    if package_name in required_packages and package_name not in ['pydantic', 'pydantic-settings']:
        operator = get_version_operator(requirements_path, package_name)
        ensure_package(package_name, required_packages[package_name], operator)
    elif package_name not in ['pydantic', 'pydantic-settings']:
        print(f"  ::  Warning: {package_name} not found in requirements.txt")

print("  ::  Package version check complete.")
# ------------------- End Version Check -------------------
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

# ------------------- RMSNorm Compatibility Patch -------------------
# Fix for PyTorch < 2.4.0 which doesn't have nn.RMSNorm
# Based on ComfyUI's actual RMSNorm implementation
import torch.nn as nn
import numbers

if not hasattr(nn, 'RMSNorm'):
    print("  ::  PyTorch RMSNorm not found, adding ComfyUI-compatible layer.")
    
    # Check if torch.nn.functional.rms_norm exists
    rms_norm_torch = None
    try:
        rms_norm_torch = torch.nn.functional.rms_norm
    except AttributeError:
        rms_norm_torch = None
    
    def rms_norm_fallback(x, weight=None, eps=1e-6):
        """Fallback RMSNorm implementation when native function unavailable"""
        if rms_norm_torch is not None and not (torch.jit.is_tracing() or torch.jit.is_scripting()):
            # Try to import comfy.model_management for proper casting
            try:
                import comfy.model_management
                cast_fn = comfy.model_management.cast_to
            except ImportError:
                # Fallback casting function if comfy not available
                cast_fn = lambda w, dtype, device: w.to(dtype=dtype, device=device) if w is not None else None
            
            if weight is None:
                return rms_norm_torch(x, (x.shape[-1],), eps=eps)
            else:
                return rms_norm_torch(x, weight.shape, weight=cast_fn(weight, dtype=x.dtype, device=x.device), eps=eps)
        else:
            # Manual implementation
            r = x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
            if weight is None:
                return r
            else:
                # Try to use comfy's cast function, fallback to simple casting
                try:
                    import comfy.model_management
                    weight_casted = comfy.model_management.cast_to(weight, dtype=x.dtype, device=x.device)
                except ImportError:
                    weight_casted = weight.to(dtype=x.dtype, device=x.device) if weight is not None else None
                return r * weight_casted
    
    class RMSNorm(nn.Module):
        def __init__(
            self,
            normalized_shape,
            eps=1e-6,
            elementwise_affine=True,
            device=None,
            dtype=None,
        ):
            factory_kwargs = {"device": device, "dtype": dtype}
            super().__init__()
            
            # Handle both int and tuple normalized_shape (like ComfyUI does)
            if isinstance(normalized_shape, numbers.Integral):
                normalized_shape = (normalized_shape,)
            self.normalized_shape = tuple(normalized_shape)
            self.eps = eps
            self.elementwise_affine = elementwise_affine
            
            if self.elementwise_affine:
                # Use empty() like ComfyUI, not ones()
                self.weight = nn.Parameter(
                    torch.empty(self.normalized_shape, **factory_kwargs)
                )
                # Initialize like LayerNorm
                nn.init.ones_(self.weight)
            else:
                self.register_parameter("weight", None)
            
            self.bias = None  # RMSNorm doesn't use bias

        def forward(self, x):
            return rms_norm_fallback(x, self.weight, self.eps)
    
    # Monkey patch nn.RMSNorm
    nn.RMSNorm = RMSNorm
    print("  ::  ComfyUI-compatible RMSNorm layer installed.")
else:
    print("  ::  PyTorch RMSNorm found, no patch needed.")
# ------------------- End RMSNorm Patch -------------------

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



