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

import torch

if torch.cuda.get_device_name().endswith("[ZLUDA]"):
    print(" ")
    print("***----------------------ZLUDA-----------------------------***")
    print("  ::  ZLUDA detected, disabling non-supported functions.")
    torch.backends.cudnn.enabled = False
    print("  ::  CuDNN, flash_sdp, math_sdp, mem_efficient_sdp disabled) ")
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    print("***--------------------------------------------------------***")
    print("  ::  Device:", torch.cuda.get_device_name())
    print(" ")
else:
    print("  ::  ZLUDA isn't detected, please try patching it.")
