import subprocess
#run this file to start your server using a mac intel.

command = "PYTORCH_ENABLE_MPS_FALLBACK=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python3 main.py --use-split-cross-attention"
subprocess.run(command, shell=True)
