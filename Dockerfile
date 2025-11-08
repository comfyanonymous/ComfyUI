FROM nvcr.io/nvidia/pytorch:25.03-py3

ENV TZ="Etc/UTC"

ENV PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync,expandable_segments:True"
ENV UV_COMPILE_BYTECODE=1
ENV UV_NO_CACHE=1
ENV UV_SYSTEM_PYTHON=1
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
ENV DEBIAN_FRONTEND=noninteractive
ENV UV_OVERRIDE=/workspace/overrides.txt

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# mitigates
# RuntimeError: Failed to import transformers.generation.utils because of the following error (look up to see its traceback):
# numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject
RUN echo "onnxruntime-gpu==1.22.0" >> /workspace/overrides.txt; pip freeze | grep nvidia >> /workspace/overrides.txt; echo "torch==2.7.0a0+7c8ec84dab.nv25.3" >> /workspace/overrides.txt; pip freeze | grep numpy >> /workspace/overrides.txt; echo "opencv-python; python_version < '0'" >> /workspace/overrides.txt; echo "opencv-contrib-python; python_version < '0'" >> /workspace/overrides.txt; echo "opencv-python-headless; python_version < '0'" >> /workspace/overrides.txt; echo "opencv-contrib-python-headless!=4.11.0.86" >> /workspace/overrides.txt; echo "sentry-sdk; python_version < '0'" >> /workspace/overrides.txt

# mitigates https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo
# mitigates AttributeError: module 'cv2.dnn' has no attribute 'DictValue' \
# see https://github.com/facebookresearch/nougat/issues/40
RUN pip install uv && uv --version && \
    apt-get update && apt-get install --no-install-recommends ffmpeg libsm6 libxext6 libcairo2-dev -y && \
    uv pip uninstall --system $(pip list --format=freeze | grep opencv) && \
    rm -rf /usr/local/lib/python3.12/dist-packages/cv2/ && \
    uv pip install wheel && \
    uv pip install --no-build-isolation "opencv-contrib-python-headless>=4.12.0.88" && \
    rm -rf /var/lib/apt/lists/*

# install sageattention
ADD pkg/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl /workspace/pkg/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl
RUN uv pip install -U --no-deps --no-build-isolation spandrel timm tensorboard poetry "flash-attn<=2.8.0" "xformers==0.0.31.post1" "file:./pkg/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl"
# this exotic command will determine the correct torchaudio to install for the image
RUN <<-EOF
python -c 'import torch, re, subprocess
torch_version_full = torch.__version__
torch_ver_match = re.match(r"(\d+\.\d+\.\d+)", torch_version_full)
if not torch_ver_match:
    raise ValueError(f"Could not parse torch version from {torch_version_full}")
torch_ver = torch_ver_match.group(1)
cuda_ver_tag = f"cu{torch.version.cuda.replace(".", "")}"
command = [
    "uv", "pip", "install", "--no-deps",
    f"torchaudio=={torch_ver}+{cuda_ver_tag}",
    "--extra-index-url", f"https://download.pytorch.org/whl/{cuda_ver_tag}",
]
subprocess.run(command, check=True)'
EOF

# sources for building this dockerfile
# use these lines to build from the local fs
ADD . /workspace/src
ARG SOURCES="comfyui[attention,comfyui_manager]@./src"
# this builds from github
# useful if you are copying and pasted in order to customize this
# ARG SOURCES="comfyui[attention,comfyui_manager]@git+https://github.com/hiddenswitch/ComfyUI.git"
ENV SOURCES=$SOURCES

RUN uv pip install $SOURCES

WORKDIR /workspace
# addresses https://github.com/pytorch/pytorch/issues/104801
# and issues reported by importing nodes_canny
# smoke test
RUN python -c "import torch; import xformers; import sageattention; import cv2; import diffusers.hooks" && comfyui --quick-test-for-ci --cpu --cwd /workspace

EXPOSE 8188
CMD ["python", "-m", "comfy.cmd.main", "--listen", "--use-sage-attention", "--reserve-vram=0", "--logging-level=INFO", "--enable-cors"]
