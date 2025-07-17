FROM nvcr.io/nvidia/pytorch:25.03-py3

ENV TZ="Etc/UTC"

ENV PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync"
ENV UV_COMPILE_BYTECODE=1
ENV UV_NO_CACHE=1
ENV UV_SYSTEM_PYTHON=1
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
ENV DEBIAN_FRONTEND=noninteractive

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# mitigates
# RuntimeError: Failed to import transformers.generation.utils because of the following error (look up to see its traceback):
# numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject
RUN pip freeze | grep numpy > numpy-override.txt

# mitigates AttributeError: module 'cv2.dnn' has no attribute 'DictValue' \
# see https://github.com/facebookresearch/nougat/issues/40
RUN apt-get update && \
    apt-get install --no-install-recommends -y ffmpeg libsm6 libxext6 && \
    pip install uv && uv --version && \
    apt-get purge -y && \
    rm -rf /var/lib/apt/lists/*

RUN uv pip uninstall --system $(pip list --format=freeze | grep opencv) && \
    rm -rf /usr/local/lib/python3.12/dist-packages/cv2/ && \
    uv pip install --no-build-isolation opencv-python-headless

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
    "uv", "pip", "install", "--no-deps", "--overrides=numpy-override.txt",
    f"torchaudio=={torch_ver}+{cuda_ver_tag}",
    "--extra-index-url", f"https://download.pytorch.org/whl/{cuda_ver_tag}",
]
subprocess.run(command, check=True)'
EOF

# sources for building this dockerfile
# use these lines to build from the local fs
# ADD . /src
# ARG SOURCES=/src
# this builds from github
ARG SOURCES="comfyui[attention,comfyui_manager]@git+https://github.com/hiddenswitch/ComfyUI.git"
ENV SOURCES=$SOURCES

RUN uv pip install --overrides=numpy-override.txt $SOURCES

WORKDIR /workspace
# addresses https://github.com/pytorch/pytorch/issues/104801
# and issues reported by importing nodes_canny
RUN comfyui --quick-test-for-ci --cpu --cwd /workspace

EXPOSE 8188
CMD ["python", "-m", "comfy.cmd.main", "--listen", "--use-sage-attention", "--reserve-vram=0", "--logging-level=INFO", "--enable-cors"]
