FROM rocm/pytorch:rocm7.0_ubuntu24.04_py3.12_pytorch_release_2.7.1

ENV TZ="Etc/UTC"

ENV UV_COMPILE_BYTECODE=1
ENV UV_NO_CACHE=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
ENV DEBIAN_FRONTEND=noninteractive

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

RUN uv pip freeze | grep torch >> /overrides.txt; uv pip freeze | grep opencv >> /overrides.txt; uv pip freeze | grep numpy >> /overrides.txt; uv pip freeze | grep rocm >> /overrides.txt; echo "sentry-sdk; python_version < '0'" >> /overrides.txt

ENV UV_OVERRIDE=/overrides.txt
env UV_PRERELEASE=allow

# mitigates AttributeError: module 'cv2.dnn' has no attribute 'DictValue' \
# see https://github.com/facebookresearch/nougat/issues/40
RUN apt-get update && \
    apt-get install --no-install-recommends -y ffmpeg libsm6 libxext6 libsndfile1 && \
    pip install uv && uv --version && \
    apt-get purge -y && \
    rm -rf /var/lib/apt/lists/*

# torchaudio
RUN uv pip install --no-deps https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0/torchaudio-2.7.1%2Brocm7.0.0.git95c61b41-cp312-cp312-linux_x86_64.whl

# sources for building this dockerfile
# use these lines to build from the local fs
ADD . /src
ARG SOURCES="comfyui[rocm,comfyui_manager]@/src"
# this builds from github
# useful if you are copying and pasted in order to customize this
# ARG SOURCES="comfyui[attention,comfyui_manager]@git+https://github.com/hiddenswitch/ComfyUI.git"
ENV SOURCES=$SOURCES
RUN uv pip install $SOURCES

WORKDIR /workspace
# addresses https://github.com/pytorch/pytorch/issues/104801
# and issues reported by importing nodes_canny
RUN python -c "import torch; import transformers; from transformers import AutoProcessor, BatchFeature; import torchaudio; import cv2" && comfyui --quick-test-for-ci --cpu --cwd /workspace

EXPOSE 8188
CMD ["python", "-m", "comfy.cmd.main", "--listen"]