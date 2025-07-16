FROM rocm/pytorch:rocm6.4.1_ubuntu24.04_py3.12_pytorch_release_2.7.1

ENV TZ="Etc/UTC"

ENV UV_COMPILE_BYTECODE=1
ENV UV_NO_CACHE=1
ENV UV_SYSTEM_PYTHON=1
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
ENV DEBIAN_FRONTEND=noninteractive

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

RUN pip freeze | grep numpy > numpy-override.txt

# mitigates AttributeError: module 'cv2.dnn' has no attribute 'DictValue' \
# see https://github.com/facebookresearch/nougat/issues/40
RUN apt-get update && \
    apt-get install --no-install-recommends -y ffmpeg libsm6 libxext6 && \
    pip install uv && uv --version && \
    apt-get purge -y && \
    rm -rf /var/lib/apt/lists/*

RUN uv pip install --overrides=numpy-override.txt "comfyui[attention,comfyui_manager]@git+https://github.com/hiddenswitch/ComfyUI.git"

WORKDIR /workspace
# addresses https://github.com/pytorch/pytorch/issues/104801
# and issues reported by importing nodes_canny
RUN comfyui --quick-test-for-ci --cpu --cwd /workspace

EXPOSE 8188
CMD ["python", "-m", "comfy.cmd.main", "--listen"]