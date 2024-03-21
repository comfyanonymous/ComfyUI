FROM nvcr.io/nvidia/pytorch:24.01-py3
RUN pip install --no-cache --no-build-isolation git+https://github.com/hiddenswitch/ComfyUI.git
EXPOSE 8188
WORKDIR /workspace
CMD ["/usr/local/bin/comfyui", "--listen"]
