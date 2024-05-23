FROM intel/intel-optimized-pytorch:2.3.0-pip-base
RUN pip install --no-cache --no-build-isolation git+https://github.com/hiddenswitch/ComfyUI.git
EXPOSE 8188
WORKDIR /workspace
CMD ["/usr/local/bin/comfyui", "--listen"]