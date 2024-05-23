FROM rocm/pytorch:rocm6.0.2_ubuntu22.04_py3.10_pytorch_2.1.2
RUN pip install --no-cache --no-build-isolation git+https://github.com/hiddenswitch/ComfyUI.git
EXPOSE 8188
WORKDIR /workspace
CMD ["/usr/local/bin/comfyui", "--listen"]
