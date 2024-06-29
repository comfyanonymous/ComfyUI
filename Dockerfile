FROM nvcr.io/nvidia/pytorch:24.03-py3
RUN pip install --no-cache --no-build-isolation git+https://github.com/hiddenswitch/ComfyUI.git
EXPOSE 8188
WORKDIR /workspace
# tries to address https://github.com/pytorch/pytorch/issues/104801
# and issues reported by importing nodes_canny
ENV PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync"
RUN comfyui --quick-test-for-ci --cpu --cwd /workspace
CMD ["/usr/local/bin/comfyui", "--listen"]
