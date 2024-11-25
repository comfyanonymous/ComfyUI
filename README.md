<div align="center">

# ComfyUI-TPU
**TPUs/XLA devices support for ComfyUI**
</div>

#### TPU/XLA Devices
Users with TPU/XLA devices can install the PyTorch XLA stable build with the following command:

```
pip install torch~=2.5.0 torch_xla[tpu]~=2.5.0 -f https://storage.googleapis.com/libtpu-releases/index.html
```

This is the command to install the nightly 2.6.0 which might have some performance improvements:

```
pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
pip install 'torch_xla[tpu] @ https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.6.0.dev-cp310-cp310-linux_x86_64.whl' -f https://storage.googleapis.com/libtpu-releases/index.html
```


To get memory info for TPU devices, install the [tpu-info](https://github.com/AI-Hypercomputer/cloud-accelerator-diagnostics/tree/main/tpu_info) package with the following command:

```
pip install tpu-info
```

To monitor tpu-info 

```
watch -n0 tpu-info
```
