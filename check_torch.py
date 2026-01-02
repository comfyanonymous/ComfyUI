import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
print('cuDNN:', torch.backends.cudnn.version())
print('GPU:', torch.cuda.get_device_name(0))
print('GPU capability:', torch.cuda.get_device_capability(0))
