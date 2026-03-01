
try:
    import transformers
    print("transformers: INSTALLED")
except ImportError:
    print("transformers: NOT INSTALLED")

try:
    import librosa
    print("librosa: INSTALLED")
except ImportError:
    print("librosa: NOT INSTALLED")

try:
    import moviepy
    print("moviepy: INSTALLED")
except ImportError:
    print("moviepy: NOT INSTALLED")

try:
    import torch
    print(f"torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
except ImportError:
    print("torch: NOT INSTALLED")
