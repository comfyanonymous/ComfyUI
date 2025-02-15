import torch

try:
    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    
    # Create tensors on the GPU
    ten1 = torch.randn((2, 4), device="cuda")
    ten2 = torch.randn((4, 8), device="cuda")
    
    # Perform matrix multiplication
    output = torch.mm(ten1, ten2)
    
    # Print success message
    print("zluda is correctly installed and working")
except Exception as e:
    # Print failure message if any exception occurs
    print(f"zluda isn't installed correctly: {e}")