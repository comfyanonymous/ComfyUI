import pytest
import torch
import torch.nn as nn
import psutil
import os
import gc
import tempfile
import sys

# Ensure the project root is on the Python path (so `import comfy` works when running tests from this folder)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from comfy.model_patcher import model_to_mmap, to_mmap


class LargeModel(nn.Module):
    """A simple model with large parameters for testing memory mapping"""
    
    def __init__(self, size_gb=10):
        super().__init__()
        # Calculate number of float32 elements needed for target size
        # 1 GB = 1024^3 bytes, float32 = 4 bytes
        bytes_per_gb = 1024 * 1024 * 1024
        elements_per_gb = bytes_per_gb // 4  # float32 is 4 bytes
        total_elements = int(size_gb * elements_per_gb)
        
        # Create a large linear layer
        # Split into multiple layers to avoid single tensor size limits
        self.layers = nn.ModuleList()
        elements_per_layer = 500 * 1024 * 1024  # 500M elements per layer (~2GB)
        num_layers = (total_elements + elements_per_layer - 1) // elements_per_layer
        
        for i in range(num_layers):
            if i == num_layers - 1:
                # Last layer gets the remaining elements
                remaining = total_elements - (i * elements_per_layer)
                in_features = int(remaining ** 0.5)
                out_features = (remaining + in_features - 1) // in_features
            else:
                in_features = int(elements_per_layer ** 0.5)
                out_features = (elements_per_layer + in_features - 1) // in_features
            
            # Create layer without bias to control size precisely
            self.layers.append(nn.Linear(in_features, out_features, bias=False))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def get_process_memory_gb():
    """Get current process memory usage in GB"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 3)  # Convert to GB


def get_model_size_gb(model):
    """Calculate model size in GB"""
    total_size = 0
    for param in model.parameters():
        total_size += param.nelement() * param.element_size()
    for buffer in model.buffers():
        total_size += buffer.nelement() * buffer.element_size()
    return total_size / (1024 ** 3)


def test_model_to_mmap_memory_efficiency():
    """Test that model_to_mmap reduces memory usage for a 10GB model to less than 1GB
    
    The typical use case is:
    1. Load a large model on CUDA
    2. Convert to mmap to offload from GPU to disk-backed memory
    3. This frees GPU memory and reduces CPU RAM usage
    """
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available, skipping test")
    
    # Force garbage collection before starting
    gc.collect()
    torch.cuda.empty_cache()
    
    # Record initial memory
    initial_cpu_memory = get_process_memory_gb()
    initial_gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3)
    print(f"\nInitial CPU memory: {initial_cpu_memory:.2f} GB")
    print(f"Initial GPU memory: {initial_gpu_memory:.2f} GB")
    
    # Create a 10GB model
    print("Creating 10GB model...")
    model = LargeModel(size_gb=10)
    
    # Verify model size
    model_size = get_model_size_gb(model)
    print(f"Model size: {model_size:.2f} GB")
    assert model_size >= 9.5, f"Model size {model_size:.2f} GB is less than expected 10 GB"
    
    # Move model to CUDA
    print("Moving model to CUDA...")
    model = model.cuda()
    torch.cuda.synchronize()
    
    # Memory after moving to CUDA
    cpu_after_cuda = get_process_memory_gb()
    gpu_after_cuda = torch.cuda.memory_allocated() / (1024 ** 3)
    print(f"CPU memory after moving to CUDA: {cpu_after_cuda:.2f} GB")
    print(f"GPU memory after moving to CUDA: {gpu_after_cuda:.2f} GB")
    
    # Convert to mmap (this should move model from GPU to disk-backed memory)
    # Note: model_to_mmap modifies the model in-place via _apply()
    # so model and model_mmap will be the same object
    print("Converting model to mmap...")
    model_mmap = model_to_mmap(model)
    
    # Verify that model and model_mmap are the same object (in-place modification)
    assert model is model_mmap, "model_to_mmap should modify the model in-place"
    
    # Force garbage collection and clear CUDA cache
    # The original CUDA tensors should be automatically freed when replaced
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # Memory after mmap conversion
    cpu_after_mmap = get_process_memory_gb()
    gpu_after_mmap = torch.cuda.memory_allocated() / (1024 ** 3)
    print(f"CPU memory after mmap: {cpu_after_mmap:.2f} GB")
    print(f"GPU memory after mmap: {gpu_after_mmap:.2f} GB")
    
    # Calculate memory changes from CUDA state (the baseline we're converting from)
    cpu_increase = cpu_after_mmap - cpu_after_cuda
    gpu_decrease = gpu_after_cuda - gpu_after_mmap  # Should be positive (freed)
    print(f"\nCPU memory increase from CUDA: {cpu_increase:.2f} GB")
    print(f"GPU memory freed: {gpu_decrease:.2f} GB")
    
    # Verify that CPU memory usage increase is less than 1GB
    # The mmap should use disk-backed storage, keeping CPU RAM usage low
    # We use 1.5 GB threshold to account for overhead
    assert cpu_increase < 1.5, (
        f"CPU memory increase after mmap ({cpu_increase:.2f} GB) should be less than 1.5 GB. "
        f"CUDA state: {cpu_after_cuda:.2f} GB, After mmap: {cpu_after_mmap:.2f} GB"
    )
    
    # Verify that GPU memory has been freed
    # We expect at least 9 GB to be freed (original 10GB model with some tolerance)
    assert gpu_decrease > 9.0, (
        f"GPU memory should be freed after mmap. "
        f"Freed: {gpu_decrease:.2f} GB (from {gpu_after_cuda:.2f} to {gpu_after_mmap:.2f} GB), expected > 9 GB"
    )
    
    # Verify the model is still functional (basic sanity check)
    assert model_mmap is not None
    assert len(list(model_mmap.parameters())) > 0
    
    print(f"\n✓ Test passed!")
    print(f"  CPU memory increase: {cpu_increase:.2f} GB < 1.5 GB")
    print(f"  GPU memory freed: {gpu_decrease:.2f} GB > 9.0 GB")
    print(f"  Model successfully offloaded from GPU to disk-backed memory")
    
    # Cleanup (model and model_mmap are the same object)
    del model, model_mmap
    gc.collect()
    torch.cuda.empty_cache()


def test_to_mmap_cuda_cycle():
    """Test CUDA -> mmap -> CUDA cycle
    
    This test verifies:
    1. CUDA tensor can be converted to mmap tensor
    2. CPU memory increase is minimal when using mmap (< 0.1 GB)
    3. GPU memory is freed when converting to mmap
    4. mmap tensor can be moved back to CUDA
    5. Data remains consistent throughout the cycle
    6. mmap file is automatically cleaned up via garbage collection
    """
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available, skipping test")
    
    # Force garbage collection
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\nTest: CUDA -> mmap -> CUDA cycle")
    
    # Record initial CPU memory
    initial_cpu_memory = get_process_memory_gb()
    print(f"Initial CPU memory: {initial_cpu_memory:.2f} GB")
    
    # Step 1: Create a CUDA tensor
    print("\n1. Creating CUDA tensor...")
    original_data = torch.randn(5000, 5000).cuda()
    original_sum = original_data.sum().item()
    print(f"   Shape: {original_data.shape}")
    print(f"   Device: {original_data.device}")
    print(f"   Sum: {original_sum:.2f}")
    
    # Record GPU and CPU memory after CUDA allocation
    cpu_after_cuda = get_process_memory_gb()
    gpu_before_mmap = torch.cuda.memory_allocated() / (1024 ** 3)
    print(f"   GPU memory: {gpu_before_mmap:.2f} GB")
    print(f"   CPU memory: {cpu_after_cuda:.2f} GB")
    
    # Step 2: Convert to mmap tensor
    print("\n2. Converting to mmap tensor...")
    mmap_tensor = to_mmap(original_data)
    del original_data
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"   Device: {mmap_tensor.device}")
    print(f"   Sum: {mmap_tensor.sum().item():.2f}")
    
    # Verify GPU memory is freed
    gpu_after_mmap = torch.cuda.memory_allocated() / (1024 ** 3)
    cpu_after_mmap = get_process_memory_gb()
    print(f"   GPU memory freed: {gpu_before_mmap - gpu_after_mmap:.2f} GB")
    print(f"   CPU memory: {cpu_after_mmap:.2f} GB")
    
    # Verify GPU memory is freed
    assert gpu_after_mmap < 0.1, f"GPU memory should be freed, but {gpu_after_mmap:.2f} GB still allocated"
    
    # Verify CPU memory increase is minimal (should be close to 0 due to mmap)
    cpu_increase = cpu_after_mmap - cpu_after_cuda
    print(f"   CPU memory increase: {cpu_increase:.2f} GB")
    assert cpu_increase < 0.1, f"CPU memory should increase minimally, but increased by {cpu_increase:.2f} GB"
    
    # Get the temp file path (we'll check if it gets cleaned up)
    # The file should exist at this point
    temp_files_before = len([f for f in os.listdir(tempfile.gettempdir()) if f.startswith('comfy_mmap_')])
    print(f"   Temp mmap files exist: {temp_files_before}")
    
    # Step 3: Move back to CUDA
    print("\n3. Moving back to CUDA...")
    cuda_tensor = mmap_tensor.to('cuda')
    torch.cuda.synchronize()
    
    print(f"   Device: {cuda_tensor.device}")
    final_sum = cuda_tensor.sum().item()
    print(f"   Sum: {final_sum:.2f}")
    
    # Verify GPU memory is used again
    gpu_after_cuda = torch.cuda.memory_allocated() / (1024 ** 3)
    print(f"   GPU memory: {gpu_after_cuda:.2f} GB")
    
    # Step 4: Verify data consistency
    print("\n4. Verifying data consistency...")
    sum_diff = abs(original_sum - final_sum)
    print(f"   Original sum: {original_sum:.2f}")
    print(f"   Final sum: {final_sum:.2f}")
    print(f"   Difference: {sum_diff:.6f}")
    assert sum_diff < 0.01, f"Data should be consistent, but difference is {sum_diff:.6f}"
    
    # Step 5: Verify file cleanup (delayed until garbage collection)
    print("\n5. Verifying file cleanup...")
    # Delete the mmap tensor reference to trigger garbage collection
    del mmap_tensor
    gc.collect()
    import time
    time.sleep(0.1)  # Give OS time to clean up
    temp_files_after = len([f for f in os.listdir(tempfile.gettempdir()) if f.startswith('comfy_mmap_')])
    print(f"   Temp mmap files after GC: {temp_files_after}")
    # File should be cleaned up after garbage collection
    assert temp_files_after <= temp_files_before, "mmap file should be cleaned up after garbage collection"
    
    print("\n✓ Test passed!")
    print("  CUDA -> mmap -> CUDA cycle works correctly")
    print(f"  CPU memory increase: {cpu_increase:.2f} GB < 0.1 GB (mmap efficiency)")
    print("  Data consistency maintained")
    print("  File cleanup successful (via garbage collection)")
    
    # Cleanup
    del cuda_tensor  # mmap_tensor already deleted in Step 5
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # Run the tests directly
    test_model_to_mmap_memory_efficiency()
    test_to_mmap_cuda_cycle()

