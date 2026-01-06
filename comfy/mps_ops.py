import torch

_LUT_CACHE = {}

def get_lut(dtype, device):
    """
    Get or create a lookup table for float8 dequantization on MPS.
    Returns a Tensor[256] of dtype=torch.float16 on the specified device.
    """
    key = (dtype, device)
    if key in _LUT_CACHE:
        return _LUT_CACHE[key]
    
    # Generate all possible 8-bit values (0-255)
    # We create them on CPU first as float8, then cast to float16, then move to MPS.
    # This acts as our decoding table.
    
    # Create uint8 pattern 0..255
    byte_pattern = torch.arange(256, dtype=torch.uint8, device="cpu")
    
    # View as the target float8 type
    # Note: We must use .view() on a tensor that has the same number of bytes.
    # We can't view uint8 as float8 directly if standard pytorch doesn't allow it easily,
    # but we can create the float8 tensor from bytes.
    
    # Actually, the easiest way to generate the LUT is:
    # 1. Create bytes 0..255
    # 2. View as float8 (on CPU, where it is supported)
    # 3. Convert to float16 (on CPU)
    # 4. Move float16 LUT to MPS
    
    try:
        f8_tensor = byte_pattern.view(dtype)
        f16_lut = f8_tensor.to(torch.float16)
        
        # Move to the requested MPS device
        lut = f16_lut.to(device)
        _LUT_CACHE[key] = lut
        return lut
    except Exception as e:
        print(f"Failed to create MPS LUT for {dtype}: {e}")
        # Fallback: return None or raise
        raise e

def mps_dequantize(qdata, scale, orig_dtype, float8_dtype):
    """
    Dequantize a uint8 tensor (representing float8 data) using a LUT on MPS.
    
    Args:
        qdata: Tensor of shape (...) with dtype=torch.uint8 (on MPS)
        scale: Tensor (scalar)
        orig_dtype: The target dtype (e.g. float16)
        float8_dtype: The original float8 dtype (torch.float8_e4m3fn or torch.float8_e5m2)
    
    Returns:
        Tensor of shape (...) with dtype=orig_dtype
    """
    lut = get_lut(float8_dtype, qdata.device)
    
    # Use index_select or advanced indexing.
    # Advanced indexing lut[qdata.long()] is generally efficient.
    # We explicitly cast to long (int64) for indexing.
    # Note: Flattening might be slightly faster depending on shape, but simple indexing is safest.
    
    # We want the LUT to be in the target orig_dtype (likely float16 or bfloat16)
    if lut.dtype != orig_dtype:
        lut = lut.to(dtype=orig_dtype)
        
    output = lut[qdata.long()]
    
    # Apply scale
    # Scale might need to be cast to orig_dtype too
    if isinstance(scale, torch.Tensor):
        scale = scale.to(dtype=orig_dtype)
        
    output.mul_(scale)
    return output
