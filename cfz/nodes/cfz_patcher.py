import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from comfy.sd import load_checkpoint_guess_config, load_checkpoint
from comfy.model_patcher import ModelPatcher
import folder_paths

# ------------------------ Optimized Quantization Logic -------------------------
def quantize_input_for_int8_matmul(input_tensor, weight_scale):
    """Quantize input tensor for optimized int8 matrix multiplication"""
    # Calculate input scale per batch/sequence dimension
    input_scale = input_tensor.abs().amax(dim=-1, keepdim=True) / 127.0
    input_scale = torch.clamp(input_scale, min=1e-8)
    
    # Quantize input to int8
    quantized_input = torch.clamp(
        (input_tensor / input_scale).round(), -128, 127
    ).to(torch.int8)
    
    # Combine input and weight scales
    combined_scale = input_scale * weight_scale
    
    # Flatten tensors for matrix multiplication if needed
    original_shape = input_tensor.shape
    if input_tensor.dim() > 2:
        quantized_input = quantized_input.flatten(0, -2).contiguous()
        combined_scale = combined_scale.flatten(0, -2).contiguous()
        # Ensure scale precision for accurate computation
        if combined_scale.dtype == torch.float16:
            combined_scale = combined_scale.to(torch.float32)
    
    return quantized_input, combined_scale, original_shape

def optimized_int8_matmul(input_tensor, quantized_weight, weight_scale, bias=None):
    """Optimized int8 matrix multiplication using torch._int_mm"""
    batch_size = input_tensor.numel() // input_tensor.shape[-1]
    
    # Performance threshold: only use optimized path for larger matrices
    # This prevents overhead from dominating small computations
    if batch_size >= 32 and input_tensor.shape[-1] >= 32:
        # Quantize input tensor for int8 computation
        q_input, combined_scale, orig_shape = quantize_input_for_int8_matmul(
            input_tensor, weight_scale
        )
        
        # Perform optimized int8 matrix multiplication
        # This is significantly faster than standard floating-point operations
        result = torch._int_mm(q_input, quantized_weight)
        
        # Dequantize result back to floating point
        result = result.to(combined_scale.dtype) * combined_scale
        
        # Reshape result back to original input dimensions
        if len(orig_shape) > 2:
            new_shape = list(orig_shape[:-1]) + [quantized_weight.shape[-1]]
            result = result.reshape(new_shape)
        
        # Add bias if present
        if bias is not None:
            result = result + bias
            
        return result
    else:
        # Fall back to standard dequantization for small matrices
        # This avoids quantization overhead when it's not beneficial
        dequantized_weight = quantized_weight.to(input_tensor.dtype) * weight_scale
        return F.linear(input_tensor, dequantized_weight, bias)

def make_optimized_quantized_forward(quant_dtype="float32", use_int8_matmul=True):
    """Create an optimized quantized forward function for neural network layers"""
    def forward(self, x):
        # Determine computation precision
        dtype = torch.float32 if quant_dtype == "float32" else torch.float16
        
        # Get input device for consistent placement
        device = x.device
        
        # Move quantized weights and scales to input device AND dtype
        qW = self.int8_weight.to(device)
        scale = self.scale.to(device, dtype=dtype)
        
        # Handle zero point for asymmetric quantization
        if hasattr(self, 'zero_point') and self.zero_point is not None:
            zp = self.zero_point.to(device, dtype=dtype)
        else:
            zp = None
        
        # Ensure input is in correct precision
        x = x.to(dtype)
        
        # Prepare bias if present - ENSURE IT'S ON THE CORRECT DEVICE
        bias = None
        if self.bias is not None:
            bias = self.bias.to(device, dtype=dtype)
        
        # Apply LoRA adaptation if present (before main computation for better accuracy)
        lora_output = None
        if hasattr(self, "lora_down") and hasattr(self, "lora_up") and hasattr(self, "lora_alpha"):
            # Ensure LoRA weights are on correct device
            lora_down = self.lora_down.to(device)
            lora_up = self.lora_up.to(device)
            lora_output = lora_up(lora_down(x)) * self.lora_alpha
        
        # Choose computation path based on layer type and optimization settings
        if isinstance(self, nn.Linear):
            # Linear layers can use optimized int8 matmul
            if (use_int8_matmul and zp is None and 
                hasattr(self, '_use_optimized_matmul') and self._use_optimized_matmul):
                # Use optimized path (only for symmetric quantization)
                result = optimized_int8_matmul(x, qW, scale, bias)
            else:
                # Standard dequantization path
                if zp is not None:
                    # Asymmetric quantization: subtract zero point then scale
                    W = (qW.to(dtype) - zp) * scale
                else:
                    # Symmetric quantization: just scale
                    W = qW.to(dtype) * scale
                result = F.linear(x, W, bias)
        
        elif isinstance(self, nn.Conv2d):
            # Convolution layers use standard dequantization
            if zp is not None:
                W = (qW.to(dtype) - zp) * scale
            else:
                W = qW.to(dtype) * scale
            result = F.conv2d(x, W, bias, self.stride, self.padding, self.dilation, self.groups)
        
        else:
            # Fallback for unsupported layer types
            return x
        
        # Add LoRA output if computed
        if lora_output is not None:
            result = result + lora_output
            
        return result
    
    return forward

def quantize_weight(weight: torch.Tensor, num_bits=8, use_asymmetric=False):
    """Quantize weights with support for both symmetric and asymmetric quantization"""
    # Determine reduction dimensions (preserve output channels)
    reduce_dim = 1 if weight.ndim == 2 else [i for i in range(weight.ndim) if i != 0]
    
    if use_asymmetric:
        # Asymmetric quantization: use full range [0, 255] for uint8
        min_val = weight.amin(dim=reduce_dim, keepdim=True)
        max_val = weight.amax(dim=reduce_dim, keepdim=True)
        scale = torch.clamp((max_val - min_val) / 255.0, min=1e-8)
        zero_point = torch.clamp((-min_val / scale).round(), 0, 255).to(torch.uint8)
        qweight = torch.clamp((weight / scale + zero_point).round(), 0, 255).to(torch.uint8)
    else:
        # Symmetric quantization: use range [-127, 127] for int8
        w_max = weight.abs().amax(dim=reduce_dim, keepdim=True)
        scale = torch.clamp(w_max / 127.0, min=1e-8)
        qweight = torch.clamp((weight / scale).round(), -128, 127).to(torch.int8)
        zero_point = None
    
    return qweight, scale.to(torch.float16), zero_point

def apply_optimized_quantization(model, use_asymmetric=False, quant_dtype="float32", 
                                use_int8_matmul=True):
    """Apply quantization with optimized inference paths to a neural network model"""
    quant_count = 0
    
    def _quantize_module(module, prefix=""):
        nonlocal quant_count
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            # Skip text encoder and CLIP-related modules to avoid conditioning issues
            if any(skip_name in full_name.lower() for skip_name in 
                   ['text_encoder', 'clip', 'embedder', 'conditioner']):
                print(f"⏭️  Skipping {full_name} (text/conditioning module)")
                _quantize_module(child, full_name)
                continue
            
            
            if isinstance(child, (nn.Linear, nn.Conv2d)):
                try:
                    # Extract and quantize weights
                    W = child.weight.data.float()
                    qW, scale, zp = quantize_weight(W, use_asymmetric=use_asymmetric)
                    
                    # Store original device info before removing weight
                    original_device = child.weight.device
                    
                    # Remove original weight parameter to save memory
                    del child._parameters["weight"]
                    
                    # Register quantized parameters as buffers (non-trainable)
                    # Keep them on CPU initially to save GPU memory
                    child.register_buffer("int8_weight", qW.to(original_device))
                    child.register_buffer("scale", scale.to(original_device))
                    if zp is not None:
                        child.register_buffer("zero_point", zp.to(original_device))
                    else:
                        child.zero_point = None
                    
                    # Configure optimization settings for this layer
                    if isinstance(child, nn.Linear) and not use_asymmetric and use_int8_matmul:
                        # Enable optimized matmul for symmetric quantized linear layers
                        child._use_optimized_matmul = True
                        # Transpose weight for optimized matmul layout
                        child.int8_weight = child.int8_weight.transpose(0, 1).contiguous()
                        # Adjust scale dimensions for matmul
                        child.scale = child.scale.squeeze(-1)
                    else:
                        child._use_optimized_matmul = False
                    
                    # Assign optimized forward function
                    child.forward = make_optimized_quantized_forward(
                        quant_dtype, use_int8_matmul
                    ).__get__(child)
                    
                    quant_count += 1
                    opt_status = "optimized" if child._use_optimized_matmul else "standard"
                    # print(f"✅ Quantized {full_name} ({opt_status})")
                    
                except Exception as e:
                    print(f"❌ Failed to quantize {full_name}: {str(e)}")
            
            # Recursively process child modules
            _quantize_module(child, full_name)
    
    _quantize_module(model)
    print(f"✅ Successfully quantized {quant_count} layers with optimized inference")
    return model

# ---------------------- ComfyUI Node Implementations ------------------------

class CheckpointLoaderQuantized2:
    """Original checkpoint loader with quantization"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                "enable_quant": ("BOOLEAN", {"default": True}),
                "use_asymmetric": ("BOOLEAN", {"default": False}),
                "quant_dtype": (["float32", "float16"], {"default": "float32"}),
                "use_int8_matmul": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_quantized"
    CATEGORY = "CFZ/loaders"
    OUTPUT_NODE = False

    def load_quantized(self, ckpt_name, enable_quant, use_asymmetric, quant_dtype, 
                      use_int8_matmul):
        """Load and optionally quantize a checkpoint with optimized inference"""
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint {ckpt_name} not found at {ckpt_path}")

        # Load checkpoint using ComfyUI's standard loading mechanism
        model_patcher, clip, vae, _ = load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings")
        )

        if enable_quant:
            # Determine quantization configuration
            quant_mode = "Asymmetric" if use_asymmetric else "Symmetric"
            matmul_mode = "Optimized Int8" if use_int8_matmul and not use_asymmetric else "Standard"
            
            print(f"🔧 Applying {quant_mode} 8-bit quantization to {ckpt_name}")
            print(f"   MatMul: {matmul_mode}, Forward: Optimized (dtype={quant_dtype})")
            
            # Apply quantization with optimizations
            apply_optimized_quantization(
                model_patcher.model, 
                use_asymmetric=use_asymmetric, 
                quant_dtype=quant_dtype,
                use_int8_matmul=use_int8_matmul
            )
        else:
            print(f"🔧 Loading {ckpt_name} without quantization")

        return (model_patcher, clip, vae)


class ModelQuantizationPatcher:
    """Quantization patcher that can be applied to any model in the workflow"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "use_asymmetric": ("BOOLEAN", {"default": False}),
                "quant_dtype": (["float32", "float16"], {"default": "float32"}),
                "use_int8_matmul": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch_model"
    CATEGORY = "CFZ/model patches"
    OUTPUT_NODE = False

    def patch_model(self, model, use_asymmetric, quant_dtype, use_int8_matmul):
        """Apply quantization to an existing model"""
        # Clone the model to avoid modifying the original
        import copy
        quantized_model = copy.deepcopy(model)
        
        # Determine quantization configuration
        quant_mode = "Asymmetric" if use_asymmetric else "Symmetric"
        matmul_mode = "Optimized Int8" if use_int8_matmul and not use_asymmetric else "Standard"
        
        print(f"🔧 Applying {quant_mode} 8-bit quantization to model")
        print(f"   MatMul: {matmul_mode}, Forward: Optimized (dtype={quant_dtype})")
        
        # Apply quantization with optimizations
        apply_optimized_quantization(
            quantized_model.model, 
            use_asymmetric=use_asymmetric, 
            quant_dtype=quant_dtype,
            use_int8_matmul=use_int8_matmul
        )
        
        return (quantized_model,)


class UNetQuantizationPatcher:
    """Specialized quantization patcher for UNet models loaded separately"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "use_asymmetric": ("BOOLEAN", {"default": False}),
                "quant_dtype": (["float32", "float16"], {"default": "float32"}),
                "use_int8_matmul": ("BOOLEAN", {"default": True}),
                "skip_input_blocks": ("BOOLEAN", {"default": False}),
                "skip_output_blocks": ("BOOLEAN", {"default": False}),
                "show_memory_usage": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch_unet"
    CATEGORY = "CFZ/model patches"
    OUTPUT_NODE = False

    def get_model_memory_usage(self, model, force_calculation=False):
        """Calculate memory usage of model parameters (CPU + GPU)"""
        total_memory = 0
        param_count = 0
        gpu_memory = 0
        
        # Count all parameters (CPU + GPU)
        for param in model.parameters():
            memory_bytes = param.data.element_size() * param.data.nelement()
            total_memory += memory_bytes
            param_count += param.data.nelement()
            
            if param.data.is_cuda:
                gpu_memory += memory_bytes
        
        # Also check for quantized buffers
        for name, buffer in model.named_buffers():
            if 'int8_weight' in name or 'scale' in name or 'zero_point' in name:
                memory_bytes = buffer.element_size() * buffer.nelement()
                total_memory += memory_bytes
                
                if buffer.is_cuda:
                    gpu_memory += memory_bytes
        
        # If force_calculation is True and nothing on GPU, return total memory as estimate
        if force_calculation and gpu_memory == 0:
            return total_memory, param_count, total_memory
        
        return total_memory, param_count, gpu_memory

    def format_memory_size(self, bytes_size):
        """Format memory size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.2f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.2f} TB"

    def patch_unet(self, model, use_asymmetric, quant_dtype, use_int8_matmul, 
                   skip_input_blocks, skip_output_blocks, show_memory_usage):
        """Apply selective quantization to UNet model with block-level control"""
        import copy
        
        # Measure original memory usage
        if show_memory_usage:
            original_memory, original_params, original_gpu = self.get_model_memory_usage(model.model, force_calculation=True)
            print(f"📊 Original Model Memory Usage:")
            print(f"   Parameters: {original_params:,}")
            print(f"   Total Size: {self.format_memory_size(original_memory)}")
            if original_gpu > 0:
                print(f"   GPU Memory: {self.format_memory_size(original_gpu)}")
            else:
                print(f"   GPU Memory: Not loaded (will use ~{self.format_memory_size(original_memory)} when loaded)")
        
        quantized_model = copy.deepcopy(model)
        
        # Determine quantization configuration
        quant_mode = "Asymmetric" if use_asymmetric else "Symmetric"
        matmul_mode = "Optimized Int8" if use_int8_matmul and not use_asymmetric else "Standard"
        
        print(f"🔧 Applying {quant_mode} 8-bit quantization to UNet")
        print(f"   MatMul: {matmul_mode}, Forward: Optimized (dtype={quant_dtype})")
        
        if skip_input_blocks or skip_output_blocks:
            print(f"   Skipping: Input blocks={skip_input_blocks}, Output blocks={skip_output_blocks}")
        
        # Apply quantization with selective skipping
        self._apply_selective_quantization(
            quantized_model.model,
            use_asymmetric=use_asymmetric,
            quant_dtype=quant_dtype,
            use_int8_matmul=use_int8_matmul,
            skip_input_blocks=skip_input_blocks,
            skip_output_blocks=skip_output_blocks
        )
        
        # Measure quantized memory usage
        if show_memory_usage:
            quantized_memory, quantized_params, quantized_gpu = self.get_model_memory_usage(quantized_model.model, force_calculation=True)
            memory_saved = original_memory - quantized_memory
            memory_reduction_pct = (memory_saved / original_memory) * 100 if original_memory > 0 else 0
            
            print(f"📊 Quantized Model Memory Usage:")
            print(f"   Parameters: {quantized_params:,}")
            print(f"   Total Size: {self.format_memory_size(quantized_memory)}")
            if quantized_gpu > 0:
                print(f"   GPU Memory: {self.format_memory_size(quantized_gpu)}")
            else:
                print(f"   GPU Memory: Not loaded (will use ~{self.format_memory_size(quantized_memory)} when loaded)")
            print(f"   Memory Saved: {self.format_memory_size(memory_saved)} ({memory_reduction_pct:.1f}%)")
            
            # Show CUDA memory info if available
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                print(f"📊 Total GPU Memory Status:")
                print(f"   Currently Allocated: {self.format_memory_size(allocated)}")
                print(f"   Reserved by PyTorch: {self.format_memory_size(reserved)}")
        
        return (quantized_model,)
    
    def _apply_selective_quantization(self, model, use_asymmetric=False, quant_dtype="float32", 
                                     use_int8_matmul=True, skip_input_blocks=False, 
                                     skip_output_blocks=False):
        """Apply quantization with selective block skipping for UNet"""
        quant_count = 0
        
        def _quantize_module(module, prefix=""):
            nonlocal quant_count
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                
                # Skip blocks based on user preference
                if skip_input_blocks and "input_blocks" in full_name:
                    print(f"⏭️  Skipping {full_name} (input block)")
                    _quantize_module(child, full_name)
                    continue
                    
                if skip_output_blocks and "output_blocks" in full_name:
                    print(f"⏭️  Skipping {full_name} (output block)")
                    _quantize_module(child, full_name)
                    continue
                
                # Skip text encoder and CLIP-related modules
                if any(skip_name in full_name.lower() for skip_name in 
                       ['text_encoder', 'clip', 'embedder', 'conditioner']):
                    print(f"⏭️  Skipping {full_name} (text/conditioning module)")
                    _quantize_module(child, full_name)
                    continue
                
                if isinstance(child, (nn.Linear, nn.Conv2d)):
                    try:
                        # Extract and quantize weights
                        W = child.weight.data.float()
                        qW, scale, zp = quantize_weight(W, use_asymmetric=use_asymmetric)
                        
                        # Store original device info before removing weight
                        original_device = child.weight.device
                        
                        # Remove original weight parameter to save memory
                        del child._parameters["weight"]
                        
                        # Register quantized parameters as buffers (non-trainable)
                        child.register_buffer("int8_weight", qW.to(original_device))
                        child.register_buffer("scale", scale.to(original_device))
                        if zp is not None:
                            child.register_buffer("zero_point", zp.to(original_device))
                        else:
                            child.zero_point = None
                        
                        # Configure optimization settings for this layer
                        if isinstance(child, nn.Linear) and not use_asymmetric and use_int8_matmul:
                            # Enable optimized matmul for symmetric quantized linear layers
                            child._use_optimized_matmul = True
                            # Transpose weight for optimized matmul layout
                            child.int8_weight = child.int8_weight.transpose(0, 1).contiguous()
                            # Adjust scale dimensions for matmul
                            child.scale = child.scale.squeeze(-1)
                        else:
                            child._use_optimized_matmul = False
                        
                        # Assign optimized forward function
                        child.forward = make_optimized_quantized_forward(
                            quant_dtype, use_int8_matmul
                        ).__get__(child)
                        
                        quant_count += 1
                        
                    except Exception as e:
                        print(f"❌ Failed to quantize {full_name}: {str(e)}")
                
                # Recursively process child modules
                _quantize_module(child, full_name)
        
        _quantize_module(model)
        print(f"✅ Successfully quantized {quant_count} layers with selective patching")

# ------------------------- Node Registration -------------------------------
NODE_CLASS_MAPPINGS = {
    "CheckpointLoaderQuantized2": CheckpointLoaderQuantized2,
    "ModelQuantizationPatcher": ModelQuantizationPatcher,
    "UNetQuantizationPatcher": UNetQuantizationPatcher,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CheckpointLoaderQuantized2": "CFZ Checkpoint Loader (Optimized)",
    "ModelQuantizationPatcher": "CFZ Model Quantization Patcher",
    "UNetQuantizationPatcher": "CFZ UNet Quantization Patcher",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
