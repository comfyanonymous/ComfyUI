import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from comfy.sd import load_checkpoint_guess_config, load_checkpoint
from comfy.model_patcher import ModelPatcher
import folder_paths

# ------------------------ Core Quantization Logic -------------------------
def make_quantized_forward(quant_dtype="float32"):
    def forward(self, x):
        dtype = torch.float32 if quant_dtype == "float32" else torch.float16

        W = self.int8_weight.to(x.device, dtype=dtype)
        if hasattr(self, 'zero_point') and self.zero_point is not None:
            zp = self.zero_point.to(x.device, dtype=dtype)
            W = W.sub(zp).mul(self.scale)
        else:
            W = W.mul(self.scale)

        bias = self.bias.to(dtype) if self.bias is not None else None

        # LoRA application (if present)
        if hasattr(self, "lora_down") and hasattr(self, "lora_up") and hasattr(self, "lora_alpha"):
            x = x + self.lora_up(self.lora_down(x)) * self.lora_alpha

        x = x.to(dtype)

        if isinstance(self, nn.Linear):
            return F.linear(x, W, bias)
        elif isinstance(self, nn.Conv2d):
            return F.conv2d(x, W, bias,
                            self.stride, self.padding,
                            self.dilation, self.groups)
        else:
            return x
    return forward



def quantize_weight(weight: torch.Tensor, num_bits=8, use_asymmetric=False):
    reduce_dim = 1 if weight.ndim == 2 else [i for i in range(weight.ndim) if i != 0]
    if use_asymmetric:
        min_val = weight.amin(dim=reduce_dim, keepdim=True)
        max_val = weight.amax(dim=reduce_dim, keepdim=True)
        scale = torch.clamp((max_val - min_val) / 255.0, min=1e-8)
        zero_point = torch.clamp((-min_val / scale).round(), 0, 255).to(torch.uint8)
        qweight = torch.clamp((weight / scale + zero_point).round(), 0, 255).to(torch.uint8)
    else:
        w_max = weight.abs().amax(dim=reduce_dim, keepdim=True)
        scale = torch.clamp(w_max / 127.0, min=1e-8)
        qweight = torch.clamp((weight / scale).round(), -128, 127).to(torch.int8)
        zero_point = None

    return qweight, scale.to(torch.float16), zero_point


def apply_quantization(model, use_asymmetric=False, quant_dtype="float32"):
    quant_count = 0

    def _quantize_module(module, prefix=""):
        nonlocal quant_count
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, (nn.Linear, nn.Conv2d)):
                try:
                    W = child.weight.data.float()
                    qW, scale, zp = quantize_weight(W, use_asymmetric=use_asymmetric)

                    del child._parameters["weight"]
                    child.register_buffer("int8_weight", qW)
                    child.register_buffer("scale", scale)
                    if zp is not None:
                        child.register_buffer("zero_point", zp)
                    else:
                        child.zero_point = None

                    child.forward = make_quantized_forward(quant_dtype).__get__(child)
                    quant_count += 1
                except Exception as e:
                    print(f"Failed to quantize {full_name}: {str(e)}")

            _quantize_module(child, full_name)

    _quantize_module(model)
    print(f"✅ Successfully quantized {quant_count} layers")
    return model

# ---------------------- ComfyUI Node Implementation ------------------------
class CheckpointLoaderQuantized:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                "enable_quant": ("BOOLEAN", {"default": True}),
                "use_asymmetric": ("BOOLEAN", {"default": False}),
                "quant_dtype": (["float32", "float16"], {"default": "float32"}),  # Toggle for precision
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_quantized"
    CATEGORY = "Loaders (Quantized)"
    OUTPUT_NODE = False

    def load_quantized(self, ckpt_name, enable_quant, use_asymmetric, quant_dtype):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint {ckpt_name} not found at {ckpt_path}")

        model_patcher, clip, vae, _ = load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings")
        )

        if enable_quant:
            mode = "Asymmetric" if use_asymmetric else "Symmetric"
            print(f"🔧 Applying {mode} 8-bit quantization to {ckpt_name} (dtype={quant_dtype})")
            apply_quantization(model_patcher.model, use_asymmetric=use_asymmetric, quant_dtype=quant_dtype)
        else:
            print(f"🔧 Loading {ckpt_name} without quantization")

        return (model_patcher, clip, vae)

# ------------------------- Node Registration -------------------------------
NODE_CLASS_MAPPINGS = {
    "CheckpointLoaderQuantized": CheckpointLoaderQuantized,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CheckpointLoaderQuantized": "CFZ Checkpoint Loader",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
