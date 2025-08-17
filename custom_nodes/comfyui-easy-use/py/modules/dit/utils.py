import torch
from comfy import model_management

def string_to_dtype(s="none", mode=None):
	s = s.lower().strip()
	if s in ["default", "as-is"]:
		return None
	elif s in ["auto", "auto (comfy)"]:
		if mode == "vae":
			return model_management.vae_device()
		elif mode == "text_encoder":
			return model_management.text_encoder_dtype()
		elif mode == "unet":
			return model_management.unet_dtype()
		else:
			raise NotImplementedError(f"Unknown dtype mode '{mode}'")
	elif s in ["none", "auto (hf)", "auto (hf/bnb)"]:
		return None
	elif s in ["fp32", "float32", "float"]:
		return torch.float32
	elif s in ["bf16", "bfloat16"]:
		return torch.bfloat16
	elif s in ["fp16", "float16", "half"]:
		return torch.float16
	elif "fp8" in s or "float8" in s:
		if "e5m2" in s:
			return torch.float8_e5m2
		elif "e4m3" in s:
			return torch.float8_e4m3fn
		else:
			raise NotImplementedError(f"Unknown 8bit dtype '{s}'")
	elif "bnb" in s:
		assert s in ["bnb8bit", "bnb4bit"], f"Unknown bnb mode '{s}'"
		return s
	elif s is None:
		return None
	else:
		raise NotImplementedError(f"Unknown dtype '{s}'")