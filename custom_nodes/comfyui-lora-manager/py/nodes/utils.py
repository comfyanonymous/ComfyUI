class AnyType(str):
  """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""

  def __ne__(self, __value: object) -> bool:
    return False

# Credit to Regis Gaughan, III (rgthree)
class FlexibleOptionalInputType(dict):
  """A special class to make flexible nodes that pass data to our python handlers.

  Enables both flexible/dynamic input types (like for Any Switch) or a dynamic number of inputs
  (like for Any Switch, Context Switch, Context Merge, Power Lora Loader, etc).

  Note, for ComfyUI, all that's needed is the `__contains__` override below, which tells ComfyUI
  that our node will handle the input, regardless of what it is.

  However, with https://github.com/comfyanonymous/ComfyUI/pull/2666 a large change would occur
  requiring more details on the input itself. There, we need to return a list/tuple where the first
  item is the type. This can be a real type, or use the AnyType for additional flexibility.

  This should be forwards compatible unless more changes occur in the PR.
  """
  def __init__(self, type):
    self.type = type

  def __getitem__(self, key):
    return (self.type, )

  def __contains__(self, key):
    return True


any_type = AnyType("*")

# Common methods extracted from lora_loader.py and lora_stacker.py
import os
import logging
import copy
import folder_paths

logger = logging.getLogger(__name__)

def extract_lora_name(lora_path):
    """Extract the lora name from a lora path (e.g., 'IL\\aorunIllstrious.safetensors' -> 'aorunIllstrious')"""
    # Get the basename without extension
    basename = os.path.basename(lora_path)
    return os.path.splitext(basename)[0]

def get_loras_list(kwargs):
    """Helper to extract loras list from either old or new kwargs format"""
    if 'loras' not in kwargs:
        return []
        
    loras_data = kwargs['loras']
    # Handle new format: {'loras': {'__value__': [...]}}
    if isinstance(loras_data, dict) and '__value__' in loras_data:
        return loras_data['__value__']
    # Handle old format: {'loras': [...]}
    elif isinstance(loras_data, list):
        return loras_data
    # Unexpected format
    else:
        logger.warning(f"Unexpected loras format: {type(loras_data)}")
        return []

def load_state_dict_in_safetensors(path, device="cpu", filter_prefix=""):
    """Simplified version of load_state_dict_in_safetensors that just loads from a local path"""  
    import safetensors.torch
    
    state_dict = {}
    with safetensors.torch.safe_open(path, framework="pt", device=device) as f:
        for k in f.keys():
            if filter_prefix and not k.startswith(filter_prefix):
                continue
            state_dict[k.removeprefix(filter_prefix)] = f.get_tensor(k)
    return state_dict

def to_diffusers(input_lora):
    """Simplified version of to_diffusers for Flux LoRA conversion"""
    import torch
    from diffusers.utils.state_dict_utils import convert_unet_state_dict_to_peft
    from diffusers.loaders import FluxLoraLoaderMixin
    
    if isinstance(input_lora, str):
        tensors = load_state_dict_in_safetensors(input_lora, device="cpu")
    else:
        tensors = {k: v for k, v in input_lora.items()}

    # Convert FP8 tensors to BF16
    for k, v in tensors.items():
        if v.dtype not in [torch.float64, torch.float32, torch.bfloat16, torch.float16]:
            tensors[k] = v.to(torch.bfloat16)
    
    new_tensors = FluxLoraLoaderMixin.lora_state_dict(tensors)
    new_tensors = convert_unet_state_dict_to_peft(new_tensors)

    return new_tensors

def nunchaku_load_lora(model, lora_name, lora_strength):
    """Load a Flux LoRA for Nunchaku model"""   
    model_wrapper = model.model.diffusion_model
    transformer = model_wrapper.model
    
    # Save the transformer temporarily
    model_wrapper.model = None
    ret_model = copy.deepcopy(model)  # copy everything except the model
    ret_model_wrapper = ret_model.model.diffusion_model
    
    # Restore the model and set it for the copy
    model_wrapper.model = transformer
    ret_model_wrapper.model = transformer
    
    # Get full path to the LoRA file
    lora_path = folder_paths.get_full_path("loras", lora_name)
    ret_model_wrapper.loras.append((lora_path, lora_strength))
    
    # Convert the LoRA to diffusers format
    sd = to_diffusers(lora_path)
    
    # Handle embedding adjustment if needed
    if "transformer.x_embedder.lora_A.weight" in sd:
        new_in_channels = sd["transformer.x_embedder.lora_A.weight"].shape[1]
        assert new_in_channels % 4 == 0
        new_in_channels = new_in_channels // 4
        
        old_in_channels = ret_model.model.model_config.unet_config["in_channels"]
        if old_in_channels < new_in_channels:
            ret_model.model.model_config.unet_config["in_channels"] = new_in_channels
    
    return ret_model