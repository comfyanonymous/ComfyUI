
checkpoint_path2 = "/apdcephfs_cq8/share_1367250/xuhuaren/kaiyuan/fork/ComfyUI/models/loras/adapter_model.safetensors"
save_path = "/apdcephfs_cq8/share_1367250/xuhuaren/kaiyuan/fork/ComfyUI/models/loras/adapter_model_convert_diffusers.safetensors"
from safetensors import safe_open
from safetensors.torch import save_file


lora_state_dict = {}
with safe_open(checkpoint_path2, framework="pt", device=0) as f:
    for k in f.keys():
        new_key = "lora_unet_" + "_".join(k[17:].split("."))
        lora_state_dict[new_key] = f.get_tensor(k) # remove 'basemodel.model'

save_file(lora_state_dict, save_path)

     
