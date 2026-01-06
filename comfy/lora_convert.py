import torch
import comfy.utils


def convert_lora_bfl_control(sd): #BFL loras for Flux
    sd_out = {}
    for k in sd:
        k_to = "diffusion_model.{}".format(k.replace(".lora_B.bias", ".diff_b").replace("_norm.scale", "_norm.scale.set_weight"))
        sd_out[k_to] = sd[k]

    sd_out["diffusion_model.img_in.reshape_weight"] = torch.tensor([sd["img_in.lora_B.weight"].shape[0], sd["img_in.lora_A.weight"].shape[1]])
    return sd_out


def convert_lora_wan_fun(sd): #Wan Fun loras
    return comfy.utils.state_dict_prefix_replace(sd, {"lora_unet__": "lora_unet_"})

def convert_uso_lora(sd):
    sd_out = {}
    for k in sd:
        tensor = sd[k]
        k_to = "diffusion_model.{}".format(k.replace(".down.weight", ".lora_down.weight")
                                           .replace(".up.weight", ".lora_up.weight")
                                           .replace(".qkv_lora2.", ".txt_attn.qkv.")
                                           .replace(".qkv_lora1.", ".img_attn.qkv.")
                                           .replace(".proj_lora1.", ".img_attn.proj.")
                                           .replace(".proj_lora2.", ".txt_attn.proj.")
                                           .replace(".qkv_lora.", ".linear1_qkv.")
                                           .replace(".proj_lora.", ".linear2.")
                                           .replace(".processor.", ".")
                                           )
        sd_out[k_to] = tensor
    return sd_out


def convert_lora(sd):
    if "img_in.lora_A.weight" in sd and "single_blocks.0.norm.key_norm.scale" in sd:
        return convert_lora_bfl_control(sd)
    if "lora_unet__blocks_0_cross_attn_k.lora_down.weight" in sd:
        return convert_lora_wan_fun(sd)
    if "single_blocks.37.processor.qkv_lora.up.weight" in sd and "double_blocks.18.processor.qkv_lora2.up.weight" in sd:
        return convert_uso_lora(sd)
    return sd
