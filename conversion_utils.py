import os

from safetensors.torch import load_file, save_file

from diffusers.utils import convert_all_state_dict_to_peft, convert_state_dict_to_kohya


def lora_convert_and_save(input_lora, output_lora):
    diffusers_state_dict = load_file(input_lora)
    peft_state_dict = convert_all_state_dict_to_peft(diffusers_state_dict)
    kohya_state_dict = convert_state_dict_to_kohya(peft_state_dict)
    save_file(kohya_state_dict, output_lora)