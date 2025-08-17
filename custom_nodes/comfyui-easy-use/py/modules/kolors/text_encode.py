import re
import random
import gc
import comfy.model_management as mm
from nodes import ConditioningConcat, ConditioningZeroOut, ConditioningSetTimestepRange, ConditioningCombine

def chatglm3_text_encode(chatglm3_model, prompt, clean_gpu=False):
    device = mm.get_torch_device()
    offload_device = mm.unet_offload_device()
    if clean_gpu:
        mm.unload_all_models()
        mm.soft_empty_cache()
    # Function to randomly select an option from the brackets

    def choose_random_option(match):
        options = match.group(1).split('|')
        return random.choice(options)

    prompt = re.sub(r'\{([^{}]*)\}', choose_random_option, prompt)

    if "|" in prompt:
        prompt = prompt.split("|")

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)

    # Define tokenizers and text encoders
    tokenizer = chatglm3_model['tokenizer']
    text_encoder = chatglm3_model['text_encoder']
    text_encoder.to(device)
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=256,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    output = text_encoder(
        input_ids=text_inputs['input_ids'],
        attention_mask=text_inputs['attention_mask'],
        position_ids=text_inputs['position_ids'],
        output_hidden_states=True)

    # [batch_size, 77, 4096]
    prompt_embeds = output.hidden_states[-2].permute(1, 0, 2).clone()
    text_proj = output.hidden_states[-1][-1, :, :].clone()  # [batch_size, 4096]
    bs_embed, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, 1, 1)
    prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)

    bs_embed = text_proj.shape[0]
    text_proj = text_proj.repeat(1, 1).view(bs_embed, -1)
    text_encoder.to(offload_device)
    if clean_gpu:
        mm.soft_empty_cache()
        gc.collect()
    return [[prompt_embeds, {"pooled_output": text_proj},]]

def chatglm3_adv_text_encode(chatglm3_model, text, clean_gpu=False):
    time_start = 0
    time_end = 1
    match = re.search(r'TIMESTEP.*$', text)
    if match:
        timestep = match.group()
        timestep = timestep.split(' ')
        timestep = timestep[0]
        text = text.replace(timestep, '')
        value = timestep.split(':')
        if len(value) >= 3:
            time_start = float(value[1])
            time_end = float(value[2])
        elif len(value) == 2:
            time_start = float(value[1])
            time_end = 1
        elif len(value) == 1:
            time_start = 0.1
            time_end = 1


    pass3 = [x.strip() for x in text.split("BREAK")]
    pass3 = [x for x in pass3 if x != '']

    if len(pass3) == 0:
        pass3 = ['']

    conditioning = None

    for text in pass3:
        cond = chatglm3_text_encode(chatglm3_model, text, clean_gpu)
        if conditioning is not None:
            conditioning = ConditioningConcat().concat(conditioning, cond)[0]
        else:
            conditioning = cond

    # setTimeStepRange
    if time_start > 0 or time_end < 1:
        conditioning_2, = ConditioningSetTimestepRange().set_range(conditioning, 0, time_start)
        conditioning_1, = ConditioningZeroOut().zero_out(conditioning)
        conditioning_1, = ConditioningSetTimestepRange().set_range(conditioning_1, time_start, time_end)
        conditioning, = ConditioningCombine().combine(conditioning_1, conditioning_2)

    return conditioning