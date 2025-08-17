from .utils import find_wildcards_seed, find_nearest_steps, is_linked_styles_selector
from .log import log_node_warn
from .translate import zh_to_en, has_chinese
from .wildcards import process_with_loras
from .adv_encode import advanced_encode

from nodes import ConditioningConcat, ConditioningCombine, ConditioningAverage, ConditioningSetTimestepRange, CLIPTextEncode

def prompt_to_cond(type, model, clip, clip_skip, lora_stack, text, prompt_token_normalization, prompt_weight_interpretation, a1111_prompt_style ,my_unique_id, prompt, easyCache, can_load_lora=True, steps=None, model_type=None):
    styles_selector = is_linked_styles_selector(prompt, my_unique_id, type)
    title = "Positive encoding" if type == 'positive' else "Negative encoding"

    # Translate cn to en
    if model_type not in ['hydit'] and text is not None and has_chinese(text):
        text = zh_to_en([text])[0]

    if model_type in ['hydit', 'flux', 'mochi']:
        log_node_warn(title + "...")
        embeddings_final, = CLIPTextEncode().encode(clip, text) if text is not None else (None,)

        return (embeddings_final, "", model, clip)

    log_node_warn(title + "...")

    positive_seed = find_wildcards_seed(my_unique_id, text, prompt)
    model, clip, text, cond_decode, show_prompt, pipe_lora_stack = process_with_loras(
        text, model, clip, type, positive_seed, can_load_lora, lora_stack, easyCache)
    wildcard_prompt = cond_decode if show_prompt or styles_selector else ""

    clipped = clip.clone()
    # 当clip模型不存在t5xxl时，可执行跳过层
    if not hasattr(clip.cond_stage_model, 't5xxl'):
        if clip_skip != 0:
            clipped.clip_layer(clip_skip)

    steps = steps if steps is not None else find_nearest_steps(my_unique_id, prompt)
    return (advanced_encode(clipped, text, prompt_token_normalization,
                            prompt_weight_interpretation, w_max=1.0,
                            apply_to_pooled='enable',
                            a1111_prompt_style=a1111_prompt_style, steps=steps) if text is not None else None, wildcard_prompt, model, clipped)

def set_cond(old_cond, new_cond, mode, average_strength, old_cond_start, old_cond_end, new_cond_start, new_cond_end):
    if not old_cond:
        return new_cond
    else:
        if mode == "replace":
            return new_cond
        elif mode == "concat":
            return ConditioningConcat().concat(new_cond, old_cond)[0]
        elif mode == "combine":
            return ConditioningCombine().combine(old_cond, new_cond)[0]
        elif mode == 'average':
            return ConditioningAverage().addWeighted(new_cond, old_cond, average_strength)[0]
        elif mode == 'timestep':
            cond_1 = ConditioningSetTimestepRange().set_range(old_cond, old_cond_start, old_cond_end)[0]
            cond_2 = ConditioningSetTimestepRange().set_range(new_cond, new_cond_start, new_cond_end)[0]
            return ConditioningCombine().combine(cond_1, cond_2)[0]