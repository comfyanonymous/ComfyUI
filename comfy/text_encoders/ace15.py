from .anima import Qwen3Tokenizer
import comfy.text_encoders.llama
from comfy import sd1_clip
import torch
import math
import comfy.utils


def sample_manual_loop_no_classes(
    model,
    ids=None,
    paddings=[],
    execution_dtype=None,
    cfg_scale: float = 2.0,
    temperature: float = 0.85,
    top_p: float = 0.9,
    top_k: int = None,
    seed: int = 1,
    min_tokens: int = 1,
    max_new_tokens: int = 2048,
    audio_start_id: int = 151669,  # The cutoff ID for audio codes
    eos_token_id: int = 151645,
):
    device = model.execution_device

    if execution_dtype is None:
        if comfy.model_management.should_use_bf16(device):
            execution_dtype = torch.bfloat16
        else:
            execution_dtype = torch.float32

    embeds, attention_mask, num_tokens, embeds_info = model.process_tokens(ids, device)
    for i, t in enumerate(paddings):
        attention_mask[i, :t] = 0
        attention_mask[i, t:] = 1

    output_audio_codes = []
    past_key_values = []
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    model_config = model.transformer.model.config

    for x in range(model_config.num_hidden_layers):
        past_key_values.append((torch.empty([embeds.shape[0], model_config.num_key_value_heads, embeds.shape[1] + min_tokens, model_config.head_dim], device=device, dtype=execution_dtype), torch.empty([embeds.shape[0], model_config.num_key_value_heads, embeds.shape[1] + min_tokens, model_config.head_dim], device=device, dtype=execution_dtype), 0))

    progress_bar = comfy.utils.ProgressBar(max_new_tokens)

    for step in range(max_new_tokens):
        outputs = model.transformer(None, attention_mask, embeds=embeds.to(execution_dtype), num_tokens=num_tokens, intermediate_output=None, dtype=execution_dtype, embeds_info=embeds_info, past_key_values=past_key_values)
        next_token_logits = model.transformer.logits(outputs[0])[:, -1]
        past_key_values = outputs[2]

        cond_logits = next_token_logits[0:1]
        uncond_logits = next_token_logits[1:2]
        cfg_logits = uncond_logits + cfg_scale * (cond_logits - uncond_logits)

        if eos_token_id is not None and eos_token_id < audio_start_id and min_tokens < step:
            eos_score = cfg_logits[:, eos_token_id].clone()

        remove_logit_value = torch.finfo(cfg_logits.dtype).min
        # Only generate audio tokens
        cfg_logits[:, :audio_start_id] = remove_logit_value

        if eos_token_id is not None and eos_token_id < audio_start_id and min_tokens < step:
            cfg_logits[:, eos_token_id] = eos_score

        if top_k is not None and top_k > 0:
            top_k_vals, _ = torch.topk(cfg_logits, top_k)
            min_val = top_k_vals[..., -1, None]
            cfg_logits[cfg_logits < min_val] = remove_logit_value

        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(cfg_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            cfg_logits[indices_to_remove] = remove_logit_value

        if temperature > 0:
            cfg_logits = cfg_logits / temperature
            next_token = torch.multinomial(torch.softmax(cfg_logits, dim=-1), num_samples=1, generator=generator).squeeze(1)
        else:
            next_token = torch.argmax(cfg_logits, dim=-1)

        token = next_token.item()

        if token == eos_token_id:
            break

        embed, _, _, _ = model.process_tokens([[token]], device)
        embeds = embed.repeat(2, 1, 1)
        attention_mask = torch.cat([attention_mask, torch.ones((2, 1), device=device, dtype=attention_mask.dtype)], dim=1)

        output_audio_codes.append(token - audio_start_id)
        progress_bar.update_absolute(step)

    return output_audio_codes


def generate_audio_codes(model, positive, negative, min_tokens=1, max_tokens=1024, seed=0):
    cfg_scale = 2.0

    positive = [[token for token, _ in inner_list] for inner_list in positive]
    negative = [[token for token, _ in inner_list] for inner_list in negative]
    positive = positive[0]
    negative = negative[0]

    neg_pad = 0
    if len(negative) < len(positive):
        neg_pad = (len(positive) - len(negative))
        negative = [model.special_tokens["pad"]] * neg_pad + negative

    pos_pad = 0
    if len(negative) > len(positive):
        pos_pad = (len(negative) - len(positive))
        positive = [model.special_tokens["pad"]] * pos_pad + positive

    paddings = [pos_pad, neg_pad]
    return sample_manual_loop_no_classes(model, [positive, negative], paddings, cfg_scale=cfg_scale, seed=seed, min_tokens=min_tokens, max_new_tokens=max_tokens)


class ACE15Tokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, name="qwen3_06b", tokenizer=Qwen3Tokenizer)

    def tokenize_with_weights(self, text, return_word_ids=False, **kwargs):
        out = {}
        lyrics = kwargs.get("lyrics", "")
        bpm = kwargs.get("bpm", 120)
        duration = kwargs.get("duration", 120)
        keyscale = kwargs.get("keyscale", "C major")
        timesignature = kwargs.get("timesignature", 2)
        language = kwargs.get("language", "en")
        seed = kwargs.get("seed", 0)

        duration = math.ceil(duration)
        meta_lm = 'bpm: {}\nduration: {}\nkeyscale: {}\ntimesignature: {}'.format(bpm, duration, keyscale, timesignature)
        lm_template = "<|im_start|>system\n# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n<|im_end|>\n<|im_start|>user\n# Caption\n{}\n{}\n<|im_end|>\n<|im_start|>assistant\n<think>\n{}\n</think>\n\n<|im_end|>\n"

        meta_cap = '- bpm: {}\n- timesignature: {}\n- keyscale: {}\n- duration: {}\n'.format(bpm, timesignature, keyscale, duration)
        out["lm_prompt"] = self.qwen3_06b.tokenize_with_weights(lm_template.format(text, lyrics, meta_lm), disable_weights=True)
        out["lm_prompt_negative"] = self.qwen3_06b.tokenize_with_weights(lm_template.format(text, lyrics, ""), disable_weights=True)

        out["lyrics"] = self.qwen3_06b.tokenize_with_weights("# Languages\n{}\n\n# Lyric{}<|endoftext|><|endoftext|>".format(language, lyrics), return_word_ids, disable_weights=True, **kwargs)
        out["qwen3_06b"] = self.qwen3_06b.tokenize_with_weights("# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n# Caption\n{}# Metas\n{}<|endoftext|>\n<|endoftext|>".format(text, meta_cap), return_word_ids, **kwargs)
        out["lm_metadata"] = {"min_tokens": duration * 5, "seed": seed}
        return out


class Qwen3_06BModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None, attention_mask=True, model_options={}):
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config={}, dtype=dtype, special_tokens={"pad": 151643}, layer_norm_hidden_state=False, model_class=comfy.text_encoders.llama.Qwen3_06B_ACE15, enable_attention_masks=attention_mask, return_attention_masks=attention_mask, model_options=model_options)

class Qwen3_2B_ACE15(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None, attention_mask=True, model_options={}):
        llama_quantization_metadata = model_options.get("llama_quantization_metadata", None)
        if llama_quantization_metadata is not None:
            model_options = model_options.copy()
            model_options["quantization_metadata"] = llama_quantization_metadata

        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config={}, dtype=dtype, special_tokens={"pad": 151643}, layer_norm_hidden_state=False, model_class=comfy.text_encoders.llama.Qwen3_2B_ACE15_lm, enable_attention_masks=attention_mask, return_attention_masks=attention_mask, model_options=model_options)

class ACE15TEModel(torch.nn.Module):
    def __init__(self, device="cpu", dtype=None, dtype_llama=None, model_options={}):
        super().__init__()
        if dtype_llama is None:
            dtype_llama = dtype

        self.qwen3_06b = Qwen3_06BModel(device=device, dtype=dtype, model_options=model_options)
        self.qwen3_2b = Qwen3_2B_ACE15(device=device, dtype=dtype_llama, model_options=model_options)
        self.dtypes = set([dtype, dtype_llama])

    def encode_token_weights(self, token_weight_pairs):
        token_weight_pairs_base = token_weight_pairs["qwen3_06b"]
        token_weight_pairs_lyrics = token_weight_pairs["lyrics"]

        self.qwen3_06b.set_clip_options({"layer": None})
        base_out, _, extra = self.qwen3_06b.encode_token_weights(token_weight_pairs_base)
        self.qwen3_06b.set_clip_options({"layer": [0]})
        lyrics_embeds, _, extra_l = self.qwen3_06b.encode_token_weights(token_weight_pairs_lyrics)

        lm_metadata = token_weight_pairs["lm_metadata"]
        audio_codes = generate_audio_codes(self.qwen3_2b, token_weight_pairs["lm_prompt"], token_weight_pairs["lm_prompt_negative"], min_tokens=lm_metadata["min_tokens"], max_tokens=lm_metadata["min_tokens"], seed=lm_metadata["seed"])

        return base_out, None, {"conditioning_lyrics": lyrics_embeds[:, 0], "audio_codes": [audio_codes]}

    def set_clip_options(self, options):
        self.qwen3_06b.set_clip_options(options)
        self.qwen3_2b.set_clip_options(options)

    def reset_clip_options(self):
        self.qwen3_06b.reset_clip_options()
        self.qwen3_2b.reset_clip_options()

    def load_sd(self, sd):
        if "model.layers.0.post_attention_layernorm.weight" in sd:
            shape = sd["model.layers.0.post_attention_layernorm.weight"].shape
            if shape[0] == 1024:
                return self.qwen3_06b.load_sd(sd)
            else:
                return self.qwen3_2b.load_sd(sd)

    def memory_estimation_function(self, token_weight_pairs, device=None):
        lm_metadata = token_weight_pairs["lm_metadata"]
        constant = 0.4375
        if comfy.model_management.should_use_bf16(device):
            constant *= 0.5

        token_weight_pairs = token_weight_pairs.get("lm_prompt", [])
        num_tokens = sum(map(lambda a: len(a), token_weight_pairs))
        num_tokens += lm_metadata['min_tokens']
        return num_tokens * constant * 1024 * 1024

def te(dtype_llama=None, llama_quantization_metadata=None):
    class ACE15TEModel_(ACE15TEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["llama_quantization_metadata"] = llama_quantization_metadata
            super().__init__(device=device, dtype_llama=dtype_llama, dtype=dtype, model_options=model_options)
    return ACE15TEModel_
