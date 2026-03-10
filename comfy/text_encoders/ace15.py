from .anima import Qwen3Tokenizer
import comfy.text_encoders.llama
from comfy import sd1_clip
import torch
import math
import yaml
import comfy.utils


def sample_manual_loop_no_classes(
    model,
    ids=None,
    execution_dtype=None,
    cfg_scale: float = 2.0,
    temperature: float = 0.85,
    top_p: float = 0.9,
    top_k: int = None,
    min_p: float = 0.000,
    seed: int = 1,
    min_tokens: int = 1,
    max_new_tokens: int = 2048,
    audio_start_id: int = 151669,  # The cutoff ID for audio codes
    audio_end_id: int = 215669,
    eos_token_id: int = 151645,
):
    if ids is None:
        return []
    device = model.execution_device

    if execution_dtype is None:
        if comfy.model_management.should_use_bf16(device):
            execution_dtype = torch.bfloat16
        else:
            execution_dtype = torch.float32

    embeds, attention_mask, num_tokens, embeds_info = model.process_tokens(ids, device)
    embeds_batch = embeds.shape[0]

    output_audio_codes = []
    past_key_values = []
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    model_config = model.transformer.model.config
    past_kv_shape = [embeds_batch, model_config.num_key_value_heads, embeds.shape[1] + min_tokens, model_config.head_dim]

    for x in range(model_config.num_hidden_layers):
        past_key_values.append((torch.empty(past_kv_shape, device=device, dtype=execution_dtype), torch.empty(past_kv_shape, device=device, dtype=execution_dtype), 0))

    progress_bar = comfy.utils.ProgressBar(max_new_tokens)

    for step in comfy.utils.model_trange(max_new_tokens, desc="LM sampling"):
        outputs = model.transformer(None, attention_mask, embeds=embeds.to(execution_dtype), num_tokens=num_tokens, intermediate_output=None, dtype=execution_dtype, embeds_info=embeds_info, past_key_values=past_key_values)
        next_token_logits = model.transformer.logits(outputs[0])[:, -1]
        past_key_values = outputs[2]

        if cfg_scale != 1.0:
            cond_logits = next_token_logits[0:1]
            uncond_logits = next_token_logits[1:2]
            cfg_logits = uncond_logits + cfg_scale * (cond_logits - uncond_logits)
        else:
            cfg_logits = next_token_logits[0:1]

        use_eos_score = eos_token_id is not None and eos_token_id < audio_start_id and min_tokens < step
        if use_eos_score:
            eos_score = cfg_logits[:, eos_token_id].clone()

        remove_logit_value = torch.finfo(cfg_logits.dtype).min
        # Only generate audio tokens
        cfg_logits[:, :audio_start_id] = remove_logit_value
        cfg_logits[:, audio_end_id:] = remove_logit_value

        if use_eos_score:
            cfg_logits[:, eos_token_id] = eos_score

        if top_k is not None and top_k > 0:
            top_k_vals, _ = torch.topk(cfg_logits, top_k)
            min_val = top_k_vals[..., -1, None]
            cfg_logits[cfg_logits < min_val] = remove_logit_value

        if min_p is not None and min_p > 0:
            probs = torch.softmax(cfg_logits, dim=-1)
            p_max = probs.max(dim=-1, keepdim=True).values
            indices_to_remove = probs < (min_p * p_max)
            cfg_logits[indices_to_remove] = remove_logit_value

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
        embeds = embed.repeat(embeds_batch, 1, 1)
        attention_mask = torch.cat([attention_mask, torch.ones((embeds_batch, 1), device=device, dtype=attention_mask.dtype)], dim=1)

        output_audio_codes.append(token - audio_start_id)
        progress_bar.update_absolute(step)

    return output_audio_codes


def generate_audio_codes(model, positive, negative, min_tokens=1, max_tokens=1024, seed=0, cfg_scale=2.0, temperature=0.85, top_p=0.9, top_k=0, min_p=0.000):
    positive = [[token for token, _ in inner_list] for inner_list in positive]
    positive = positive[0]

    if cfg_scale != 1.0:
        negative = [[token for token, _ in inner_list] for inner_list in negative]
        negative = negative[0]

        neg_pad = 0
        if len(negative) < len(positive):
            neg_pad = (len(positive) - len(negative))
            negative = [model.special_tokens["pad"]] * neg_pad + negative

        pos_pad = 0
        if len(negative) > len(positive):
            pos_pad = (len(negative) - len(positive))
            positive = [model.special_tokens["pad"]] * pos_pad + positive

        ids = [positive, negative]
    else:
        ids = [positive]

    return sample_manual_loop_no_classes(model, ids, cfg_scale=cfg_scale, temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p, seed=seed, min_tokens=min_tokens, max_new_tokens=max_tokens)


class ACE15Tokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, name="qwen3_06b", tokenizer=Qwen3Tokenizer)

    def _metas_to_cot(self, *, return_yaml: bool = False, **kwargs) -> str:
        user_metas = {
            k: kwargs.pop(k)
            for k in ("bpm", "duration", "keyscale", "timesignature")
            if k in kwargs
        }
        timesignature = user_metas.get("timesignature")
        if isinstance(timesignature, str) and timesignature.endswith("/4"):
            user_metas["timesignature"] = timesignature[:-2]
        user_metas = {
            k: v if not isinstance(v, str) or not v.isdigit() else int(v)
            for k, v in user_metas.items()
            if v not in {"unspecified", None}
        }
        if len(user_metas):
            meta_yaml = yaml.dump(user_metas, allow_unicode=True, sort_keys=True).strip()
        else:
            meta_yaml = ""
        return f"<think>\n{meta_yaml}\n</think>" if not return_yaml else meta_yaml

    def _metas_to_cap(self, **kwargs) -> str:
        use_keys = ("bpm", "timesignature", "keyscale", "duration")
        user_metas = { k: kwargs.pop(k, "N/A") for k in use_keys }
        timesignature = user_metas.get("timesignature")
        if isinstance(timesignature, str) and timesignature.endswith("/4"):
            user_metas["timesignature"] = timesignature[:-2]
        duration = user_metas["duration"]
        if duration == "N/A":
            user_metas["duration"] = "30 seconds"
        elif isinstance(duration, (str, int, float)):
            user_metas["duration"] = f"{math.ceil(float(duration))} seconds"
        else:
            raise TypeError("Unexpected type for duration key, must be str, int or float")
        return "\n".join(f"- {k}: {user_metas[k]}" for k in use_keys)

    def tokenize_with_weights(self, text, return_word_ids=False, **kwargs):
        text = text.strip()
        text_negative = kwargs.get("caption_negative", text).strip()
        lyrics = kwargs.get("lyrics", "")
        lyrics_negative = kwargs.get("lyrics_negative", lyrics)
        duration = kwargs.get("duration", 120)
        if isinstance(duration, str):
            duration = float(duration.split(None, 1)[0])
        language = kwargs.get("language")
        seed = kwargs.get("seed", 0)

        generate_audio_codes = kwargs.get("generate_audio_codes", True)
        cfg_scale = kwargs.get("cfg_scale", 2.0)
        temperature = kwargs.get("temperature", 0.85)
        top_p = kwargs.get("top_p", 0.9)
        top_k = kwargs.get("top_k", 0.0)
        min_p = kwargs.get("min_p", 0.000)

        duration = math.ceil(duration)
        kwargs["duration"] = duration
        tokens_duration = duration * 5
        min_tokens = int(kwargs.get("min_tokens", tokens_duration))
        max_tokens = int(kwargs.get("max_tokens", tokens_duration))

        metas_negative = {
            k.rsplit("_", 1)[0]: kwargs.pop(k)
            for k in ("bpm_negative", "duration_negative", "keyscale_negative", "timesignature_negative", "language_negative", "caption_negative")
            if k in kwargs
        }
        if not kwargs.get("use_negative_caption"):
            _ = metas_negative.pop("caption", None)

        cot_text = self._metas_to_cot(caption=text, **kwargs)
        cot_text_negative = "<think>\n\n</think>" if not metas_negative else self._metas_to_cot(**metas_negative)
        meta_cap = self._metas_to_cap(**kwargs)

        lm_template = "<|im_start|>system\n# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n<|im_end|>\n<|im_start|>user\n# Caption\n{}\n\n# Lyric\n{}\n<|im_end|>\n<|im_start|>assistant\n{}\n\n<|im_end|>\n"
        lyrics_template = "# Languages\n{}\n\n# Lyric\n{}<|endoftext|><|endoftext|>"
        qwen3_06b_template = "# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n# Caption\n{}\n\n# Metas\n{}\n<|endoftext|>\n<|endoftext|>"

        llm_prompts = {
            "lm_prompt": lm_template.format(text, lyrics.strip(), cot_text),
            "lm_prompt_negative": lm_template.format(text_negative, lyrics_negative.strip(), cot_text_negative),
            "lyrics": lyrics_template.format(language if language is not None else "", lyrics),
            "qwen3_06b": qwen3_06b_template.format(text, meta_cap),
        }

        out = {
            prompt_key: self.qwen3_06b.tokenize_with_weights(
                prompt,
                prompt_key == "qwen3_06b" and return_word_ids,
                disable_weights = True,
                **kwargs,
            )
            for prompt_key, prompt in llm_prompts.items()
        }
        out["lm_metadata"] = {"min_tokens": min_tokens,
                              "max_tokens": max_tokens,
                              "seed": seed,
                              "generate_audio_codes": generate_audio_codes,
                              "cfg_scale": cfg_scale,
                              "temperature": temperature,
                              "top_p": top_p,
                              "top_k": top_k,
                              "min_p": min_p,
                              }
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

class Qwen3_4B_ACE15(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None, attention_mask=True, model_options={}):
        llama_quantization_metadata = model_options.get("llama_quantization_metadata", None)
        if llama_quantization_metadata is not None:
            model_options = model_options.copy()
            model_options["quantization_metadata"] = llama_quantization_metadata

        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config={}, dtype=dtype, special_tokens={"pad": 151643}, layer_norm_hidden_state=False, model_class=comfy.text_encoders.llama.Qwen3_4B_ACE15_lm, enable_attention_masks=attention_mask, return_attention_masks=attention_mask, model_options=model_options)

class ACE15TEModel(torch.nn.Module):
    def __init__(self, device="cpu", dtype=None, dtype_llama=None, lm_model=None, model_options={}):
        super().__init__()
        if dtype_llama is None:
            dtype_llama = dtype

        model = None
        self.constant = 0.4375
        if lm_model == "qwen3_4b":
            model = Qwen3_4B_ACE15
            self.constant = 0.5625
        elif lm_model == "qwen3_2b":
            model = Qwen3_2B_ACE15

        self.lm_model = lm_model
        self.qwen3_06b = Qwen3_06BModel(device=device, dtype=dtype, model_options=model_options)
        if model is not None:
            setattr(self, self.lm_model, model(device=device, dtype=dtype_llama, model_options=model_options))

        self.dtypes = set([dtype, dtype_llama])

    def encode_token_weights(self, token_weight_pairs):
        token_weight_pairs_base = token_weight_pairs["qwen3_06b"]
        token_weight_pairs_lyrics = token_weight_pairs["lyrics"]

        self.qwen3_06b.set_clip_options({"layer": None})
        base_out, _, extra = self.qwen3_06b.encode_token_weights(token_weight_pairs_base)
        self.qwen3_06b.set_clip_options({"layer": [0]})
        lyrics_embeds, _, extra_l = self.qwen3_06b.encode_token_weights(token_weight_pairs_lyrics)

        out = {"conditioning_lyrics": lyrics_embeds[:, 0]}

        lm_metadata = token_weight_pairs["lm_metadata"]
        if lm_metadata["generate_audio_codes"]:
            audio_codes = generate_audio_codes(getattr(self, self.lm_model, self.qwen3_06b), token_weight_pairs["lm_prompt"], token_weight_pairs["lm_prompt_negative"], min_tokens=lm_metadata["min_tokens"], max_tokens=lm_metadata["min_tokens"], seed=lm_metadata["seed"], cfg_scale=lm_metadata["cfg_scale"], temperature=lm_metadata["temperature"], top_p=lm_metadata["top_p"], top_k=lm_metadata["top_k"], min_p=lm_metadata["min_p"])
            out["audio_codes"] = [audio_codes]

        return base_out, None, out

    def set_clip_options(self, options):
        self.qwen3_06b.set_clip_options(options)
        lm_model = getattr(self, self.lm_model, None)
        if lm_model is not None:
            lm_model.set_clip_options(options)

    def reset_clip_options(self):
        self.qwen3_06b.reset_clip_options()
        lm_model = getattr(self, self.lm_model, None)
        if lm_model is not None:
            lm_model.reset_clip_options()

    def load_sd(self, sd):
        if "model.layers.0.post_attention_layernorm.weight" in sd:
            shape = sd["model.layers.0.post_attention_layernorm.weight"].shape
            if shape[0] == 1024:
                return self.qwen3_06b.load_sd(sd)
            else:
                return getattr(self, self.lm_model).load_sd(sd)

    def memory_estimation_function(self, token_weight_pairs, device=None):
        lm_metadata = token_weight_pairs.get("lm_metadata", {})
        constant = self.constant
        if comfy.model_management.should_use_bf16(device):
            constant *= 0.5

        token_weight_pairs = token_weight_pairs.get("lm_prompt", [])
        num_tokens = sum(map(lambda a: len(a), token_weight_pairs))
        num_tokens += lm_metadata.get("min_tokens", 0)
        return num_tokens * constant * 1024 * 1024

def te(dtype_llama=None, llama_quantization_metadata=None, lm_model="qwen3_2b"):
    class ACE15TEModel_(ACE15TEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["llama_quantization_metadata"] = llama_quantization_metadata
            super().__init__(device=device, dtype_llama=dtype_llama, lm_model=lm_model, dtype=dtype, model_options=model_options)
    return ACE15TEModel_
