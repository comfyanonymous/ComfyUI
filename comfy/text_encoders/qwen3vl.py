import math
import os
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import Qwen2Tokenizer

import comfy.model_management
import comfy.text_encoders.qwen_vl
import comfy.utils
from comfy import sd1_clip
from comfy.ldm.modules.attention import optimized_attention_for_device
from comfy.text_encoders.hidream_o1 import IMAGE_TOKEN_ID
from comfy.text_encoders.llama import BaseGenerate, BaseLlama, Llama2_
from comfy.text_encoders.qwen35 import Qwen35VisionModel


@dataclass
class Qwen3VLTextConfig:
    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5000000.0
    transformer_type: str = "llama"
    head_dim: int = 128
    rms_norm_add: bool = False
    mlp_activation: str = "silu"
    qkv_bias: bool = False
    rope_dims: list = field(default_factory=lambda: [24, 20, 20])
    rope_scale: float = None
    interleaved_mrope: bool = True
    q_norm: str = "gemma3"
    k_norm: str = "gemma3"
    final_norm: bool = True
    lm_head: bool = True
    stop_tokens: list = field(default_factory=lambda: [151645, 151643])


QWEN3VL_MODELS = {
    "qwen3vl_4b": {
        "hidden_size": 2560,
        "intermediate_size": 9728,
        "vision": {
            "hidden_size": 1024,
            "num_heads": 16,
            "intermediate_size": 4096,
            "depth": 24,
            "out_hidden_size": 2560,
            "deepstack_visual_indexes": [5, 11, 17],
        },
    },
    "qwen3vl_8b": {
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "vision": {
            "hidden_size": 1152,
            "num_heads": 16,
            "intermediate_size": 4304,
            "depth": 27,
            "out_hidden_size": 4096,
            "deepstack_visual_indexes": [8, 16, 24],
        },
    },
}

QWEN3VL_VISION_DEFAULTS = {
    "hidden_size": 1152,
    "num_heads": 16,
    "intermediate_size": 4304,
    "depth": 27,
    "patch_size": 16,
    "temporal_patch_size": 2,
    "in_channels": 3,
    "spatial_merge_size": 2,
    "num_position_embeddings": 2304,
    "out_hidden_size": 4096,
    "deepstack_visual_indexes": [8, 16, 24],
}


def _make_config(model_type, config_dict={}):
    overrides = QWEN3VL_MODELS.get(model_type, {}).copy()
    overrides.pop("vision", None)
    overrides.update(config_dict)
    return Qwen3VLTextConfig(**overrides)


def _expanded_token_ids(tokens, embeds_info, seq_len):
    ids = [0] * seq_len
    expanded_idx = 0
    embed_map = {info["index"]: info for info in embeds_info}
    for token in tokens:
        if expanded_idx in embed_map:
            info = embed_map[expanded_idx]
            fill_id = IMAGE_TOKEN_ID if info.get("type") == "image" else 0
            for i in range(info["size"]):
                if expanded_idx + i < seq_len:
                    ids[expanded_idx + i] = fill_id
            expanded_idx += info["size"]
        elif isinstance(token, int):
            if expanded_idx < seq_len:
                ids[expanded_idx] = int(token)
            expanded_idx += 1
        else:
            expanded_idx += 1
    return ids


class Qwen3VLVisionPatchMerger(torch.nn.Module):
    def __init__(self, hidden_size, spatial_merge_size, out_hidden_size, use_postshuffle_norm=False, device=None, dtype=None, ops=None):
        super().__init__()
        merge_dim = hidden_size * (spatial_merge_size ** 2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = ops.LayerNorm(merge_dim if use_postshuffle_norm else hidden_size, eps=1e-6, device=device, dtype=dtype)
        self.linear_fc1 = ops.Linear(merge_dim, merge_dim, device=device, dtype=dtype)
        self.linear_fc2 = ops.Linear(merge_dim, out_hidden_size, device=device, dtype=dtype)
        self.merge_dim = merge_dim

    def forward(self, x):
        if self.use_postshuffle_norm:
            x = self.norm(x.view(-1, self.merge_dim))
        else:
            x = self.norm(x).view(-1, self.merge_dim)
        return self.linear_fc2(F.gelu(self.linear_fc1(x)))


class Qwen3VLVisionModel(Qwen35VisionModel):
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__(config, device=device, dtype=dtype, ops=ops)
        self.deepstack_visual_indexes = config["deepstack_visual_indexes"]
        self.merger = Qwen3VLVisionPatchMerger(
            config["hidden_size"],
            config["spatial_merge_size"],
            config["out_hidden_size"],
            use_postshuffle_norm=False,
            device=device,
            dtype=dtype,
            ops=ops,
        )
        self.deepstack_merger_list = torch.nn.ModuleList([
            Qwen3VLVisionPatchMerger(
                config["hidden_size"],
                config["spatial_merge_size"],
                config["out_hidden_size"],
                use_postshuffle_norm=True,
                device=device,
                dtype=dtype,
                ops=ops,
            )
            for _ in self.deepstack_visual_indexes
        ])

    def forward(self, x, grid_thw):
        x = self.patch_embed(x)
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw).to(x.device)
        x = x + pos_embeds
        rotary_pos_emb = self.rot_pos_emb(grid_thw).to(x.device)
        seq_len = x.shape[0]
        x = x.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos = emb.cos().unsqueeze(-2)
        sin = emb.sin().unsqueeze(-2)
        sin_half = sin.shape[-1] // 2
        position_embeddings = (cos, sin[..., :sin_half], -sin[..., sin_half:])
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        optimized_attention = optimized_attention_for_device(x.device, mask=False, small_input=True)
        deepstack_features = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings, optimized_attention=optimized_attention)
            if i in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(i)](x)
                deepstack_features.append(deepstack_feature)
        merged = self.merger(x)
        return merged, deepstack_features


class Qwen3VL(BaseLlama, BaseGenerate, torch.nn.Module):
    model_type = "qwen3vl_8b"

    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = _make_config(self.model_type, config_dict)
        self.num_layers = config.num_hidden_layers
        self.model = Llama2_(config, device=device, dtype=dtype, ops=operations)
        vision_overrides = QWEN3VL_MODELS.get(self.model_type, {}).get("vision", {})
        vision_config = {**QWEN3VL_VISION_DEFAULTS, **vision_overrides}
        self.visual = Qwen3VLVisionModel(vision_config, device=device, dtype=dtype, ops=operations)
        self.dtype = dtype

    def preprocess_embed(self, embed, device):
        if embed["type"] == "image":
            image, grid = comfy.text_encoders.qwen_vl.process_qwen2vl_images(
                embed["data"],
                min_pixels=65536,
                max_pixels=16777216,
                patch_size=16,
                image_mean=[0.5, 0.5, 0.5],
                image_std=[0.5, 0.5, 0.5],
            )
            image_embeds, deepstack_embeds = self.visual(image.to(device, dtype=torch.float32), grid)
            return image_embeds, {"grid": grid, "deepstack": deepstack_embeds}
        return None, None

    def _deepstack_from_embeds_info(self, embeds, embeds_info):
        visual_pos_masks = None
        deepstack_visual_embeds = None
        for e in embeds_info:
            if e.get("type") != "image":
                continue
            extra = e.get("extra", None)
            if extra is None:
                continue
            deepstack = extra.get("deepstack", None)
            if deepstack is None:
                continue
            start = e.get("index")
            end = start + e.get("size")
            if visual_pos_masks is None:
                visual_pos_masks = torch.zeros((embeds.shape[0], embeds.shape[1]), device=embeds.device, dtype=torch.bool)
                deepstack_visual_embeds = [[] for _ in range(len(deepstack))]
            visual_pos_masks[:, start:end] = True
            for i, d in enumerate(deepstack):
                if embeds.shape[0] > 1:
                    d = d.repeat(embeds.shape[0], 1)
                deepstack_visual_embeds[i].append(d)

        if visual_pos_masks is None:
            return None, None

        return visual_pos_masks, [torch.cat(d, dim=0) for d in deepstack_visual_embeds]

    def _deepstack_process(self, hidden_states, visual_pos_masks, visual_embeds):
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        hidden_states = hidden_states.clone()
        hidden_states[visual_pos_masks, :] = hidden_states[visual_pos_masks, :] + visual_embeds
        return hidden_states

    def _position_ids_from_embeds(self, embeds, embeds_info):
        grid = None
        position_ids = None
        offset = 0
        for e in embeds_info:
            if e.get("type") == "image":
                extra = e.get("extra", None)
                grid = extra.get("grid", None) if isinstance(extra, dict) else extra
                start = e.get("index")
                if position_ids is None:
                    position_ids = torch.zeros((3, embeds.shape[1]), device=embeds.device, dtype=torch.long)
                    position_ids[:, :start] = torch.arange(0, start, device=embeds.device)
                end = e.get("size") + start
                len_max = int(grid.max()) // 2
                start_next = len_max + start
                position_ids[:, end:] = torch.arange(start_next + offset, start_next + (embeds.shape[1] - end) + offset, device=embeds.device)
                position_ids[0, start:end] = start + offset
                max_d = int(grid[0][1]) // 2
                position_ids[1, start:end] = torch.arange(start + offset, start + max_d + offset, device=embeds.device).unsqueeze(1).repeat(1, math.ceil((end - start) / max_d)).flatten(0)[:end - start]
                max_d = int(grid[0][2]) // 2
                position_ids[2, start:end] = torch.arange(start + offset, start + max_d + offset, device=embeds.device).unsqueeze(0).repeat(math.ceil((end - start) / max_d), 1).flatten(0)[:end - start]
                offset += len_max - (end - start)

        if grid is None:
            return None, 0

        return position_ids, int(position_ids.max().item()) + 1 - embeds.shape[1]

    def forward(self, x, attention_mask=None, embeds=None, num_tokens=None, intermediate_output=None, final_layer_norm_intermediate=True, dtype=None, embeds_info=[], past_key_values=None, position_ids=None):
        if embeds is not None:
            x = embeds
        else:
            x = self.model.embed_tokens(x, out_dtype=dtype)

        seq_len = x.shape[1]
        past_len = 0
        if past_key_values is not None and len(past_key_values) > 0:
            past_len = self.model.get_past_len(past_key_values)

        if position_ids is None:
            if embeds is not None:
                position_ids, _ = self._position_ids_from_embeds(embeds, embeds_info)
            if position_ids is None:
                position_ids = torch.arange(past_len, past_len + seq_len, device=x.device).unsqueeze(0)

        freqs_cis = self.model.compute_freqs_cis(position_ids, x.device)

        mask = None
        if attention_mask is not None:
            mask = 1.0 - attention_mask.to(x.dtype).reshape((attention_mask.shape[0], 1, -1, attention_mask.shape[-1])).expand(attention_mask.shape[0], 1, seq_len, attention_mask.shape[-1])
            mask = mask.masked_fill(mask.to(torch.bool), torch.finfo(x.dtype).min / 4)

        if seq_len > 1:
            causal_mask = torch.empty(past_len + seq_len, past_len + seq_len, dtype=x.dtype, device=x.device).fill_(torch.finfo(x.dtype).min / 4).triu_(1)
            if mask is not None:
                mask += causal_mask
            else:
                mask = causal_mask

        optimized_attention = optimized_attention_for_device(x.device, mask=mask is not None, small_input=True)

        intermediate = None
        all_intermediate = None
        only_layers = None
        if intermediate_output is not None:
            if isinstance(intermediate_output, list):
                all_intermediate = []
                only_layers = set(intermediate_output)
            elif intermediate_output == "all":
                all_intermediate = []
                intermediate_output = None
            elif intermediate_output < 0:
                intermediate_output = len(self.model.layers) + intermediate_output

        visual_pos_masks, deepstack_visual_embeds = self._deepstack_from_embeds_info(x, embeds_info)

        next_key_values = []
        for i, layer in enumerate(self.model.layers):
            if all_intermediate is not None:
                if only_layers is None or (i in only_layers):
                    all_intermediate.append(x.unsqueeze(1).clone())

            past_kv = None
            if past_key_values is not None:
                past_kv = past_key_values[i] if len(past_key_values) > 0 else []

            x, current_kv = layer(
                x=x,
                attention_mask=mask,
                freqs_cis=freqs_cis,
                optimized_attention=optimized_attention,
                past_key_value=past_kv,
            )

            if deepstack_visual_embeds is not None and i in range(len(deepstack_visual_embeds)):
                x = self._deepstack_process(x, visual_pos_masks, deepstack_visual_embeds[i])

            if current_kv is not None:
                next_key_values.append(current_kv)

            if i == intermediate_output:
                intermediate = x.clone()

        if self.model.norm is not None:
            x = self.model.norm(x)

        if all_intermediate is not None:
            if only_layers is None or ((i + 1) in only_layers):
                all_intermediate.append(x.unsqueeze(1).clone())

        if all_intermediate is not None:
            intermediate = torch.cat(all_intermediate, dim=1)

        if intermediate is not None and final_layer_norm_intermediate and self.model.norm is not None:
            intermediate = self.model.norm(intermediate)

        if len(next_key_values) > 0:
            return x, intermediate, next_key_values
        else:
            return x, intermediate

    def generate(self, embeds=None, embeds_info=[], do_sample=True, max_length=256, temperature=1.0, top_k=50, top_p=0.9, min_p=0.0, repetition_penalty=1.0, seed=42, stop_tokens=None, initial_tokens=[], execution_dtype=None, presence_penalty=0.0, initial_input_ids=None):
        device = embeds.device

        if stop_tokens is None:
            stop_tokens = self.model.config.stop_tokens

        if execution_dtype is None:
            if comfy.model_management.should_use_bf16(device):
                execution_dtype = torch.bfloat16
            else:
                execution_dtype = torch.float32
        embeds = embeds.to(execution_dtype)

        if embeds.ndim == 2:
            embeds = embeds.unsqueeze(0)

        prompt_position_ids, position_delta = self._position_ids_from_embeds(embeds, embeds_info)

        max_cache_len = embeds.shape[1] + max_length
        past_key_values = self.init_kv_cache(embeds.shape[0], max_cache_len, device, execution_dtype)

        generator = torch.Generator(device=device).manual_seed(seed) if do_sample else None

        generated_token_ids = []
        pbar = comfy.utils.ProgressBar(max_length)
        current_position_ids = prompt_position_ids
        current_embeds_info = embeds_info
        for _ in tqdm(range(max_length), desc="Generating tokens"):
            x, _, past_key_values = self.forward(
                None,
                embeds=embeds,
                attention_mask=None,
                past_key_values=past_key_values,
                position_ids=current_position_ids,
                embeds_info=current_embeds_info,
            )
            logits = self.logits(x)[:, -1]
            next_token = self.sample_token(logits, temperature, top_k, top_p, min_p, repetition_penalty, initial_tokens + generated_token_ids, generator, do_sample=do_sample, presence_penalty=presence_penalty)
            token_id = next_token[0].item()
            generated_token_ids.append(token_id)

            embeds = self.model.embed_tokens(next_token).to(execution_dtype)
            current_embeds_info = []
            if prompt_position_ids is not None:
                past_len = self.model.get_past_len(past_key_values)
                current_position_ids = torch.full((3, 1), past_len + position_delta, device=device, dtype=torch.long)
            else:
                current_position_ids = None
            pbar.update(1)

            if token_id in stop_tokens:
                break

        return generated_token_ids


class Qwen3VLTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}, embedding_size=4096, embedding_key="qwen3vl_8b"):
        tokenizer_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "qwen25_tokenizer")
        super().__init__(tokenizer_path, pad_with_end=False, embedding_directory=embedding_directory, embedding_size=embedding_size, embedding_key=embedding_key, tokenizer_class=Qwen2Tokenizer,
                         has_start_token=False, has_end_token=False, pad_to_max_length=False, max_length=99999999, min_length=1, pad_token=151643, tokenizer_data=tokenizer_data)


class Qwen3VLImageTokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}, model_type="qwen3vl_8b"):
        embedding_size = QWEN3VL_MODELS.get(model_type, {}).get("hidden_size", 4096)
        tokenizer = lambda *a, **kw: Qwen3VLTokenizer(*a, **kw, embedding_size=embedding_size, embedding_key=model_type)
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, name=model_type, tokenizer=tokenizer)
        self.llama_template = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        self.llama_template_images = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"

    def tokenize_with_weights(self, text, return_word_ids=False, llama_template=None, images=[], prevent_empty_text=False, thinking=False, **kwargs):
        image = kwargs.get("image", None)
        if image is not None and len(images) == 0:
            images = [image[i:i + 1] for i in range(image.shape[0])]

        skip_template = kwargs.get("skip_template", False)
        if text.startswith("<|im_start|>"):
            skip_template = True
        if prevent_empty_text and text == "":
            text = " "

        if skip_template:
            llama_text = text
        else:
            if llama_template is not None:
                template = llama_template
            elif len(images) == 0:
                template = self.llama_template
            else:
                template = self.llama_template_images
                if len(images) > 1:
                    vision_block = "<|vision_start|><|image_pad|><|vision_end|>"
                    template = template.replace(vision_block, vision_block * len(images), 1)
            llama_text = template.format(text)

        tokens = super().tokenize_with_weights(llama_text, return_word_ids=return_word_ids, disable_weights=True, **kwargs)
        key_name = next(iter(tokens))
        embed_count = 0
        qwen_tokens = tokens[key_name]
        for r in qwen_tokens:
            for i in range(len(r)):
                if r[i][0] == IMAGE_TOKEN_ID:
                    if len(images) > embed_count:
                        r[i] = ({"type": "image", "data": images[embed_count], "original_type": "image"},) + r[i][1:]
                        embed_count += 1
        return tokens


class Qwen3VLClipModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="hidden", layer_idx=-2, dtype=None, attention_mask=True, model_options={}, model_type="qwen3vl_8b"):
        class Qwen3VL_(Qwen3VL):
            pass
        Qwen3VL_.model_type = model_type

        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config={},
                         dtype=dtype, special_tokens={"pad": 151643}, layer_norm_hidden_state=False,
                         model_class=Qwen3VL_, enable_attention_masks=attention_mask,
                         return_attention_masks=attention_mask, model_options=model_options)

    def generate(self, tokens, do_sample, max_length, temperature, top_k, top_p, min_p, repetition_penalty, seed, presence_penalty=0.0):
        if isinstance(tokens, dict):
            tokens = next(iter(tokens.values()))
        tokens_only = [[t[0] for t in b] for b in tokens]
        embeds, _, _, embeds_info = sd1_clip.SDClipModel.process_tokens(self, tokens_only, self.execution_device)
        initial_token_ids = [_expanded_token_ids(tokens_only[0], embeds_info, embeds.shape[1])]
        input_ids = torch.tensor(initial_token_ids, device=self.execution_device)
        return self.transformer.generate(
            embeds,
            embeds_info=embeds_info,
            do_sample=do_sample,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
            initial_tokens=initial_token_ids[0],
            presence_penalty=presence_penalty,
            initial_input_ids=input_ids,
        )


class Qwen3VLTEModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}, model_type="qwen3vl_8b"):
        clip_model = lambda **kw: Qwen3VLClipModel(**kw, model_type=model_type)
        super().__init__(device=device, dtype=dtype, name=model_type, clip_model=clip_model, model_options=model_options)


def tokenizer(model_type="qwen3vl_8b"):
    class Qwen3VLImageTokenizer_(Qwen3VLImageTokenizer):
        def __init__(self, embedding_directory=None, tokenizer_data={}):
            super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, model_type=model_type)
    return Qwen3VLImageTokenizer_


def te(dtype_llama=None, llama_quantization_metadata=None, model_type="qwen3vl_8b"):
    class Qwen3VLTEModel_(Qwen3VLTEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if dtype_llama is not None:
                dtype = dtype_llama
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["quantization_metadata"] = llama_quantization_metadata
            super().__init__(device=device, dtype=dtype, model_options=model_options, model_type=model_type)
    return Qwen3VLTEModel_
