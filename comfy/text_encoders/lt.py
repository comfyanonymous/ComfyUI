from comfy import sd1_clip
import os
from transformers import T5TokenizerFast
from .spiece_tokenizer import SPieceTokenizer
import comfy.text_encoders.genmo
import torch
import comfy.utils
import math
import itertools

class T5XXLTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "t5_tokenizer")
        super().__init__(tokenizer_path, embedding_directory=embedding_directory, pad_with_end=False, embedding_size=4096, embedding_key='t5xxl', tokenizer_class=T5TokenizerFast, has_start_token=False, pad_to_max_length=False, max_length=99999999, min_length=128, tokenizer_data=tokenizer_data) #pad to 128?


class LTXVT5Tokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, clip_name="t5xxl", tokenizer=T5XXLTokenizer)


def ltxv_te(*args, **kwargs):
    return comfy.text_encoders.genmo.mochi_te(*args, **kwargs)


class Gemma3_Tokenizer():
    def state_dict(self):
        return {"spiece_model": self.tokenizer.serialize_model()}

    def tokenize_with_weights(self, text, return_word_ids=False, image=None, llama_template=None, skip_template=True, **kwargs):
        self.llama_template = "<start_of_turn>system\nYou are a helpful assistant.<end_of_turn>\n<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n"
        self.llama_template_images = "<start_of_turn>system\nYou are a helpful assistant.<end_of_turn>\n<start_of_turn>user\n\n<image_soft_token>{}<end_of_turn>\n\n<start_of_turn>model\n"

        if image is None:
            images = []
        else:
            samples = image.movedim(-1, 1)
            total = int(896 * 896)

            scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
            width = round(samples.shape[3] * scale_by)
            height = round(samples.shape[2] * scale_by)

            s = comfy.utils.common_upscale(samples, width, height, "area", "disabled").movedim(1, -1)
            images = [s[:, :, :, :3]]

        if text.startswith('<start_of_turn>'):
            skip_template = True

        if skip_template:
            llama_text = text
        else:
            if llama_template is None:
                if len(images) > 0:
                    llama_text = self.llama_template_images.format(text)
                else:
                    llama_text = self.llama_template.format(text)
            else:
                llama_text = llama_template.format(text)

        text_tokens = super().tokenize_with_weights(llama_text, return_word_ids)

        if len(images) > 0:
            embed_count = 0
            for r in text_tokens:
                for i, token in enumerate(r):
                    if token[0] == 262144 and embed_count < len(images):
                        r[i] = ({"type": "image", "data": images[embed_count]},) + token[1:]
                        embed_count += 1
        return text_tokens

class Gemma3_12BTokenizer(Gemma3_Tokenizer, sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer = tokenizer_data.get("spiece_model", None)
        special_tokens = {"<image_soft_token>": 262144, "<end_of_turn>": 106}
        super().__init__(tokenizer, pad_with_end=False, embedding_size=3840, embedding_key='gemma3_12b', tokenizer_class=SPieceTokenizer, has_end_token=False, pad_to_max_length=False, max_length=99999999, min_length=1024, pad_left=True, disable_weights=True, tokenizer_args={"add_bos": True, "add_eos": False, "special_tokens": special_tokens}, tokenizer_data=tokenizer_data)


class LTXAVGemmaTokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, name="gemma3_12b", tokenizer=Gemma3_12BTokenizer)


class Gemma3_12BModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="all", layer_idx=None, dtype=None, attention_mask=True, model_options={}):
        llama_quantization_metadata = model_options.get("llama_quantization_metadata", None)
        if llama_quantization_metadata is not None:
            model_options = model_options.copy()
            model_options["quantization_metadata"] = llama_quantization_metadata
        self.dtypes = set()
        self.dtypes.add(dtype)
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config={}, dtype=dtype, special_tokens={"start": 2, "pad": 0}, layer_norm_hidden_state=False, model_class=comfy.text_encoders.llama.Gemma3_12B, enable_attention_masks=attention_mask, return_attention_masks=attention_mask, model_options=model_options)

    def generate(self, tokens, do_sample, max_length, temperature, top_k, top_p, min_p, repetition_penalty, seed):
        tokens_only = [[t[0] for t in b] for b in tokens]
        embeds, _, _, embeds_info = self.process_tokens(tokens_only, self.execution_device)
        comfy.utils.normalize_image_embeddings(embeds, embeds_info, self.transformer.model.config.hidden_size ** 0.5)
        return self.transformer.generate(embeds, do_sample, max_length, temperature, top_k, top_p, min_p, repetition_penalty, seed, stop_tokens=[106])  # 106 is <end_of_turn>

class LTXAVTEModel(torch.nn.Module):
    def __init__(self, dtype_llama=None, device="cpu", dtype=None, model_options={}):
        super().__init__()
        self.dtypes = set()
        self.dtypes.add(dtype)
        self.compat_mode = False

        self.gemma3_12b = Gemma3_12BModel(device=device, dtype=dtype_llama, model_options=model_options, layer="all", layer_idx=None)
        self.dtypes.add(dtype_llama)

        operations = self.gemma3_12b.operations # TODO
        self.text_embedding_projection = operations.Linear(3840 * 49, 3840, bias=False, dtype=dtype, device=device)

    def enable_compat_mode(self):  # TODO: remove
        from comfy.ldm.lightricks.embeddings_connector import Embeddings1DConnector
        operations = self.gemma3_12b.operations
        dtype = self.text_embedding_projection.weight.dtype
        device = self.text_embedding_projection.weight.device
        self.audio_embeddings_connector = Embeddings1DConnector(
            split_rope=True,
            double_precision_rope=True,
            dtype=dtype,
            device=device,
            operations=operations,
        )

        self.video_embeddings_connector = Embeddings1DConnector(
            split_rope=True,
            double_precision_rope=True,
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.compat_mode = True

    def set_clip_options(self, options):
        self.execution_device = options.get("execution_device", self.execution_device)
        self.gemma3_12b.set_clip_options(options)

    def reset_clip_options(self):
        self.gemma3_12b.reset_clip_options()
        self.execution_device = None

    def encode_token_weights(self, token_weight_pairs):
        token_weight_pairs = token_weight_pairs["gemma3_12b"]

        out, pooled, extra = self.gemma3_12b.encode_token_weights(token_weight_pairs)
        out = out[:, :, -torch.sum(extra["attention_mask"]).item():]
        out_device = out.device
        if comfy.model_management.should_use_bf16(self.execution_device):
            out = out.to(device=self.execution_device, dtype=torch.bfloat16)
        out = out.movedim(1, -1).to(self.execution_device)
        out = 8.0 * (out - out.mean(dim=(1, 2), keepdim=True)) / (out.amax(dim=(1, 2), keepdim=True) - out.amin(dim=(1, 2), keepdim=True) + 1e-6)
        out = out.reshape((out.shape[0], out.shape[1], -1))
        out = self.text_embedding_projection(out)
        out = out.float()

        if self.compat_mode:
            out_vid = self.video_embeddings_connector(out)[0]
            out_audio = self.audio_embeddings_connector(out)[0]
            out = torch.concat((out_vid, out_audio), dim=-1)

        return out.to(out_device), pooled

    def generate(self, tokens, do_sample, max_length, temperature, top_k, top_p, min_p, repetition_penalty, seed):
        return self.gemma3_12b.generate(tokens["gemma3_12b"], do_sample, max_length, temperature, top_k, top_p, min_p, repetition_penalty, seed)

    def load_sd(self, sd):
        if "model.layers.47.self_attn.q_norm.weight" in sd:
            return self.gemma3_12b.load_sd(sd)
        else:
            sdo = comfy.utils.state_dict_prefix_replace(sd, {"text_embedding_projection.aggregate_embed.weight": "text_embedding_projection.weight"}, filter_keys=True)
            if len(sdo) == 0:
                sdo = sd

            missing_all = []
            unexpected_all = []

            for prefix, component in [("text_embedding_projection.", self.text_embedding_projection)]:
                component_sd = {k.replace(prefix, ""): v for k, v in sdo.items() if k.startswith(prefix)}
                if component_sd:
                    missing, unexpected = component.load_state_dict(component_sd, strict=False, assign=getattr(self, "can_assign_sd", False))
                    missing_all.extend([f"{prefix}{k}" for k in missing])
                    unexpected_all.extend([f"{prefix}{k}" for k in unexpected])

            if "model.diffusion_model.audio_embeddings_connector.transformer_1d_blocks.2.attn1.to_q.bias" not in sd:  # TODO: remove
                ww = sd.get("model.diffusion_model.audio_embeddings_connector.transformer_1d_blocks.0.attn1.to_q.bias", None)
                if ww is not None:
                    if ww.shape[0] == 3840:
                        self.enable_compat_mode()
                        sdv = comfy.utils.state_dict_prefix_replace(sd, {"model.diffusion_model.video_embeddings_connector.": ""}, filter_keys=True)
                        self.video_embeddings_connector.load_state_dict(sdv, strict=False, assign=getattr(self, "can_assign_sd", False))
                        sda = comfy.utils.state_dict_prefix_replace(sd, {"model.diffusion_model.audio_embeddings_connector.": ""}, filter_keys=True)
                        self.audio_embeddings_connector.load_state_dict(sda, strict=False, assign=getattr(self, "can_assign_sd", False))

            return (missing_all, unexpected_all)

    def memory_estimation_function(self, token_weight_pairs, device=None):
        constant = 6.0
        if comfy.model_management.should_use_bf16(device):
            constant /= 2.0

        token_weight_pairs = token_weight_pairs.get("gemma3_12b", [])
        m = min([sum(1 for _ in itertools.takewhile(lambda x: x[0] == 0, sub)) for sub in token_weight_pairs])

        num_tokens = sum(map(lambda a: len(a), token_weight_pairs)) - m
        num_tokens = max(num_tokens, 642)
        return num_tokens * constant * 1024 * 1024

def ltxav_te(dtype_llama=None, llama_quantization_metadata=None):
    class LTXAVTEModel_(LTXAVTEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["llama_quantization_metadata"] = llama_quantization_metadata
            if dtype_llama is not None:
                dtype = dtype_llama
            super().__init__(dtype_llama=dtype_llama, device=device, dtype=dtype, model_options=model_options)
    return LTXAVTEModel_

def gemma3_te(dtype_llama=None, llama_quantization_metadata=None):
    class Gemma3_12BModel_(Gemma3_12BModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["llama_quantization_metadata"] = llama_quantization_metadata
            if dtype_llama is not None:
                dtype = dtype_llama
            super().__init__(device=device, dtype=dtype, model_options=model_options)
    return Gemma3_12BModel_
