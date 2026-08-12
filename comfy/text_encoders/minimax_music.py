import torch
from tokenizers import Tokenizer

import comfy.ops
from comfy.ldm.minimax_music.ar import CFG_SCALE, CFG_TOP_K, MAX_AUDIO_FRAMES, MiniMaxMusic3AR
from comfy.ldm.minimax_music.prompt import SPECIAL_TOKEN_IDS, build_prompt


MODEL_CONFIG = {
    "vocab_size": 200000,
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "max_position_embeddings": 10240,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "head_dim": 128,
    "audio_vocab_size": 1024,
    "audio_num_codebooks": 8,
    "decoder_num_heads": 16,
    "decoder_intermediate_size": 6144,
    "decoder_num_layers": 4,
}


class MiniMaxMusic3Tokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer_json = tokenizer_data.get("tokenizer_json")
        if tokenizer_json is None:
            raise ValueError("MiniMax Music3 text encoder checkpoint is missing tokenizer_json")
        if torch.is_tensor(tokenizer_json):
            tokenizer_json = tokenizer_json.detach().cpu().numpy().tobytes()
        self.tokenizer_json = tokenizer_json
        self.tokenizer = Tokenizer.from_str(tokenizer_json.decode("utf-8"))
        for token, expected in SPECIAL_TOKEN_IDS.items():
            if self.tokenizer.token_to_id(token) != expected:
                raise ValueError(f"MiniMax Music3 tokenizer mismatch for {token}")

    def tokenize_with_weights(self, text, return_word_ids=False, **kwargs):
        prompt = build_prompt(text, kwargs.get("lyrics", ""))
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False).ids
        return {
            "minimax_music3": [[(token, 1.0) for token in token_ids]],
            "seed": int(kwargs.get("seed", 0)),
            "max_audio_frames": int(kwargs.get("max_audio_frames", MAX_AUDIO_FRAMES)),
            "cfg_scale": float(kwargs.get("cfg_scale", CFG_SCALE)),
            "top_k": int(kwargs.get("top_k", CFG_TOP_K)),
        }

    def state_dict(self):
        return {"tokenizer_json": torch.frombuffer(bytearray(self.tokenizer_json), dtype=torch.uint8)}

    def decode(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


class MiniMaxMusic3TEModel(MiniMaxMusic3AR):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        operations = model_options.get("custom_operations", comfy.ops.manual_cast)
        super().__init__(MODEL_CONFIG, dtype, device, operations)
        self.dtypes = {dtype}
        self.execution_device = device

    def set_clip_options(self, options):
        self.execution_device = options.get("execution_device", self.execution_device)

    def reset_clip_options(self):
        pass

    def get_dynamic_vram__units(self):
        return [self.model.audio_decoder, *self.model.layers]

    def encode_token_weights(self, token_weight_pairs):
        token_ids = [token for token, _ in token_weight_pairs["minimax_music3"][0]]
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        seed = token_weight_pairs["seed"]
        max_audio_frames = token_weight_pairs["max_audio_frames"]
        cfg_scale = token_weight_pairs["cfg_scale"]
        top_k = token_weight_pairs["top_k"]
        hidden = self.generate(input_ids, seed, max_audio_frames, self.execution_device, cfg_scale, top_k)
        return hidden.unsqueeze(0), None, {}

    def load_sd(self, state_dict):
        return self.load_state_dict(state_dict, strict=False, assign=getattr(self, "can_assign_sd", False))
