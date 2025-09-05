import asyncio
import base64
import math
import torch
import numpy as np
from io import BytesIO
from dataclasses import dataclass
from typing import List, Optional, Union, Any
from copy import deepcopy
from transformers import AutoTokenizer
from transformers.cache_utils import StaticCache
from dataclasses import asdict
from loguru import logger
import librosa
import os
from tokenizer import HiggsAudioTokenizer

import json
from huggingface_hub import snapshot_download
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING
from model import HiggsAudioConfig

class HiggsAudioEncoderConfig(PretrainedConfig):
    """Configuration of the Audio encoder in Higgs-Audio."""

    model_type = "higgs_audio_encoder"

    def __init__(
        self,
        num_mel_bins=128,
        encoder_layers=32,
        encoder_attention_heads=20,
        encoder_ffn_dim=5120,
        encoder_layerdrop=0.0,
        d_model=1280,
        dropout=0.0,
        attention_dropout=0.0,
        activation_function="gelu",
        activation_dropout=0.0,
        scale_embedding=False,
        init_std=0.02,
        max_source_positions=1500,
        pad_token_id=128001,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_mel_bins = num_mel_bins
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_function = activation_function
        self.activation_dropout = activation_dropout
        self.encoder_layerdrop = encoder_layerdrop
        self.num_hidden_layers = encoder_layers
        self.init_std = init_std
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        self.max_source_positions = max_source_positions
        self.pad_token_id = pad_token_id


class HiggsAudioConfig(PretrainedConfig):
    
    model_type = "higgs_audio"
    is_composition = True

    def __init__(
        self,
        text_config=None,
        audio_encoder_config=None,
        audio_tokenizer_config=None,
        audio_adapter_type="stack",
        audio_embed_avg=False,
        audio_ffn_hidden_size=4096,
        audio_ffn_intermediate_size=14336,
        audio_dual_ffn_layers=None,
        audio_decoder_proj_num_layers=0,
        encode_whisper_embed=True,
        encode_audio_in_tokens=False,
        use_delay_pattern=False,
        skip_audio_tower=False,
        use_audio_out_embed_projector=False,
        use_audio_out_self_attention=False,
        use_rq_transformer=False,
        rq_transformer_hidden_size=None,
        rq_transformer_intermediate_size=None,
        rq_transformer_num_attention_heads=None,
        rq_transformer_num_key_value_heads=None,
        rq_transformer_num_hidden_layers=3,
        audio_num_codebooks=12,
        audio_codebook_size=1024,
        audio_stream_bos_id=1024,
        audio_stream_eos_id=1025,
        audio_bos_token="<|audio_bos|>",
        audio_eos_token="<|audio_eos|>",
        audio_out_bos_token="<|audio_out_bos|>",
        audio_in_token="<|AUDIO|>",
        audio_out_token="<|AUDIO_OUT|>",
        audio_in_token_idx=128015,
        audio_out_token_idx=128016,
        pad_token_id=128001,
        audio_out_bos_token_id=128013,
        audio_eos_token_id=128012,
        **kwargs,
    ):


        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "llama"
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["llama"]()

        assert audio_adapter_type in [
            "stack",
            "dual_ffn",
            "dual_ffn_fast_forward",
        ], f"Invalid audio adapter type: {audio_adapter_type}"
        if audio_adapter_type.startswith("dual_ffn"):
            assert audio_dual_ffn_layers is not None, (
                "audio_dual_ffn_layers must be specified when using dual_ffn adapter."
            )
        self.text_config = text_config
        self.audio_encoder_config = audio_encoder_config
        self.audio_tokenizer_config = audio_tokenizer_config
        self.audio_adapter_type = audio_adapter_type
        self.audio_embed_avg = audio_embed_avg
        self.audio_ffn_hidden_size = audio_ffn_hidden_size
        self.audio_ffn_intermediate_size = audio_ffn_intermediate_size
        self.audio_dual_ffn_layers = audio_dual_ffn_layers
        self.audio_decoder_proj_num_layers = audio_decoder_proj_num_layers
        self.encode_whisper_embed = encode_whisper_embed
        self.encode_audio_in_tokens = encode_audio_in_tokens
        self.use_delay_pattern = use_delay_pattern
        self.skip_audio_tower = skip_audio_tower
        self.use_audio_out_embed_projector = use_audio_out_embed_projector
        self.use_audio_out_self_attention = use_audio_out_self_attention

        self.use_rq_transformer = use_rq_transformer

        if self.use_rq_transformer:
            assert not self.use_delay_pattern, "Delay pattern is not supported if you turned on RQ-Transformer!"
        self.rq_transformer_hidden_size = rq_transformer_hidden_size
        self.rq_transformer_intermediate_size = rq_transformer_intermediate_size
        self.rq_transformer_num_attention_heads = rq_transformer_num_attention_heads
        self.rq_transformer_num_key_value_heads = rq_transformer_num_key_value_heads
        self.rq_transformer_num_hidden_layers = rq_transformer_num_hidden_layers

        if use_rq_transformer:
            # For RQ-Transformer, we set the hidden_size to the same as the text model's hidden size if it is not specified.
            if self.rq_transformer_hidden_size is None:
                self.rq_transformer_hidden_size = text_config.hidden_size
            assert self.rq_transformer_hidden_size % 128 == 0
            if self.rq_transformer_intermediate_size is None:
                self.rq_transformer_intermediate_size = text_config.intermediate_size
            if self.rq_transformer_num_attention_heads is None:
                self.rq_transformer_num_attention_heads = self.rq_transformer_hidden_size // 128
            if self.rq_transformer_num_key_value_heads is None:
                self.rq_transformer_num_key_value_heads = self.rq_transformer_hidden_size // 128 // 4
            assert self.rq_transformer_hidden_size % self.rq_transformer_num_attention_heads == 0
            assert self.rq_transformer_hidden_size % self.rq_transformer_num_key_value_heads == 0

        self.audio_num_codebooks = audio_num_codebooks
        self.audio_codebook_size = audio_codebook_size
        self.audio_bos_token = audio_bos_token
        self.audio_eos_token = audio_eos_token
        self.audio_out_bos_token = audio_out_bos_token
        self.audio_in_token = audio_in_token
        self.audio_out_token = audio_out_token
        self.audio_in_token_idx = audio_in_token_idx
        self.audio_out_token_idx = audio_out_token_idx
        self.audio_stream_bos_id = audio_stream_bos_id
        self.audio_stream_eos_id = audio_stream_eos_id
        self.audio_out_bos_token_id = audio_out_bos_token_id
        self.audio_eos_token_id = audio_eos_token_id

        super().__init__(**kwargs)
        self.pad_token_id = pad_token_id

from model import HiggsAudioModel
from preprocess import HiggsAudioSampleCollator, ChatMLSample, ChatMLDatasetSample, prepare_chatml_sample, Message

def load_higgs_audio_tokenizer(tokenizer_name_or_path, device="cuda"):
    is_local = os.path.exists(tokenizer_name_or_path)
    if not is_local:
        tokenizer_path = snapshot_download(tokenizer_name_or_path)
    else:
        tokenizer_path = tokenizer_name_or_path
    config_path = os.path.join(tokenizer_path, "config.json")
    model_path = os.path.join(tokenizer_path, "model.pth")
    config = json.load(open(config_path))
    model = HiggsAudioTokenizer(
        **config,
        device=device,
    )
    parameter_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(parameter_dict, strict=False)
    model.to(device)
    model.eval()
    return model

def revert_delay_pattern(data):
    """Convert samples encoded with delay pattern back to the original form.

    Args:
        data (:obj:`torch.Tensor`):
            The data with delay pattern applied. It will have shape (num_codebooks, seq_len + num_codebooks - 1).

    Returns:
        ret (:obj:`torch.Tensor`):
            Recovered data with delay pattern removed. It will have shape (num_codebooks, seq_len).
    """
    assert len(data.shape) == 2
    out_l = []
    num_codebooks = data.shape[0]
    for i in range(num_codebooks):
        out_l.append(data[i : (i + 1), i : (data.shape[1] - num_codebooks + 1 + i)])
    return torch.cat(out_l, dim=0)


@dataclass
class HiggsAudioStreamerDelta:
    """Represents a chunk of generated content, either text or audio tokens."""

    text: Optional[str] = None
    text_tokens: Optional[torch.Tensor] = None
    audio_tokens: Optional[torch.Tensor] = None
    finish_reason: Optional[str] = None



@dataclass
class HiggsAudioResponse:
    audio: Optional[np.ndarray] = None
    generated_audio_tokens: Optional[np.ndarray] = None
    sampling_rate: Optional[int] = None
    generated_text: str = ""
    generated_text_tokens: Optional[np.ndarray] = None
    usage: Optional[dict] = None


class HiggsAudioServeEngine:
    def __init__(
        self,
        model_name_or_path: str,
        audio_tokenizer_name_or_path: str,
        tokenizer_name_or_path: Optional[str] = None,
        device: str = "cuda",
        torch_dtype: Union[torch.dtype, str] = torch.float16,
        kv_cache_lengths: List[int] = [1024, 4096, 8192],  # Multiple KV cache sizes
    ):

        self.device = device
        self.model_name_or_path = model_name_or_path
        self.torch_dtype = torch_dtype
        torch.set_default_device("cuda")

        # Initialize model and tokenizer
        config = HiggsAudioConfig.from_pretrained(
            "bosonai/higgs-audio-v2-generation-3B-base",
            #trust_remote_code=True  
        )
        #config.num_hidden_layers = config.num_hidden_layers // 2
        # ---- Audio Config ----
        #config.audio_dual_ffn_layers = config.audio_dual_ffn_layers[:12]
        #config.audio_ffn_hidden_size //= 2                # 3072 → 1536
        #config.audio_ffn_intermediate_size //= 2          # 8192 → 4096
        
        # ---- Text Config ----
        #config.text_config.hidden_size //= 2              # 3072 → 1536
        #config.text_config.intermediate_size //= 2        # 8192 → 4096
        #config.text_config.num_attention_heads //= 2      # 24 → 12
        #config.text_config.num_key_value_heads //= 2      # 8 → 4
        #config.text_config.num_hidden_layers //= 2        # 28 → 14
        #config.text_config.head_dim //= 2                 # 128 → 64
        
        # ---- Shared ----
        #config.hidden_size //= 2                          # 3072 → 1536

        self.model = HiggsAudioModel.from_pretrained(model_name_or_path, torch_dtype = torch_dtype, config = HiggsAudioConfig.from_pretrained(model_name_or_path))#(config = config, device = device, operations = torch.nn, dtype = torch_dtype)
        print(self.model.device)
        self.model.config = config
        logger.info(f"Loaded model from {model_name_or_path}, dtype: {self.model.dtype}")

        if tokenizer_name_or_path is None:
            tokenizer_name_or_path = model_name_or_path
        logger.info(f"Loading tokenizer from {tokenizer_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

        logger.info(f"Initializing Higgs Audio Tokenizer")
        self.audio_tokenizer = load_higgs_audio_tokenizer(audio_tokenizer_name_or_path, device=device)

        self.audio_num_codebooks = self.model.config.audio_num_codebooks
        self.audio_codebook_size = self.model.config.audio_codebook_size
        self.audio_tokenizer_tps = self.audio_tokenizer.tps
        self.samples_per_token = int(self.audio_tokenizer.sampling_rate // self.audio_tokenizer_tps)
        self.hamming_window_len = 2 * self.audio_num_codebooks * self.samples_per_token
        # Set the audio special tokens

        # Prepare KV caches for different lengths
        cache_config = deepcopy(self.model.config.text_config)
        cache_config.num_hidden_layers = self.model.config.text_config.num_hidden_layers
        if self.model.config.audio_dual_ffn_layers:
            cache_config.num_hidden_layers += len(self.model.config.audio_dual_ffn_layers)
        # A list of KV caches for different lengths
        self.kv_caches = {
            length: StaticCache(
                config=cache_config,
                max_batch_size=1,
                max_cache_len=length,
                device=self.model.device,
                dtype=self.model.dtype,
            )
            for length in sorted(kv_cache_lengths)
        }

        # Reuse collator to prepare inference samples
        self.collator = HiggsAudioSampleCollator(
            audio_in_token_id=self.model.config.audio_in_token_idx,
            audio_out_token_id=self.model.config.audio_out_token_idx,
            audio_stream_bos_id=self.model.config.audio_stream_bos_id,
            audio_stream_eos_id=self.model.config.audio_stream_eos_id,
            pad_token_id=self.model.config.pad_token_id,
            return_audio_in_tokens=False,
            use_delay_pattern=self.model.config.use_delay_pattern,
            audio_num_codebooks=self.model.config.audio_num_codebooks,
            round_to=1,
        )

        # Capture CUDA graphs for each KV cache length
        #if device == "cuda":
         #   logger.info(f"Capturing CUDA graphs for each KV cache length")
          #  self.model.capture_model(self.kv_caches.values())

    def _prepare_inputs(self, chat_ml_sample: ChatMLSample, force_audio_gen: bool = False):
        input_tokens, audio_contents, _ = prepare_chatml_sample(
            chat_ml_sample,
            self.tokenizer,
        )

        postfix = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        if force_audio_gen:
            postfix += "<|audio_out_bos|>"
        postfix = self.tokenizer.encode(postfix, add_special_tokens=False)
        input_tokens.extend(postfix)

        # Configure the audio inputs
        audio_ids_l = []
        for audio_content in audio_contents:
            if audio_content.audio_url not in ["placeholder", ""]:
                raw_audio, _ = librosa.load(audio_content.audio_url, sr=self.audio_tokenizer.sampling_rate)
            elif audio_content.raw_audio is not None:
                raw_audio, _ = librosa.load(
                    BytesIO(base64.b64decode(audio_content.raw_audio)), sr=self.audio_tokenizer.sampling_rate
                )
            else:
                raw_audio = None

            if raw_audio is not None:
                audio_ids = self.audio_tokenizer.encode(raw_audio, self.audio_tokenizer.sampling_rate)
                audio_ids_l.append(audio_ids.squeeze(0).cpu())

        if len(audio_ids_l) > 0:
            audio_ids_start = torch.tensor(
                np.cumsum(np.array([0] + [audio_ids.shape[1] for audio_ids in audio_ids_l])),
                dtype=torch.long,
                device=self.device,
            )[0:-1]
            audio_ids_concat = torch.cat(audio_ids_l, dim=1)
        else:
            audio_ids_start = None
            audio_ids_concat = None

        sample = ChatMLDatasetSample(
            input_ids=torch.LongTensor(input_tokens),
            label_ids=None,
            audio_ids_concat=audio_ids_concat,
            audio_ids_start=audio_ids_start,
            audio_waveforms_concat=None,
            audio_waveforms_start=None,
            audio_sample_rate=None,
            audio_speaker_indices=None,
        )
        data = self.collator([sample])
        inputs = asdict(data)
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.model.device)

        return inputs

    def _prepare_kv_caches(self):
        for kv_cache in self.kv_caches.values():
            kv_cache.reset()

    def generate(
        self,
        chat_ml_sample: ChatMLSample,
        max_new_tokens: int,
        temperature: float = 0.7,
        top_k: Optional[int] = None,
        top_p: float = 0.95,
        stop_strings: Optional[List[str]] = None,
        force_audio_gen: bool = False,
        ras_win_len: Optional[int] = 7,
        ras_win_max_num_repeat: int = 2,
        seed: Optional[int] = None,
    ):

        # Default stop strings
        if stop_strings is None:
            stop_strings = ["<|end_of_text|>", "<|eot_id|>"]
        if ras_win_len is not None and ras_win_len <= 0:
            ras_win_len = None

        with torch.no_grad():
            inputs = self._prepare_inputs(chat_ml_sample, force_audio_gen=force_audio_gen)
            prompt_token_ids = inputs["input_ids"][0].cpu().numpy()

            self._prepare_kv_caches()
            from autoregressive_sampling import auto_sample
            outputs = auto_sample(
                self.model,
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                stop_strings=stop_strings,
                tokenizer=self.tokenizer,
                do_sample=False if temperature == 0.0 else True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                past_key_values_buckets=self.kv_caches,
                ras_win_len=ras_win_len,
                ras_win_max_num_repeat=ras_win_max_num_repeat,
                seed=seed,
            )

            if len(outputs) > 0:
                wv_list = []
                for output_audio in outputs:
                    vq_code = revert_delay_pattern(output_audio).clip(0, self.audio_codebook_size - 1)[:, 1:-1]
                    wv_numpy = self.audio_tokenizer.decode(vq_code.unsqueeze(0))[0, 0]
                    wv_list.append(wv_numpy)
                wv_numpy = torch.cat(wv_list)
                wv_numpy = wv_numpy.cpu().numpy()
            else:
                wv_numpy = None
            # We only support one request at a time now
            #generated_text_tokens = outputs[0][0].cpu().numpy()[len(prompt_token_ids) :]
            #generated_text = self.tokenizer.decode(generated_text_tokens)
            #print(generated_text)
            generated_audio_tokens = outputs[0].cpu().numpy()
            return HiggsAudioResponse(
                audio=wv_numpy,
                generated_audio_tokens=generated_audio_tokens,
                sampling_rate=self.audio_tokenizer.sampling_rate,
                #generated_text=generated_text,
                #generated_text_tokens=generated_text_tokens,
                usage={
                    "prompt_tokens": prompt_token_ids.shape[0],
                   # "completion_tokens": generated_text_tokens.shape[0] + generated_audio_tokens.shape[1],
                    "total_tokens": (
                        prompt_token_ids.shape[0] #+ generated_text_tokens.shape[0] + generated_audio_tokens.shape[1]
                    ),
                    "cached_tokens": 0,
                },
            )

def get_zero_shot_input_sample():
    system_prompt = (
        "Generate audio following instruction.\n\n<|scene_desc_start|>\nSPEAKER0: british accent\n<|scene_desc_end|>"
    )

    messages = [
        Message(
            role="system",
            content=system_prompt,
        ),
        Message(
            role="user",
            content="Hey, everyone! Welcome back to Tech Talk Tuesdays.\n"
            "It's your host, Alex, and today, we're diving into a topic that's become absolutely crucial in the tech world — deep learning.\n"
            "And let's be honest, if you've been even remotely connected to tech, AI, or machine learning lately, you know that deep learning is everywhere.",
        ),
    ]
    chat_ml_sample = ChatMLSample(messages=messages)
    return chat_ml_sample

def _update_model_kwargs_for_generation(
    self,
    outputs,
    model_kwargs: dict[str, Any],
    num_new_tokens: int = 1,
) -> dict[str, Any]:

    # past_key_values will be the standard name for kv cache naming
    model_kwargs["past_key_values"] = outputs.past_key_values

    # thinking above removing token_type_ids
    # update token_type_ids with last value
    if "token_type_ids" in model_kwargs:
        token_type_ids = model_kwargs["token_type_ids"]
        model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

    # update attention mask
    if "attention_mask" in model_kwargs:
        attention_mask = model_kwargs["attention_mask"]
        model_kwargs["attention_mask"] = torch.cat(
            [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
        )

    if model_kwargs.get("use_cache", True):
        model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens
    else:
        past_positions = model_kwargs.pop("cache_position")
        new_positions = torch.arange(
            past_positions[-1] + 1, past_positions[-1] + num_new_tokens + 1, dtype=past_positions.dtype
        ).to(past_positions.device)
        model_kwargs["cache_position"] = torch.cat((past_positions, new_positions))
    return model_kwargs

MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"
def main():
    input_sample = get_zero_shot_input_sample()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    torch.manual_seed(2025)
    torch.cuda.manual_seed_all(2025)
    torch.cuda.manual_seed(2025)
    serve_engine = HiggsAudioServeEngine(
        MODEL_PATH,
        AUDIO_TOKENIZER_PATH,
        device=device,
    )
    import time
    logger.info("Starting generation...")
    start_time = time.time()
    output: HiggsAudioResponse = serve_engine.generate(
        chat_ml_sample=input_sample,
        max_new_tokens=1024,
        top_k=50,
        stop_strings=["<|end_of_text|>", "<|eot_id|>"],
    )
    elapsed_time = time.time() - start_time
    logger.info(f"Generation time: {elapsed_time:.2f} seconds")
    print(output)
    import torchaudio
    torchaudio.save(f"output_.wav", torch.from_numpy(output.audio)[None, :], output.sampling_rate)
    logger.info(f"Generated text:\n{output.generated_text}")
    logger.info(f"Saved audio to output_.wav")


if __name__ == "__main__":
    main()