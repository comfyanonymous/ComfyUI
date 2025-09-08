import os
import torch
import comfy.ops
import torch.nn as nn
from transformers import AutoTokenizer
from comfy.ldm.higgsv2.tokenizer import HiggsAudioTokenizer
from comfy.ldm.higgsv2.preprocess import HiggsAudioSampleCollator

class DummyTokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        pass

def revert_delay_pattern_vectorized(data: torch.Tensor) -> torch.Tensor:
    num_codebooks, total_len = data.shape
    seq_len = total_len - num_codebooks + 1

    col_idx = torch.arange(seq_len, device=data.device)[None, :] \
             + torch.arange(num_codebooks, device=data.device)[:, None]
    out = data[torch.arange(num_codebooks)[:, None], col_idx]
    return out

class HiggsTokenizer(nn.Module):
    def __init__(self, device, dtype, model_options={}, **kwargs):
        super().__init__()

        self.dtype = torch.float32
        self.device = device
        self.dtypes = [torch.float32]

        here = os.path.dirname(__file__)
        tokenizer_path = os.path.join(here, "higgs_text_tokenizer")

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        scaled_fp8 = model_options.get("scaled_fp8", None)

        if scaled_fp8 is not None:
            operations = comfy.ops.scaled_fp8_ops(fp8_matrix_mult=False, override_dtype=scaled_fp8)
        else:
            operations = comfy.ops.manual_cast

        self.audio_codebook_size = 1024
        self.audio_tokenizer = HiggsAudioTokenizer(device = device, dtype = dtype, operations = operations)

        if scaled_fp8 is not None:
            self.audio_tokenizer.scaled_fp8 = torch.nn.Parameter(torch.tensor([], dtype=scaled_fp8))

        self.collator = HiggsAudioSampleCollator(
            audio_in_token_id = 128015,
            audio_out_token_id = 128016,
            audio_stream_bos_id = 1024,
            audio_stream_eos_id = 1025,
            pad_token_id = 128001,
            return_audio_in_tokens = False,
            use_delay_pattern = True,
            audio_num_codebooks = 8,
            round_to = 1,
        )

        postfix = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        self.postfix = postfix + "<|audio_out_bos|>" # force audio generation

    def decode_tokens(self, audio_tokens):
        outputs = []

        # due to instability issues, I had to convert the audio tokenizer to float32, avoiding outputing nans
        self.audio_tokenizer = self.audio_tokenizer.to(self.dtype)
        torch.cuda.synchronize()

        for audio in audio_tokens:
            vq_code = revert_delay_pattern_vectorized(audio).clip(0, self.audio_codebook_size - 1)[:, 1:-1]
            wv_numpy = self.audio_tokenizer.decode(vq_code.unsqueeze(0))[0, 0]
            outputs.append(wv_numpy)

        # currently only supports one batch size
        return (None, {"waveform": torch.cat(outputs, dim = 0).unsqueeze(0).unsqueeze(1), "sample_rate": self.audio_tokenizer.sample_rate}) # audio only

    def load_state_dict(self, sd, strict = False):
        return self.audio_tokenizer.load_state_dict(sd, strict = strict)

    def state_dict(self):
        return self.audio_tokenizer.state_dict()
