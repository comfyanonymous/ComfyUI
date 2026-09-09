import torch
import pytest
from torch import nn

from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE


def make_vae(finalized_parts):
    vae = MiniMaxH3VideoVAE.__new__(MiniMaxH3VideoVAE)
    nn.Module.__init__(vae)
    vae.tokens_chunk_size = 5
    vae.token_drop = 3
    vae.token_overlap = 2
    vae.vae_ratio_t = 4
    vae.frame_pre_padding = 3
    vae.frame_overlap = 5
    vae.clip_length = 17
    vae._adaptive_decode = lambda z: z.repeat_interleave(4, dim=2)

    def finalize(part):
        finalized_parts.append(part.shape[2])
        return part.mul(0.25).add(0.5).clamp_(0, 1)

    vae._finalize_pixels = finalize
    return vae


def decode_temporal_reference(vae, z, output_buffer):
    chunk_dec = vae.tokens_chunk_size * vae.vae_ratio_t
    split_count = int(vae.token_drop > 0) + 1
    pad_tokens, num_chunks = vae._decode_temporal_chunks(z.shape[2])
    if pad_tokens > 0:
        z = torch.cat([z, z[:, :, -1:, :, :].repeat(1, 1, pad_tokens, 1, 1)], dim=2)

    dec = output_buffer
    dec_overlap = None
    write_pos = 0

    def write_part(part):
        nonlocal write_pos
        part_frames = part.shape[2]
        if part_frames <= 0:
            return
        part = vae._finalize_pixels(part)
        copy_frames = min(part_frames, max(0, dec.shape[2] - write_pos))
        if copy_frames > 0:
            dec[:, :, write_pos:write_pos + copy_frames, :, :].copy_(part[:, :, :copy_frames, :, :])
            write_pos += copy_frames

    for i in range(num_chunks):
        t_start_idx = i * vae.tokens_chunk_size
        t_end_idx = t_start_idx + vae.tokens_chunk_size + vae.token_overlap
        clip_dec = vae._adaptive_decode(z[:, :, t_start_idx:t_end_idx, :, :])

        for j in range(split_count):
            f_start_idx = j * chunk_dec
            f_end_idx = min(f_start_idx + chunk_dec, clip_dec.shape[2])
            clip_dec_chunk = clip_dec[:, :, f_start_idx:f_end_idx, :, :]
            clip_dec_chunk = clip_dec_chunk[:, :, vae.frame_pre_padding:, :, :]

            if j == 0:
                if dec_overlap is not None:
                    clip_dec_chunk = vae.blend(dec_overlap, clip_dec_chunk, vae.frame_overlap, dim=-3)
                    dec_overlap = None
                write_part(clip_dec_chunk)
            else:
                dec_overlap = clip_dec_chunk.contiguous()

        if i == num_chunks - 1 and dec_overlap is not None:
            write_part(dec_overlap)

    return dec


@torch.inference_mode()
@pytest.mark.parametrize("length", [2, 7, 11, 17])
def test_decode_temporal_skips_finalization_for_discarded_tail(length):
    z = torch.arange(length * 6, dtype=torch.float32).reshape(1, 3, length, 2, 1)

    reference_parts = []
    reference_vae = make_vae(reference_parts)
    pad_tokens, chunks = reference_vae._decode_temporal_chunks(length)
    frames = reference_vae._decode_temporal_frame_plan(length + pad_tokens, chunks, pad_tokens)
    reference = decode_temporal_reference(reference_vae, z.clone(), torch.empty(1, 3, frames, 2, 1))

    candidate_parts = []
    candidate = make_vae(candidate_parts).decode_temporal(
        z.clone(), torch.empty(1, 3, frames, 2, 1)
    )

    assert torch.equal(candidate, reference)
    assert sum(candidate_parts) <= sum(reference_parts)
    if length in (2, 11):
        assert sum(candidate_parts) < sum(reference_parts)


@torch.inference_mode()
def test_decode_temporal_does_not_finalize_when_output_is_full():
    finalized_parts = []
    vae = make_vae(finalized_parts)
    z = torch.ones(1, 3, 7, 2, 1)

    output = vae.decode_temporal(z, torch.empty(1, 3, 0, 2, 1))

    assert output.shape[2] == 0
    assert finalized_parts == []
