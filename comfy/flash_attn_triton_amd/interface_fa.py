import os
import torch
from comfy.flash_attn_triton_amd.fwd_prefill import attention_prefill_forward_triton_impl
from comfy.flash_attn_triton_amd.bwd_prefill import attention_prefill_backward_triton_impl
from comfy.flash_attn_triton_amd.fwd_decode import attention_decode_forward_triton_impl
from comfy.flash_attn_triton_amd.fwd_ref import attention_forward_pytorch_ref_impl
from comfy.flash_attn_triton_amd.bwd_ref import attention_backward_pytorch_ref_impl
from comfy.flash_attn_triton_amd.utils import MetaData, get_shape_from_layout


USE_REF = os.environ.get('FLASH_ATTENTION_TRITON_AMD_REF', '0').lower() in ('1', 'true', 'yes')


def fwd(q,
    k,
    v,
    o,
    alibi_slopes,
    dropout_p,
    softmax_scale,
    causal,
    window_size_left, window_size_right, softcap, # pylint: disable=unused-argument
    return_softmax,
    gen_ # pylint: disable=unused-argument
):
    if dropout_p != 0.0:
        raise ValueError("dropout is not supported on AMD's Triton Backend yet")

    if o is None:
        o = torch.empty_like(q)

    # Setup metadata
    metadata = MetaData(sm_scale=softmax_scale)
    metadata.max_seqlens_q = q.shape[1]
    metadata.max_seqlens_k = k.shape[1]
    metadata.layout = "bshd"
    if return_softmax:
        metadata.return_scores = True

    batch, nheads_q, nheads_k, head_size, _, _ = get_shape_from_layout(q, k, metadata.layout) # pylint: disable=unused-variable

    if causal:
        metadata.need_causal()

    if alibi_slopes is not None:
        metadata.need_alibi(alibi_slopes, batch, nheads_q)

    if dropout_p > 0.0:
        metadata.need_dropout(dropout_p, return_softmax)

    # Check arguments
    metadata.check_args(q, k, v, o)
    if USE_REF:
        (output,
        softmax_lse,
        exp_scores,
        _,
        _,
        _,
        _) = attention_forward_pytorch_ref_impl(
                                                q,
                                                k,
                                                v,
                                                metadata.sm_scale,
                                                metadata.causal,
                                                metadata.layout,
                                                metadata.cu_seqlens_q,
                                                metadata.cu_seqlens_k,
                                                metadata.max_seqlens_q,
                                                metadata.max_seqlens_k,
                                                metadata.use_exp2)
        o.copy_(output)
    else:
        (_,
        softmax_lse,
        exp_scores,
        _,
        _,
        _,
        _,
        _,
        _) = attention_prefill_forward_triton_impl(
                                                q,
                                                k,
                                                v,
                                                o,
                                                metadata.sm_scale,
                                                metadata.alibi_slopes,
                                                metadata.causal,
                                                metadata.bias,
                                                metadata.dropout_p,
                                                metadata.layout,
                                                metadata.cu_seqlens_q,
                                                metadata.cu_seqlens_k,
                                                metadata.max_seqlens_q,
                                                metadata.max_seqlens_k,
                                                metadata.return_scores,
                                                metadata.use_exp2)

    return o, softmax_lse, exp_scores, None


def bwd(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    dq,
    dk,
    dv,
    alibi_slopes,
    dropout_p,
    softmax_scale,
    causal,
    window_size_left, window_size_right, softcap, deterministic, gen_, rng_state, # pylint: disable=unused-argument
):
    if dropout_p != 0.0:
        raise ValueError("dropout is not supported on AMD yet")

    if USE_REF:
        dq_ref, dk_ref, dv_ref, delta_ref = attention_backward_pytorch_ref_impl(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            softmax_scale,
            causal,
            "bshd",
            None,
            None,
            None,
            None,
            False,
        )
        dq.copy_(dq_ref)
        dk.copy_(dk_ref)
        dv.copy_(dv_ref)
        delta = delta_ref
    else:
        dq_triton, dk_triton, dv_triton, delta_triton, _, _ = attention_prefill_backward_triton_impl( # pylint: disable=unused-variable
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            softmax_scale,
            alibi_slopes,
            causal,
            "bshd",
            None,
            None,
            None,
            None,
            False,
        )
        delta = delta_triton

    return dq, dk, dv, delta


def varlen_fwd(
    q,
    k,
    v,
    o,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_k, leftpad_k, block_table_, # pylint: disable=unused-argument
    alibi_slopes,\
    max_seqlen_q,
    max_seqlen_k,
    dropout_p,
    softmax_scale,
    zero_tensors, # pylint: disable=unused-argument
    causal,
    window_size_left, window_size_right, softcap, # pylint: disable=unused-argument
    return_softmax,
    gen_ # pylint: disable=unused-argument
):
    if dropout_p != 0.0:
        raise ValueError("dropout is not supported on AMD's Triton Backend yet")

    if o is None:
        o = torch.empty_like(q)

    # Setup metadata
    metadata = MetaData(sm_scale=softmax_scale)
    if return_softmax:
        metadata.return_scores = True
    metadata.set_varlen_params(cu_seqlens_q, cu_seqlens_k)  # set layout to "thd" and other metdata

    # get shapes
    batch, nheads_q, nheads_k, head_size , seqlen_q, seqlen_k = get_shape_from_layout(q, k, metadata.layout, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k) # pylint: disable=unused-variable

    if causal:
        metadata.need_causal()

    if alibi_slopes is not None:
        metadata.need_alibi(alibi_slopes, batch, nheads_q)

    if dropout_p > 0.0:
        metadata.need_dropout(dropout_p, return_softmax)

    # Check arguments
    metadata.check_args(q, k, v, o)
    if o is None:
        o = torch.empty_like(q, dtype=v.dtype)

    if USE_REF:
        (output,
        softmax_lse,
        exp_scores,
        _,
        _,
        _,
        _) = attention_forward_pytorch_ref_impl(
                                                q,
                                                k,
                                                v,
                                                metadata.sm_scale,
                                                metadata.causal,
                                                metadata.layout,
                                                metadata.cu_seqlens_q,
                                                metadata.cu_seqlens_k,
                                                metadata.max_seqlens_q,
                                                metadata.max_seqlens_k,
                                                metadata.use_exp2)
        o.copy_(output)
    else:
        (_,
        softmax_lse,
        exp_scores,
        _,
        _,
        _,
        _,
        _,
        _) = attention_prefill_forward_triton_impl(
                                                    q,
                                                    k,
                                                    v,
                                                    o,
                                                    metadata.sm_scale,
                                                    metadata.alibi_slopes,
                                                    metadata.causal,
                                                    metadata.bias,
                                                    metadata.dropout_p,
                                                    metadata.layout,
                                                    metadata.cu_seqlens_q,
                                                    metadata.cu_seqlens_k,
                                                    metadata.max_seqlens_q,
                                                    metadata.max_seqlens_k,
                                                    metadata.return_scores,
                                                    metadata.use_exp2)

    return o, softmax_lse, exp_scores, None


def varlen_bwd(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    dq,
    dk,
    dv,
    cu_seqlens_q,
    cu_seqlens_k,
    alibi_slopes,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p,
    softmax_scale,
    zero_tensors, # pylint: disable=unused-argument
    causal,
    window_size_left, window_size_right, softcap, deterministic, gen_, rng_state, # pylint: disable=unused-argument
):
    if dropout_p != 0.0:
        raise ValueError("dropout is not supported on AMD yet")

    if USE_REF:
        dq_ref, dk_ref, dv_ref, delta_ref = attention_backward_pytorch_ref_impl(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            softmax_scale,
            causal,
            "thd",
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            False,
        )
        dq.copy_(dq_ref)
        dk.copy_(dk_ref)
        dv.copy_(dv_ref)
        delta = delta_ref
    else:
        dq_triton, dk_triton, dv_triton, delta_triton, _, _ = attention_prefill_backward_triton_impl( # pylint: disable=unused-variable
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            softmax_scale,
            alibi_slopes,
            causal,
            "thd",
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            False,
        )
        delta = delta_triton

    return dq, dk, dv, delta


def fwd_kvcache(
    q,
    k_cache,
    v_cache,
    k,
    v,
    cache_seqlens,
    rotary_cos, rotary_sin, # pylint: disable=unused-argument
    cache_batch_idx,
    cache_leftpad, block_table, # pylint: disable=unused-argument
    alibi_slopes,
    out,
    softmax_scale,
    causal,
    window_size_left, window_size_right, softcap, rotary_interleaved, num_splits, # pylint: disable=unused-argument
):
    if out is None:
        out = torch.empty_like(q)

    # fill metadata
    metadata = MetaData(sm_scale=softmax_scale)
    metadata.layout = "bshd"
    metadata.max_seqlens_q = q.shape[1]
    metadata.max_seqlens_k = k_cache.shape[1]
    metadata.cache_seqlens = cache_seqlens
    metadata.cache_batch_idx = cache_batch_idx

    if k is not None and v is not None:
        metadata.new_kv = True
        metadata.seqlen_new = k.shape[1]
        metadata.k_new = k
        metadata.v_new = v

    if causal:
        metadata.need_causal()

    if alibi_slopes is not None:
        batch, _ , nheads_q, _= q.shape
        metadata.need_alibi(alibi_slopes, batch, nheads_q)

    # launch kernel
    # TODO: pass output as an arg. Maybe we are copying output which is causing slow down
    output, softmax_lse = attention_decode_forward_triton_impl(
        q,
        k_cache,
        v_cache,
        metadata.sm_scale,
        metadata.causal,
        metadata.alibi_slopes,
        metadata.layout,
        metadata.cache_seqlens,
        metadata.cache_batch_idx,
        metadata.new_kv,
        metadata.k_new,
        metadata.v_new,
    )
    return output, softmax_lse
