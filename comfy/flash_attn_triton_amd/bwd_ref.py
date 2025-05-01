import math
import torch


def attention_backward_core_ref_impl(
    do, q, k, v, o, softmax_lse, sm_scale, causal, use_exp2
):
    # cast to float32
    do = do.to(torch.float32)
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    v = v.to(torch.float32)
    o = o.to(torch.float32)
    softmax_lse = softmax_lse.to(torch.float32)

    # recompute attention_scores. Make sure it matches the forward impl. i.e. It use float32
    attention_scores = torch.matmul(q.to(torch.float32), k.transpose(-2, -1).to(torch.float32))

    # scale scores
    attention_scaled_scores = sm_scale * attention_scores

    # Apply causal mask if necessary
    if causal:
        L_q, L_k = q.shape[1], k.shape[1]
        row_idx = torch.arange(L_q, device=q.device).unsqueeze(1)
        col_idx = torch.arange(L_k, device=q.device).unsqueeze(0)
        col_offset = L_q-L_k
        causal_mask = row_idx >= (col_offset + col_idx)
        # set -inf to places the causal mask is false
        attention_scaled_scores = attention_scaled_scores.masked_fill(
             torch.logical_not(causal_mask.unsqueeze(0)), float('-inf')
        )

    # compute probabilities using softmax_lse
    if use_exp2:
        RCP_LN = 1 / math.log(2)
        attention_scaled_scores_base2 = attention_scaled_scores * RCP_LN
        softmax_lse_base2 = softmax_lse * RCP_LN
        softmax_lse_3d =  softmax_lse_base2.unsqueeze(-1)
        p = torch.exp2(attention_scaled_scores_base2 - softmax_lse_3d)
    else:
        softmax_lse_3d =  softmax_lse.unsqueeze(-1)
        p = torch.exp(attention_scaled_scores - softmax_lse_3d)

    # compute gradient wrt v
    dv = torch.matmul(p.transpose(-2, -1), do.to(torch.float32))

    # compute dp
    dp = torch.matmul(do, v.transpose(-2, -1))

    # calculate ds using dp
    delta = torch.sum(o * do, axis=-1).to(torch.float32)  # what OAI kernel uses
    delta_3d = delta.unsqueeze(-1)
    ds = (p * (dp - delta_3d)) * sm_scale

    # compute gradient wrt k
    dk = torch.matmul(ds.transpose(-2, -1), q.to(torch.float32))

    # compute gradient wrt q
    dq = torch.matmul(ds, k.to(torch.float32))

    # cast back to original dtype
    dq = dq.to(torch.float16)
    dk = dk.to(torch.float16)
    dv = dv.to(torch.float16)

    # remove d dim with size 1
    delta = delta_3d.squeeze(-1)

    return dq, dk, dv, delta

def attention_varlen_backward_pytorch_ref_impl(
    do,
    q,
    k,
    v,
    o,
    softmax_lse,
    sm_scale,
    causal,
    layout,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q, max_seqlen_k, # pylint: disable=unused-argument
    use_exp2,
):
    # Ensure the layout is 'thd'
    if layout != 'thd':
        raise ValueError(f"Unsupported layout {layout}. Expected 'thd'.")

    batch_size = cu_seqlens_q.shape[0] - 1
    num_heads = q.shape[1]
    head_dim = q.shape[2] # pylint: disable=unused-variable

    # Pre-allocate outputs
    total_L_q = q.shape[0]
    total_L_k = k.shape[0] # pylint: disable=unused-variable

    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    # delta has the same shape as softmax_lse: [total_L_q, num_heads]
    delta = torch.zeros((total_L_q, num_heads), dtype=torch.float32, device=o.device)

    for i in range(batch_size):
        # Get the start and end indices for the current sequence
        start_q = cu_seqlens_q[i].item()
        end_q = cu_seqlens_q[i + 1].item()
        start_k = cu_seqlens_k[i].item()
        end_k = cu_seqlens_k[i + 1].item()

        # Extract q_i, k_i, v_i, do_i, o_i, softmax_lse_i
        q_i = q[start_q:end_q, :, :]      # [L_q_i, num_heads, head_dim]
        k_i = k[start_k:end_k, :, :]      # [L_k_i, num_heads, head_dim]
        v_i = v[start_k:end_k, :, :]      # [L_k_i, num_heads, head_dim]
        do_i = do[start_q:end_q, :, :]    # [L_q_i, num_heads, head_dim]
        o_i = o[start_q:end_q, :, :]      # [L_q_i, num_heads, head_dim]
        # softmax_lse has shape [total_L_q, num_heads]
        softmax_lse_i = softmax_lse[start_q:end_q, :]  # [L_q_i, num_heads]
        softmax_lse_i = softmax_lse_i.transpose(0, 1)  # [num_heads, L_q_i]

        # Permute to [num_heads, L_q_i, head_dim]
        q_i = q_i.permute(1, 0, 2)
        k_i = k_i.permute(1, 0, 2)
        v_i = v_i.permute(1, 0, 2)
        do_i = do_i.permute(1, 0, 2)
        o_i = o_i.permute(1, 0, 2)
        # softmax_lse_i is already in [num_heads, L_q_i]

        # Call the core backward function for this sequence
        dq_i, dk_i, dv_i, delta_i = attention_backward_core_ref_impl(
            do_i,
            q_i,
            k_i,
            v_i,
            o_i,
            softmax_lse_i,
            sm_scale,
            causal,
            use_exp2
        )

        # Convert back to 'thd' layout
        dq_i = dq_i.permute(1, 0, 2)  # [L_q_i, num_heads, head_dim]
        dk_i = dk_i.permute(1, 0, 2)  # [L_k_i, num_heads, head_dim]
        dv_i = dv_i.permute(1, 0, 2)  # [L_k_i, num_heads, head_dim]

        # Place outputs in pre-allocated tensors
        dq[start_q:end_q, :, :] = dq_i
        dk[start_k:end_k, :, :] += dk_i  # Accumulate gradients for shared keys
        dv[start_k:end_k, :, :] += dv_i  # Accumulate gradients for shared values
        # delta_i has shape [num_heads, L_q_i]
        delta_i = delta_i.transpose(1, 0)  # [L_q_i, num_heads]
        delta[start_q:end_q, :] = delta_i

    return dq, dk, dv, delta

def attention_vanilla_backward_pytorch_ref_impl(
    do,
    q,
    k,
    v,
    o,
    softmax_lse,
    sm_scale,
    causal,
    layout,
    use_exp2,
):
    if layout == "bshd":
        do = do.transpose(1, 2).contiguous()
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        o = o.transpose(1, 2).contiguous()
    elif layout == "bhsd":
        pass
    else:
        raise ValueError(f"Unknown layout {layout}")

    # Prepare tensors in [batch_size * num_heads, seq_len, head_dim] format
    batch_size, num_heads, seq_len_q, head_dim = q.shape
    seq_len_k = k.shape[2]

    # Merge batch and heads dimensions
    do = do.reshape(batch_size * num_heads, seq_len_q, head_dim)
    q = q.reshape(batch_size * num_heads, seq_len_q, head_dim)
    k = k.reshape(batch_size * num_heads, seq_len_k, head_dim)
    v = v.reshape(batch_size * num_heads, seq_len_k, head_dim)
    softmax_lse = softmax_lse.reshape(batch_size * num_heads, seq_len_q)
    o = o.reshape(batch_size * num_heads, seq_len_q, head_dim)

    dq, dk, dv, delta = attention_backward_core_ref_impl(
        do,
        q,
        k,
        v,
        o,
        softmax_lse,
        sm_scale,
        causal,
        use_exp2
    )

    # Reshape outputs back to [batch_size, num_heads, seq_len, head_dim]
    dq = dq.reshape(batch_size, num_heads, seq_len_q, head_dim)
    dk = dk.reshape(batch_size, num_heads, seq_len_k, head_dim)
    dv = dv.reshape(batch_size, num_heads, seq_len_k, head_dim)
    delta = delta.reshape(batch_size, num_heads, seq_len_q)

    # Go back to original layout
    if layout == "bshd":
        dq = dq.transpose(1, 2)
        dk = dk.transpose(1, 2)
        dv = dv.transpose(1, 2)
    elif layout == "bhsd":
        pass
    else:
        raise ValueError(f"Unknown layout {layout}")

    return dq, dk, dv, delta


def attention_backward_pytorch_ref_impl(
    do,
    q,
    k,
    v,
    o,
    softmax_lse,
    sm_scale,
    causal,
    layout,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    use_exp2
):
    if layout == "thd":
        dq, dk, dv, delta = attention_varlen_backward_pytorch_ref_impl(
            do,
            q,
            k,
            v,
            o,
            softmax_lse,
            sm_scale,
            causal,
            layout,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            use_exp2,
        )
    else:
        dq, dk, dv, delta = attention_vanilla_backward_pytorch_ref_impl(
            do,
            q,
            k,
            v,
            o,
            softmax_lse,
            sm_scale,
            causal,
            layout,
            use_exp2,
        )

    return dq, dk, dv, delta
