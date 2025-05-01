import math
import torch


def attention_forward_core_ref_impl(q, k, v, sm_scale, causal, use_exp2):
    # Compute attention scores
    attention_scores = torch.matmul(q.to(torch.float32), k.transpose(-2, -1).to(torch.float32))

    # Scale scores
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


    # Compute max for numerical stability
    max_scores = torch.max(attention_scaled_scores, dim=-1, keepdim=True)[0]
    if causal:
        # Replace -inf in max_scores with zeros to avoid NaN in subtraction
        max_scores = torch.where(
            torch.isinf(max_scores), torch.zeros_like(max_scores), max_scores
        )

    # Shift scores
    attention_shifted_scaled_scores = attention_scaled_scores - max_scores

    # Exponentiate
    if use_exp2:
        RCP_LN = 1 / math.log(2)
        exp_scores = torch.exp2(RCP_LN * attention_shifted_scaled_scores)
    else:
        exp_scores = torch.exp(attention_shifted_scaled_scores)

    # Sum of exponentials
    sum_exp_scores = torch.sum(exp_scores, dim=-1, keepdim=True)
    if causal:
        # if sum of exp scores is 0.0 it means scores where -inf, we cannot compute softmax and softmax_lse. Setting to 1 deals with -inf case cleanly
        sum_exp_scores = torch.where(
        sum_exp_scores == 0,
        torch.ones_like(sum_exp_scores),
        sum_exp_scores
        )

    # Compute softmax probabilities
    softmax = exp_scores / sum_exp_scores

    # Compute log-sum-exp
    if use_exp2:
        LN2 = math.log(2)
        RCP_LN = 1 / math.log(2)
        max_scores_base2 = max_scores * RCP_LN
        softmax_lse_base2 = max_scores_base2 + torch.log2(sum_exp_scores)
        softmax_lse = softmax_lse_base2 * LN2
        softmax_lse.squeeze_(-1)
    else:
        softmax_lse = max_scores + torch.log(sum_exp_scores)
        softmax_lse = softmax_lse.squeeze(-1)

    # Compute output
    o = torch.matmul(softmax, v.to(torch.float32)).to(torch.float16)

    return o, softmax_lse, exp_scores, softmax, attention_shifted_scaled_scores, attention_scaled_scores, attention_scores

def attention_vanilla_forward_pytorch_ref_impl(q, k, v, sm_scale, causal, layout, use_exp2):
    """Compute reference output and softmax_lse using PyTorch's built-in function"""

    # Ensure the layout is 'bhsd'
    if layout == "bshd":
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
    elif layout != "bhsd":
        raise ValueError(f"Unknown layout {layout}")

    # Prepare tensors in [batch_size * num_heads, seq_len, head_dim] format
    batch_size, num_heads, seq_len_q, head_dim = q.shape
    seq_len_k = k.shape[2]

    # Merge batch and heads dimensions
    q = q.reshape(batch_size * num_heads, seq_len_q, head_dim)
    k = k.reshape(batch_size * num_heads, seq_len_k, head_dim)
    v = v.reshape(batch_size * num_heads, seq_len_k, head_dim)

    # Call the core attention function
    o, softmax_lse, exp_scores, softmax, attention_shifted_scaled_scores, attention_scaled_scores, attention_scores = attention_forward_core_ref_impl(
        q, k, v, sm_scale, causal, use_exp2
    )

    # Reshape outputs back to [batch_size, num_heads, seq_len, head_dim]
    o = o.reshape(batch_size, num_heads, seq_len_q, head_dim)
    softmax_lse = softmax_lse.reshape(batch_size, num_heads, seq_len_q)
    exp_scores = exp_scores.reshape(batch_size, num_heads, seq_len_q, seq_len_k)
    softmax = softmax.reshape(batch_size, num_heads, seq_len_q, seq_len_k)
    attention_shifted_scaled_scores = attention_shifted_scaled_scores.reshape(batch_size, num_heads, seq_len_q, seq_len_k)
    attention_scaled_scores = attention_scaled_scores.reshape(batch_size, num_heads, seq_len_q, seq_len_k)
    attention_scores = attention_scores.reshape(batch_size, num_heads, seq_len_q, seq_len_k)

    # Restore original layout if necessary
    if layout == "bshd":
        o = o.transpose(1, 2)

    return o, softmax_lse, exp_scores, softmax, attention_shifted_scaled_scores, attention_scaled_scores, attention_scores

def attention_varlen_forward_pytorch_ref_impl(
    q,
    k,
    v,
    sm_scale,
    causal,
    layout,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q, max_seqlen_k, # pylint: disable=unused-argument
    use_exp2
):
    # Ensure the layout is 'thd'
    if layout != 'thd':
        raise ValueError(f"Unsupported layout {layout}. Expected 'thd'.")

    batch_size = cu_seqlens_q.shape[0] - 1
    num_heads = q.shape[1]
    head_dim = q.shape[2]

    # Pre-allocate outputs
    total_L_q = q.shape[0]
    total_L_k = k.shape[0] # pylint: disable=unused-variable

    o = torch.empty((total_L_q, num_heads, head_dim), dtype=q.dtype, device=q.device)
    softmax_lse = torch.empty((total_L_q, num_heads), dtype=torch.float32, device=q.device)

    for i in range(batch_size):
        # Get the start and end indices for the current sequence
        start_q = cu_seqlens_q[i].item()
        end_q = cu_seqlens_q[i + 1].item()
        start_k = cu_seqlens_k[i].item()
        end_k = cu_seqlens_k[i + 1].item()

        # Extract q_i, k_i, v_i
        q_i = q[start_q:end_q, :, :]  # [L_q_i, num_heads, head_dim]
        k_i = k[start_k:end_k, :, :]  # [L_k_i, num_heads, head_dim]
        v_i = v[start_k:end_k, :, :]  # [L_k_i, num_heads, head_dim]

        # Permute to [num_heads, L_q_i, head_dim]
        q_i = q_i.permute(1, 0, 2)
        k_i = k_i.permute(1, 0, 2)
        v_i = v_i.permute(1, 0, 2)

        # Call the core attention function for this sequence
        (
            o_i,
            softmax_lse_i,
            exp_scores_i,
            softmax_i,
            attention_shifted_scaled_scores_i,
            attention_scaled_scores_i,
            attention_scores_i,
        ) = attention_forward_core_ref_impl(q_i, k_i, v_i, sm_scale, causal, use_exp2)

        # Convert back to 'thd' layout and float16
        o_i = o_i.permute(1, 0, 2).to(torch.float16)  # [L_q_i, num_heads, head_dim]

        # Place outputs in pre-allocated tensors
        o[start_q:end_q, :, :] = o_i
        softmax_lse[start_q:end_q, :] = softmax_lse_i.transpose(0, 1)  # Transpose to [L_q_i, num_heads]

        # For variable-sized outputs, map them into the preallocated tensors
        # exp_scores_i: [num_heads, L_q_i, L_k_i] -> [L_q_i, num_heads, L_k_i]
        exp_scores_i = exp_scores_i.permute(1, 0, 2)
        softmax_i = softmax_i.permute(1, 0, 2)
        attention_shifted_scaled_scores_i = attention_shifted_scaled_scores_i.permute(1, 0, 2)
        attention_scaled_scores_i = attention_scaled_scores_i.permute(1, 0, 2)
        attention_scores_i = attention_scores_i.permute(1, 0, 2)

    return (
        o,
        softmax_lse,
        None,
        None,
        None,
        None,
        None,
    )


def attention_forward_pytorch_ref_impl(
    q,
    k,
    v,
    sm_scale,
    causal,
    layout,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    use_exp2
    ):
     # compute reference
    if layout == "thd":
        (
            o_ref,
            softmax_lse_ref,
            exp_scores_ref,
            softmax_ref,
            attention_shifted_scaled_scores_ref,
            attention_scaled_scores_ref,
            attention_scores_ref,
        ) = attention_varlen_forward_pytorch_ref_impl(
            q.clone(),
            k.clone(),
            v.clone(),
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
        (
            o_ref,
            softmax_lse_ref,
            exp_scores_ref,
            softmax_ref,
            attention_shifted_scaled_scores_ref,
            attention_scaled_scores_ref,
            attention_scores_ref,
        ) = attention_vanilla_forward_pytorch_ref_impl(
            q.clone(), k.clone(), v.clone(), sm_scale, causal, layout, use_exp2
        )

    return (
            o_ref,
            softmax_lse_ref,
            exp_scores_ref,
            softmax_ref,
            attention_shifted_scaled_scores_ref,
            attention_scaled_scores_ref,
            attention_scores_ref,
    )


def compute_alibi_tensor_ref(alibi_slopes, seqlen_q, seqlen_k):
    q_idx = torch.arange(seqlen_q, dtype=torch.int32, device="cuda").unsqueeze(-1)  # (N_CTX_Q, 1)
    k_idx = torch.arange(seqlen_k, dtype=torch.int32, device="cuda").unsqueeze(0)  # (1, N_CTX_K)
    relative_pos = torch.abs(q_idx + seqlen_k - seqlen_q - k_idx)  # (N_CTX_Q, N_CTX_K)
    return -1 * alibi_slopes.unsqueeze(-1).unsqueeze(-1) * relative_pos  # (Z, H, N_CTX_Q, N_CTX_K)
