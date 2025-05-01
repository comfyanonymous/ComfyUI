import torch
import triton
import triton.language as tl
from comfy.flash_attn_triton_amd.utils import get_shape_from_layout, get_strides_from_layout


@triton.jit
def _bwd_preprocess_use_o(
    Out,
    DO,
    Delta,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_doz, stride_doh, stride_dom, stride_dok, # pylint: disable=unused-argument
    stride_deltaz, stride_deltah, stride_deltam,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    ACTUAL_BLOCK_DMODEL: tl.constexpr,
    N_CTX_Q: tl.constexpr,
    Z: tl.constexpr, # pylint: disable=unused-argument
    H: tl.constexpr,
    IS_VARLEN: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    # Compute batch and head indices
    off_z = pid_bh // H
    off_h = pid_bh % H

    if IS_VARLEN:
        # Compute sequence lengths for the current batch
        q_start = tl.load(cu_seqlens_q + off_z)
        q_end = tl.load(cu_seqlens_q + off_z + 1)
        k_start = tl.load(cu_seqlens_k + off_z)
        k_end = tl.load(cu_seqlens_k + off_z + 1)

        # Compute actual sequence lengths
        N_CTX_Q = q_end - q_start
        N_CTX_K = k_end - k_start # pylint: disable=unused-variable
    else:
        q_start = 0
        k_start = 0
        N_CTX_Q = max_seqlen_q
        N_CTX_K = max_seqlen_k # pylint: disable=unused-variable

    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, BLOCK_DMODEL)

    # create masks
    mask_m = off_m < N_CTX_Q
    mask_d = off_d < ACTUAL_BLOCK_DMODEL

    # compute offsets
    o_offset = Out + off_z * stride_oz + off_h * stride_oh + q_start * stride_om
    do_offset = DO + off_z * stride_oz + off_h * stride_oh + q_start * stride_om

    # compute pointers
    out_ptrs = o_offset + off_m[:, None] * stride_om + off_d[None, :] * stride_ok
    do_ptrs = do_offset + off_m[:, None] * stride_dom + off_d[None, :] * stride_dok

    # load
    o = tl.load(out_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    # compute delta
    delta = tl.sum(o * do, axis=1)

    # write-back delta
    delta_offset = Delta + off_z * stride_deltaz + off_h * stride_deltah + q_start * stride_deltam
    delta_ptrs = delta_offset + off_m * stride_deltam
    tl.store(delta_ptrs, delta, mask=mask_m)


@triton.jit
def _bwd_kernel_one_col_block(
    Q,
    K,
    V,
    sm_scale,
    Out, DO, DQ, DK, DV, L, D, # pylint: disable=unused-argument
    q_offset,
    k_offset,
    v_offset,
    do_offset,
    dq_offset,
    dk_offset,
    dv_offset,
    d_offset,
    l_offset,
    stride_dq_all, stride_qz, stride_qh, # pylint: disable=unused-argument
    stride_qm,
    stride_qk,
    stride_kz, stride_kh, # pylint: disable=unused-argument
    stride_kn,
    stride_kk,
    stride_vz, stride_vh, # pylint: disable=unused-argument
    stride_vn,
    stride_vk,
    stride_deltaz,  stride_deltah, # pylint: disable=unused-argument
    stride_deltam,
    Z, H, # pylint: disable=unused-argument
    N_CTX_Q,
    N_CTX_K,
    off_h, off_z, off_hz, # pylint: disable=unused-argument
    start_n,
    num_block_m,
    num_block_n, # pylint: disable=unused-argument
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    ACTUAL_BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SEQUENCE_PARALLEL: tl.constexpr,
    CAUSAL: tl.constexpr,
    USE_EXP2: tl.constexpr,
):
    if CAUSAL:
        # TODO: Causal can skip more blocks with something like lo = start_m * BLOCK_M
        lo = 0
    else:
        lo = 0

    # initialize col and head offsets
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # masks
    mask_n = offs_n < N_CTX_K
    mask_d = offs_d < ACTUAL_BLOCK_DMODEL
    kv_mask = mask_n[:, None] & mask_d[None, :]

    # initialize grad accumulators
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

    # load k and v once per column block
    k_ptrs = k_offset + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_ptrs = v_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
    k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
    v = tl.load(v_ptrs, mask=kv_mask, other=0.0)

    # loop over rows
    for start_m in range(lo, num_block_m * BLOCK_M, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        q_ptrs = q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        dq_ptrs = dq_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        do_ptrs = do_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk

        # update mask as row block changes
        mask_m = offs_m < N_CTX_Q
        q_mask = mask_m[:, None] & mask_d[None, :]

        # load q, k, v, do on-chip
        q = tl.load(q_ptrs, mask=q_mask, other=0.0)
        do = tl.load(do_ptrs, mask=q_mask, other=0.0)

        # recompute p = softmax(qk, dim=-1).T
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))

        if CAUSAL:
            col_offset = N_CTX_Q - N_CTX_K
            causal_mask = offs_m[:, None] >= (col_offset + offs_n[None, :])
            qk = tl.where(causal_mask, qk, float("-inf"))

        l_ptrs = l_offset + offs_m * stride_deltam
        l_i = tl.load(l_ptrs, mask=mask_m)

        # compute p
        if USE_EXP2:
            RCP_LN2: tl.constexpr = 1.4426950408889634
            qk *= sm_scale * RCP_LN2
            l_i *= RCP_LN2
            p = tl.math.exp2(qk - l_i[:, None])
        else:
            qk *= sm_scale
            p = tl.math.exp(qk - l_i[:, None])

        # mask block in the cases where the data is smaller the block size
        p_mask = mask_m[:, None] & mask_n[None, :]
        p = tl.where(p_mask, p, 0.0)

        # compute dv
        dv += tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do)

        # compute dp
        dp = tl.dot(do, tl.trans(v))

        # compute ds , ds = p * (dp - delta[:, None])
        d_ptrs = d_offset + offs_m * stride_deltam
        Di = tl.load(d_ptrs, mask=mask_m)
        ds = (p * (dp - Di[:, None])) * sm_scale
        ds = tl.where(p_mask, ds, 0.0).to(Q.dtype.element_ty)

        # compute dk = dot(ds.T, q)
        dk += tl.dot(tl.trans(ds), q)

        # compute dq
        if SEQUENCE_PARALLEL:
            dq = tl.dot(ds, k)
        else:
            dq = tl.load(dq_ptrs, mask=q_mask, other=0.0)
            dq += tl.dot(ds, k)
        tl.store(dq_ptrs, dq.to(Q.dtype.element_ty), mask=q_mask)

    # write-back dv and dk
    dk_ptrs = dk_offset + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    dv_ptrs = dv_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk

    # write-back
    tl.store(dk_ptrs, dk.to(K.dtype.element_ty), mask=kv_mask)
    tl.store(dv_ptrs, dv.to(V.dtype.element_ty), mask=kv_mask)

@triton.jit
def _bwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    Out,
    DO,
    DQ,
    DK,
    DV,
    L,
    D,
    stride_dq_all,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_deltaz,
    stride_deltah,
    stride_deltam,
    Z,
    H,
    num_block_m,
    num_block_n,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    ACTUAL_BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SEQUENCE_PARALLEL: tl.constexpr,
    CAUSAL: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    # program ids
    off_hz = tl.program_id(0)
    if SEQUENCE_PARALLEL:
        start_n = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    if IS_VARLEN:
        # Compute sequence lengths for the current batch
        q_start = tl.load(cu_seqlens_q + off_z)
        q_end = tl.load(cu_seqlens_q + off_z + 1)
        k_start = tl.load(cu_seqlens_k + off_z)
        k_end = tl.load(cu_seqlens_k + off_z + 1)

        # Compute actual sequence lengths
        N_CTX_Q = q_end - q_start
        N_CTX_K = k_end - k_start
    else:
        q_start = 0
        k_start = 0
        N_CTX_Q = max_seqlen_q
        N_CTX_K = max_seqlen_k

    # input tensor offsets
    q_offset = Q + off_z * stride_qz + off_h * stride_qh + q_start * stride_qm
    k_offset = K + off_z * stride_kz + off_h * stride_kh + k_start * stride_kn
    v_offset = V + off_z * stride_vz + off_h * stride_vh + k_start * stride_vn
    do_offset = DO + off_z * stride_qz + off_h * stride_qh + q_start * stride_qm
    l_offset = L + off_z * stride_deltaz + off_h * stride_deltah + q_start * stride_deltam
    d_offset = D + off_z * stride_deltaz + off_h * stride_deltah + q_start * stride_deltam

    # output tensor offsets
    dk_offset = DK + off_z * stride_kz + off_h * stride_kh + k_start * stride_kn
    dv_offset = DV + off_z * stride_vz + off_h * stride_vh + k_start * stride_vn
    if SEQUENCE_PARALLEL:
        dq_offset = DQ + start_n * stride_dq_all + off_z * stride_qz + off_h * stride_qh + q_start * stride_qm
    else:
        dq_offset = DQ + off_z * stride_qz + off_h * stride_qh + q_start * stride_qm

    # inner loop
    if SEQUENCE_PARALLEL:
        _bwd_kernel_one_col_block(
            Q,
            K,
            V,
            sm_scale,
            Out,
            DO,
            DQ,
            DK,
            DV,
            L,
            D,
            q_offset,
            k_offset,
            v_offset,
            do_offset,
            dq_offset,
            dk_offset,
            dv_offset,
            d_offset,
            l_offset,
            stride_dq_all,
            stride_qz,
            stride_qh,
            stride_qm,
            stride_qk,
            stride_kz,
            stride_kh,
            stride_kn,
            stride_kk,
            stride_vz,
            stride_vh,
            stride_vn,
            stride_vk,
            stride_deltaz,
            stride_deltah,
            stride_deltam,
            Z,
            H,
            N_CTX_Q,
            N_CTX_K,
            off_h,
            off_z,
            off_hz,
            start_n,
            num_block_m,
            num_block_n,
            BLOCK_M=BLOCK_M,
            BLOCK_DMODEL=BLOCK_DMODEL,
            ACTUAL_BLOCK_DMODEL=ACTUAL_BLOCK_DMODEL,
            BLOCK_N=BLOCK_N,
            SEQUENCE_PARALLEL=SEQUENCE_PARALLEL,
            CAUSAL=CAUSAL,
            USE_EXP2=USE_EXP2,
        )
    else:
        for start_n in range(0, num_block_n):
            _bwd_kernel_one_col_block(
                Q,
                K,
                V,
                sm_scale,
                Out,
                DO,
                DQ,
                DK,
                DV,
                L,
                D,
                q_offset,
                k_offset,
                v_offset,
                do_offset,
                dq_offset,
                dk_offset,
                dv_offset,
                d_offset,
                l_offset,
                stride_dq_all,
                stride_qz,
                stride_qh,
                stride_qm,
                stride_qk,
                stride_kz,
                stride_kh,
                stride_kn,
                stride_kk,
                stride_vz,
                stride_vh,
                stride_vn,
                stride_vk,
                stride_deltaz,
                stride_deltah,
                stride_deltam,
                Z,
                H,
                N_CTX_Q,
                N_CTX_K,
                off_h,
                off_z,
                off_hz,
                start_n,
                num_block_m,
                num_block_n,
                BLOCK_M=BLOCK_M,
                BLOCK_DMODEL=BLOCK_DMODEL,
                ACTUAL_BLOCK_DMODEL=ACTUAL_BLOCK_DMODEL,
                BLOCK_N=BLOCK_N,
                SEQUENCE_PARALLEL=SEQUENCE_PARALLEL,
                CAUSAL=CAUSAL,
                USE_EXP2=USE_EXP2,
            )


# NOTE: smaller blocks have lower accuracy. more accumlation error probably 128 * 128 seems good but leads to oom. 64 * 64 has accumlation errors but no oom.
def attention_prefill_backward_triton_impl(
    do,
    q,
    k,
    v,
    o,
    softmax_lse,
    dq,
    dk,
    dv,
    sm_scale: float,
    alibi_slopes, # pylint: disable=unused-argument
    causal,
    layout: str,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q: int,
    max_seqlen_k: int,
    use_exp2: bool,
    sequence_parallel = True,
):
    # make contigious
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    softmax_lse = softmax_lse.contiguous()

    # get strides and shape
    batch, nheads_q, nheads_k, head_size, max_seqlen_q, max_seqlen_k = get_shape_from_layout(q, k, layout, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k) # pylint: disable=unused-variable
    q_strides, k_strides, v_strides, o_strides = get_strides_from_layout(q, k, v, o, layout)
    stride_qz, stride_qh, stride_qm, stride_qk =  q_strides
    stride_kz, stride_kh, stride_kn, stride_kk = k_strides
    stride_vz, stride_vh, stride_vn, stride_vk = v_strides
    stride_oz, stride_oh, stride_om, stride_ok = o_strides
    batch_headsize = batch * nheads_q
    is_varlen = layout == "thd"

    # FIXME: some configs lead to oom for some reason when using 64 x 64 blocks
    if max_seqlen_q <= 32 or max_seqlen_k <= 32:
        BLOCK_M = 32
        BLOCK_N = 32
    else:
        BLOCK_M = 64
        BLOCK_N = 64
    num_warps = 4 # NOTE: originial is 8. changing it to 1 caused issues be careful
    num_stages = 1
    waves_per_eu = 1

    # divide up the problem
    num_blocks_m = triton.cdiv(max_seqlen_q, BLOCK_M)
    num_blocks_n = triton.cdiv(max_seqlen_k, BLOCK_N)

    # get closest power of 2 over or equal to 32.
    padded_d_model = 1 << (head_size - 1).bit_length()
    padded_d_model = max(padded_d_model, 16)
    BLOCK_DMODEL = padded_d_model
    ACTUAL_BLOCK_DMODEL = head_size

    do = do.contiguous()
    # NOTE: we might need to copy the output tensor if they are not continuous or have other issues
    copy_back = {"dq": False, "dk": False, "dv": False}

    dq_og = None
    # deal with dq
    if dq is None:
        if sequence_parallel:
            dq = torch.zeros((num_blocks_n,) + q.shape, device=q.device, dtype=q.dtype)
        else:
            dq = torch.zeros(q.shape, device=q.device, dtype=q.dtype)
    else:
        dq_og = dq
        if not dq.is_contiguous():
            dq = dq.contiguous()
            copy_back["dq"] = True

        if sequence_parallel:
            dq = torch.zeros((num_blocks_n,) + q.shape, device=q.device, dtype=q.dtype)
            copy_back["dq"] = True
        else:
            # NOTE: the kernel does inplace accumlation so dq has to be zeros. This avoids the case where we are passed empty dq and it is not all zeros
            dq.zero_()
    stride_dq_all = dq.stride()[0]

    dk_og = None
    dv_og = None
    # deal with dk, dv
    if (dk is None) or (dv is None):
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
    else:
        if not dk.is_contiguous():
            dk_og = dk
            dk = dk.contiguous()
            copy_back["dk"] = True

        if not dv.is_contiguous():
            dv_og = dv
            dv = dv.contiguous()
            copy_back["dv"] = True

    # assert contigious
    assert do.is_contiguous()
    assert q.is_contiguous()
    assert k.is_contiguous()
    assert v.is_contiguous()
    assert o.is_contiguous()
    assert softmax_lse.is_contiguous()

    # init delta
    delta = torch.empty_like(softmax_lse)
    if is_varlen:
        stride_deltam, stride_deltah = delta.stride()
        stride_deltaz = 0
    else:
        stride_deltaz, stride_deltah, stride_deltam = delta.stride()

    _bwd_preprocess_use_o[(num_blocks_m, batch_headsize)](
        o,
        do,
        delta,
        stride_oz, stride_oh, stride_om, stride_ok,
        stride_oz, stride_oh, stride_om, stride_ok,
        stride_deltaz, stride_deltah, stride_deltam,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL=BLOCK_DMODEL,
        ACTUAL_BLOCK_DMODEL=ACTUAL_BLOCK_DMODEL,
        N_CTX_Q=max_seqlen_q,
        Z=batch,
        H=nheads_q,
        IS_VARLEN=is_varlen
    )

    _bwd_kernel[(batch_headsize, num_blocks_n if sequence_parallel else 1)](
        q,
        k,
        v,
        sm_scale,
        o,
        do,
        dq,
        dk,
        dv,
        softmax_lse,
        delta,
        stride_dq_all,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vn, stride_vk,
        stride_deltaz, stride_deltah, stride_deltam,
        batch,
        nheads_q,
        num_blocks_m,
        num_blocks_n,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        ACTUAL_BLOCK_DMODEL=ACTUAL_BLOCK_DMODEL,
        SEQUENCE_PARALLEL=sequence_parallel,
        CAUSAL=causal,
        USE_EXP2=use_exp2,
        num_warps=num_warps,
        num_stages=num_stages,
        waves_per_eu = waves_per_eu,
        IS_VARLEN=is_varlen
    )

    if sequence_parallel:
        dq = dq.sum(dim=0)

    if copy_back["dq"]:
        dq_og.copy_(dq)
        dq = dq_og
    if copy_back["dk"]:
        dk_og.copy_(dk)
        dk = dk_og
    if copy_back["dv"]:
        dv_og.copy_(dv)
        dv = dv_og

    return dq, dk, dv, delta, None, None
