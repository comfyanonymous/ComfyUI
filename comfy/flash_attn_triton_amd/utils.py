import os
import torch
import triton


AUTOTUNE = os.environ.get('FLASH_ATTENTION_TRITON_AMD_AUTOTUNE', '0').lower() in ('1', 'true', 'yes')
PERF = os.environ.get('FLASH_ATTENTION_TRITON_AMD_PERF', '0').lower() in ('1', 'true', 'yes')


class MetaData():
    cu_seqlens_q = None
    cu_seqlens_k = None
    max_seqlens_q = 0
    max_seqlens_k = 0
    bias = None
    alibi_slopes = None
    causal = False
    num_contexts = 0
    varlen = False
    layout = None
    cache_seqlens = None
    cache_batch_idx = None
    new_kv = False
    seqlen_new = None
    k_new = None
    v_new = None
    dropout_p, return_scores= 0.0, False
    # NOTE: scale sm_scale by log_2(e) and use 2^x in the loop as we do not have native e^x support in HW.
    use_exp2 = False

    def __repr__(self) -> str:
        return (f"MetaData(\n"
                f"  sm_scale={self.sm_scale},\n"
                f"  cu_seqlens_q={self.cu_seqlens_q},\n"
                f"  cu_seqlens_k={self.cu_seqlens_k},\n"
                f"  max_seqlens_q={self.max_seqlens_q},\n"
                f"  max_seqlens_k={self.max_seqlens_k},\n"
                f"  bias={self.bias},\n"
                f"  alibi_slopes={self.alibi_slopes},\n"
                f"  causal={self.causal},\n"
                f"  num_contexts={self.num_contexts},\n"
                f"  varlen={self.varlen},\n"
                f"  layout={self.layout},\n"
                f"  cache_seqlens={self.cache_seqlens},\n"
                f"  cache_batch_idx={self.cache_batch_idx},\n"
                f"  new_kv={self.new_kv},\n"
                f"  seqlen_new={self.seqlen_new},\n"
                f"  k_new={self.k_new},\n"
                f"  v_new={self.v_new},\n"
                f"  dropout_p={self.dropout_p},\n"
                f"  return_scores={self.return_scores}\n"
                f")")

    def __init__(self, sm_scale=1.0):
        self.sm_scale = sm_scale

    def set_varlen_params(self, cu_seqlens_q, cu_seqlens_k):
        self.varlen = True
        self.layout = 'thd'
        self.cu_seqlens_q = cu_seqlens_q
        self.cu_seqlens_k = cu_seqlens_k
        # Without "varlen", there should still be one sequence.
        assert len(cu_seqlens_q) >= 2
        assert len(cu_seqlens_q) == len(cu_seqlens_k)
        self.num_contexts = len(cu_seqlens_q) - 1
        for i in range(0, self.num_contexts):
            self.max_seqlens_q = max(cu_seqlens_q[i + 1].item() - cu_seqlens_q[i].item(), self.max_seqlens_q)
            self.max_seqlens_k = max(cu_seqlens_k[i + 1].item() - cu_seqlens_k[i].item(), self.max_seqlens_k)

    def need_bias(self, bias, batch, nheads, seqlen_q, seqlen_k): # pylint: disable=unused-argument
        assert bias.is_cuda
        assert bias.dim() == 4
        assert bias.shape[0] == 1
        assert bias.shape[2:] == (seqlen_q, seqlen_k)
        self.bias = bias

    def need_alibi(self, alibi_slopes, batch, nheads):
        assert alibi_slopes.is_cuda
        assert alibi_slopes.dim() == 2
        assert alibi_slopes.shape[0] == batch
        assert alibi_slopes.shape[1] == nheads
        self.alibi_slopes = alibi_slopes

    def need_causal(self):
        self.causal = True

    def need_dropout(self, dropout_p, return_scores):
        self.dropout_p = dropout_p
        self.return_scores = return_scores

    def check_args(self, q, k, v, o):
        assert q.dim() == k.dim() and q.dim() == v.dim()

        batch, nheads_q, nheads_k, head_size, _, _ = get_shape_from_layout(q, k, self.layout, self.cu_seqlens_q, self.cu_seqlens_k, self.max_seqlens_q, self.max_seqlens_k) # pylint: disable=unused-variable
        if self.varlen:
            assert q.dim() == 3
            assert self.cu_seqlens_q is not None
            assert self.cu_seqlens_k is not None
            assert len(self.cu_seqlens_q) == len(self.cu_seqlens_k)
            # TODO: Remove once bias is supported with varlen
            assert self.bias is None
            # TODO:Remove once dropout is supported with varlen
            assert self.dropout_p == 0.0
            # assert not self.return_scores
        else:
            assert q.dim() == 4
            assert self.max_seqlens_q > 0 and self.max_seqlens_k > 0
            assert self.cu_seqlens_q is None and self.cu_seqlens_k is None
        assert k.shape == v.shape
        assert q.shape[-1] == k.shape[-1] and q.shape[-1] == v.shape[-1]
        # TODO: Change assert if we support qkl f8 and v f16
        assert q.dtype == k.dtype and q.dtype == v.dtype
        assert head_size <= 256
        assert o.shape == q.shape
        assert (nheads_q % nheads_k) == 0
        assert self.layout is not None
        assert self.layout == 'thd' or not self.varlen

def input_helper(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout, device="cuda", DEBUG_INPUT=False):
    torch.manual_seed(20)

    # Initialize q, k, v
    if layout == 'bhsd':
        q_tensor_shape = (Z, HQ, N_CTX_Q, D_HEAD)
        k_tensor_shape = (Z, HK, N_CTX_K, D_HEAD)
    elif layout == 'bshd':
        q_tensor_shape = (Z, N_CTX_Q, HQ, D_HEAD)
        k_tensor_shape = (Z, N_CTX_K, HK, D_HEAD)
    else:
        assert False, f'Got unsupported tensor layout: {layout}'

    q = None
    k = None
    v = None

    if DEBUG_INPUT:
        if layout == "bhsd":
            q = torch.arange(N_CTX_Q, dtype=dtype, device=device).view(1, 1, N_CTX_Q, 1).expand(*q_tensor_shape).contiguous().requires_grad_()
            k = torch.arange(N_CTX_K, dtype=dtype, device=device).view(1, 1, N_CTX_K, 1).expand(*k_tensor_shape).contiguous().requires_grad_()
            v = torch.arange(N_CTX_K, dtype=dtype, device=device).view(1, 1, N_CTX_K, 1).expand(*k_tensor_shape).contiguous().requires_grad_()
        elif layout == "bshd":
            q = torch.arange(N_CTX_Q, dtype=dtype, device=device).view(1, N_CTX_Q, 1, 1).expand(*q_tensor_shape).contiguous().requires_grad_()
            k = torch.arange(N_CTX_K, dtype=dtype, device=device).view(1, N_CTX_K, 1, 1).expand(*k_tensor_shape).contiguous().requires_grad_()
            v = torch.arange(N_CTX_K, dtype=dtype, device=device).view(1, N_CTX_K, 1, 1).expand(*k_tensor_shape).contiguous().requires_grad_()
    else:
        q = torch.randn(q_tensor_shape, dtype=dtype, device=device, requires_grad=True)
        k = torch.randn(k_tensor_shape, dtype=dtype, device=device, requires_grad=True)
        v = torch.randn(k_tensor_shape, dtype=dtype, device=device, requires_grad=True)

    if DEBUG_INPUT:
        sm_scale = 1
    else:
        sm_scale = D_HEAD**-0.5
    input_metadata = MetaData(sm_scale=sm_scale)
    input_metadata.max_seqlens_q = N_CTX_Q
    input_metadata.max_seqlens_k = N_CTX_K
    input_metadata.layout = layout
    return q, k, v, input_metadata


def varlen_input_helper(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, device="cuda", equal_seqlens=False, DEBUG_INPUT=False):
    torch.manual_seed(20)

    # Random or equal sequence lengths based on 'equal_seqlens' flag
    if not equal_seqlens:
        max_seqlens_q = N_CTX_Q // Z
        max_seqlens_k = N_CTX_K // Z
        seqlens_q = torch.randint(1, max_seqlens_q + 1, (Z,), dtype=torch.int32)
        seqlens_k = torch.randint(1, max_seqlens_k + 1, (Z,), dtype=torch.int32)
    else:
        seqlens_q = torch.full((Z,), N_CTX_Q // Z, dtype=torch.int32)
        seqlens_k = torch.full((Z,), N_CTX_K // Z, dtype=torch.int32)

    # Calculate cumulative sequence lengths
    cu_seqlens_q = torch.cat([torch.tensor([0], dtype=torch.int32), seqlens_q.cumsum(dim=0)])
    cu_seqlens_k = torch.cat([torch.tensor([0], dtype=torch.int32), seqlens_k.cumsum(dim=0)])
    cu_seqlens_q = cu_seqlens_q.to(device=device).to(torch.int32)
    cu_seqlens_k = cu_seqlens_k.to(device=device).to(torch.int32)

    # Total lengths
    total_q = cu_seqlens_q[-1].item()
    total_k = cu_seqlens_k[-1].item()

    if DEBUG_INPUT:
        # Initialize q, k, v with deterministic values
        q = torch.arange(total_q, dtype=dtype, device=device).view(total_q, 1, 1)
        q = q.expand(total_q, HQ, D_HEAD).contiguous().requires_grad_()
        k = torch.arange(total_k, dtype=dtype, device=device).view(total_k, 1, 1)
        k = k.expand(total_k, HK, D_HEAD).contiguous().requires_grad_()
        v = torch.arange(total_k, dtype=dtype, device=device).view(total_k, 1, 1)
        v = v.expand(total_k, HK, D_HEAD).contiguous().requires_grad_()
        sm_scale = 1
    else:
        # Initialize q, k, v with random values
        q = torch.randn((total_q, HQ, D_HEAD), dtype=dtype, device=device).requires_grad_()
        k = torch.randn((total_k, HK, D_HEAD), dtype=dtype, device=device).requires_grad_()
        v = torch.randn((total_k, HK, D_HEAD), dtype=dtype, device=device).requires_grad_()
        sm_scale = D_HEAD ** -0.5

    input_metadata = MetaData(sm_scale=sm_scale)
    input_metadata.set_varlen_params(cu_seqlens_q, cu_seqlens_k)
    return q, k, v, input_metadata


def get_shape_from_layout(q, k, layout, cu_seqlens_q = None, cu_seqlens_k = None, max_seqlen_q=None, max_seqlen_k=None):
    if layout == 'bhsd':
        batch_q, nheads_q, max_seqlen_q, head_size_q = q.shape
        batch_k, nheads_k, max_seqlen_k, head_size_k = k.shape
    elif layout == 'bshd':
        batch_q, max_seqlen_q, nheads_q, head_size_q = q.shape
        batch_k, max_seqlen_k, nheads_k, head_size_k = k.shape
    elif  layout == 'thd':
        batch_q, max_seqlen_q, nheads_q, head_size_q = len(cu_seqlens_q) - 1, max_seqlen_q, q.shape[1], q.shape[2] # pylint: disable=self-assigning-variable
        batch_k, max_seqlen_k, nheads_k, head_size_k = len(cu_seqlens_k) - 1, max_seqlen_k, k.shape[1], k.shape[2] # pylint: disable=self-assigning-variable
    else:
        assert False, "Got unsupported layout."

    # assert
    assert batch_q == batch_k
    assert head_size_q == head_size_k

    return batch_q, nheads_q, nheads_k, head_size_q, max_seqlen_q, max_seqlen_k


def get_strides_from_layout(q, k, v, o, layout):
    if layout == 'thd':
        q_strides = (0, q.stride(1), q.stride(0), q.stride(2))
        k_strides = (0, k.stride(1), k.stride(0), k.stride(2))
        v_strides = (0, v.stride(1), v.stride(0), v.stride(2))
        o_strides = (0, o.stride(1), o.stride(0), o.stride(2))
    elif layout == 'bhsd':
        q_strides = (q.stride(0), q.stride(1), q.stride(2), q.stride(3))
        k_strides = (k.stride(0), k.stride(1), k.stride(2), k.stride(3))
        v_strides = (v.stride(0), v.stride(1), v.stride(2), v.stride(3))
        o_strides = (o.stride(0), o.stride(1), o.stride(2), o.stride(3))
    elif layout == 'bshd':
        q_strides = (q.stride(0), q.stride(2), q.stride(1), q.stride(3))
        k_strides = (k.stride(0), k.stride(2), k.stride(1), k.stride(3))
        v_strides = (v.stride(0), v.stride(2), v.stride(1), v.stride(3))
        o_strides = (o.stride(0), o.stride(2), o.stride(1), o.stride(3))
    else:
        assert False, 'Got unsupported layout.'
    return q_strides, k_strides, v_strides, o_strides


def get_padded_headsize(size):
    # Get closest power of 2 over or equal to 32.
    padded_d_model = 1 << (size - 1).bit_length()
    # Smallest head_dim supported is 16. If smaller, the tile in the
    # kernel is padded - there is no padding in memory for any dims.
    padded_d_model = max(padded_d_model, 16)
    return padded_d_model


def _strides(x: torch.Tensor, *stride_names: str):
    if x is None:
        return {f"stride_{s}": 0 for i, s in enumerate(stride_names)}

    assert x.ndim == len(stride_names)
    return {f"stride_{s}": x.stride(i) for i, s in enumerate(stride_names)}


def get_input_shapes():
    cases = [(max(1, 2**(16 - i)), 1, 2**i, 16, 1, 128)
             for i in range(8, 18)] + [(max(1, 2**(16 - i)), 1, 2**i, 16, 2, 128) for i in range(8, 18)]
    return cases


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

def is_cdna():
    return is_hip() and triton.runtime.driver.active.get_current_target().arch.startswith('gfx9')

def is_rdna():
    return is_hip() and triton.runtime.driver.active.get_current_target().arch.startswith("gfx1")
