import torch
import triton
import triton.language as tl
from triton import Config
from typing import Tuple


"""
simplified explanation of the scaled int8 matmul algorithm
adopted from deepseek scaled FP8 matmul and jetfire paper
https://arxiv.org/abs/2403.12422
https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/kernel.py

                                                     N dimension →  
                                               INT8 weights                 scaler per block
                                               ┌-----┬-----┬─────┬─────┐    ┌-----┬-----┬─────┬─────┐
                                               : b00 : b01 : b02 | b03 |    :     :     :     |     |
                                               ├-----┼-----┼─────┼─────┤    :b_s00:b_s10:b_s20|b_s30|
                                           K   : b10 : b11 : b12 | b13 |    :     :     :     |     |
                                          dim  ├-----┼-----┼─────┼─────┤    ├-----┼-----┼─────┼─────┤
                                           ↓   | b20 | b21 | b22 | b23 |    |     |     |     |     |
                                               ├─────┼─────┼─────┼─────┤    |b_s01|b_s11|b_s21|b_s31|
                                               | b30 | b31 | b32 | b33 |    |     |     |     |     |
                                               └─────┴─────┴─────┴─────┘    └─────┴─────┴─────┴─────┘
                                               ┌-----┬-----┐
                                               : b00 : b01 :
     ├─── blk ───┤                             ├-----┼-----┤
                                               : b10 : b11 :
            K dimension →                      └-----┴-----┘                                
     INT8 activations
     ┌-----┬-----┬─────┬─────┐   ┌-----┬-----┐ ┌-----┬-----┐   ┌-----------┐   ┌-----┬-----┐   ┌-----┬-----┐
     : a00 : a01 : a02 | a03 |   : a00 : a01 : :  @  :  @  :   :   a_s00   :   :     :     :   :acc00:acc01:
     ├-----┼-----┼─────┼─────┤   ├-----┼-----┤ ├-----┼-----┤ * ├-----------┤ * :b_s00:b_s10: = ├-----┼-----┤ 
 M   : a10 : a11 : a12 | a13 |   : a10 : a11 : :  @  :  @  :   :   a_s10   :   :     :     :   :acc10:acc11:
dim  ├-----┼-----┼─────┼─────┤   └-----┴-----┘ └-----┴-----┘   └-----------┘   └-----┴-----┘   └-----┴-----┘
 ↓   | a20 | a21 | a22 | a23 |   INT8 matmul acc in INT32      rescale the FP32 intermediate   accumulate
     ├─────┼─────┼─────┼─────┤   then cast to FP32             "rank 1" hadamard scaler        intermediate
     | a30 | a31 | a32 | a33 |  
     └─────┴─────┴─────┴─────┘  
     scaler per block
     ┌-----------┬───────────┐
     :   a_s00   :   a_s01   |
     ├-----------┼───────────┤
     :   a_s10   :   a_s11   |
     ├-----------┼───────────┤
     |   a_s20   |   a_s21   |
     ├───────────┼───────────┤
     |   a_s30   |   a_s31   |
     └───────────┴───────────┘
"""


@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    """
    Quantizes the input tensor `x_ptr` and stores the result in `y_ptr` and the scaling factor in `s_ptr`.

    Args:
        x_ptr (triton.Pointer): Pointer to the input tensor.
        y_ptr (triton.Pointer): Pointer to the output tensor where quantized values will be stored.
        s_ptr (triton.Pointer): Pointer to the output tensor where scaling factors will be stored.
        BLOCK_SIZE (tl.constexpr): The size of the block to be processed by each program instance.

    Returns:
        None
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    amax = tl.max(tl.abs(x))  # reduction
    # amax = tl.maximum(amax, 1e-4) # clamp to 1e-4
    s = amax / 127.0
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)


def act_quant(
    x: torch.Tensor, block_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.int8`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert (
        x.size(-1) % block_size == 0
    ), f"Last dimension size must be divisible by block_size (block_size={block_size})"
    y = torch.empty_like(x, dtype=torch.int8)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)
    # Grid size should match number of scale elements (one program per block)
    # Each program processes block_size elements and writes one scale value
    num_programs = s.numel()  # Number of blocks = number of scale elements
    grid = lambda meta: (num_programs,)
    act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size)
    return y, s


@triton.jit
def act_dequant_kernel(x_ptr, s_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
    """
    Dequantizes the input tensor `x_ptr` using scaling factors from `s_ptr`.

    Args:
        x_ptr (triton.Pointer): Pointer to the quantized input tensor.
        s_ptr (triton.Pointer): Pointer to the scaling factors.
        y_ptr (triton.Pointer): Pointer to the output tensor where dequantized values will be stored.
        BLOCK_SIZE (tl.constexpr): The size of the block to be processed by each program instance.

    Returns:
        None
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    s = tl.load(s_ptr + pid)
    y = x * s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)


def act_dequant(
    x: torch.Tensor, s: torch.Tensor, block_size: int = 128, output_dtype: torch.dtype = None
) -> torch.Tensor:
    """
    Dequantizes the activation tensor `x` using the provided scale tensor.

    Args:
        x (torch.Tensor): The quantized activation tensor. Must be contiguous and its last dimension size must be divisible by `block_size`.
        s (torch.Tensor): The scale tensor with shape (*batch_dims, last_dim // block_size).
        block_size (int, optional): The block size to use for dequantization. Defaults to 128.
        output_dtype (torch.dtype, optional): Target dtype for output. Defaults to torch.get_default_dtype().

    Returns:
        torch.Tensor: The dequantized activation tensor of the same shape as `x`.
    """
    assert x.is_contiguous() and s.is_contiguous(), "Input tensors must be contiguous"
    assert (
        x.size(-1) % block_size == 0
    ), f"Last dimension size must be divisible by block_size (block_size={block_size})"
    
    if output_dtype is None:
        output_dtype = torch.get_default_dtype()
    
    y = torch.empty_like(x, dtype=output_dtype)
    # Grid size should match number of scale elements (one program per block)
    num_programs = s.numel()  # Number of blocks = number of scale elements
    grid = lambda meta: (num_programs,)
    act_dequant_kernel[grid](x, s, y, BLOCK_SIZE=block_size)
    return y


@triton.jit
def weight_quant_kernel(x_ptr, y_ptr, s_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    """
    Quantizes weights using block-wise quantization.

    Args:
        x_ptr (tl.pointer): Pointer to the input weights.
        y_ptr (tl.pointer): Pointer to the output buffer for quantized weights.
        s_ptr (tl.pointer): Pointer to the output buffer for scaling factors.
        M (int): Number of rows in the weight matrix.
        N (int): Number of columns in the weight matrix.
        BLOCK_SIZE (tl.constexpr): Size of the block for quantization.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    
    # Compute per-block absolute maximum
    amax = tl.max(tl.abs(x))
    s = amax / 127.0
    #s = tl.maximum(s, 1e-8)  # Prevent division by zero
    
    # Quantize
    y = x / s
    #y = tl.maximum(tl.minimum(y, 127.0), -127.0)  # Clamp
    y = y.to(y_ptr.dtype.element_ty)
    
    tl.store(y_ptr + offs, y, mask=mask)
    tl.store(s_ptr + pid_m * n + pid_n, s)


def weight_quant(
    x: torch.Tensor, block_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the weight tensor using block-wise quantization.

    Args:
        x (torch.Tensor): The weight tensor of shape (M, N).
        block_size (int, optional): The block size to use for quantization. Defaults to 128.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.int8`.
            - A tensor of scaling factors with shape (M//block_size, N//block_size) and dtype `torch.float32`.

    Raises:
        AssertionError: If `x` is not contiguous or if its dimensions are not 2.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert x.dim() == 2, "Input tensor must have 2 dimensions"
    M, N = x.size()
    assert M % block_size == 0 and N % block_size == 0, \
        f"Dimensions must be divisible by block_size={block_size}, got shape {x.shape}"
    
    y = torch.empty_like(x, dtype=torch.int8)
    s = x.new_empty(M // block_size, N // block_size, dtype=torch.float32)
    
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE"]),
        triton.cdiv(N, meta["BLOCK_SIZE"]),
    )
    weight_quant_kernel[grid](x, y, s, M, N, BLOCK_SIZE=block_size)
    return y, s


@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    """
    Dequantizes weights using the provided scaling factors and stores the result.

    Args:
        x_ptr (tl.pointer): Pointer to the quantized weights.
        s_ptr (tl.pointer): Pointer to the scaling factors.
        y_ptr (tl.pointer): Pointer to the output buffer for dequantized weights.
        M (int): Number of rows in the weight matrix.
        N (int): Number of columns in the weight matrix.
        BLOCK_SIZE (tl.constexpr): Size of the block for tiling.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


def weight_dequant(
    x: torch.Tensor, s: torch.Tensor, block_size: int = 128, output_dtype: torch.dtype = None
) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor.

    Args:
        x (torch.Tensor): The quantized weight tensor of shape (M, N).
        s (torch.Tensor): The scale tensor of shape (M//block_size, N//block_size).
        block_size (int, optional): The block size to use for dequantization. Defaults to 128.
        output_dtype (torch.dtype, optional): Target dtype for output. Defaults to torch.get_default_dtype().

    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `x`.

    Raises:
        AssertionError: If `x` or `s` are not contiguous or if their dimensions are not 2.
    """
    assert x.is_contiguous() and s.is_contiguous(), "Input tensors must be contiguous"
    assert x.dim() == 2 and s.dim() == 2, "Input tensors must have 2 dimensions"
    M, N = x.size()
    
    if output_dtype is None:
        output_dtype = torch.get_default_dtype()
    
    y = torch.empty_like(x, dtype=output_dtype)
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE"]),
        triton.cdiv(N, meta["BLOCK_SIZE"]),
    )
    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


# matmul intermediate block size is hardcoded to 128
int8_gemm_configs = [
    Config(
        {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n, "BLOCK_SIZE_K": 128},
        num_stages=num_stages,
        num_warps=8,
    )
    for block_m in [128, 256]  # >= 128 for consistency with out_block_size
    for block_n in [128, 256]  # >= 128 required for out_block_size compatibility
    for num_stages in [3, 4, 5]
]


#@triton.autotune(configs=int8_gemm_configs, key=["N", "K"])
@triton.jit
def int8_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_s_ptr,
    b_s_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Performs a matrix multiplication operation on INT8 matrices with scaling factors.

    Args:
        a_ptr (tl.tensor): Pointer to the first input matrix A.
        b_ptr (tl.tensor): Pointer to the second input matrix B.
        c_ptr (tl.tensor): Pointer to the output matrix C.
        a_s_ptr (tl.tensor): Pointer to the scaling factors for matrix A.
        b_s_ptr (tl.tensor): Pointer to the scaling factors for matrix B.
        M (int): Number of rows in matrix A and C.
        N (tl.constexpr): Number of columns in matrix B and C.
        K (tl.constexpr): Number of columns in matrix A and rows in matrix B.
        BLOCK_SIZE_M (tl.constexpr): Block size for the M dimension.
        BLOCK_SIZE_N (tl.constexpr): Block size for the N dimension.
        BLOCK_SIZE_K (tl.constexpr): Block size for the K dimension.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    a_s_ptrs = a_s_ptr + offs_m * k
    
    # FIXED: Weight scale indexing for 2D scale array (N_blocks, K_blocks)
    # b_s has shape (N//BLOCK_SIZE_K, K//BLOCK_SIZE_K) stored in row-major
    # For N tile pid_n, we need scales[pid_n, :] across K iterations
    # Address calculation: scale[pid_n, i] = base + pid_n * stride + i
    k_blocks = k  # Number of K blocks for clarity
    b_s_base = b_s_ptr + pid_n * k_blocks

    # Create accumulators outside the loop for better performance
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k_blocks):
        # Load int8 data - use other=0 (int) not 0.0 (float) to preserve int8 type
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0)
        a_s = tl.load(a_s_ptrs)
        # FIXED: Load single scalar weight scale for (pid_n, i) block pair
        b_s = tl.load(b_s_base + i)
        # INT8 matmul → INT32 acc, then cast to FP32 and apply per-block scaling
        dot_prod = tl.dot(a, b, out_dtype=tl.int32)  # int8 × int8 → int32
        accumulator += dot_prod.to(tl.float32) * a_s[:, None] * b_s
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
        a_s_ptrs += 1
    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


#@triton.autotune(configs=int8_gemm_configs, key=["N", "K"])
@triton.jit
def int8_gemm_addmm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    a_s_ptr,
    b_s_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """
    Fused INT8 matrix multiplication with bias addition (addmm).
    Computes: C = A @ B + bias
    
    This kernel fuses the bias addition into the matmul, avoiding an extra memory write/read cycle.

    Args:
        a_ptr (tl.tensor): Pointer to the first input matrix A (INT8).
        b_ptr (tl.tensor): Pointer to the second input matrix B (INT8).
        c_ptr (tl.tensor): Pointer to the output matrix C.
        bias_ptr (tl.tensor): Pointer to the bias vector (1D, length N).
        a_s_ptr (tl.tensor): Pointer to the scaling factors for matrix A.
        b_s_ptr (tl.tensor): Pointer to the scaling factors for matrix B.
        M (int): Number of rows in matrix A and C.
        N (tl.constexpr): Number of columns in matrix B and C.
        K (tl.constexpr): Number of columns in matrix A and rows in matrix B.
        BLOCK_SIZE_M (tl.constexpr): Block size for the M dimension.
        BLOCK_SIZE_N (tl.constexpr): Block size for the N dimension.
        BLOCK_SIZE_K (tl.constexpr): Block size for the K dimension.
        HAS_BIAS (tl.constexpr): Whether bias is provided.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    a_s_ptrs = a_s_ptr + offs_m * k
    
    # FIXED: Weight scale indexing for 2D scale array (N_blocks, K_blocks)
    # b_s has shape (N//BLOCK_SIZE_K, K//BLOCK_SIZE_K) stored in row-major
    # For N tile pid_n, we need scales[pid_n, :] across K iterations
    # Address calculation: scale[pid_n, i] = base + pid_n * stride + i
    k_blocks = k  # Number of K blocks for clarity
    b_s_base = b_s_ptr + pid_n * k_blocks

    # Accumulate matmul result
    # Create accumulators outside the loop for better performance
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k_blocks):
        # Load int8 data - use other=0 (int) not 0.0 (float) to preserve int8 type
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0)
        a_s = tl.load(a_s_ptrs)
        # FIXED: Load single scalar weight scale for (pid_n, i) block pair
        b_s = tl.load(b_s_base + i)
        # INT8 matmul → INT32 acc, then cast to FP32 and apply per-block scaling
        dot_prod = tl.dot(a, b, out_dtype=tl.int32)  # int8 × int8 → int32
        accumulator += dot_prod.to(tl.float32) * a_s[:, None] * b_s
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
        a_s_ptrs += 1
    
    # Add bias if provided (fused operation)
    if HAS_BIAS:
        bias_ptrs = bias_ptr + offs_n[None, :]
        bias = tl.load(bias_ptrs, mask=offs_n[None, :] < N, other=0.0)
        accumulator += bias  # Broadcast bias across M dimension
    
    # Store result
    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def int8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor):
    """
    Perform a matrix multiplication using INT8 precision.
    
    Expected tensor shapes:
    - a: [..., K] where ... can be any batch dimensions
    - b: [N, K] (weight matrix in standard format, kernel transposes internally)
    - a_s: [..., K//block_size]
    - b_s: [N//block_size, K//block_size]

    Args:
        a (torch.Tensor): The first input matrix, must be contiguous.
        a_s (torch.Tensor): The scaling factor for the first input matrix, must be contiguous.
        b (torch.Tensor): The second input matrix [N, K], must be contiguous.
        b_s (torch.Tensor): The scaling factor for the second input matrix, must be contiguous.

    Returns:
        torch.Tensor: The result of the matrix multiplication.
    """
    assert a.is_contiguous() and b.is_contiguous(), "Input tensors must be contiguous"
    assert (
        a_s.is_contiguous() and b_s.is_contiguous()
    ), "Scaling factor tensors must be contiguous"
    assert b.dim() == 2, f"Expected b to be 2D, got shape {b.shape}"
    
    K = a.size(-1)
    M = a.numel() // K
    # b has shape [N, K], extract N from first dimension
    N = b.shape[0]
    
    # Validate shapes
    assert b.size(1) == K, f"Shape mismatch: b.shape={b.shape}, expected [..., {K}]"
    
    # Output tensor (same batch shape as input, last dim = N)
    # let's use float16 as output dtype
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.float16)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    int8_gemm_kernel[grid](a, b, c, a_s, b_s, M, N, K, BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128)
    return c


def int8_addmm(
    a: torch.Tensor, 
    a_s: torch.Tensor, 
    b: torch.Tensor, 
    b_s: torch.Tensor,
    bias: torch.Tensor = None
):
    """
    Fused INT8 matrix multiplication with bias addition (addmm).
    Computes: output = (a @ b) + bias
    
    Expected tensor shapes:
    - a: [..., K] where ... can be any batch dimensions
    - b: [N, K] (weight matrix in standard format, kernel transposes internally)
    - a_s: [..., K//block_size]
    - b_s: [N//block_size, K//block_size]
    - bias: [N] (optional)
    
    This is more efficient than separate matmul + bias add operations as it:
    1. Avoids an extra memory write/read cycle
    2. Fuses the bias addition into the matmul kernel
    3. Better utilizes GPU memory bandwidth

    Args:
        a (torch.Tensor): The first input matrix (INT8), must be contiguous.
        a_s (torch.Tensor): The scaling factors for the first input matrix, must be contiguous.
        b (torch.Tensor): The second input matrix (INT8) [N, K], must be contiguous.
        b_s (torch.Tensor): The scaling factors for the second input matrix, must be contiguous.
        bias (torch.Tensor, optional): The bias vector (1D, length N). If None, only matmul is performed.

    Returns:
        torch.Tensor: The result of the fused matrix multiplication and bias addition.
    
    Example:
        >>> a_int8, a_scale = act_quant(input_tensor, block_size=128)
        >>> b_int8, b_scale = weight_quant(weight_tensor, block_size=128)
        >>> bias = torch.randn(output_features)
        >>> output = int8_addmm(a_int8, a_scale, b_int8, b_scale, bias)
    """
    assert a.is_contiguous() and b.is_contiguous(), "Input tensors must be contiguous"
    assert (
        a_s.is_contiguous() and b_s.is_contiguous()
    ), "Scaling factor tensors must be contiguous"
    assert b.dim() == 2, f"Expected b to be 2D, got shape {b.shape}"
    
    K = a.size(-1)
    M = a.numel() // K
    # b has shape [N, K], extract N from first dimension
    N = b.shape[0]
    
    # Validate shapes
    assert b.size(1) == K, f"Shape mismatch: b.shape={b.shape}, expected [..., {K}]"
    
    # Output tensor (same batch shape as input, last dim = N)
    # let's use float16 as output dtype
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.float16)
    
    # Handle bias
    has_bias = bias is not None
    if has_bias:
        assert bias.is_contiguous(), "Bias tensor must be contiguous"
        assert bias.dim() == 1 and bias.size(0) == N, \
            f"Bias must be 1D with length {N}, got shape {bias.shape}"
        bias_ptr = bias
    else:
        # Create a dummy pointer (won't be used due to HAS_BIAS=False)
        bias_ptr = c
    
    # Launch kernel
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    int8_gemm_addmm_kernel[grid](
        a, b, c, bias_ptr, a_s, b_s, M, N, K, HAS_BIAS=has_bias, BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128
    )
    return c


# ==============================================================================
# Fused INT8 GEMM + Quantization Kernels
# ==============================================================================
# 
# Architecture Overview:
# ----------------------
# 1. Kernels compute matmul and quantize PER-ROW for activation format
#    - Each row gets its own scale for the N-range of the tile
#    - Kernel output: c_scale shape is (M, N/BLOCK_SIZE_N)
#    - BLOCK_SIZE_M, BLOCK_SIZE_N are tile sizes from autotuner (e.g., 16-64, 32-128)
#    - This matches activation quantization: per-row, block-wise along N
# 
# 2. Wrapper functions convert to final activation format
#    - Kernel output: (M, N/BLOCK_SIZE_N)
#    - Target format: (*batch_dims, N/out_block_size)
#    - If BLOCK_SIZE_N == out_block_size: already correct, just reshape
#    - If BLOCK_SIZE_N != out_block_size: replicate or merge scales
# 
# 3. Benefits:
#    - Accurate: per-row scales match activation quantization format
#    - Efficient: single max reduction per row per tile
#    - Compatible: direct output in activation format
#    - Better precision: each row has independent scales
# 
# ==============================================================================

#@triton.autotune(configs=int8_gemm_configs, key=["N", "K"])
@triton.heuristics({
    'NUM_BLOCKS': lambda args: args["BLOCK_SIZE_N"] // args["out_block_size"],
})
@triton.jit
def int8_gemm_quant_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    c_s_ptr,
    a_s_ptr,
    b_s_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    out_block_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    """
    Fused INT8 matrix multiplication with output quantization.
    Computes: C_int8, C_scale = quantize(A @ B)
    
    This kernel fuses matmul and block-wise quantization in a single pass.
    Quantizes at out_block_size granularity (like act_quant_kernel).

    Args:
        a_ptr: Pointer to INT8 activations
        b_ptr: Pointer to INT8 weights
        c_ptr: Pointer to INT8 output
        c_s_ptr: Pointer to output scales (shape: M x N/out_block_size)
        a_s_ptr: Pointer to activation scales
        b_s_ptr: Pointer to weight scales
        M: Number of rows in A and C
        N: Number of columns in B and C
        K: Inner dimension (columns in A, rows in B)
        out_block_size: Block size for output quantization
        BLOCK_SIZE_M/N/K: Tile sizes for matmul
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    a_s_ptrs = a_s_ptr + offs_m * k
    
    # FIXED: Weight scale indexing for 2D scale array (N_blocks, K_blocks)
    # b_s has shape (N//BLOCK_SIZE_K, K//BLOCK_SIZE_K) stored in row-major
    # For N tile pid_n, we need scales[pid_n, :] across K iterations
    k_blocks = k  # Number of K blocks for clarity
    b_s_base = b_s_ptr + pid_n * k_blocks

    # Accumulate matmul result
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k_blocks):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0)
        a_s = tl.load(a_s_ptrs)
        # FIXED: Load single scalar weight scale for (pid_n, i) block pair
        b_s = tl.load(b_s_base + i)
        dot_prod = tl.dot(a, b, out_dtype=tl.int32)
        accumulator += dot_prod.to(tl.float32) * a_s[:, None] * b_s
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
        a_s_ptrs += 1
    
    # Quantize in activation format: per-row, block-wise at out_block_size granularity
    # Reshape accumulator to separate blocks: (BLOCK_SIZE_M, BLOCK_SIZE_N) -> (BLOCK_SIZE_M, NUM_BLOCKS, out_block_size)
    accumulator_reshaped = tl.reshape(accumulator, (BLOCK_SIZE_M, NUM_BLOCKS, out_block_size))
    
    # Compute max per block: reduce over out_block_size dimension
    # Shape: (BLOCK_SIZE_M, NUM_BLOCKS)
    block_max = tl.max(tl.abs(accumulator_reshaped), axis=2)
    block_scale = tl.maximum(block_max / 127.0, 1e-8)
    
    # Reshape scales for broadcasting: (BLOCK_SIZE_M, NUM_BLOCKS) -> (BLOCK_SIZE_M, NUM_BLOCKS, 1)
    block_scale_broadcast = tl.reshape(block_scale, (BLOCK_SIZE_M, NUM_BLOCKS, 1))
    
    # Quantize: accumulator -> int8
    quantized = accumulator_reshaped / block_scale_broadcast
    quantized = tl.maximum(tl.minimum(quantized, 127.0), -127.0)
    quantized_int8 = quantized.to(c_ptr.dtype.element_ty)
    
    # Reshape back to 2D: (BLOCK_SIZE_M, NUM_BLOCKS, out_block_size) -> (BLOCK_SIZE_M, BLOCK_SIZE_N)
    quantized_int8 = tl.reshape(quantized_int8, (BLOCK_SIZE_M, BLOCK_SIZE_N))
    
    # Store quantized output
    offs_m_actual = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n_actual = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_m_actual[:, None] < M) & (offs_n_actual[None, :] < N)
    c_ptrs = c_ptr + offs_m_actual[:, None] * N + offs_n_actual[None, :]
    tl.store(c_ptrs, quantized_int8, mask=mask)
    
    # Store scales: (BLOCK_SIZE_M, NUM_BLOCKS) scales for this tile
    # Scale layout: (M, N//out_block_size) - matches activation format directly!
    # This tile covers M range [pid_m*BLOCK_SIZE_M : (pid_m+1)*BLOCK_SIZE_M]
    #               N range [pid_n*BLOCK_SIZE_N : (pid_n+1)*BLOCK_SIZE_N]
    # N block indices: [pid_n * NUM_BLOCKS : (pid_n+1) * NUM_BLOCKS]
    n_scale_stride = N // out_block_size  # Total number of N blocks
    offs_m_scale = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n_scale = pid_n * NUM_BLOCKS + tl.arange(0, NUM_BLOCKS)
    scale_ptrs = c_s_ptr + offs_m_scale[:, None] * n_scale_stride + offs_n_scale[None, :]
    scale_mask = (offs_m_scale[:, None] < M) & (offs_n_scale[None, :] < n_scale_stride)
    tl.store(scale_ptrs, block_scale, mask=scale_mask)


#@triton.autotune(configs=int8_gemm_configs, key=["N", "K"])
@triton.heuristics({
    'NUM_BLOCKS': lambda args: args["BLOCK_SIZE_N"] // args["out_block_size"],
})
@triton.jit
def int8_gemm_addmm_quant_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    c_s_ptr,
    bias_ptr,
    a_s_ptr,
    b_s_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    out_block_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """
    Fused INT8 matrix multiplication with bias addition and output quantization.
    Computes: C_int8, C_scale = quantize(A @ B + bias)
    
    This kernel fuses matmul, bias addition, and block-wise quantization.
    Quantizes at out_block_size granularity (like act_quant_kernel).

    Args:
        a_ptr: Pointer to INT8 activations
        b_ptr: Pointer to INT8 weights
        c_ptr: Pointer to INT8 output
        c_s_ptr: Pointer to output scales (shape: M x N/out_block_size)
        bias_ptr: Pointer to bias vector
        a_s_ptr: Pointer to activation scales
        b_s_ptr: Pointer to weight scales
        M: Number of rows in A and C
        N: Number of columns in B and C
        K: Inner dimension
        out_block_size: Block size for output quantization
        BLOCK_SIZE_M/N/K: Tile sizes for matmul
        HAS_BIAS: Whether bias is provided
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    a_s_ptrs = a_s_ptr + offs_m * k
    
    # FIXED: Weight scale indexing for 2D scale array (N_blocks, K_blocks)
    # b_s has shape (N//BLOCK_SIZE_K, K//BLOCK_SIZE_K) stored in row-major
    # For N tile pid_n, we need scales[pid_n, :] across K iterations
    k_blocks = k  # Number of K blocks for clarity
    b_s_base = b_s_ptr + pid_n * k_blocks

    # Accumulate matmul result
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k_blocks):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0)
        a_s = tl.load(a_s_ptrs)
        # FIXED: Load single scalar weight scale for (pid_n, i) block pair
        b_s = tl.load(b_s_base + i)
        dot_prod = tl.dot(a, b, out_dtype=tl.int32)
        accumulator += dot_prod.to(tl.float32) * a_s[:, None] * b_s
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
        a_s_ptrs += 1
    
    # Add bias if provided
    if HAS_BIAS:
        bias_ptrs = bias_ptr + offs_n[None, :]
        bias = tl.load(bias_ptrs, mask=offs_n[None, :] < N, other=0.0)
        accumulator += bias
    
    # Quantize in activation format: per-row, block-wise at out_block_size granularity
    # Reshape accumulator to separate blocks: (BLOCK_SIZE_M, BLOCK_SIZE_N) -> (BLOCK_SIZE_M, NUM_BLOCKS, out_block_size)
    accumulator_reshaped = tl.reshape(accumulator, (BLOCK_SIZE_M, NUM_BLOCKS, out_block_size))
    
    # Compute max per block: reduce over out_block_size dimension
    # Shape: (BLOCK_SIZE_M, NUM_BLOCKS)
    block_max = tl.max(tl.abs(accumulator_reshaped), axis=2)
    block_scale = tl.maximum(block_max / 127.0, 1e-8)
    
    # Reshape scales for broadcasting: (BLOCK_SIZE_M, NUM_BLOCKS) -> (BLOCK_SIZE_M, NUM_BLOCKS, 1)
    block_scale_broadcast = tl.reshape(block_scale, (BLOCK_SIZE_M, NUM_BLOCKS, 1))
    
    # Quantize: accumulator -> int8
    quantized = accumulator_reshaped / block_scale_broadcast
    quantized = tl.maximum(tl.minimum(quantized, 127.0), -127.0)
    quantized_int8 = quantized.to(c_ptr.dtype.element_ty)
    
    # Reshape back to 2D: (BLOCK_SIZE_M, NUM_BLOCKS, out_block_size) -> (BLOCK_SIZE_M, BLOCK_SIZE_N)
    quantized_int8 = tl.reshape(quantized_int8, (BLOCK_SIZE_M, BLOCK_SIZE_N))
    
    # Store quantized output
    offs_m_actual = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n_actual = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_m_actual[:, None] < M) & (offs_n_actual[None, :] < N)
    c_ptrs = c_ptr + offs_m_actual[:, None] * N + offs_n_actual[None, :]
    tl.store(c_ptrs, quantized_int8, mask=mask)
    
    # Store scales: (BLOCK_SIZE_M, NUM_BLOCKS) scales for this tile
    # Scale layout: (M, N//out_block_size) - matches activation format directly!
    # This tile covers M range [pid_m*BLOCK_SIZE_M : (pid_m+1)*BLOCK_SIZE_M]
    #               N range [pid_n*BLOCK_SIZE_N : (pid_n+1)*BLOCK_SIZE_N]
    # N block indices: [pid_n * NUM_BLOCKS : (pid_n+1) * NUM_BLOCKS]
    n_scale_stride = N // out_block_size  # Total number of N blocks
    offs_m_scale = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n_scale = pid_n * NUM_BLOCKS + tl.arange(0, NUM_BLOCKS)
    scale_ptrs = c_s_ptr + offs_m_scale[:, None] * n_scale_stride + offs_n_scale[None, :]
    scale_mask = (offs_m_scale[:, None] < M) & (offs_n_scale[None, :] < n_scale_stride)
    tl.store(scale_ptrs, block_scale, mask=scale_mask)


def int8_gemm_quant(
    a: torch.Tensor, 
    a_s: torch.Tensor, 
    b: torch.Tensor, 
    b_s: torch.Tensor,
    out_block_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused INT8 GEMM with output quantization.
    Computes: C_int8, C_scale = quantize(A @ B)
    
    This avoids materializing the full-precision intermediate result.
    
    The kernel produces scales in activation format directly: (*batch_dims, N/out_block_size).

    Args:
        a: INT8 activations [..., K]
        a_s: Activation scales [..., K//block_size]
        b: INT8 weights [N, K]
        b_s: Weight scales [N//block_size, K//block_size]
        out_block_size: Block size for output quantization (default: 128)

    Returns:
        Tuple of (quantized output INT8, output scales in activation format)
    """
    assert a.is_contiguous() and b.is_contiguous(), "Input tensors must be contiguous"
    assert a_s.is_contiguous() and b_s.is_contiguous(), "Scaling tensors must be contiguous"
    assert b.dim() == 2, f"Expected b to be 2D, got shape {b.shape}"
    
    K = a.size(-1)
    M = a.numel() // K
    N = b.shape[0]
    batch_shape = a.size()[:-1]
    
    assert b.size(1) == K, f"Shape mismatch: b.shape={b.shape}, expected [..., {K}]"
    assert N % out_block_size == 0, f"N={N} must be divisible by out_block_size={out_block_size}"
    
    # Allocate output tensors
    c = a.new_empty(*batch_shape, N, dtype=torch.int8)
    
    # Allocate scales in activation format directly: (M, N//out_block_size)
    n_blocks = N // out_block_size
    c_s = a.new_empty(M, n_blocks, dtype=torch.float32)
    
    # Launch kernel
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    
    int8_gemm_quant_kernel[grid](
        a, b, c, c_s, a_s, b_s, M, N, K, out_block_size, BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128
    )
    
    # Reshape scales to match batch dimensions: (M, n_blocks) -> (*batch_dims, n_blocks)
    if len(batch_shape) > 0:
        c_s = c_s.reshape(*batch_shape, n_blocks)
    
    return c, c_s


def int8_addmm_quant(
    a: torch.Tensor, 
    a_s: torch.Tensor, 
    b: torch.Tensor, 
    b_s: torch.Tensor,
    bias: torch.Tensor = None,
    out_block_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused INT8 addmm with output quantization.
    Computes: C_int8, C_scale = quantize(A @ B + bias)
    
    This fuses matmul, bias addition, and quantization in a single kernel pass.
    
    The kernel produces scales in activation format directly: (*batch_dims, N/out_block_size).

    Args:
        a: INT8 activations [..., K]
        a_s: Activation scales [..., K//block_size]
        b: INT8 weights [N, K]
        b_s: Weight scales [N//block_size, K//block_size]
        bias: Optional bias vector [N]
        out_block_size: Block size for output quantization (default: 128)

    Returns:
        Tuple of (quantized output INT8, output scales in activation format)
    """
    assert a.is_contiguous() and b.is_contiguous(), "Input tensors must be contiguous"
    assert a_s.is_contiguous() and b_s.is_contiguous(), "Scaling tensors must be contiguous"
    assert b.dim() == 2, f"Expected b to be 2D, got shape {b.shape}"
    
    K = a.size(-1)
    M = a.numel() // K
    N = b.shape[0]
    batch_shape = a.size()[:-1]
    
    assert b.size(1) == K, f"Shape mismatch: b.shape={b.shape}, expected [..., {K}]"
    assert N % out_block_size == 0, f"N={N} must be divisible by out_block_size={out_block_size}"
    
    # Allocate output tensors
    c = a.new_empty(*batch_shape, N, dtype=torch.int8)
    
    # Allocate scales in activation format directly: (M, N//out_block_size)
    n_blocks = N // out_block_size
    c_s = a.new_empty(M, n_blocks, dtype=torch.float32)
    
    # Handle bias
    has_bias = bias is not None
    if has_bias:
        assert bias.is_contiguous(), "Bias tensor must be contiguous"
        assert bias.dim() == 1 and bias.size(0) == N, \
            f"Bias must be 1D with length {N}, got shape {bias.shape}"
        bias_ptr = bias
    else:
        bias_ptr = c  # Dummy pointer
    
    # Launch kernel
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    
    int8_gemm_addmm_quant_kernel[grid](
        a, b, c, c_s, bias_ptr, a_s, b_s, M, N, K, out_block_size, HAS_BIAS=has_bias, BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128
    )
    
    # Reshape scales to match batch dimensions: (M, n_blocks) -> (*batch_dims, n_blocks)
    if len(batch_shape) > 0:
        c_s = c_s.reshape(*batch_shape, n_blocks)
    
    return c, c_s


# ==============================================================================
# INT8 GELU Kernel
# ==============================================================================

# Autotuning configs for GELU kernel
# Note: BLOCK_N must be >= quantization block_size (typically 128) and divisible by it
# BLOCK_M can be any size since we don't block in M dimension for activations
int8_gelu_configs = [
    Config(
        {"BLOCK_M": block_m, "BLOCK_N": block_n},
        num_stages=num_stages,
        num_warps=num_warps,
    )
    for block_m in [64, 128, 256]
    for block_n in [128, 256]  # Must be >= block_size and divisible by it
    for num_stages in [2, 3, 4]
    for num_warps in [4, 8]
]


#@triton.autotune(configs=int8_gelu_configs, key=["M", "N"])
@triton.heuristics({
    'BLOCK_SM': lambda args: args["BLOCK_M"],  # For activations, no blocking in M dimension
    'BLOCK_SN': lambda args: args["BLOCK_N"] // args["BLOCK_SIZE"],
})
@triton.jit
def int8_gelu_kernel(
    output_ptr,
    output_scale_ptr,
    input_ptr,
    input_scale_ptr,
    M,
    N: tl.constexpr,
    SM,
    SN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_SM: tl.constexpr,
    BLOCK_SN: tl.constexpr,
):
    """
    Fused INT8 GELU with block-wise quantization.
    
    Computes: output_int8, output_scale = quantize(gelu(dequantize(input_int8, input_scale)))
    
    For activation quantization, we only block along the last dimension (N).
    Each row gets its own set of scales along N.
    
    Scale tensor layout:
    - Input scales: (M, N // BLOCK_SIZE) - one scale per row per block in N
    - Within each tile (BLOCK_M x BLOCK_N), we load (BLOCK_M, BLOCK_N // BLOCK_SIZE) scales
    
    This kernel:
    1. Loads INT8 input and its block-wise scales
    2. Dequantizes to float
    3. Applies GELU activation
    4. Quantizes output back to INT8 with new block-wise scales
    
    Args:
        output_ptr: Pointer to INT8 output tensor
        output_scale_ptr: Pointer to output scales
        input_ptr: Pointer to INT8 input tensor
        input_scale_ptr: Pointer to input scales
        M: Number of rows
        N: Number of columns
        SM: Number of rows in scale tensor (= M for activations)
        SN: Number of scale blocks in N dimension (= N // BLOCK_SIZE)
        BLOCK_SIZE: Quantization block size (e.g., 128)
        BLOCK_M: Tile size in M dimension
        BLOCK_N: Tile size in N dimension
        BLOCK_SM: Number of rows per tile (= BLOCK_M for activations)
        BLOCK_SN: Number of scale blocks per tile in N dimension (= BLOCK_N // BLOCK_SIZE)
    """
    # Block PID
    pid = tl.program_id(0)
    NUM_BLOCK_N = tl.cdiv(N, BLOCK_N)
    pid_m = pid // NUM_BLOCK_N
    pid_n = pid % NUM_BLOCK_N
    
    # Offsets for data
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Load input data
    input_ptrs = input_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    input_data = tl.load(input_ptrs, mask=mask, other=0).to(tl.int32)
    
    # Load input scales
    # Scale dimensions: (SM, SN) where SM = M, SN = N // BLOCK_SIZE
    # For this tile: load (BLOCK_M, BLOCK_N // BLOCK_SIZE) scales
    offs_sm = pid_m * BLOCK_SM + tl.arange(0, BLOCK_SM)
    offs_sn = pid_n * BLOCK_SN + tl.arange(0, BLOCK_SN)
    scale_ptrs = input_scale_ptr + offs_sm[:, None] * SN + offs_sn[None, :]
    scale_mask = (offs_sm[:, None] < SM) & (offs_sn[None, :] < SN)
    input_scales = tl.load(scale_ptrs, mask=scale_mask, other=1.0)
    
    # Reshape for broadcasting
    # Data: (BLOCK_M, BLOCK_N) -> (BLOCK_M, BLOCK_SN, BLOCK_SIZE)
    # Scales: (BLOCK_M, BLOCK_SN) -> (BLOCK_M, BLOCK_SN, 1)
    input_data = tl.reshape(input_data, (BLOCK_M, BLOCK_SN, BLOCK_SIZE))
    input_scales = tl.reshape(input_scales, (BLOCK_M, BLOCK_SN, 1))
    
    # Dequantize
    input_fp32 = input_data.to(tl.float32) * input_scales
    
    # Apply GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    sqrt_2 = 1.41421356237
    erf_input = input_fp32 / sqrt_2
    erf_val = tl.math.erf(erf_input)
    gelu_output = input_fp32 * 0.5 * (1.0 + erf_val)
    
    # Compute output scales per block
    # Shape: (BLOCK_M, BLOCK_SN, BLOCK_SIZE) -> (BLOCK_M, BLOCK_SN)
    abs_output = tl.abs(gelu_output)
    max_val = tl.max(abs_output, axis=2)  # Reduce over BLOCK_SIZE dimension
    output_scales = tl.maximum(max_val / 127.0, 1e-8)
    
    # Reshape scales for broadcasting: (BLOCK_M, BLOCK_SN) -> (BLOCK_M, BLOCK_SN, 1)
    output_scales_broadcast = tl.reshape(output_scales, (BLOCK_M, BLOCK_SN, 1))
    
    # Quantize output
    quantized = gelu_output / output_scales_broadcast
    quantized = tl.maximum(tl.minimum(quantized, 127.0), -127.0)
    quantized_int8 = quantized.to(tl.int8)
    
    # Reshape back to 2D
    quantized_int8 = tl.reshape(quantized_int8, (BLOCK_M, BLOCK_N))
    
    # Store quantized output
    output_ptrs = output_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(output_ptrs, quantized_int8, mask=mask)
    
    # Store output scales
    output_scale_ptrs = output_scale_ptr + offs_sm[:, None] * SN + offs_sn[None, :]
    tl.store(output_scale_ptrs, output_scales, mask=scale_mask)


def int8_gelu(
    x: torch.Tensor,
    s_x: torch.Tensor,
    block_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused INT8 GELU activation with block-wise quantization.
    
    Computes: y_int8, y_scale = quantize(gelu(dequantize(x, s_x)))
    
    This avoids materializing the full-precision intermediate result.
    
    Args:
        x: INT8 input tensor of any shape
        s_x: Input scales with shape (*batch_dims, last_dim // block_size)
        block_size: Quantization block size (default: 128)
    
    Returns:
        Tuple of (quantized output INT8, output scales)
    
    Note:
        The kernel requires tile sizes >= block_size. This is automatically
        handled by the autotuner, which uses BLOCK_M, BLOCK_N >= 128.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert s_x.is_contiguous(), "Scale tensor must be contiguous"
    assert x.size(-1) % block_size == 0, \
        f"Last dimension must be divisible by block_size={block_size}"
    assert block_size == 128, \
        f"Only block_size=128 is currently supported in autotuner configs (got {block_size})"
    
    # Handle multi-dimensional tensors by reshaping to 2D
    original_shape = x.shape
    batch_shape = original_shape[:-1]
    N = original_shape[-1]
    
    if x.dim() > 2:
        x = x.reshape(-1, N)
        s_x = s_x.reshape(-1, s_x.size(-1))
    
    M = x.size(0)
    SM = M  # For activations, we don't block in M dimension
    SN = N // block_size
    
    # Allocate output tensors
    y = torch.empty_like(x, dtype=torch.int8)
    s_y = torch.empty_like(s_x, dtype=torch.float32)
    
    # Launch kernel
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    
    int8_gelu_kernel[grid](
        y, s_y, x, s_x,
        M, N, SM, SN,
        BLOCK_SIZE=block_size, BLOCK_M=128, BLOCK_N=128, BLOCK_SM=128
    )
    
    # Reshape back to original batch dimensions
    if len(batch_shape) > 0:
        y = y.reshape(*batch_shape, N)
        s_y = s_y.reshape(*batch_shape, SN)
    
    return y, s_y
