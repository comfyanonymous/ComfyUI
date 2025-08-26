import torch

"""
x: 512x1024 w:1024x1024
- For TensorWise scaling, a and b should be float8, scales should be float and singletons.
- For RowWise scaling, a and b should be float8, scales should be float, scale_a should be (512, 1) and scale_b should be (1, 1024), and both should be contiguous.
- For BlockWise 1x128 scaling, a and b should be float8, scales should be float, scale_a should be (512, 8) and scale_b should be (8, 1024), and both should be outer-dim-major.
- For BlockWise 128x128 scaling, a and b should be float8, scales should be float, scale_a should be (4, 8) and scale_b should be (8, 8), and both should be near-inner-dim-major (with 16-byte aligned strides).
- For Blockwise 1x32 scaling, a and b should be float8, scales should be float8_e8m0fnu, scale_a should have 16384 elements and scale_b should have 32768 elements, and both should be contiguous.
- For Blockwise 1x16 scaling, a and b should be float4 (packed 2x), scales should be float8_e4m3fn, scale_a should have 65536 elements and scale_b should have 131072 elements, and both should be contiguous.
"""
Q_TYPES = [torch.float8_e4m3fn, torch.float4_e2m1fn_x2]

def dynamic_tensor_quantizer(x: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype):
    input_scale = torch.abs(x).max() / torch.finfo(dtype).max
    x = (x / input_scale).clamp(torch.finfo(dtype).min, torch.finfo(dtype).max).to(dtype=dtype)
    return x, input_scale.float()

def mxfp8_quantizer(x: torch.Tensor, dtype: torch.dtype):
    block_size = 32
    orig_shape = x.shape
    x = x.reshape(-1, block_size)
    scale = (torch.amax(torch.abs(x), dim=-1) / torch.finfo(dtype).max)
    x = (x / scale.unsqueeze(-1)).clamp(torch.finfo(dtype).min, torch.finfo(dtype).max).to(dtype=dtype).contiguous()
    x = x.view(orig_shape)

    return x, scale.to(dtype=torch.float8_e8m0fnu).contiguous()

def tensor_quantizer(x: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype):
    x = (x / scale).clamp(torch.finfo(dtype).min, torch.finfo(dtype).max).to(dtype=dtype).contiguous()
    return x, scale.float()

def nvfp4_quantizer(x: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype):
    """
        orig_shape = x.shape
        x = x.reshape(orig_shape[0], -1, block_size)
        max_abs = torch.amax(torch.abs(x), dim=-1)
        block_scale = (max_abs / torch.finfo(torch.float4_e2m1fn_x2.max))-float()
        scaled_block_scales = block_scale / scale
        scaled_block_scales_fp8 = torch.clamp(
            scaled_block_scales, min=E4M3_EPS, max=F8E4M3_MAX
        ).to(torch.float8_e4m3fn)
        scaled_block_scales_fp32 = scaled_block_scales_fp8.to(torch.float32)
        # We "temporarily" dequant the scaled_block_scales_fp32 to get the per_tensor_scale
        # To apply to data
        total_scale = scale * scaled_block_scales_fp32
        data_scaled = x / total_scale.unsqueeze(-1)
        out_scales = scaled_block_scales_fp8

        data_scaled = torch.clamp(data_scaled, -F4_E2M1_MAX, F4_E2M1_MAX)
        data_scaled = data_scaled.view(orig_shape)
        data_lp = f32_to_f4_unpacked(data_scaled)
        # TODO: NotImplementedError: "copy_kernel" not implemented for 'Float4_e2m1fn_x2'
        # data_lp = pack_uint4(data_lp).view(torch.float4_e2m1fn_x2)
        data_lp = pack_uint4(data_lp)
        return out_scales, data_lp
    """
    block_size: int = 16
    raise NotImplementedError


def tensor_dequantizer(x: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype):
    x = (x.to(dtype=scale.dtype) * scale).to(dtype=dtype)
    return x

def woq_fwd(self, x):
    dq_weight = self.dequantizer(self.weight, self.scale_weight, x.dtype)
    bias = self.bias
    if bias is not None and bias.dtype == self.weight.dtype:
        bias = self.dequantizer(bias, self.scale_weight, x.dtype)
    return torch.nn.functional.linear(x, dq_weight, bias)

def quantized_fwd(self, input):
    tensor_2d = False
    if len(input.shape) == 2:
        tensor_2d = True
        input = input.unsqueeze(1)

    input_shape = input.shape
    input_dtype = input.dtype
    assert len(input_shape) == 3, "input must be 3D"

    q_input, input_scale = self.quantizer(input, self.scale_input, self.weight.dtype)
    q_input = q_input.reshape(-1, input_shape[2])
    o = torch._scaled_mm(q_input, self.weight.T, scale_a=input_scale, scale_b=self.scale_weight.float(),
                         bias=self.bias, out_dtype=input_dtype)
    if isinstance(o, tuple):
        o = o[0]
    if tensor_2d:
        return o.reshape(input_shape[0], -1)
    return o.reshape((-1, input_shape[1], self.weight.shape[0]))
