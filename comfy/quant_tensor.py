import torch

Q_TYPES = [torch.float8_e4m3fn, torch.float8_e5m2]


def get_quantizer_with_constraints(target_dtype: torch.dtype):
    if target_dtype in Q_TYPES:
        q_fn = dynamic_tensor_quantizer
    else:
        raise ValueError(f"Unsupported dtype {target_dtype}")

    alignment_check_fn = lambda x: x.shape[0] % 16 or x.shape[1] % 16

    def fn(x, **kwargs):
        if alignment_check_fn(x):
            return x, None
        if x.dtype == target_dtype:
            return x, None
        return q_fn(x, dtype=target_dtype, **kwargs)

    return fn


def dynamic_tensor_quantizer(x: torch.Tensor, dtype: torch.dtype, *args, **kwargs):
    input_scale = torch.abs(x).max() / torch.finfo(dtype).max
    x = (x / input_scale).clamp(torch.finfo(dtype).min, torch.finfo(dtype).max).to(dtype=dtype)
    return x, input_scale.float()


def tensor_quantizer(x: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype):
    x = (x / scale).clamp(torch.finfo(dtype).min, torch.finfo(dtype).max).to(dtype=dtype).contiguous()
    return x, scale.float()


def tensor_dequantizer(x: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype):
    x = x.to(dtype=dtype) * scale.to(dtype=dtype)
    return x


def woq_fwd(self, x):
    dq_weight = self.dequantizer(self.weight, scale=self.scale_weight, dtype=x.dtype)
    bias = self.bias
    if bias is not None and bias.dtype == self.weight.dtype:
        bias = self.dequantizer(bias, torch.ones_like(self.scale_weight), x.dtype)
    return torch.nn.functional.linear(x, dq_weight, bias)


def quantized_fwd(self, input):
    tensor_2d = False
    if len(input.shape) == 2:
        tensor_2d = True
        input = input.unsqueeze(1)

    input_shape = input.shape
    input_dtype = input.dtype
    assert len(input_shape) == 3, "input must be 3D"

    scale_input = getattr(self, "scale_input", None)
    q_input, scale_input = self.quantizer(input, scale=scale_input, dtype=self.weight.dtype)
    q_input = q_input.reshape(-1, input_shape[2])
    o = torch._scaled_mm(q_input, self.weight.T, scale_a=scale_input, scale_b=self.scale_weight.float(),
                         bias=self.bias, out_dtype=input_dtype)
    if isinstance(o, tuple):
        o = o[0]
    if tensor_2d:
        return o.reshape(input_shape[0], -1)
    return o.reshape((-1, input_shape[1], self.weight.shape[0]))
