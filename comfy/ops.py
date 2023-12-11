import torch
from contextlib import contextmanager

class Linear(torch.nn.Linear):
    def reset_parameters(self):
        return None

class Conv2d(torch.nn.Conv2d):
    def reset_parameters(self):
        return None

class Conv3d(torch.nn.Conv3d):
    def reset_parameters(self):
        return None

class GroupNorm(torch.nn.GroupNorm):
    def reset_parameters(self):
        return None

class LayerNorm(torch.nn.LayerNorm):
    def reset_parameters(self):
        return None

def conv_nd(dims, *args, **kwargs):
    if dims == 2:
        return Conv2d(*args, **kwargs)
    elif dims == 3:
        return Conv3d(*args, **kwargs)
    else:
        raise ValueError(f"unsupported dimensions: {dims}")

def cast_bias_weight(s, input):
    bias = None
    if s.bias is not None:
        bias = s.bias.to(device=input.device, dtype=input.dtype)
    weight = s.weight.to(device=input.device, dtype=input.dtype)
    return weight, bias

class manual_cast:
    class Linear(Linear):
        def forward(self, input):
            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.linear(input, weight, bias)

    class Conv2d(Conv2d):
        def forward(self, input):
            weight, bias = cast_bias_weight(self, input)
            return self._conv_forward(input, weight, bias)

    class Conv3d(Conv3d):
        def forward(self, input):
            weight, bias = cast_bias_weight(self, input)
            return self._conv_forward(input, weight, bias)

    class GroupNorm(GroupNorm):
        def forward(self, input):
            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.group_norm(input, self.num_groups, weight, bias, self.eps)

    class LayerNorm(LayerNorm):
        def forward(self, input):
            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.layer_norm(input, self.normalized_shape, weight, bias, self.eps)

    @classmethod
    def conv_nd(s, dims, *args, **kwargs):
        if dims == 2:
            return s.Conv2d(*args, **kwargs)
        elif dims == 3:
            return s.Conv3d(*args, **kwargs)
        else:
            raise ValueError(f"unsupported dimensions: {dims}")

@contextmanager
def use_comfy_ops(device=None, dtype=None): # Kind of an ugly hack but I can't think of a better way
    old_torch_nn_linear = torch.nn.Linear
    force_device = device
    force_dtype = dtype
    def linear_with_dtype(in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
        if force_device is not None:
            device = force_device
        if force_dtype is not None:
            dtype = force_dtype
        return Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)

    torch.nn.Linear = linear_with_dtype
    try:
        yield
    finally:
        torch.nn.Linear = old_torch_nn_linear
