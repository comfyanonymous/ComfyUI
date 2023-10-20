import torch
from contextlib import contextmanager

class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        return torch.nn.functional.linear(input, self.weight, self.bias)

class Conv2d(torch.nn.Conv2d):
    def reset_parameters(self):
        return None

def conv_nd(dims, *args, **kwargs):
    if dims == 2:
        return Conv2d(*args, **kwargs)
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
