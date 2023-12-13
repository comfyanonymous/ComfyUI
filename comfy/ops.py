import torch
from contextlib import contextmanager

class disable_weight_init:
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

    @classmethod
    def conv_nd(s, dims, *args, **kwargs):
        if dims == 2:
            return s.Conv2d(*args, **kwargs)
        elif dims == 3:
            return s.Conv3d(*args, **kwargs)
        else:
            raise ValueError(f"unsupported dimensions: {dims}")

def cast_bias_weight(s, input):
    bias = None
    if s.bias is not None:
        bias = s.bias.to(device=input.device, dtype=input.dtype)
    weight = s.weight.to(device=input.device, dtype=input.dtype)
    return weight, bias

class manual_cast(disable_weight_init):
    class Linear(disable_weight_init.Linear):
        def forward(self, input):
            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.linear(input, weight, bias)

    class Conv2d(disable_weight_init.Conv2d):
        def forward(self, input):
            weight, bias = cast_bias_weight(self, input)
            return self._conv_forward(input, weight, bias)

    class Conv3d(disable_weight_init.Conv3d):
        def forward(self, input):
            weight, bias = cast_bias_weight(self, input)
            return self._conv_forward(input, weight, bias)

    class GroupNorm(disable_weight_init.GroupNorm):
        def forward(self, input):
            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.group_norm(input, self.num_groups, weight, bias, self.eps)

    class LayerNorm(disable_weight_init.LayerNorm):
        def forward(self, input):
            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.layer_norm(input, self.normalized_shape, weight, bias, self.eps)
