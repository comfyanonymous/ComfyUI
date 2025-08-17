import torch
class FunctionAdaCoF(torch.autograd.Function):
    # end
    @staticmethod
    def forward(ctx, input, weight, offset_i, offset_j, dilation):
        raise NotImplementedError()
