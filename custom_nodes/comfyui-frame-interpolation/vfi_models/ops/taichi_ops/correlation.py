import torch

class _FunctionCorrelation(torch.autograd.Function):
    @staticmethod
    def forward(self, first, second):
        raise NotImplementedError()

def FunctionCorrelation(tenFirst, tenSecond):
    raise NotImplementedError()
    return _FunctionCorrelation.apply(tenFirst, tenSecond)

class ModuleCorrelation(torch.nn.Module):
    def __init__(self):
        raise NotImplementedError()
        super(ModuleCorrelation, self).__init__()