import torch
def waste_cpu_resource():
    x = torch.rand(1, 1e6, dtype=torch.float64).cpu()
    return x.numpy()[0, 1]