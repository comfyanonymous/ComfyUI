# replaced torch.ops.torch_cluster.fps with a manual implementation
# to avoid having torch_cluster downloaded as dependency
# also the dependency takes a long time to install

import torch
from torch import Tensor
import math

def fps(src: Tensor, batch: Tensor, sampling_ratio: float, start_random: bool = True):

    # manually create the pointer vector
    assert src.size(0) == batch.numel()

    batch_size = int(batch.max()) + 1
    deg = src.new_zeros(batch_size, dtype = torch.long)

    deg.scatter_add_(0, batch, torch.ones_like(batch))

    ptr_vec = deg.new_zeros(batch_size + 1)
    torch.cumsum(deg, 0, out=ptr_vec[1:])

    #return fps_sampling(src, ptr_vec, ratio)
    sampled_indicies = []

    for b in range(batch_size):
        # start and the end of each batch
        start, end = ptr_vec[b].item(), ptr_vec[b + 1].item()
        # points from the point cloud
        points = src[start:end]

        num_points = points.size(0)
        num_samples = max(1, math.ceil(num_points * sampling_ratio))

        selected = torch.zeros(num_samples, device = src.device, dtype = torch.long)
        distances = torch.full((num_points,), float("inf"), device = src.device)

        # select a random start point
        if start_random:
            farthest = torch.randint(0, num_points, (1,), device = src.device)
        else: farthest = torch.tensor([0], device = src.device, dtype = torch.long)

        for i in range(num_samples):
            selected[i] = farthest
            centroid = points[farthest].squeeze(0)
            dist = torch.norm(points - centroid, dim = 1) # compute euclidean distance
            distances = torch.minimum(distances, dist)
            farthest = torch.argmax(distances)

        sampled_indicies.append(torch.arange(start, end)[selected])

    return torch.cat(sampled_indicies, dim = 0)


def test_fps():

    torch.manual_seed(2025)

    # 2 batches with different numbers of points
    points = torch.tensor([
        [0.0, 0.0, 0.0],   # batch 0
        [1.0, 0.0, 0.0],   # batch 0
        [2.0, 0.0, 0.0],   # batch 0
        [0.0, 1.0, 0.0],   # batch 1
        [0.0, 2.0, 0.0],   # batch 1
        [0.0, 3.0, 0.0]    # batch 1
    ], dtype=torch.float)

    batch = torch.tensor([0, 0, 0, 1, 1, 1])  # batch IDs

    ratio = 0.5  # sample 50% of points per batch
    # jit compilation for speedups
    #optimized_fps = torch.compile(fps)

    outputs = fps(points, batch, ratio, start_random = True) 
    #outputs2 = torch.ops.torch_cluster.fps(points, batch, ratio, True) # shouldn't work

    print(outputs)

if __name__ == "__main__":
    test_fps()
    