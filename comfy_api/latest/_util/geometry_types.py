import torch


class VOXEL:
    def __init__(self, data: torch.Tensor):
        self.data = data


class MESH:
    def __init__(self, vertices: torch.Tensor, faces: torch.Tensor):
        self.vertices = vertices
        self.faces = faces
