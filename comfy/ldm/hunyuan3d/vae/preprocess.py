import trimesh
import torch
import numpy as np

def normalize_mesh(mesh, scale = 0.9999):
    """Normalize mesh to fit in [-scale, scale]. Translate mesh so its center is [0,0,0]"""

    bbox = mesh.bounds
    center = (bbox[1] + bbox[0]) / 2

    max_extent = (bbox[1] - bbox[0]).max()
    mesh.apply_translation(-center)
    mesh.apply_scale((2 * scale) / max_extent)

    return mesh

def sample_pointcloud(mesh, num = 200000):
    """ Uniformly sample points from the surface of the mesh """

    points, face_idx = mesh.sample(num, return_index = True)
    normals = mesh.face_normals[face_idx]
    return torch.from_numpy(points.astype(np.float32)), torch.from_numpy(normals.astype(np.float32))

def detect_sharp_edges(mesh, threshold=0.985):
    """Return edge indices (a, b) that lie on sharp boundaries of the mesh."""

    V, F = mesh.vertices, mesh.faces
    VN, FN = mesh.vertex_normals, mesh.face_normals

    sharp_mask = np.ones(V.shape[0])
    for i in range(3):
        indices = F[:, i]
        alignment = np.einsum('ij,ij->i', VN[indices], FN)
        dot_stack = np.stack((sharp_mask[indices], alignment), axis=-1)
        sharp_mask[indices] = np.min(dot_stack, axis=-1)

    edge_a = np.concatenate([F[:, 0], F[:, 1], F[:, 2]])
    edge_b = np.concatenate([F[:, 1], F[:, 2], F[:, 0]])
    sharp_edges = (sharp_mask[edge_a] < threshold) & (sharp_mask[edge_b] < threshold)

    return edge_a[sharp_edges], edge_b[sharp_edges]


def sharp_sample_pointcloud(mesh, num = 16384):
    """ Sample points preferentially from sharp edges in the mesh. """

    edge_a, edge_b = detect_sharp_edges(mesh)
    V, VN = mesh.vertices, mesh.vertex_normals

    va, vb = V[edge_a], V[edge_b]
    na, nb = VN[edge_a], VN[edge_b]

    edge_lengths = np.linalg.norm(vb - va, axis=-1)
    weights = edge_lengths / edge_lengths.sum()

    indices = np.searchsorted(np.cumsum(weights), np.random.rand(num))
    t = np.random.rand(num, 1)

    samples = t * va[indices] + (1 - t) * vb[indices]
    normals = t * na[indices] + (1 - t) * nb[indices]

    return samples.astype(np.float32), normals.astype(np.float32)

def load_surface_sharpedge(mesh, num_points=4096, num_sharp_points=4096, sharpedge_flag = True, device = "cuda"):
    """Load a surface with optional sharp-edge annotations from a trimesh mesh."""

    try:
        mesh_full = trimesh.util.concatenate(mesh.dump())
    except Exception:
        mesh_full = trimesh.util.concatenate(mesh)

    mesh_full = normalize_mesh(mesh_full)

    faces = mesh_full.faces
    vertices = mesh_full.vertices
    origin_face_count = faces.shape[0]

    mesh_surface = trimesh.Trimesh(vertices=vertices, faces=faces[:origin_face_count])
    mesh_fill = trimesh.Trimesh(vertices=vertices, faces=faces[origin_face_count:])

    area_surface = mesh_surface.area
    area_fill = mesh_fill.area
    total_area = area_surface + area_fill

    sample_num = 499712 // 2
    fill_ratio = area_fill / total_area if total_area > 0 else 0

    num_fill = int(sample_num * fill_ratio)
    num_surface = sample_num - num_fill

    surf_pts, surf_normals = sample_pointcloud(mesh_surface, num_surface)
    fill_pts, fill_normals = (torch.zeros(0, 3), torch.zeros(0, 3)) if num_fill == 0 else sample_pointcloud(mesh_fill, num_fill)

    sharp_pts, sharp_normals = sharp_sample_pointcloud(mesh_surface, sample_num)

    def assemble_tensor(points, normals, label=None):

        data = torch.cat([points, normals], dim=1).half().to(device)

        if label is not None:
            label_tensor = torch.full((data.shape[0], 1), float(label), dtype=torch.float16).to(device)
            data = torch.cat([data, label_tensor], dim=1)

        return data

    surface = assemble_tensor(torch.cat([surf_pts.to(device), fill_pts.to(device)], dim=0),
                              torch.cat([surf_normals.to(device), fill_normals.to(device)], dim=0),
                              label = 0 if sharpedge_flag else None)
    
    sharp_surface = assemble_tensor(torch.from_numpy(sharp_pts), torch.from_numpy(sharp_normals),
                                    label = 1 if sharpedge_flag else None)

    rng = np.random.default_rng()

    surface = surface[rng.choice(surface.shape[0], num_points, replace = False)]
    sharp_surface = sharp_surface[rng.choice(sharp_surface.shape[0], num_sharp_points, replace = False)]

    full = torch.cat([surface, sharp_surface], dim = 0).unsqueeze(0)

    return full

class SharpEdgeSurfaceLoader:
    """ Load mesh surface and sharp edge samples. """

    def __init__(self, num_uniform_points = 8192, num_sharp_points = 8192):

        self.num_uniform_points = num_uniform_points
        self.num_sharp_points = num_sharp_points
        self.total_points = num_uniform_points + num_sharp_points

    def __call__(self, mesh_input, device = "cuda"):
        mesh = self._load_mesh(mesh_input)
        return load_surface_sharpedge(mesh, self.num_uniform_points, self.num_sharp_points, device = device)

    @staticmethod
    def _load_mesh(mesh_input):

        if isinstance(mesh_input, str):
            mesh = trimesh.load(mesh_input, force="mesh", merge_primitives = True)
        else:
            mesh = mesh_input

        if isinstance(mesh, trimesh.Scene):
            combined = None
            for obj in mesh.geometry.values():
                combined = obj if combined is None else combined + obj
            return combined
        
        return mesh

def test_preprocess():
    torch.set_default_device("cpu")
    torch.manual_seed(2025)
    np.random.seed(2025)
    loader = SharpEdgeSurfaceLoader(
        num_sharp_points = 0,
        num_uniform_points = 81920,
    )

    mesh_demo = 'rock.glb'
    surface = loader(mesh_demo, device = "cpu").to(dtype = torch.float16)
    print(surface)

if __name__ == "__main__":
    test_preprocess()