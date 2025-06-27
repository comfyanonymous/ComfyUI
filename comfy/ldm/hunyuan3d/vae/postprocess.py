import torch
from skimage import measure
from dataclasses import dataclass
import numpy as np

@dataclass
class Latent2MeshOutput():
    # mesh for vertices and faces
    mesh_v: None
    mesh_f: None

class SufraceExtractor():
    def compute_box_stat(self, bounds, octree_resolution: int):

        # if float, turn it into a cube
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]

        bbox_min, bbox_max = np.array(bounds[0:3]), np.array(bounds[3:6])
        bbox_size = bbox_max - bbox_min
        grid_size = [int(octree_resolution) + 1, int(octree_resolution) + 1, int(octree_resolution) + 1]
        return grid_size, bbox_min, bbox_size
    
    def run(self, grid_logit, *, bounds, octree_res, **kwargs):
        # grid_logit from volume decoder
        # use marching cube algo to turn an sdf to a mesh
        vertices, faces, _, _ = measure.marching_cubes(grid_logit.cpu().numpy(),
                                           0.0,
                                           method = "lewiner")
        
        grid_size, bbox_min, bbox_size = self.compute_box_stat(bounds = bounds, octree_resolution = octree_res)
        vertices = vertices / grid_size * bbox_size + bbox_min

        return vertices, faces
    
    def __call__(self, grid_logits, **kwds):
        
        outputs = []
        # loop over the batches
        for i in range(grid_logits.shape[0]):
            try:
                # process each batch
                vertices, faces = self.run(grid_logits[i], **kwds)
                vertices = vertices.astype(np.float32)
                faces = np.ascontiguousarray(faces)
                outputs.append(Latent2MeshOutput(mesh_v = vertices, mesh_f = faces))

            except Exception:
                import traceback
                traceback.print_exc()
                outputs.append(None)

        return outputs
    
################################################
# Volume Decoder
################################################

class VanillaVolumeDecoder():
    @torch.no_grad()
    def __call__(self, latents: torch.Tensor, geo_decoder: callable, octree_res: int, bounds = 1.01,
                 num_chunks: int = 10_000):
        
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]

        bbox_min, bbox_max = torch.tensor(bounds[:3]), torch.tensor(bounds[3:])

        x = torch.linspace(bbox_min[0], bbox_max[0], int(octree_res) + 1, dtype = torch.float32)
        y = torch.linspace(bbox_min[1], bbox_max[1], int(octree_res) + 1, dtype = torch.float32)
        z = torch.linspace(bbox_min[2], bbox_max[2], int(octree_res) + 1, dtype = torch.float32)

        [xs, ys, zs] = torch.meshgrid(x, y, z, indexing = "ij")
        xyz = torch.stack((xs, ys, zs), axis=-1).to(latents.device, dtype = latents.dtype).contiguous().reshape(-1, 3)
        grid_size = [int(octree_res) + 1, int(octree_res) + 1, int(octree_res) + 1]

        batch_logits = []
        for start in range(0, xyz.shape[0], num_chunks):
            chunk_queries = xyz[start: start + num_chunks, :]
            chunk_queries = chunk_queries.unsqueeze(0).repeat(latents.shape[0], 1, 1)
            logits = geo_decoder(queries = chunk_queries, latents = latents)
            batch_logits.append(logits)

        grid_logits = torch.cat(batch_logits, dim = 1)
        grid_logits = grid_logits.view((latents.shape[0], *grid_size)).float()

        return grid_logits

def export_to_trimesh(mesh_output):
    import trimesh
    
    if isinstance(mesh_output, list):
        outputs = []
        for mesh in mesh_output:
            if mesh is None:
                outputs.append(None)
            else:
                mesh.mesh_f = mesh.mesh_f[:, ::-1]
                mesh_output = trimesh.Trimesh(mesh.mesh_v, mesh.mesh_f)
                outputs.append(mesh_output)
        return outputs
    else:
        mesh_output.mesh_f = mesh_output.mesh_f[:, ::-1]
        mesh_output = trimesh.Trimesh(mesh_output.mesh_v, mesh_output.mesh_f)
        return mesh_output
        