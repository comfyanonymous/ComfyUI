import torch
from torch import Tensor
import math
import numpy as np
from skimage import measure
from dataclasses import dataclass
import torch.nn as nn

import sys, os; 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from comfy.ldm.hunyuan3d.vae import (
    CrossAttentionDecoder, Transformer, ResidualCrossAttentionBlock, FourierEmbedder, VanillaVolumeDecoder
)
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
class PointCrossAttention(nn.Module):
    def __init__(self,
        num_latents: int,
        downsample_ratio: float,
        pc_size: int,
        pc_sharpedge_size: int,
        point_feats: int,
        width: int,
        heads: int,
        layers: int,
        fourier_embedder,
        normal_pe: bool = False,
        qkv_bias: bool = False,
        use_ln_post: bool = True,
        qk_norm: bool = True):

        super().__init__()

        self.fourier_embedder = fourier_embedder

        self.pc_size = pc_size
        self.normal_pe = normal_pe
        self.downsample_ratio = downsample_ratio
        self.pc_sharpedge_size = pc_sharpedge_size
        self.num_latents = num_latents
        self.point_feats = point_feats

        self.input_proj = nn.Linear(self.fourier_embedder.out_dim + point_feats, width)

        self.cross_attn = ResidualCrossAttentionBlock(
            width = width,
            heads = heads,
            qkv_bias = qkv_bias,
            qk_norm = qk_norm
        )

        self.self_attn = None
        if layers > 0:
            self.self_attn = Transformer(
                width = width,
                heads = heads,
                qkv_bias = qkv_bias,
                qk_norm = qk_norm,
                layers = layers
            )

        if use_ln_post:
            self.ln_post = nn.LayerNorm(width)
        else:
            self.ln_post = None

    def sample_points_and_latents(self, point_cloud: torch.Tensor, features: torch.Tensor):

        """
        Subsample points randomly from the point cloud (input_pc)
        Further sample the subsampled points to get query_pc
        take the fourier embeddings for both input and query pc

        Mental Note: FPS-sampled points (query_pc) act as latent tokens that attend to and learn from the broader context in input_pc.
        Goal: get a smaller represenation (query_pc) to represent the entire scence structure by learning from a broader subset (input_pc).
        More computationally efficient.

        Features are additional information for each point in the cloud
        """

        B, _, D = point_cloud.shape

        num_latents = int(self.num_latents)

        num_random_query = self.pc_size / (self.pc_size + self.pc_sharpedge_size) * num_latents
        num_sharpedge_query = num_latents - num_random_query

        # Split random and sharpedge surface points
        random_pc, sharpedge_pc = torch.split(point_cloud, [self.pc_size, self.pc_sharpedge_size], dim=1)

        # assert statements 
        assert random_pc.shape[1] <= self.pc_size, "Random surface points size must be less than or equal to pc_size"
        assert sharpedge_pc.shape[1] <= self.pc_sharpedge_size, "Sharpedge surface points size must be less than or equal to pc_sharpedge_size"

        input_random_pc_size = int(num_random_query * self.downsample_ratio)
        random_query_pc, random_input_pc, random_idx_pc, random_idx_query = \
            self.subsample(pc = random_pc, num_query = num_random_query, input_pc_size = input_random_pc_size)
        
        input_sharpedge_pc_size = int(num_sharpedge_query * self.downsample_ratio)

        if input_sharpedge_pc_size == 0:
            sharpedge_input_pc = torch.zeros(B, 0, D, dtype = random_input_pc.dtype).to(point_cloud.device)
            sharpedge_query_pc = torch.zeros(B, 0, D, dtype= random_query_pc.dtype).to(point_cloud.device)

        else: sharpedge_query_pc, sharpedge_input_pc, sharpedge_idx_pc, sharpedge_idx_query = \
            self.subsample(pc = sharpedge_pc, num_query = num_sharpedge_query, input_pc_size = input_sharpedge_pc_size)
        
        # concat the random and sharpedges
        query_pc = torch.cat([random_query_pc, sharpedge_query_pc], dim = 1)
        input_pc = torch.cat([random_input_pc, sharpedge_input_pc], dim = 1)

        query = self.fourier_embedder(query_pc)
        data = self.fourier_embedder(input_pc)

        if self.point_feats > 0:
            random_surface_features, sharpedge_surface_features = torch.split(features, [self.pc_size, self.pc_sharpedge_size], dim = 1)

            input_random_surface_features, query_random_features = \
                self.handle_features(features = random_surface_features, idx_pc = random_idx_pc, batch_size = B,
                                     input_pc_size = input_random_pc_size, idx_query = random_idx_query)
            
            if input_sharpedge_pc_size == 0:
                input_sharpedge_surface_features = torch.zeros(B, 0, self.point_feats,
                                                               dtype = input_random_surface_features.dtype, device = point_cloud.device)
                
                query_sharpedge_features = torch.zeros(B, 0, self.point_feats,
                                                       dtype = query_random_features.dtype, device = point_cloud.device)
            else:

                input_sharpedge_surface_features, query_sharpedge_features = \
                    self.handle_features(idx_pc = sharpedge_idx_pc, features = sharpedge_surface_features,
                                         batch_size = B, idx_query = sharpedge_idx_query, input_pc_size = input_sharpedge_pc_size)

            query_features = torch.cat([query_random_features, query_sharpedge_features], dim = 1)
            input_features = torch.cat([input_random_surface_features, input_sharpedge_surface_features], dim = 1)

            if self.normal_pe:
                # apply the fourier embeddings on the first 3 dims (xyz)
                input_features_pe = self.fourier_embedder(input_features[..., :3])
                query_features_pe = self.fourier_embedder(query_features[..., :3])
                # replace the first 3 dims with the new PE ones
                input_features = torch.cat([input_features_pe, input_features[..., :3]], dim = -1)
                query_features = torch.cat([query_features_pe, query_features[..., :3]], dim = -1)

            # concat at the channels dim
            query = torch.cat([query, query_features], dim = -1)
            data = torch.cat([data, input_features], dim = -1)

        # don't return pc_info to avoid unnecessary memory usuage
        return query.view(B, -1, query.shape[-1]), data.view(B, -1, data.shape[-1])

    def forward(self, point_cloud: torch.Tensor, features: torch.Tensor):
        
        query, data = self.sample_points_and_latents(point_cloud = point_cloud, features = features)

        # apply projections
        query = self.input_proj(query)
        data = self.input_proj(data)

        # apply cross attention between query and data
        latents = self.cross_attn(query, data)

        if self.self_attn is not None:
            latents = self.self_attn(latents)

        if self.ln_post is not None:
            latents = self.ln_post(latents)

        return latents


    def subsample(self, pc, num_query, input_pc_size: int):

        """
        num_query: number of points to keep after FPS
        input_pc_size: number of points to select before FPS
        """

        B, _, D = pc.shape
        query_ratio = num_query / input_pc_size

        # random subsampling of points inside the point cloud
        idx_pc = torch.randperm(pc.shape[1], device = pc.device)[:input_pc_size]
        input_pc = pc[:, idx_pc, :]

        # flatten to allow applying fps across the whole batch
        flattent_input_pc = input_pc.view(B * input_pc_size, D)

        # construct a batch_down tensor to tell fps
        # which points belong to which batch
        N_down = int(flattent_input_pc.shape[0] / B)
        batch_down = torch.arange(B).to(pc.device)
        batch_down = torch.repeat_interleave(batch_down, N_down)

        idx_query = fps(flattent_input_pc, batch_down, sampling_ratio = query_ratio)
        query_pc = flattent_input_pc[idx_query].view(B, -1, D)

        return query_pc, input_pc, idx_pc, idx_query
    
    def handle_features(self, features, idx_pc, input_pc_size, batch_size: int, idx_query):

        B = batch_size

        input_surface_features = features[:, idx_pc, :]
        flattent_input_features = input_surface_features.view(B * input_pc_size, -1)
        query_features = flattent_input_features[idx_query].view(B, -1, 
                                                                 flattent_input_features.shape[-1])
        
        return input_surface_features, query_features
    
    def forward(self, pc, feats):
        """

        Args:
            pc (torch.FloatTensor): [B, N, 3]
            feats (torch.FloatTensor or None): [B, N, C]

        Returns:

        """

        query, data = self.sample_points_and_latents(pc, feats)

        query = self.input_proj(query)
        query = query
        data = self.input_proj(data)
        data = data

        latents = self.cross_attn(query, data)
        if self.self_attn is not None:
            latents = self.self_attn(latents)

        if self.ln_post is not None:
            latents = self.ln_post(latents)

        return latents

@dataclass
class Latent2MeshOutput():
    # mesh for vertices and faces
    vertices: None
    faces: None

class SurfaceExtractor():
    def compute_box_stat(self, bounds, octree_resolution: int):

        # if float, turn it into a cube
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]

        bbox_min, bbox_max = np.array(bounds[0:3]), np.array(bounds[3:6])
        bbox_size = bbox_max - bbox_min
        grid_size = [int(octree_resolution) + 1, int(octree_resolution) + 1, int(octree_resolution) + 1]
        return grid_size, bbox_min, bbox_size
    
    def run(self, grid_logit, *, bounds, octree_resolution, level: float = 0.0, **kwargs):
        # grid_logit from volume decoder
        # use marching cube algo to turn an sdf to a mesh
        vertices, faces, _, _ = measure.marching_cubes(grid_logit.cpu().numpy(),
                                           level,
                                           method = "lewiner")
        
        grid_size, bbox_min, bbox_size = self.compute_box_stat(bounds = bounds, octree_resolution = octree_resolution)
        vertices = vertices / grid_size * bbox_size + bbox_min

        return vertices, faces
    
    def __call__(self, grid_logits, **kwds):
        
        outputs = []
        veritces_list = []
        faces_list = []
        # loop over the batches
        for i in range(grid_logits.shape[0]):
            try:
                # process each batch
                vertices, faces = self.run(grid_logits[i], **kwds)
                vertices = vertices.astype(np.float32)
                faces = np.ascontiguousarray(faces)
                #outputs.append(Latent2MeshOutput(vertices = vertices, faces = faces))
                veritces_list.append(vertices)
                faces_list.append(faces)

            except Exception:
                import traceback
                traceback.print_exc()
                outputs.append(None)

        return outputs
    
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

    import trimesh

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
        import trimesh

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

class FourierEmbedder(nn.Module):
    def __init__(self, num_freq: int = 8, input_dim: int = 3, include_pi: bool = False):
        super().__init__()

        frequencies = 2.0 ** torch.arange(
            num_freq,
            dtype = torch.float32
        )

        if include_pi:
            frequencies *= torch.pi

        self.register_buffer("frequencies", frequencies, persistent = False)

        self.out_dim = input_dim * (num_freq * 2 + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        embed = (x[..., None].contiguous() * self.frequencies).view(*x.shape[:-1], -1)
        return torch.cat((x, embed.sin(), embed.cos()), dim = -1)
    
class DiagonalGaussianDistribution:
    def __init__(self, params: torch.Tensor, feature_dim: int = -1):

        # divide quant channels (8) into mean and log variance
        self.mean, self.logvar = torch.chunk(params, 2, dim = feature_dim)

        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)

    def sample(self):

        eps = torch.randn_like(self.std)
        z = self.mean + eps * self.std

        return z
    
class VAE(nn.Module):
    def __init__(self,
        *,
        num_latents: int = 4096,
        embed_dim: int = 64,
        width: int = 1024,
        heads: int = 16,
        num_decoder_layers: int = 16,
        num_encoder_layers: int = 8,
        pc_size: int = 81920,
        pc_sharpedge_size: int = 0,
        point_feats: int = 4,
        downsample_ratio: int = 20,
        geo_decoder_downsample_ratio: int = 1,
        geo_decoder_mlp_expand_ratio: int = 4,
        geo_decoder_ln_post: bool = True,
        num_frequencies: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        drop_path_rate: float = 0.0,
        include_pi: bool = False,
        scale_factor: float = 1.0039506158752403
        ):

        super().__init__()

        self.latent_shape = (num_latents, embed_dim)
        self.scale_factor = scale_factor

        self.fourier_embedder = FourierEmbedder(num_freq = num_frequencies, include_pi = include_pi)

        self.encoder = PointCrossAttention(layers = num_encoder_layers,
                                           num_latents = num_latents,
                                           downsample_ratio = downsample_ratio,
                                           heads = heads,
                                           pc_size = pc_size,
                                           width = width,
                                           point_feats = point_feats,
                                           fourier_embedder = self.fourier_embedder,
                                           pc_sharpedge_size = pc_sharpedge_size) 
        
        self.transformer = Transformer(
            width=width,
            layers=num_decoder_layers,
            heads=heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            drop_path_rate=drop_path_rate
        )

        self.geo_decoder = CrossAttentionDecoder(
            fourier_embedder = self.fourier_embedder,
            out_channels = 1,
            mlp_expand_ratio = geo_decoder_mlp_expand_ratio,
            downsample_ratio = geo_decoder_downsample_ratio,
            enable_ln_post = geo_decoder_ln_post,
            width=width // geo_decoder_downsample_ratio,
            heads=heads // geo_decoder_downsample_ratio,
            qkv_bias = qkv_bias,
            qk_norm= qk_norm
        )

        self.pre_kl = nn.Linear(width, embed_dim * 2)
        self.post_kl = nn.Linear(embed_dim, width)

        self.volume_decoder = VanillaVolumeDecoder()
        self.surface_extractor = SurfaceExtractor()
        

    def forward(self):
        pass

    def encode(self, surface):

        pc, feats = surface[:, :, :3], surface[:, :, 3:]
        latents = self.encoder(pc, feats)
        
        moments = self.pre_kl(latents)
        posterior = DiagonalGaussianDistribution(moments, feature_dim = -1)

        latents = posterior.sample()

        return latents

    def decode(self, latents, **kwargs):
        to_mesh = kwargs.pop("to_mesh", True)
        latents = self.post_kl(latents)
        latents = self.transformer(latents)

        if not to_mesh:
            return latents

        grid_logits = self.volume_decoder(latents = latents, geo_decoder = self.geo_decoder, **kwargs)
        mesh = self.surface_extractor(grid_logits, **kwargs)

        return mesh

def load_vae(vae):

    DEBUG = False
    
    checkpoint = "model.fp16.ckpt"
    missing, unexpected = vae.load_state_dict(torch.load(checkpoint), strict = not DEBUG)

    if DEBUG:
        print(f"Missing {len(missing)}: ", missing)
        print(f"\nUnexpected {len(unexpected)}: ", unexpected)

    return vae

    