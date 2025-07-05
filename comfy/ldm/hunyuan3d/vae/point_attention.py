from transformer import MLP
from transformer import Transformer
import torch.nn.functional as F
import torch.nn as nn
from fps import fps
import torch

class QKVMultiheadCrossAttention(nn.Module):
    def __init__(
        self,
        heads: int,
        n_data = None,
        width=None,
        qk_norm=False,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.heads = heads
        self.n_data = n_data
        self.q_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()

    def forward(self, q, kv):

        _, n_ctx, _ = q.shape
        bs, n_data, width = kv.shape

        attn_ch = width // self.heads // 2
        q = q.view(bs, n_ctx, self.heads, -1)

        kv = kv.view(bs, n_data, self.heads, -1)
        k, v = torch.split(kv, attn_ch, dim=-1)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k, v = [t.permute(0, 2, 1, 3) for t in (q, k, v)]
        out = F.scaled_dot_product_attention(q, k, v)

        out = out.transpose(1, 2).reshape(bs, n_ctx, -1)

        return out


class MultiheadCrossAttention(nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        qkv_bias: bool = False,
        n_data = None,
        norm_layer = nn.LayerNorm,
        qk_norm: bool = False,
        kv_cache: bool = False,
    ):
        super().__init__()

        self.c_q = nn.Linear(width, width, bias=qkv_bias)
        self.c_kv = nn.Linear(width, width * 2, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)

        self.attention = QKVMultiheadCrossAttention(
            heads = heads,
            n_data = n_data,
            width = width,
            norm_layer = norm_layer,
            qk_norm = qk_norm
        )

        self.kv_cache = kv_cache
        self.data = None

    def forward(self, x, data):
        x = self.c_q(x)

        if self.kv_cache:
            if self.data is None:
                self.data = self.c_kv(data)

            data = self.data
        else:
            data = self.c_kv(data)

        x = self.attention(x, data)
        x = self.c_proj(x)

        return x


class ResidualCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        n_data: int = None,
        mlp_expand_ratio: int = 4,
        qkv_bias: bool = False,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False
    ):
        super().__init__()

        self.attn = MultiheadCrossAttention(
            n_data=n_data,
            width = width,
            heads=heads,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            qk_norm=qk_norm
        )
        
        self.ln_1 = norm_layer(width, elementwise_affine = True, eps = 1e-6)
        self.ln_2 = norm_layer(width, elementwise_affine = True, eps = 1e-6)
        self.ln_3 = norm_layer(width, elementwise_affine = True, eps = 1e-6)

        self.mlp = MLP(width=width, ratio = mlp_expand_ratio)

    def forward(self, x: torch.Tensor, data: torch.Tensor):
        x = x + self.attn(self.ln_1(x), self.ln_2(data))
        x = x + self.mlp(self.ln_3(x))
        return x
    
class CrossAttentionDecoder(nn.Module):
    def __init__(
            self,
            num_latents: int,
            out_channels: int,
            fourier_embedder,
            width: int,
            heads: int,
            mlp_expand_ratio: int = 4,
            downsample_ratio: int = 1,
            enable_ln_post: bool = True,
            qkv_bias: bool = False,
            qk_norm: bool = False):
        
        super().__init__()

        self.enable_ln_post = enable_ln_post
        self.fourier_embedder = fourier_embedder
        self.downsample_ratio = downsample_ratio

        self.query_proj = nn.Linear(self.fourier_embedder.out_dim, width)

        if self.downsample_ratio != 1:
            self.latents_proj = nn.Linear(width * downsample_ratio, width)

        if self.enable_ln_post == False:
            qk_norm = False

        self.cross_attn_decoder = ResidualCrossAttentionBlock(
            n_data=num_latents,
            width=width,
            mlp_expand_ratio=mlp_expand_ratio,
            heads=heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm
        )

        if self.enable_ln_post:
            self.ln_post = nn.LayerNorm(width)

        self.output_proj = nn.Linear(width, out_channels)
        self.count = 0

    def forward(self, queries = None, query_embeddings = None, latents = None):

        if query_embeddings is None:
            query_embeddings = self.query_proj(self.fourier_embedder(queries).to(latents.dtype))

        self.count += query_embeddings.shape[1]

        if self.downsample_ratio != 1:
            latents = self.latents_proj(latents)

        x = self.cross_attn_decoder(query_embeddings, latents)

        if self.enable_ln_post:
            x = self.ln_post(x)

        out = self.output_proj(x)

        return out
        

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
                n_ctx = num_latents,
                width = width,
                heads = heads,
                qkv_bias = qkv_bias,
                qk_norm = qk_norm,
                depth = layers
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
def test_point_cross_attention():

    from vae import FourierEmbedder

    torch.manual_seed(2025)
    B = 2                     # batch size
    D = 3                     # point dimension (x, y, z)
    F = 16                    # feature dimension

    pc_random = 96            # number of random surface points
    pc_sharpedge = 32         # number of sharpedge points
    total_points = pc_random + pc_sharpedge  # = 128

    L = 32                    # num_latents (final tokens)
    downsample_ratio = 2.0    # modest oversampling
    width = 128
    heads = 4
    layers = 2


    point_cloud = torch.randn(B, total_points, D)
    features     = torch.randn(B, total_points, F)

    embedder = FourierEmbedder()

    model = PointCrossAttention(
        num_latents=L,
        downsample_ratio=downsample_ratio,
        pc_size=pc_random,
        pc_sharpedge_size=pc_sharpedge,
        point_feats=F,
        width=width,
        heads=heads,
        layers=layers,
        fourier_embedder=embedder,
        use_ln_post=True,
        qkv_bias=True,
        qk_norm=False
    )


    output = model(point_cloud, features)
    print(output[0])
if __name__ == '__main__':
    test_point_cross_attention()