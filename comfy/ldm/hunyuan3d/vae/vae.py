import torch
import torch.nn as nn
from transformer import Transformer
from postprocess import VanillaVolumeDecoder, SufraceExtractor
from point_attention import PointCrossAttention, CrossAttentionDecoder

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
        self.scale = scale_factor

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
            n_ctx=num_latents,
            width=width,
            depth=num_decoder_layers,
            heads=heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            drop_path_rate=drop_path_rate
        )

        self.geo_decoder = CrossAttentionDecoder(
            fourier_embedder = self.fourier_embedder,
            out_channels = 1,
            num_latents = num_latents,
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
        self.surface_extractor = SufraceExtractor()
        

    def forward(self):
        pass

    def encode(self, surface):

        pc, feats = surface[:, :, :3], surface[:, :, 3:]
        latents = self.encoder(pc, feats)
        
        moments = self.pre_kl(latents)
        posterior = DiagonalGaussianDistribution(moments, feature_dim = -1)

        latents = posterior.sample()

        return latents

    def decode(self, latents, to_mesh: bool = True, **kwargs):

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

def test_vae():

    torch.manual_seed(2025)
    vae = VAE()
    vae = load_vae(vae)

    from preprocess import SharpEdgeSurfaceLoader
    from postprocess import export_to_trimesh

    loader = SharpEdgeSurfaceLoader(
        num_sharp_points = 0,
        num_uniform_points = 81920,
    )

    mesh_demo = 'Duck.glb'
    surface = loader(mesh_demo).to(dtype = torch.float16)

    latents = vae.encode(surface)

    mesh = vae.decode(latents,
                      num_chunks = 20000,
                      octree_res = 256,
                      to_mesh = True)

    mesh = export_to_trimesh(mesh)[0]

    mesh.export("duck_recreated.glb")

if __name__ == "__main__":
    test_vae()