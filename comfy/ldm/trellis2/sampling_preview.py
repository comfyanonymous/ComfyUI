"""Side-channel for per-step sampling previews of the Trellis2/Pixal3D cascade.

The sampler callback only receives the denoised latent `x0`, but a 3D preview of
the texture stage also needs the sparse voxel coords (the latent alone is just an
unordered [B, C, N, 1] feature stack). The diffusion model's forward writes the
current stage context here each step; the latent previewer reads it. Single prompt
samples one stage at a time, so a module-level holder is safe.

Intentionally dependency-free to avoid import cycles with comfy.ldm.trellis2.model.
"""

_context = {}

# Fitted texture latent -> base-color factors: (W [C, 3], b [3]) on CPU, or None.
# Calibrated by VaeDecodeTextureTrellis from real decoded albedo and read by the
# latent previewer for a faithful texture-stage color preview.
_tex_rgb = None


def set_context(mode=None, coords=None, coord_counts=None, model_frame=None):
    _context["mode"] = mode
    _context["coords"] = coords
    _context["coord_counts"] = coord_counts
    _context["model_frame"] = model_frame


def get_context():
    return _context


def clear():
    _context.clear()


def set_tex_rgb(W, b):
    global _tex_rgb
    _tex_rgb = (W, b)


def get_tex_rgb():
    return _tex_rgb
