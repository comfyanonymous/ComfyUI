"""Named constants for the SeedVR2 integration, grouped by provenance.

Provenance prefixes:
- ``SEEDVR2_*``   - introduced by this integration (no external origin); rationale inline.
- ``BYTEDANCE_*`` - ported from the official ByteDance-Seed/SeedVR release; each cites
                    the upstream config/source path it was lifted from.
- unprefixed standards (``ROPE_THETA``, ``CIELAB_*``, ``D65_*``) - published literature /
                    ISO / CIE values; cite the standard.
"""

# --------------------------------------------------------------------------------------
# A. Progressive-sampler chunk-size law  (SEEDVR2 - this integration's VRAM experiment)
#    n_max(frames/chunk) = SEEDVR2_CHUNK_FRAMES_PER_GB * (free_GB - SEEDVR2_CHUNK_GB_MARGIN)
#    rounded to the 4n+1 grid. Fit on 22 blocked-5090 cells, validated on a real RTX 4070
#    (3b and 7b). Resolution-independent (the VAE tiling sets the wall, not the DiT).
# --------------------------------------------------------------------------------------
SEEDVR2_CHUNK_GB_MARGIN = 3        # fixed VRAM overhead before chunks scale (GiB)
SEEDVR2_CHUNK_FRAMES_PER_GB = 4    # empirical slope: pixel frames admitted per free GiB

# --------------------------------------------------------------------------------------
# B. Fork heuristics  (SEEDVR2 - this integration)
# --------------------------------------------------------------------------------------
SEEDVR2_7B_VID_DIM = 3072          # runtime 3b-vs-7b sentinel; tested against vid_dim.
                                   # (3072 is ByteDance's 7b vid_dim; the sentinel use is ours.)
SEEDVR2_OOM_BACKOFF_DIVISOR = 2    # auto-chunk OOM retry: halve the chunk and retry.
SEEDVR2_DTYPE_BYTES_FLOOR = 4      # per-element byte floor for memory math (fp32 worst case).
SEEDVR2_7B_MLP_CHUNK = 8192        # 7b MLP token-chunk to bound peak VRAM.
SEEDVR2_ROPE_PARTIAL_CHUNK_TOKENS = 4096  # partial-RoPE application token-chunk.
SEEDVR2_LATENT_CHANNELS = 16       # SeedVR2 latent channel count (== BYTEDANCE latent_channels).
SEEDVR2_COND_CHANNELS = 17         # conditioning channels = vid_in_channels(33) - latent(16).
SEEDVR2_DEFAULT_TEMPORAL_SIZE = 16 # default VAE temporal tile when unset.

# Color-correction memory model (fork tuning; per-frame VRAM estimate for chunk sizing)
SEEDVR2_COLOR_MEM_HEADROOM = 0.75  # fraction of free VRAM usable per color-correction chunk.
SEEDVR2_LAB_SCALE_MULTIPLIER = 13  # per-frame byte multiplier, LAB path.
SEEDVR2_WAVELET_SCALE_MULTIPLIER = 10  # per-frame byte multiplier, wavelet path.
SEEDVR2_ADAIN_SCALE_MULTIPLIER = 6     # per-frame byte multiplier, AdaIN path.

# --------------------------------------------------------------------------------------
# C. ByteDance config / source  (BYTEDANCE - cite ByteDance-Seed/SeedVR)
# --------------------------------------------------------------------------------------
BYTEDANCE_VAE_SCALING_FACTOR = 0.9152   # configs_3b/main.yaml:57 (scaling_factor); latent denorm.
BYTEDANCE_VAE_SHIFTING_FACTOR = 0.0     # infer.py (shifting_factor default); latent denorm shift.
BYTEDANCE_VAE_CONV_MEM_GIB = 0.5        # configs_3b/main.yaml:54 (conv_max_mem).
BYTEDANCE_VAE_NORM_MEM_GIB = 0.5        # configs_3b/main.yaml:55 (norm_max_mem).
BYTEDANCE_LOGVAR_CLAMP_MIN = -30.0      # video_vae_v3/modules/types.py:28.
BYTEDANCE_LOGVAR_CLAMP_MAX = 20.0       # video_vae_v3/modules/types.py:28.
BYTEDANCE_GN_CHUNKS_FP16 = 4            # causal_inflation_lib.py:351 (GroupNorm chunk count, fp16).
BYTEDANCE_GN_CHUNKS_FP32 = 2            # causal_inflation_lib.py:351 (GroupNorm chunk count, fp32).
BYTEDANCE_CONTIGUOUS_BATCH_THRESHOLD = 64  # attn_video_vae.py:308 (force .contiguous() above this b*t).
BYTEDANCE_BLOCK_OUT_CHANNELS = (128, 256, 512, 512)  # s8_c16_t4_inflation_sd3.yaml:7-11.
BYTEDANCE_SLICING_SAMPLE_MIN = 4        # s8_c16_t4_inflation_sd3.yaml:22 (slicing_sample_min_size).
BYTEDANCE_VAE_TEMPORAL_DOWNSAMPLE = 4   # infer.py:230 (temporal_downsample_factor); the 4n+1 factor.
BYTEDANCE_VAE_SPATIAL_DOWNSAMPLE = 8    # infer.py:231 (spatial_downsample_factor).
BYTEDANCE_SCHEDULE_T = 1000.0           # configs_3b/main.yaml:65 (schedule.T); timestep range.
BYTEDANCE_SPATIAL_DIVISOR = 16          # inference_seedvr2_3b.py:241 (DivisibleCrop((16,16))).
BYTEDANCE_720P_REF_AREA = 45 * 80       # dit_v2/window.py:32 (720p reference area for window scaling).
BYTEDANCE_MAX_TEMPORAL_WINDOW = 30      # dit_v2/window.py:35 (max temporal window frames).
BYTEDANCE_ROPE_MAX_FREQ = 256           # dit_v2/rope.py:31 (pixel-RoPE max frequency).
BYTEDANCE_SINUSOIDAL_DIM = 256          # dit_3b/nadit.py:120 (timestep sinusoidal embed dim).
# Resolution-dependent timestep-shift linear fits: (x1, y1, x2, y2) for get_lin_function.
BYTEDANCE_IMG_SHIFT_FIT = (256 * 256, 1.0, 1024 * 1024, 3.2)            # infer.py:242.
BYTEDANCE_VID_SHIFT_FIT = (256 * 256 * 37, 1.0, 1280 * 720 * 145, 5.0)  # infer.py:243.

# --------------------------------------------------------------------------------------
# D. Published standards (cite the literature)
# --------------------------------------------------------------------------------------
ROPE_THETA = 10000   # RoPE base; Su et al., "RoFormer", arXiv:2104.09864.

# CIELAB f(t) piecewise constants and D65 white point (CIE 15 colorimetry; CIE D65).
CIELAB_DELTA = 6.0 / 29.0          # CIE 15 (delta).
CIELAB_KAPPA = (29.0 / 3.0) ** 3   # CIE 15 (kappa).
D65_WHITE_X = 0.95047              # CIE D65 standard illuminant Xn (Yn = 1).
D65_WHITE_Z = 1.08883              # CIE D65 standard illuminant Zn.
WAVELET_DECOMP_LEVELS = 5          # wavelet color-fix decomposition depth (GIMP/Krita; StableSR).

# NOTE: the sRGB<->XYZ D65 3x3 matrices (IEC 61966-2-1) remain inline in the color code and
# are named (SRGB_TO_XYZ_D65 / XYZ_TO_SRGB_D65) during the color-module extraction, where the
# exact existing coefficients move verbatim rather than being retyped here.
