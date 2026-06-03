import torch
import torch.nn.functional as F
from torch import Tensor

from comfy.ldm.seedvr.model import safe_pad_operation
from comfy.ldm.seedvr.vae import safe_interpolate_operation
from comfy.ldm.seedvr.constants import (
    CIELAB_DELTA,
    CIELAB_KAPPA,
    D65_WHITE_X,
    D65_WHITE_Z,
    WAVELET_DECOMP_LEVELS,
)


def wavelet_blur(image: Tensor, radius):
    max_safe_radius = max(1, min(image.shape[-2:]) // 8)
    if radius > max_safe_radius:
        radius = max_safe_radius

    num_channels = image.shape[1]

    kernel_vals = [
        [0.0625, 0.125, 0.0625],
        [0.125,  0.25,  0.125],
        [0.0625, 0.125, 0.0625],
    ]
    kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device)
    kernel = kernel[None, None].repeat(num_channels, 1, 1, 1)

    image = safe_pad_operation(image, (radius, radius, radius, radius), mode='replicate')
    output = F.conv2d(image, kernel, groups=num_channels, dilation=radius)

    return output

def wavelet_decomposition(image: Tensor, levels: int = WAVELET_DECOMP_LEVELS):
    high_freq = torch.zeros_like(image)

    for i in range(levels):
        radius = 2 ** i
        low_freq = wavelet_blur(image, radius)
        high_freq.add_(image).sub_(low_freq)
        image = low_freq

    return high_freq, low_freq

def wavelet_reconstruction(content_feat: Tensor, style_feat: Tensor) -> Tensor:

    if content_feat.shape != style_feat.shape:
        # Resize style to match content spatial dimensions
        if len(content_feat.shape) >= 3:
            # safe_interpolate_operation handles FP16 conversion automatically
            style_feat = safe_interpolate_operation(
                style_feat,
                size=content_feat.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

    # Decompose both features into frequency components
    content_high_freq, content_low_freq = wavelet_decomposition(content_feat)
    del content_low_freq  # Free memory immediately

    style_high_freq, style_low_freq = wavelet_decomposition(style_feat)
    del style_high_freq  # Free memory immediately

    if content_high_freq.shape != style_low_freq.shape:
        style_low_freq = safe_interpolate_operation(
            style_low_freq,
            size=content_high_freq.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

    content_high_freq.add_(style_low_freq)

    return content_high_freq.clamp_(-1.0, 1.0)

def _histogram_matching_channel(source: Tensor, reference: Tensor, device: torch.device) -> Tensor:
    original_shape = source.shape

    # Flatten
    source_flat = source.flatten()
    reference_flat = reference.flatten()

    # Sort both arrays
    source_sorted, source_indices = torch.sort(source_flat)
    reference_sorted, _ = torch.sort(reference_flat)
    del reference_flat

    # Quantile mapping
    n_source = len(source_sorted)
    n_reference = len(reference_sorted)

    if n_source == n_reference:
        matched_sorted = reference_sorted
    else:
        # Interpolate reference to match source quantiles
        source_quantiles = torch.linspace(0, 1, n_source, device=device)
        ref_indices = (source_quantiles * (n_reference - 1)).long()
        ref_indices.clamp_(0, n_reference - 1)
        matched_sorted = reference_sorted[ref_indices]
        del source_quantiles, ref_indices, reference_sorted

    del source_sorted, source_flat

    # Reconstruct using argsort (portable across CUDA/ROCm/MPS)
    inverse_indices = torch.argsort(source_indices)
    del source_indices
    matched_flat = matched_sorted[inverse_indices]
    del matched_sorted, inverse_indices

    return matched_flat.reshape(original_shape)

def _lab_to_rgb_batch(lab: Tensor, device: torch.device, matrix_inv: Tensor, epsilon: float, kappa: float) -> Tensor:
    """Convert batch of CIELAB images to RGB color space."""
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]

    # LAB to XYZ
    fy = (L + 16.0) / 116.0
    fx = a.div(500.0).add_(fy)
    fz = fy - b / 200.0
    del L, a, b

    # XYZ transformation
    x = torch.where(
        fx > epsilon,
        torch.pow(fx, 3.0),
        fx.mul(116.0).sub_(16.0).div_(kappa)
    )
    y = torch.where(
        fy > epsilon,
        torch.pow(fy, 3.0),
        fy.mul(116.0).sub_(16.0).div_(kappa)
    )
    z = torch.where(
        fz > epsilon,
        torch.pow(fz, 3.0),
        fz.mul(116.0).sub_(16.0).div_(kappa)
    )
    del fx, fy, fz

    # Apply D65 white point (in-place)
    x.mul_(D65_WHITE_X)
    # y *= 1.00000  # (no-op, skip)
    z.mul_(D65_WHITE_Z)

    xyz = torch.stack([x, y, z], dim=1)
    del x, y, z

    # Matrix multiplication: XYZ -> RGB
    B, C, H, W = xyz.shape
    xyz_flat = xyz.permute(0, 2, 3, 1).reshape(-1, 3)
    del xyz

    # Ensure dtype consistency for matrix multiplication
    xyz_flat = xyz_flat.to(dtype=matrix_inv.dtype)
    rgb_linear_flat = torch.matmul(xyz_flat, matrix_inv.T)
    del xyz_flat

    rgb_linear = rgb_linear_flat.reshape(B, H, W, 3).permute(0, 3, 1, 2)
    del rgb_linear_flat

    # Apply inverse gamma correction (delinearize)
    mask = rgb_linear > 0.0031308
    rgb = torch.where(
        mask,
        torch.pow(torch.clamp(rgb_linear, min=0.0), 1.0 / 2.4).mul_(1.055).sub_(0.055),
        rgb_linear * 12.92
    )
    del mask, rgb_linear

    return torch.clamp(rgb, 0.0, 1.0)

def _rgb_to_lab_batch(rgb: Tensor, device: torch.device, matrix: Tensor, epsilon: float, kappa: float) -> Tensor:
    """Convert batch of RGB images to CIELAB color space using D65 illuminant."""
    # Apply sRGB gamma correction (linearize)
    mask = rgb > 0.04045
    rgb_linear = torch.where(
        mask,
        torch.pow((rgb + 0.055) / 1.055, 2.4),
        rgb / 12.92
    )
    del mask

    # Matrix multiplication: RGB -> XYZ
    B, C, H, W = rgb_linear.shape
    rgb_flat = rgb_linear.permute(0, 2, 3, 1).reshape(-1, 3)
    del rgb_linear

    # Ensure dtype consistency for matrix multiplication
    rgb_flat = rgb_flat.to(dtype=matrix.dtype)
    xyz_flat = torch.matmul(rgb_flat, matrix.T)
    del rgb_flat

    xyz = xyz_flat.reshape(B, H, W, 3).permute(0, 3, 1, 2)
    del xyz_flat

    # Normalize by D65 white point (in-place)
    xyz[:, 0].div_(D65_WHITE_X)  # X
    # xyz[:, 1] /= 1.00000  # Y (no-op, skip)
    xyz[:, 2].div_(D65_WHITE_Z)  # Z

    # XYZ to LAB transformation
    epsilon_cubed = epsilon ** 3
    mask = xyz > epsilon_cubed
    f_xyz = torch.where(
        mask,
        torch.pow(xyz, 1.0 / 3.0),
        xyz.mul(kappa).add_(16.0).div_(116.0)
    )
    del xyz, mask

    # Extract channels and compute LAB
    L = f_xyz[:, 1].mul(116.0).sub_(16.0)  # Lightness [0, 100]
    a = (f_xyz[:, 0] - f_xyz[:, 1]).mul_(500.0)  # Green-Red [-128, 127]
    b = (f_xyz[:, 1] - f_xyz[:, 2]).mul_(200.0)  # Blue-Yellow [-128, 127]
    del f_xyz

    return torch.stack([L, a, b], dim=1)

def lab_color_transfer(
    content_feat: Tensor,
    style_feat: Tensor,
    luminance_weight: float = 0.8
) -> Tensor:
    content_feat = wavelet_reconstruction(content_feat, style_feat)

    if content_feat.shape != style_feat.shape:
        style_feat = safe_interpolate_operation(
            style_feat,
            size=content_feat.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

    device = content_feat.device

    def ensure_float32_precision(c):
        orig_dtype = c.dtype
        c = c.float()
        return c, orig_dtype
    content_feat, original_dtype = ensure_float32_precision(content_feat)
    style_feat, _ = ensure_float32_precision(style_feat)

    rgb_to_xyz_matrix = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], dtype=torch.float32, device=device)

    xyz_to_rgb_matrix = torch.tensor([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252]
    ], dtype=torch.float32, device=device)

    epsilon = CIELAB_DELTA
    kappa = CIELAB_KAPPA

    content_feat.add_(1.0).mul_(0.5).clamp_(0.0, 1.0)
    style_feat.add_(1.0).mul_(0.5).clamp_(0.0, 1.0)

    # Convert to LAB color space
    content_lab = _rgb_to_lab_batch(content_feat, device, rgb_to_xyz_matrix, epsilon, kappa)
    del content_feat

    style_lab = _rgb_to_lab_batch(style_feat, device, rgb_to_xyz_matrix, epsilon, kappa)
    del style_feat, rgb_to_xyz_matrix

    # Match chrominance channels (a*, b*) for accurate color transfer
    matched_a = _histogram_matching_channel(content_lab[:, 1], style_lab[:, 1], device)
    matched_b = _histogram_matching_channel(content_lab[:, 2], style_lab[:, 2], device)

    # Handle luminance with weighted blending
    if luminance_weight < 1.0:
        # Partially match luminance for better overall color accuracy
        matched_L = _histogram_matching_channel(content_lab[:, 0], style_lab[:, 0], device)
        # Blend: preserve some content L* for detail, adopt some style L* for color
        result_L = content_lab[:, 0].mul(luminance_weight).add_(matched_L.mul(1.0 - luminance_weight))
        del matched_L
    else:
        # Fully preserve content luminance
        result_L = content_lab[:, 0]

    del content_lab, style_lab

    # Reconstruct LAB with corrected channels
    result_lab = torch.stack([result_L, matched_a, matched_b], dim=1)
    del result_L, matched_a, matched_b

    # Convert back to RGB
    result_rgb = _lab_to_rgb_batch(result_lab, device, xyz_to_rgb_matrix, epsilon, kappa)
    del result_lab, xyz_to_rgb_matrix

    # Convert back to [-1, 1] range (in-place)
    result = result_rgb.mul_(2.0).sub_(1.0)
    del result_rgb

    result = result.to(original_dtype)

    return result


def wavelet_color_transfer(content_feat: Tensor, style_feat: Tensor) -> Tensor:
    return wavelet_reconstruction(content_feat, style_feat)


def adain_color_transfer(content_feat: Tensor, style_feat: Tensor, eps: float = 1e-5) -> Tensor:
    if content_feat.shape != style_feat.shape:
        style_feat = safe_interpolate_operation(
            style_feat,
            size=content_feat.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )

    original_dtype = content_feat.dtype
    content_feat = content_feat.float()
    style_feat = style_feat.float()

    b, c = content_feat.shape[:2]
    content_flat = content_feat.reshape(b, c, -1)
    style_flat = style_feat.reshape(b, c, -1)

    content_mean = content_flat.mean(dim=2).reshape(b, c, 1, 1)
    content_std = (content_flat.var(dim=2, correction=0) + eps).sqrt().reshape(b, c, 1, 1)
    style_mean = style_flat.mean(dim=2).reshape(b, c, 1, 1)
    style_std = (style_flat.var(dim=2, correction=0) + eps).sqrt().reshape(b, c, 1, 1)
    del content_flat, style_flat

    normalized = (content_feat - content_mean) / content_std
    del content_mean, content_std
    result = normalized * style_std + style_mean
    del normalized, style_mean, style_std

    result = result.clamp_(-1.0, 1.0)
    if result.dtype != original_dtype:
        result = result.to(original_dtype)
    return result
