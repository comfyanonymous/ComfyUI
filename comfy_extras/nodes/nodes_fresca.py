# Code based on https://github.com/WikiChao/FreSca (MIT License)
import torch
import torch.fft as fft


def Fourier_filter(x, scale_low=1.0, scale_high=1.5, freq_cutoff=20):
    """
    Apply frequency-dependent scaling to an image tensor using Fourier transforms.

    Parameters:
        x:           Input tensor of shape (B, C, H, W)
        scale_low:   Scaling factor for low-frequency components (default: 1.0)
        scale_high:  Scaling factor for high-frequency components (default: 1.5)
        freq_cutoff: Number of frequency indices around center to consider as low-frequency (default: 20)

    Returns:
        x_filtered: Filtered version of x in spatial domain with frequency-specific scaling applied.
    """
    # Preserve input dtype and device
    dtype, device = x.dtype, x.device

    # Convert to float32 for FFT computations
    x = x.to(torch.float32)

    # 1) Apply FFT and shift low frequencies to center
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))

    # Initialize mask with high-frequency scaling factor
    mask = torch.ones(x_freq.shape, device=device) * scale_high
    m = mask
    for d in range(len(x_freq.shape) - 2):
        dim = d + 2
        cc = x_freq.shape[dim] // 2
        f_c = min(freq_cutoff, cc)
        m = m.narrow(dim, cc - f_c, f_c * 2)

    # Apply low-frequency scaling factor to center region
    m[:] = scale_low

    # 3) Apply frequency-specific scaling
    x_freq = x_freq * mask

    # 4) Convert back to spatial domain
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real

    # 5) Restore original dtype
    x_filtered = x_filtered.to(dtype)

    return x_filtered


class FreSca:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "scale_low": ("FLOAT", {"default": 1.0, "min": 0, "max": 10, "step": 0.01,
                                        "tooltip": "Scaling factor for low-frequency components"}),
                "scale_high": ("FLOAT", {"default": 1.25, "min": 0, "max": 10, "step": 0.01,
                                        "tooltip": "Scaling factor for high-frequency components"}),
                "freq_cutoff": ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1,
                                        "tooltip": "Number of frequency indices around center to consider as low-frequency"}),
            }
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "_for_testing"
    DESCRIPTION = "Applies frequency-dependent scaling to the guidance"
    def patch(self, model, scale_low, scale_high, freq_cutoff):
        def custom_cfg_function(args):
            cond = args["conds_out"][0]
            uncond = args["conds_out"][1]

            guidance = cond - uncond
            filtered_guidance = Fourier_filter(
                guidance,
                scale_low=scale_low,
                scale_high=scale_high,
                freq_cutoff=freq_cutoff,
            )
            filtered_cond = filtered_guidance + uncond

            return [filtered_cond, uncond]

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(custom_cfg_function)

        return (m,)


NODE_CLASS_MAPPINGS = {
    "FreSca": FreSca,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FreSca": "FreSca",
}
