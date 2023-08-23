#From https://github.com/kornia/kornia
import math

import torch
import torch.nn.functional as F


def get_canny_nms_kernel(device=None, dtype=None):
    """Utility function that returns 3x3 kernels for the Canny Non-maximal suppression."""
    return torch.tensor(
        [
            [[[0.0, 0.0, 0.0], [0.0, 1.0, -1.0], [0.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]],
            [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]],
            [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]],
        ],
        device=device,
        dtype=dtype,
    )


def get_hysteresis_kernel(device=None, dtype=None):
    """Utility function that returns the 3x3 kernels for the Canny hysteresis."""
    return torch.tensor(
        [
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]],
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
        ],
        device=device,
        dtype=dtype,
    )

def gaussian_blur_2d(img, kernel_size, sigma):
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)

    pdf = torch.exp(-0.5 * (x / sigma).pow(2))

    x_kernel = pdf / pdf.sum()
    x_kernel = x_kernel.to(device=img.device, dtype=img.dtype)

    kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :])
    kernel2d = kernel2d.expand(img.shape[-3], 1, kernel2d.shape[0], kernel2d.shape[1])

    padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]

    img = torch.nn.functional.pad(img, padding, mode="reflect")
    img = torch.nn.functional.conv2d(img, kernel2d, groups=img.shape[-3])

    return img

def get_sobel_kernel2d(device=None, dtype=None):
    kernel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=device, dtype=dtype)
    kernel_y = kernel_x.transpose(0, 1)
    return torch.stack([kernel_x, kernel_y])

def spatial_gradient(input, normalized: bool = True):
    r"""Compute the first order image derivative in both x and y using a Sobel operator.
    .. image:: _static/img/spatial_gradient.png
    Args:
        input: input image tensor with shape :math:`(B, C, H, W)`.
        mode: derivatives modality, can be: `sobel` or `diff`.
        order: the order of the derivatives.
        normalized: whether the output is normalized.
    Return:
        the derivatives of the input feature map. with shape :math:`(B, C, 2, H, W)`.
    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       filtering_edges.html>`__.
    Examples:
        >>> input = torch.rand(1, 3, 4, 4)
        >>> output = spatial_gradient(input)  # 1x3x2x4x4
        >>> output.shape
        torch.Size([1, 3, 2, 4, 4])
    """
    # KORNIA_CHECK_IS_TENSOR(input)
    # KORNIA_CHECK_SHAPE(input, ['B', 'C', 'H', 'W'])

    # allocate kernel
    kernel = get_sobel_kernel2d(device=input.device, dtype=input.dtype)
    if normalized:
        kernel = normalize_kernel2d(kernel)

    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...]

    # Pad with "replicate for spatial dims, but with zeros for channel
    spatial_pad = [kernel.size(1) // 2, kernel.size(1) // 2, kernel.size(2) // 2, kernel.size(2) // 2]
    out_channels: int = 2
    padded_inp = torch.nn.functional.pad(input.reshape(b * c, 1, h, w), spatial_pad, 'replicate')
    out = F.conv2d(padded_inp, tmp_kernel, groups=1, padding=0, stride=1)
    return out.reshape(b, c, out_channels, h, w)

def rgb_to_grayscale(image, rgb_weights = None):
    r"""Convert a RGB image to grayscale version of image.

    .. image:: _static/img/rgb_to_grayscale.png

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: RGB image to be converted to grayscale with shape :math:`(*,3,H,W)`.
        rgb_weights: Weights that will be applied on each channel (RGB).
            The sum of the weights should add up to one.
    Returns:
        grayscale version of the image with shape :math:`(*,1,H,W)`.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       color_conversions.html>`__.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> gray = rgb_to_grayscale(input) # 2x1x4x5
    """

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    if rgb_weights is None:
        # 8 bit images
        if image.dtype == torch.uint8:
            rgb_weights = torch.tensor([76, 150, 29], device=image.device, dtype=torch.uint8)
        # floating point images
        elif image.dtype in (torch.float16, torch.float32, torch.float64):
            rgb_weights = torch.tensor([0.299, 0.587, 0.114], device=image.device, dtype=image.dtype)
        else:
            raise TypeError(f"Unknown data type: {image.dtype}")
    else:
        # is tensor that we make sure is in the same device/dtype
        rgb_weights = rgb_weights.to(image)

    # unpack the color image channels with RGB order
    r: Tensor = image[..., 0:1, :, :]
    g: Tensor = image[..., 1:2, :, :]
    b: Tensor = image[..., 2:3, :, :]

    w_r, w_g, w_b = rgb_weights.unbind()
    return w_r * r + w_g * g + w_b * b

def canny(
    input,
    low_threshold = 0.1,
    high_threshold = 0.2,
    kernel_size  = 5,
    sigma = 1,
    hysteresis = True,
    eps = 1e-6,
):
    r"""Find edges of the input image and filters them using the Canny algorithm.
    .. image:: _static/img/canny.png
    Args:
        input: input image tensor with shape :math:`(B,C,H,W)`.
        low_threshold: lower threshold for the hysteresis procedure.
        high_threshold: upper threshold for the hysteresis procedure.
        kernel_size: the size of the kernel for the gaussian blur.
        sigma: the standard deviation of the kernel for the gaussian blur.
        hysteresis: if True, applies the hysteresis edge tracking.
            Otherwise, the edges are divided between weak (0.5) and strong (1) edges.
        eps: regularization number to avoid NaN during backprop.
    Returns:
        - the canny edge magnitudes map, shape of :math:`(B,1,H,W)`.
        - the canny edge detection filtered by thresholds and hysteresis, shape of :math:`(B,1,H,W)`.
    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       canny.html>`__.
    Example:
        >>> input = torch.rand(5, 3, 4, 4)
        >>> magnitude, edges = canny(input)  # 5x3x4x4
        >>> magnitude.shape
        torch.Size([5, 1, 4, 4])
        >>> edges.shape
        torch.Size([5, 1, 4, 4])
    """
    # KORNIA_CHECK_IS_TENSOR(input)
    # KORNIA_CHECK_SHAPE(input, ['B', 'C', 'H', 'W'])
    # KORNIA_CHECK(
    #     low_threshold <= high_threshold,
    #     "Invalid input thresholds. low_threshold should be smaller than the high_threshold. Got: "
    #     f"{low_threshold}>{high_threshold}",
    # )
    # KORNIA_CHECK(0 < low_threshold < 1, f'Invalid low threshold. Should be in range (0, 1). Got: {low_threshold}')
    # KORNIA_CHECK(0 < high_threshold < 1, f'Invalid high threshold. Should be in range (0, 1). Got: {high_threshold}')

    device = input.device
    dtype = input.dtype

    # To Grayscale
    if input.shape[1] == 3:
        input = rgb_to_grayscale(input)

    # Gaussian filter
    blurred: Tensor = gaussian_blur_2d(input, kernel_size, sigma)

    # Compute the gradients
    gradients: Tensor = spatial_gradient(blurred, normalized=False)

    # Unpack the edges
    gx: Tensor = gradients[:, :, 0]
    gy: Tensor = gradients[:, :, 1]

    # Compute gradient magnitude and angle
    magnitude: Tensor = torch.sqrt(gx * gx + gy * gy + eps)
    angle: Tensor = torch.atan2(gy, gx)

    # Radians to Degrees
    angle = 180.0 * angle / math.pi

    # Round angle to the nearest 45 degree
    angle = torch.round(angle / 45) * 45

    # Non-maximal suppression
    nms_kernels: Tensor = get_canny_nms_kernel(device, dtype)
    nms_magnitude: Tensor = F.conv2d(magnitude, nms_kernels, padding=nms_kernels.shape[-1] // 2)

    # Get the indices for both directions
    positive_idx: Tensor = (angle / 45) % 8
    positive_idx = positive_idx.long()

    negative_idx: Tensor = ((angle / 45) + 4) % 8
    negative_idx = negative_idx.long()

    # Apply the non-maximum suppression to the different directions
    channel_select_filtered_positive: Tensor = torch.gather(nms_magnitude, 1, positive_idx)
    channel_select_filtered_negative: Tensor = torch.gather(nms_magnitude, 1, negative_idx)

    channel_select_filtered: Tensor = torch.stack(
        [channel_select_filtered_positive, channel_select_filtered_negative], 1
    )

    is_max: Tensor = channel_select_filtered.min(dim=1)[0] > 0.0

    magnitude = magnitude * is_max

    # Threshold
    edges: Tensor = F.threshold(magnitude, low_threshold, 0.0)

    low: Tensor = magnitude > low_threshold
    high: Tensor = magnitude > high_threshold

    edges = low * 0.5 + high * 0.5
    edges = edges.to(dtype)

    # Hysteresis
    if hysteresis:
        edges_old: Tensor = -torch.ones(edges.shape, device=edges.device, dtype=dtype)
        hysteresis_kernels: Tensor = get_hysteresis_kernel(device, dtype)

        while ((edges_old - edges).abs() != 0).any():
            weak: Tensor = (edges == 0.5).float()
            strong: Tensor = (edges == 1).float()

            hysteresis_magnitude: Tensor = F.conv2d(
                edges, hysteresis_kernels, padding=hysteresis_kernels.shape[-1] // 2
            )
            hysteresis_magnitude = (hysteresis_magnitude == 1).any(1, keepdim=True).to(dtype)
            hysteresis_magnitude = hysteresis_magnitude * weak + strong

            edges_old = edges.clone()
            edges = hysteresis_magnitude + (hysteresis_magnitude == 0) * weak * 0.5

        edges = hysteresis_magnitude

    return magnitude, edges


class Canny:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",),
                                "low_threshold": ("FLOAT", {"default": 0.4, "min": 0.01, "max": 0.99, "step": 0.01}),
                                "high_threshold": ("FLOAT", {"default": 0.8, "min": 0.01, "max": 0.99, "step": 0.01})
                                }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "detect_edge"

    CATEGORY = "image/preprocessors"

    def detect_edge(self, image, low_threshold, high_threshold):
        output = canny(image.movedim(-1, 1), low_threshold, high_threshold)
        img_out = output[1].repeat(1, 3, 1, 1).movedim(1, -1)
        return (img_out,)

NODE_CLASS_MAPPINGS = {
    "Canny": Canny,
}
