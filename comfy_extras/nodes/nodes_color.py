import colour
import numpy as np
import torch
from scipy.interpolate import interpn
from scipy.interpolate.interpnd import LinearNDInterpolator
from scipy.ndimage import gaussian_filter
from scipy.optimize._lsap import linear_sum_assignment
from sklearn.cluster import KMeans

import comfy.model_management
from comfy.nodes.package_typing import CustomNode


class ColorPaletteExtractor(CustomNode):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "num_colors": ("INT", {"default": 8, "min": 2, "max": 64, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "PALETTE")
    FUNCTION = "extract_palette"
    CATEGORY = "image/color"

    def extract_palette(self, image, num_colors):
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        image_np = image.squeeze().cpu().numpy()

        pixels = image_np.reshape(-1, 3)

        kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)

        colors = kmeans.cluster_centers_

        _, counts = np.unique(kmeans.labels_, return_counts=True)
        sorted_indices = np.argsort(counts)[::-1]
        colors = colors[sorted_indices]

        palette_height = 512
        palette_width = 512
        palette_image = np.zeros((palette_height, palette_width * num_colors, 3), dtype=np.uint8)
        for i, color in enumerate(colors):
            palette_image[:, i * palette_width:(i + 1) * palette_width] = color

        palette_tensor = torch.from_numpy(palette_image).float() / 255.0
        palette_tensor = palette_tensor.unsqueeze(0).to(comfy.model_management.get_torch_device())

        color_array = (colors * 255).astype(np.uint8)

        return palette_tensor, color_array


class ImageBasedColorRemap(CustomNode):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "reference_image": ("IMAGE",),
                "palette_size": ("INT", {"default": 8, "min": 2, "max": 64, "step": 1}),
                "lut_size": ("INT", {"default": 33, "min": 8, "max": 64, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "remap_colors"
    CATEGORY = "image/color"

    def remap_colors(self, image, reference_image, palette_size, lut_size):
        # Ensure images are in the correct format (B, H, W, C)
        image = image.squeeze(0) if image.ndim == 4 else image
        reference_image = reference_image.squeeze(0) if reference_image.ndim == 4 else reference_image

        # Convert torch tensors to numpy arrays
        image_np = image.cpu().numpy()
        reference_np = reference_image.cpu().numpy()

        # Extract palettes using k-means clustering
        image_palette = self.extract_palette(image_np, palette_size)
        reference_palette = self.extract_palette(reference_np, palette_size)

        # Convert palettes to LAB color space
        image_palette_lab = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(image_palette))
        reference_palette_lab = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(reference_palette))

        # Align palettes
        aligned_image_palette_lab, aligned_reference_palette_lab = self.align_palettes(image_palette_lab, reference_palette_lab)

        # Create 3D LUT in LAB space
        lut = self.create_color_remap_3dlut(aligned_image_palette_lab, aligned_reference_palette_lab, size=lut_size)

        # Apply 3D LUT
        image_lab = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(image_np))
        remapped_lab = self.apply_lut(image_lab, lut, lut_size)
        remapped_rgb = colour.XYZ_to_sRGB(colour.Lab_to_XYZ(remapped_lab))

        # Clip values to ensure they're in the valid range
        remapped_rgb = np.clip(remapped_rgb, 0, 1)

        # Convert back to torch tensor
        remapped_tensor = torch.from_numpy(remapped_rgb).float()
        remapped_tensor = remapped_tensor.unsqueeze(0)  # Add batch dimension

        return (remapped_tensor,)

    def extract_palette(self, image, palette_size):
        pixels = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=palette_size, random_state=42, n_init=10)
        kmeans.fit(pixels)
        return kmeans.cluster_centers_

    def align_palettes(self, palette1, palette2):
        distances = np.linalg.norm(palette1[:, np.newaxis] - palette2, axis=2)
        row_ind, col_ind = linear_sum_assignment(distances)
        return palette1, palette2[col_ind]

    def create_color_remap_3dlut(self, original_colors, target_colors, size=33):
        # Create a regular grid in LAB space
        L = np.linspace(0, 100, size)
        a = np.linspace(-128, 127, size)
        b = np.linspace(-128, 127, size)
        grid = np.meshgrid(L, a, b, indexing='ij')

        # Reshape the grid for KNN
        grid_points = np.vstack([g.ravel() for g in grid]).T

        # Use KNN to find the nearest neighbor for each grid point
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(original_colors)
        _, indices = nn.kneighbors(grid_points)

        # Create the LUT using the target colors
        lut = target_colors[indices.ravel()]
        lut = lut.reshape(size, size, size, 3)

        # Apply Gaussian smoothing to create a more continuous mapping
        for i in range(3):
            lut[:,:,:,i] = gaussian_filter(lut[:,:,:,i], sigma=1)

        return lut

    def apply_lut(self, image_lab, lut, lut_size):
        points = (
            np.linspace(0, 100, lut_size),  # L
            np.linspace(-128, 127, lut_size),  # a
            np.linspace(-128, 127, lut_size)  # b
        )

        xi = np.stack([
            image_lab[..., 0],
            image_lab[..., 1],
            image_lab[..., 2]
        ], axis=-1)

        remapped_lab = np.zeros_like(image_lab)
        for i in range(3):
            remapped_lab[..., i] = interpn(
                points,
                lut[..., i],
                xi,
                method='linear',
                bounds_error=False,
                fill_value=None
            )

        return remapped_lab


NODE_CLASS_MAPPINGS = {
    "ColorPaletteExtractor": ColorPaletteExtractor,
    "ImageBasedColorRemap": ImageBasedColorRemap,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorPaletteExtractor": "Extract Color Palette",
    "ImageBasedColorRemap": "Image-Based Color Remap",

}
