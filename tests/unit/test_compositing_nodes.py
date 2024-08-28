import torch

from comfy_extras.nodes.nodes_compositing import Posterize, EnhanceContrast


def test_posterize():
    posterize_node = Posterize()

    # Create a sample image
    sample_image = torch.rand((1, 64, 64, 3))

    # Test with different levels
    for levels in [2, 4, 8, 16]:
        result = posterize_node.posterize(sample_image, levels)
        assert isinstance(result[0], torch.Tensor)
        assert result[0].shape == sample_image.shape

        # Check if the unique values are within the expected range
        unique_values = torch.unique(result[0])
        assert len(unique_values) <= levels


def test_enhance_contrast():
    enhance_contrast_node = EnhanceContrast()

    # Create a sample image
    sample_image = torch.rand((1, 64, 64, 3))

    # Test Histogram Equalization
    result = enhance_contrast_node.enhance_contrast(sample_image, "Histogram Equalization", 0.03, 2.0, 98.0)
    assert isinstance(result[0], torch.Tensor)
    assert result[0].shape == sample_image.shape

    # Test Adaptive Equalization
    result = enhance_contrast_node.enhance_contrast(sample_image, "Adaptive Equalization", 0.05, 2.0, 98.0)
    assert isinstance(result[0], torch.Tensor)
    assert result[0].shape == sample_image.shape

    # Test Contrast Stretching
    result = enhance_contrast_node.enhance_contrast(sample_image, "Contrast Stretching", 0.03, 1.0, 99.0)
    assert isinstance(result[0], torch.Tensor)
    assert result[0].shape == sample_image.shape
