# Adapted from https://github.com/lodestone-rock/flow
from functools import lru_cache

import torch
from torch import nn

from comfy.ldm.flux.layers import RMSNorm


class NerfEmbedder(nn.Module):
    """
    An embedder module that combines input features with a 2D positional
    encoding that mimics the Discrete Cosine Transform (DCT).

    This module takes an input tensor of shape (B, P^2, C), where P is the
    patch size, and enriches it with positional information before projecting
    it to a new hidden size.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_size_input: int,
        max_freqs: int,
        dtype=None,
        device=None,
        operations=None,
    ):
        """
        Initializes the NerfEmbedder.

        Args:
            in_channels (int): The number of channels in the input tensor.
            hidden_size_input (int): The desired dimension of the output embedding.
            max_freqs (int): The number of frequency components to use for both
                             the x and y dimensions of the positional encoding.
                             The total number of positional features will be max_freqs^2.
        """
        super().__init__()
        self.dtype = dtype
        self.max_freqs = max_freqs
        self.hidden_size_input = hidden_size_input

        # A linear layer to project the concatenated input features and
        # positional encodings to the final output dimension.
        self.embedder = nn.Sequential(
            operations.Linear(in_channels + max_freqs**2, hidden_size_input, dtype=dtype, device=device)
        )

    @lru_cache(maxsize=4)
    def fetch_pos(self, patch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Generates and caches 2D DCT-like positional embeddings for a given patch size.

        The LRU cache is a performance optimization that avoids recomputing the
        same positional grid on every forward pass.

        Args:
            patch_size (int): The side length of the square input patch.
            device: The torch device to create the tensors on.
            dtype: The torch dtype for the tensors.

        Returns:
            A tensor of shape (1, patch_size^2, max_freqs^2) containing the
            positional embeddings.
        """
        # Create normalized 1D coordinate grids from 0 to 1.
        pos_x = torch.linspace(0, 1, patch_size, device=device, dtype=dtype)
        pos_y = torch.linspace(0, 1, patch_size, device=device, dtype=dtype)

        # Create a 2D meshgrid of coordinates.
        pos_y, pos_x = torch.meshgrid(pos_y, pos_x, indexing="ij")

        # Reshape positions to be broadcastable with frequencies.
        # Shape becomes (patch_size^2, 1, 1).
        pos_x = pos_x.reshape(-1, 1, 1)
        pos_y = pos_y.reshape(-1, 1, 1)

        # Create a 1D tensor of frequency values from 0 to max_freqs-1.
        freqs = torch.linspace(0, self.max_freqs - 1, self.max_freqs, dtype=dtype, device=device)

        # Reshape frequencies to be broadcastable for creating 2D basis functions.
        # freqs_x shape: (1, max_freqs, 1)
        # freqs_y shape: (1, 1, max_freqs)
        freqs_x = freqs[None, :, None]
        freqs_y = freqs[None, None, :]

        # A custom weighting coefficient, not part of standard DCT.
        # This seems to down-weight the contribution of higher-frequency interactions.
        coeffs = (1 + freqs_x * freqs_y) ** -1

        # Calculate the 1D cosine basis functions for x and y coordinates.
        # This is the core of the DCT formulation.
        dct_x = torch.cos(pos_x * freqs_x * torch.pi)
        dct_y = torch.cos(pos_y * freqs_y * torch.pi)

        # Combine the 1D basis functions to create 2D basis functions by element-wise
        # multiplication, and apply the custom coefficients. Broadcasting handles the
        # combination of all (pos_x, freqs_x) with all (pos_y, freqs_y).
        # The result is flattened into a feature vector for each position.
        dct = (dct_x * dct_y * coeffs).view(1, -1, self.max_freqs ** 2)

        return dct

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the embedder.

        Args:
            inputs (Tensor): The input tensor of shape (B, P^2, C).

        Returns:
            Tensor: The output tensor of shape (B, P^2, hidden_size_input).
        """
        # Get the batch size, number of pixels, and number of channels.
        B, P2, C = inputs.shape

        # Infer the patch side length from the number of pixels (P^2).
        patch_size = int(P2 ** 0.5)

        input_dtype = inputs.dtype
        inputs = inputs.to(dtype=self.dtype)

        # Fetch the pre-computed or cached positional embeddings.
        dct = self.fetch_pos(patch_size, inputs.device, self.dtype)

        # Repeat the positional embeddings for each item in the batch.
        dct = dct.repeat(B, 1, 1)

        # Concatenate the original input features with the positional embeddings
        # along the feature dimension.
        inputs = torch.cat((inputs, dct), dim=-1)

        # Project the combined tensor to the target hidden size.
        return self.embedder(inputs).to(dtype=input_dtype)


class NerfGLUBlock(nn.Module):
    """
    A NerfBlock using a Gated Linear Unit (GLU) like MLP.
    """
    def __init__(self, hidden_size_s: int, hidden_size_x: int, mlp_ratio, dtype=None, device=None, operations=None):
        super().__init__()
        # The total number of parameters for the MLP is increased to accommodate
        # the gate, value, and output projection matrices.
        # We now need to generate parameters for 3 matrices.
        total_params = 3 * hidden_size_x**2 * mlp_ratio
        self.param_generator = operations.Linear(hidden_size_s, total_params, dtype=dtype, device=device)
        self.norm = RMSNorm(hidden_size_x, dtype=dtype, device=device, operations=operations)
        self.mlp_ratio = mlp_ratio


    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        batch_size, num_x, hidden_size_x = x.shape
        mlp_params = self.param_generator(s)

        # Split the generated parameters into three parts for the gate, value, and output projection.
        fc1_gate_params, fc1_value_params, fc2_params = mlp_params.chunk(3, dim=-1)

        # Reshape the parameters into matrices for batch matrix multiplication.
        fc1_gate = fc1_gate_params.view(batch_size, hidden_size_x, hidden_size_x * self.mlp_ratio)
        fc1_value = fc1_value_params.view(batch_size, hidden_size_x, hidden_size_x * self.mlp_ratio)
        fc2 = fc2_params.view(batch_size, hidden_size_x * self.mlp_ratio, hidden_size_x)

        # Normalize the generated weight matrices as in the original implementation.
        fc1_gate = torch.nn.functional.normalize(fc1_gate, dim=-2)
        fc1_value = torch.nn.functional.normalize(fc1_value, dim=-2)
        fc2 = torch.nn.functional.normalize(fc2, dim=-2)

        res_x = x
        x = self.norm(x)

        # Apply the final output projection.
        x = torch.bmm(torch.nn.functional.silu(torch.bmm(x, fc1_gate)) * torch.bmm(x, fc1_value), fc2)

        return x + res_x


class NerfFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm = RMSNorm(hidden_size, dtype=dtype, device=device, operations=operations)
        self.linear = operations.Linear(hidden_size, out_channels, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMSNorm normalizes over the last dimension, but our channel dim (C) is at dim=1.
        # So we temporarily move the channel dimension to the end for the norm operation.
        return self.linear(self.norm(x.movedim(1, -1))).movedim(-1, 1)


class NerfFinalLayerConv(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm = RMSNorm(hidden_size, dtype=dtype, device=device, operations=operations)
        self.conv = operations.Conv2d(
            in_channels=hidden_size,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            dtype=dtype,
            device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMSNorm normalizes over the last dimension, but our channel dim (C) is at dim=1.
        # So we temporarily move the channel dimension to the end for the norm operation.
        return self.conv(self.norm(x.movedim(1, -1)).movedim(-1, 1))
