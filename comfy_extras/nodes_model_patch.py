import torch
from torch import nn
import folder_paths
import comfy.utils
import comfy.ops
import comfy.model_management
import comfy.ldm.common_dit
import comfy.latent_formats


class BlockWiseControlBlock(torch.nn.Module):
    # [linear, gelu, linear]
    def __init__(self, dim: int = 3072, device=None, dtype=None, operations=None):
        super().__init__()
        self.x_rms = operations.RMSNorm(dim, eps=1e-6)
        self.y_rms = operations.RMSNorm(dim, eps=1e-6)
        self.input_proj = operations.Linear(dim, dim)
        self.act = torch.nn.GELU()
        self.output_proj = operations.Linear(dim, dim)

    def forward(self, x, y):
        x, y = self.x_rms(x), self.y_rms(y)
        x = self.input_proj(x + y)
        x = self.act(x)
        x = self.output_proj(x)
        return x


class QwenImageBlockWiseControlNet(torch.nn.Module):
    def __init__(
        self,
        num_layers: int = 60,
        in_dim: int = 64,
        additional_in_dim: int = 0,
        dim: int = 3072,
        device=None, dtype=None, operations=None
    ):
        super().__init__()
        self.additional_in_dim = additional_in_dim
        self.img_in = operations.Linear(in_dim + additional_in_dim, dim, device=device, dtype=dtype)
        self.controlnet_blocks = torch.nn.ModuleList(
            [
                BlockWiseControlBlock(dim, device=device, dtype=dtype, operations=operations)
                for _ in range(num_layers)
            ]
        )

    def process_input_latent_image(self, latent_image):
        latent_image[:, :16] = comfy.latent_formats.Wan21().process_in(latent_image[:, :16])
        patch_size = 2
        hidden_states = comfy.ldm.common_dit.pad_to_patch_size(latent_image, (1, patch_size, patch_size))
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(orig_shape[0], orig_shape[1], orig_shape[-2] // 2, 2, orig_shape[-1] // 2, 2)
        hidden_states = hidden_states.permute(0, 2, 4, 1, 3, 5)
        hidden_states = hidden_states.reshape(orig_shape[0], (orig_shape[-2] // 2) * (orig_shape[-1] // 2), orig_shape[1] * 4)
        return self.img_in(hidden_states)

    def control_block(self, img, controlnet_conditioning, block_id):
        return self.controlnet_blocks[block_id](img, controlnet_conditioning)


class SigLIPMultiFeatProjModel(torch.nn.Module):
    """
    SigLIP Multi-Feature Projection Model for processing style features from different layers
    and projecting them into a unified hidden space.

    Args:
        siglip_token_nums (int): Number of SigLIP tokens, default 257
        style_token_nums (int): Number of style tokens, default 256
        siglip_token_dims (int): Dimension of SigLIP tokens, default 1536
        hidden_size (int): Hidden layer size, default 3072
        context_layer_norm (bool): Whether to use context layer normalization, default False
    """

    def __init__(
        self,
        siglip_token_nums: int = 729,
        style_token_nums: int = 64,
        siglip_token_dims: int = 1152,
        hidden_size: int = 3072,
        context_layer_norm: bool = True,
        device=None, dtype=None, operations=None
    ):
        super().__init__()

        # High-level feature processing (layer -2)
        self.high_embedding_linear = nn.Sequential(
            operations.Linear(siglip_token_nums, style_token_nums),
            nn.SiLU()
        )
        self.high_layer_norm = (
            operations.LayerNorm(siglip_token_dims) if context_layer_norm else nn.Identity()
        )
        self.high_projection = operations.Linear(siglip_token_dims, hidden_size, bias=True)

        # Mid-level feature processing (layer -11)
        self.mid_embedding_linear = nn.Sequential(
            operations.Linear(siglip_token_nums, style_token_nums),
            nn.SiLU()
        )
        self.mid_layer_norm = (
            operations.LayerNorm(siglip_token_dims) if context_layer_norm else nn.Identity()
        )
        self.mid_projection = operations.Linear(siglip_token_dims, hidden_size, bias=True)

        # Low-level feature processing (layer -20)
        self.low_embedding_linear = nn.Sequential(
            operations.Linear(siglip_token_nums, style_token_nums),
            nn.SiLU()
        )
        self.low_layer_norm = (
            operations.LayerNorm(siglip_token_dims) if context_layer_norm else nn.Identity()
        )
        self.low_projection = operations.Linear(siglip_token_dims, hidden_size, bias=True)

    def forward(self, siglip_outputs):
        """
        Forward pass function

        Args:
            siglip_outputs: Output from SigLIP model, containing hidden_states

        Returns:
            torch.Tensor: Concatenated multi-layer features with shape [bs, 3*style_token_nums, hidden_size]
        """
        dtype = next(self.high_embedding_linear.parameters()).dtype

        # Process high-level features (layer -2)
        high_embedding = self._process_layer_features(
            siglip_outputs[2],
            self.high_embedding_linear,
            self.high_layer_norm,
            self.high_projection,
            dtype
        )

        # Process mid-level features (layer -11)
        mid_embedding = self._process_layer_features(
            siglip_outputs[1],
            self.mid_embedding_linear,
            self.mid_layer_norm,
            self.mid_projection,
            dtype
        )

        # Process low-level features (layer -20)
        low_embedding = self._process_layer_features(
            siglip_outputs[0],
            self.low_embedding_linear,
            self.low_layer_norm,
            self.low_projection,
            dtype
        )

        # Concatenate features from all layersmodel_patch
        return torch.cat((high_embedding, mid_embedding, low_embedding), dim=1)

    def _process_layer_features(
        self,
        hidden_states: torch.Tensor,
        embedding_linear: nn.Module,
        layer_norm: nn.Module,
        projection: nn.Module,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Helper function to process features from a single layer

        Args:
            hidden_states: Input hidden states [bs, seq_len, dim]
            embedding_linear: Embedding linear layer
            layer_norm: Layer normalization
            projection: Projection layer
            dtype: Target data type

        Returns:
            torch.Tensor: Processed features [bs, style_token_nums, hidden_size]
        """
        # Transform dimensions: [bs, seq_len, dim] -> [bs, dim, seq_len] -> [bs, dim, style_token_nums] -> [bs, style_token_nums, dim]
        embedding = embedding_linear(
            hidden_states.to(dtype).transpose(1, 2)
        ).transpose(1, 2)

        # Apply layer normalization
        embedding = layer_norm(embedding)

        # Project to target hidden space
        embedding = projection(embedding)

        return embedding

class ModelPatchLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "name": (folder_paths.get_filename_list("model_patches"), ),
                              }}
    RETURN_TYPES = ("MODEL_PATCH",)
    FUNCTION = "load_model_patch"
    EXPERIMENTAL = True

    CATEGORY = "advanced/loaders"

    def load_model_patch(self, name):
        model_patch_path = folder_paths.get_full_path_or_raise("model_patches", name)
        sd = comfy.utils.load_torch_file(model_patch_path, safe_load=True)
        dtype = comfy.utils.weight_dtype(sd)

        if 'controlnet_blocks.0.y_rms.weight' in sd:
            additional_in_dim = sd["img_in.weight"].shape[1] - 64
            model = QwenImageBlockWiseControlNet(additional_in_dim=additional_in_dim, device=comfy.model_management.unet_offload_device(), dtype=dtype, operations=comfy.ops.manual_cast)
        elif 'feature_embedder.mid_layer_norm.bias' in sd:
            sd = comfy.utils.state_dict_prefix_replace(sd, {"feature_embedder.": ""}, filter_keys=True)
            model = SigLIPMultiFeatProjModel(device=comfy.model_management.unet_offload_device(), dtype=dtype, operations=comfy.ops.manual_cast)

        model.load_state_dict(sd)
        model = comfy.model_patcher.ModelPatcher(model, load_device=comfy.model_management.get_torch_device(), offload_device=comfy.model_management.unet_offload_device())
        return (model,)


class DiffSynthCnetPatch:
    def __init__(self, model_patch, vae, image, strength, mask=None):
        self.model_patch = model_patch
        self.vae = vae
        self.image = image
        self.strength = strength
        self.mask = mask
        self.encoded_image = model_patch.model.process_input_latent_image(self.encode_latent_cond(image))
        self.encoded_image_size = (image.shape[1], image.shape[2])

    def encode_latent_cond(self, image):
        latent_image = self.vae.encode(image)
        if self.model_patch.model.additional_in_dim > 0:
            if self.mask is None:
                mask_ = torch.ones_like(latent_image)[:, :self.model_patch.model.additional_in_dim // 4]
            else:
                mask_ = comfy.utils.common_upscale(self.mask.mean(dim=1, keepdim=True), latent_image.shape[-1], latent_image.shape[-2], "bilinear", "none")

            return torch.cat([latent_image, mask_], dim=1)
        else:
            return latent_image

    def __call__(self, kwargs):
        x = kwargs.get("x")
        img = kwargs.get("img")
        block_index = kwargs.get("block_index")
        spacial_compression = self.vae.spacial_compression_encode()
        if self.encoded_image is None or self.encoded_image_size != (x.shape[-2] * spacial_compression, x.shape[-1] * spacial_compression):
            image_scaled = comfy.utils.common_upscale(self.image.movedim(-1, 1), x.shape[-1] * spacial_compression, x.shape[-2] * spacial_compression, "area", "center")
            loaded_models = comfy.model_management.loaded_models(only_currently_used=True)
            self.encoded_image = self.model_patch.model.process_input_latent_image(self.encode_latent_cond(image_scaled.movedim(1, -1)))
            self.encoded_image_size = (image_scaled.shape[-2], image_scaled.shape[-1])
            comfy.model_management.load_models_gpu(loaded_models)

        img[:, :self.encoded_image.shape[1]] += (self.model_patch.model.control_block(img[:, :self.encoded_image.shape[1]], self.encoded_image.to(img.dtype), block_index) * self.strength)
        kwargs['img'] = img
        return kwargs

    def to(self, device_or_dtype):
        if isinstance(device_or_dtype, torch.device):
            self.encoded_image = self.encoded_image.to(device_or_dtype)
        return self

    def models(self):
        return [self.model_patch]

class QwenImageDiffsynthControlnet:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "model_patch": ("MODEL_PATCH",),
                              "vae": ("VAE",),
                              "image": ("IMAGE",),
                              "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                              },
                "optional": {"mask": ("MASK",)}}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "diffsynth_controlnet"
    EXPERIMENTAL = True

    CATEGORY = "advanced/loaders/qwen"

    def diffsynth_controlnet(self, model, model_patch, vae, image, strength, mask=None):
        model_patched = model.clone()
        image = image[:, :, :, :3]
        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            if mask.ndim == 4:
                mask = mask.unsqueeze(2)
            mask = 1.0 - mask

        model_patched.set_model_double_block_patch(DiffSynthCnetPatch(model_patch, vae, image, strength, mask))
        return (model_patched,)


class UsoStyleProjectorPatch:
    def __init__(self, model_patch, encoded_image):
        self.model_patch = model_patch
        self.encoded_image = encoded_image

    def __call__(self, kwargs):
        txt_ids = kwargs.get("txt_ids")
        txt = kwargs.get("txt")
        siglip_embedding = self.model_patch.model(self.encoded_image.to(txt.dtype)).to(txt.dtype)
        txt = torch.cat([siglip_embedding, txt], dim=1)
        kwargs['txt'] = txt
        kwargs['txt_ids'] = torch.cat([torch.zeros(siglip_embedding.shape[0], siglip_embedding.shape[1], 3, dtype=txt_ids.dtype, device=txt_ids.device), txt_ids], dim=1)
        return kwargs

    def to(self, device_or_dtype):
        if isinstance(device_or_dtype, torch.device):
            self.encoded_image = self.encoded_image.to(device_or_dtype)
        return self

    def models(self):
        return [self.model_patch]


class USOStyleReference:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",),
                             "model_patch": ("MODEL_PATCH",),
                             "clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_patch"
    EXPERIMENTAL = True

    CATEGORY = "advanced/model_patches/flux"

    def apply_patch(self, model, model_patch, clip_vision_output):
        encoded_image = torch.stack((clip_vision_output.all_hidden_states[:, -20], clip_vision_output.all_hidden_states[:, -11], clip_vision_output.penultimate_hidden_states))
        model_patched = model.clone()
        model_patched.set_model_post_input_patch(UsoStyleProjectorPatch(model_patch, encoded_image))
        return (model_patched,)


NODE_CLASS_MAPPINGS = {
    "ModelPatchLoader": ModelPatchLoader,
    "QwenImageDiffsynthControlnet": QwenImageDiffsynthControlnet,
    "USOStyleReference": USOStyleReference,
}
