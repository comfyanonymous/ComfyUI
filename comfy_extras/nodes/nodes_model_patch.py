import torch
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
        # TODO: this node will work with more types of model patches
        additional_in_dim = sd["img_in.weight"].shape[1] - 64
        model = QwenImageBlockWiseControlNet(additional_in_dim=additional_in_dim, device=comfy.model_management.unet_offload_device(), dtype=dtype, operations=comfy.ops.manual_cast)
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
        if self.encoded_image is None or self.encoded_image.shape[1:] != img.shape[1:]:
            spacial_compression = self.vae.spacial_compression_encode()
            image_scaled = comfy.utils.common_upscale(self.image.movedim(-1, 1), x.shape[-1] * spacial_compression, x.shape[-2] * spacial_compression, "area", "center")
            loaded_models = comfy.model_management.loaded_models(only_currently_used=True)
            self.encoded_image = self.model_patch.model.process_input_latent_image(self.encode_latent_cond(image_scaled.movedim(1, -1)))
            comfy.model_management.load_models_gpu(loaded_models)

        img = img + (self.model_patch.model.control_block(img, self.encoded_image.to(img.dtype), block_index) * self.strength)
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


NODE_CLASS_MAPPINGS = {
    "ModelPatchLoader": ModelPatchLoader,
    "QwenImageDiffsynthControlnet": QwenImageDiffsynthControlnet,
}
