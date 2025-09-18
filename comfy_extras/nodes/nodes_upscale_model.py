import logging
from typing import Optional, Any

import torch
from spandrel import ModelLoader, ImageModelDescriptor

from comfy import model_management
from comfy import utils
from comfy.component_model.tensor_types import RGBImageBatch
from comfy.model_downloader import get_filename_list_with_downloadable, KNOWN_UPSCALERS, get_or_download
from comfy.model_management import load_models_gpu
from comfy.model_management_types import ModelManageableStub

logger = logging.getLogger(__name__)
try:
    from spandrel_extra_arches import EXTRA_REGISTRY  # pylint: disable=import-error
    from spandrel import MAIN_REGISTRY

    MAIN_REGISTRY.add(*EXTRA_REGISTRY)
    logger.debug("Successfully imported spandrel_extra_arches: support for non commercial upscale models.")
except:
    pass


class UpscaleModelManageable(ModelManageableStub):
    def __init__(self, model_descriptor: ImageModelDescriptor, ckpt_name: str):
        self.ckpt_name = ckpt_name
        self.model_descriptor = model_descriptor
        self.model = model_descriptor.model
        self.load_device = model_management.get_torch_device()
        self.offload_device = model_management.unet_offload_device()
        self._input_size = (1, 512, 512)
        self._input_channels = model_descriptor.input_channels
        self._output_channels = model_descriptor.output_channels
        self.tile = 512

    @property
    def current_device(self) -> torch.device:
        return self.model_descriptor.device

    @property
    def input_size(self) -> tuple[int, int, int]:
        return self._input_size

    @input_size.setter
    def input_size(self, size: tuple[int, int, int]):
        self._input_size = size

    @property
    def scale(self) -> int:
        if not hasattr(self.model_descriptor, "scale"):
            return 1
        return self.model_descriptor.scale

    @property
    def output_size(self) -> tuple[int, int, int]:
        return (self._input_size[0],
                self._input_size[1] * self.scale,
                self._input_size[2] * self.scale)

    def set_input_size_from_images(self, images: RGBImageBatch):
        if images.ndim != 4:
            raise ValueError("Input must be a 4D tensor (batch, height, width, channels)")
        if images.shape[-1] != 3:
            raise ValueError("Input must have 3 channels (RGB)")
        self._input_size = (images.shape[0], images.shape[1], images.shape[2])

    def is_clone(self, other: Any) -> bool:
        return isinstance(other, UpscaleModelManageable) and self.model is other.model

    def clone_has_same_weights(self, clone) -> bool:
        return self.is_clone(clone)

    def model_size(self) -> int:
        model_params_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        dtype_size = torch.finfo(self.model_dtype()).bits // 8
        batch_size = self._input_size[0]
        input_size = batch_size * min(self.tile, self._input_size[1]) * min(self.tile, self._input_size[2]) * self._input_channels * dtype_size
        output_size = batch_size * min(self.tile * self.scale, self.output_size[1]) * min(self.tile * self.scale, self.output_size[2]) * self._output_channels * dtype_size

        return model_params_size + input_size + output_size

    def model_patches_to(self, arg: torch.device | torch.dtype):
        if isinstance(arg, torch.device):
            self.model.to(device=arg)
        else:
            self.model.to(dtype=arg)

    def model_dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def patch_model(self, device_to: torch.device | None = None, lowvram_model_memory: int = 0, load_weights: bool = True, force_patch_weights: bool = False) -> torch.nn.Module:
        self.model.to(device=device_to)
        return self.model

    def unpatch_model(self, device_to: torch.device | None = None, unpatch_weights: Optional[bool] = False) -> torch.nn.Module:
        self.model.to(device=device_to)
        return self.model

    def __str__(self):
        if self.ckpt_name is not None:
            return f"<UpscaleModelManageable for {self.ckpt_name} ({self.model.__class__.__name__})>"
        else:
            return f"<UpscaleModelManageable for {self.model.__class__.__name__}>"


class UpscaleModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model_name": (get_filename_list_with_downloadable("upscale_models"),),
                             }}

    RETURN_TYPES = ("UPSCALE_MODEL",)
    FUNCTION = "load_model"

    CATEGORY = "loaders"

    def load_model(self, model_name):
        model_path = get_or_download("upscale_models", model_name, KNOWN_UPSCALERS)
        sd = utils.load_torch_file(model_path, safe_load=True)
        if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
            sd = utils.state_dict_prefix_replace(sd, {"module.": ""})
        out = ModelLoader().load_from_state_dict(sd).eval()

        if not isinstance(out, ImageModelDescriptor):
            raise Exception("Upscale model must be a single-image model.")

        return (UpscaleModelManageable(out, model_name),)


class ImageUpscaleWithModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"upscale_model": ("UPSCALE_MODEL",),
                             "image": ("IMAGE",),
                             }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = "image/upscaling"

    def upscale(self, upscale_model: UpscaleModelManageable, image: RGBImageBatch):
        upscale_model.set_input_size_from_images(image)
        load_models_gpu([upscale_model])

        in_img = image.movedim(-1, -3).to(upscale_model.current_device, dtype=upscale_model.model_dtype())

        tile = upscale_model.tile
        overlap = 32

        oom = True
        s = None
        while oom:
            try:
                steps = in_img.shape[0] * utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
                pbar = utils.ProgressBar(steps)
                s = utils.tiled_scale(in_img, lambda a: upscale_model.model(a), tile_x=tile, tile_y=tile, overlap=overlap, upscale_amount=upscale_model.scale, pbar=pbar)
                oom = False
            except model_management.OOM_EXCEPTION as e:
                tile //= 2
                overlap //= 2
                if tile < 64 or overlap < 4:
                    raise e
            except RuntimeError as exc_info:
                if "have 1 channels, but got 3 channels instead" in str(exc_info):
                    # convert RGB to luminance (assuming sRGB)

                    rgb_weights = torch.tensor([0.2126, 0.7152, 0.0722], device=in_img.device, dtype=in_img.dtype)
                    in_img = (in_img * rgb_weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
                    continue
                else:
                    raise exc_info

        # upscale_model.to("cpu")
        s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)

        if s.shape[-1] == 1:
            s = s.expand(-1, -1, -1, 3)

        del in_img
        return (s,)


NODE_CLASS_MAPPINGS = {
    "UpscaleModelLoader": UpscaleModelLoader,
    "ImageUpscaleWithModel": ImageUpscaleWithModel
}
