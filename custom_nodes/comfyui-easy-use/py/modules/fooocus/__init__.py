#credit to Acly for this module
#from https://github.com/Acly/comfyui-inpaint-nodes
import torch
import torch.nn.functional as F
import comfy
from comfy.model_base import BaseModel
from comfy.model_patcher import ModelPatcher
from comfy.model_management import cast_to_device

from ...libs.log import log_node_warn, log_node_error, log_node_info

class InpaintHead(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = torch.nn.Parameter(torch.empty(size=(320, 5, 3, 3), device="cpu"))

    def __call__(self, x):
        x = F.pad(x, (1, 1, 1, 1), "replicate")
        return F.conv2d(x, weight=self.head)

# injected_model_patcher_calculate_weight = False
# original_calculate_weight = None

class applyFooocusInpaint:
    def calculate_weight_patched(self, patches, weight, key, intermediate_dtype=torch.float32):
        remaining = []

        for p in patches:
            alpha = p[0]
            v = p[1]

            is_fooocus_patch = isinstance(v, tuple) and len(v) == 2 and v[0] == "fooocus"
            if not is_fooocus_patch:
                remaining.append(p)
                continue

            if alpha != 0.0:
                v = v[1]
                w1 = cast_to_device(v[0], weight.device, torch.float32)
                if w1.shape == weight.shape:
                    w_min = cast_to_device(v[1], weight.device, torch.float32)
                    w_max = cast_to_device(v[2], weight.device, torch.float32)
                    w1 = (w1 / 255.0) * (w_max - w_min) + w_min
                    weight += alpha * cast_to_device(w1, weight.device, weight.dtype)
                else:
                    print(
                        f"[ApplyFooocusInpaint] Shape mismatch {key}, weight not merged ({w1.shape} != {weight.shape})"
                    )

        if len(remaining) > 0:
            return self.original_calculate_weight(remaining, weight, key, intermediate_dtype)
        return weight

    def __enter__(self):
        try:
            print("[comfyui-easy-use] Injecting patched comfy.lora.calculate_weight.calculate_weight")
            self.original_calculate_weight = comfy.lora.calculate_weight
            comfy.lora.calculate_weight = self.calculate_weight_patched
        except AttributeError:
            print("[comfyui-easy-use] Injecting patched comfy.model_patcher.ModelPatcher.calculate_weight")
            self.original_calculate_weight = ModelPatcher.calculate_weight
            ModelPatcher.calculate_weight = self.calculate_weight_patched

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            comfy.lora.calculate_weight = self.original_calculate_weight
        except:
            ModelPatcher.calculate_weight = self.original_calculate_weight

# def inject_patched_calculate_weight():
#     global injected_model_patcher_calculate_weight
#     if not injected_model_patcher_calculate_weight:
#         try:
#             print("[comfyui-easy-use] Injecting patched comfy.lora.calculate_weight.calculate_weight")
#             original_calculate_weight = comfy.lora.calculate_weight
#             comfy.lora.original_calculate_weight = original_calculate_weight
#             comfy.lora.calculate_weight = calculate_weight_patched
#         except AttributeError:
#             print("[comfyui-easy-use] Injecting patched comfy.model_patcher.ModelPatcher.calculate_weight")
#             original_calculate_weight = ModelPatcher.calculate_weight
#             ModelPatcher.original_calculate_weight = original_calculate_weight
#             ModelPatcher.calculate_weight = calculate_weight_patched
#         injected_model_patcher_calculate_weight = True


class InpaintWorker:
    def __init__(self, node_name):
        self.node_name = node_name if node_name is not None else ""

    def load_fooocus_patch(self, lora: dict, to_load: dict):
        patch_dict = {}
        loaded_keys = set()
        for key in to_load.values():
            if value := lora.get(key, None):
                patch_dict[key] = ("fooocus", value)
                loaded_keys.add(key)

        not_loaded = sum(1 for x in lora if x not in loaded_keys)
        if not_loaded > 0:
            log_node_info(self.node_name,
                f"{len(loaded_keys)} Lora keys loaded, {not_loaded} remaining keys not found in model."
            )
        return patch_dict

    def _input_block_patch(self, h: torch.Tensor, transformer_options: dict):
        if transformer_options["block"][1] == 0:
            if self._inpaint_block is None or self._inpaint_block.shape != h.shape:
                assert self._inpaint_head_feature is not None
                batch = h.shape[0] // self._inpaint_head_feature.shape[0]
                self._inpaint_block = self._inpaint_head_feature.to(h).repeat(batch, 1, 1, 1)
            h = h + self._inpaint_block
        return h

    def patch(self, model, latent, patch):
        base_model: BaseModel = model.model
        latent_pixels = base_model.process_latent_in(latent["samples"])
        noise_mask = latent["noise_mask"].round()
        latent_mask = F.max_pool2d(noise_mask, (8, 8)).round().to(latent_pixels)

        inpaint_head_model, inpaint_lora = patch
        feed = torch.cat([latent_mask, latent_pixels], dim=1)
        inpaint_head_model.to(device=feed.device, dtype=feed.dtype)
        self._inpaint_head_feature = inpaint_head_model(feed)
        self._inpaint_block = None

        lora_keys = comfy.lora.model_lora_keys_unet(model.model, {})
        lora_keys.update({x: x for x in base_model.state_dict().keys()})
        loaded_lora = self.load_fooocus_patch(inpaint_lora, lora_keys)

        m = model.clone()
        m.set_model_input_block_patch(self._input_block_patch)
        patched = m.add_patches(loaded_lora, 1.0)
        m.model_options['transformer_options']['fooocus'] = True
        not_patched_count = sum(1 for x in loaded_lora if x not in patched)
        if not_patched_count > 0:
            log_node_error(self.node_name, f"Failed to patch {not_patched_count} keys")

        # inject_patched_calculate_weight()
        return (m,)