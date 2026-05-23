from __future__ import annotations

import copy
import logging
from inspect import cleandoc
from typing import TYPE_CHECKING
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

if TYPE_CHECKING:
    from comfy.model_patcher import ModelPatcher
    from comfy.sd import CLIP, VAE
import comfy.model_management
import comfy.multigpu


class MultiGPUCFGSplitNode(io.ComfyNode):
    """
    Prepares model to have sampling accelerated via splitting work units.

    Should be placed after nodes that modify the model object itself, such as compile or attention-switch nodes.

    Other than those exceptions, this node can be placed in any order.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MultiGPU_WorkUnits",
            display_name="MultiGPU CFG Split",
            category="advanced/multigpu",
            description=cleandoc(cls.__doc__),
            inputs=[
                io.Model.Input("model"),
                io.Int.Input("max_gpus", default=2, min=1, step=1),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model: ModelPatcher, max_gpus: int) -> io.NodeOutput:
        model = comfy.multigpu.create_multigpu_deepclones(model, max_gpus, reuse_loaded=True)
        return io.NodeOutput(model)


def _remember_base_devices(patcher: ModelPatcher):
    """Stash the original load/offload device on the underlying model.

    Stored on patcher.model (which is shared across patcher clones), so
    repeated selector applications can recover the loader's original
    routing when the user picks "default".
    """
    if not hasattr(patcher.model, "_select_base_load_device"):
        patcher.model._select_base_load_device = patcher.load_device
        patcher.model._select_base_offload_device = patcher.offload_device


def _apply_patcher_device(patcher: ModelPatcher, resolved, base_offload_override=None):
    """Apply *resolved* to a freshly-cloned patcher; respect base devices on default.

    Returns the (possibly newly-replaced) patcher. For CPU on a dynamic
    patcher, also tries to downgrade to a plain ModelPatcher so the
    dynamic-only code paths are bypassed (best-effort: silently keeps
    the dynamic patcher if downgrade is not supported).
    """
    _remember_base_devices(patcher)
    base_load = patcher.model._select_base_load_device
    base_offload = base_offload_override if base_offload_override is not None else patcher.model._select_base_offload_device

    if resolved is None:
        # "default" -> reset routing to whatever the loader produced
        patcher.load_device = base_load
        patcher.offload_device = base_offload
    elif resolved.type == "cpu":
        if patcher.is_dynamic():
            try:
                patcher = patcher.clone(disable_dynamic=True)
            except Exception:
                # Downgrade unavailable (no cached_patcher_init); fall
                # back to the existing dynamic patcher.
                pass
        patcher.load_device = resolved
        patcher.offload_device = resolved
    else:
        patcher.load_device = resolved
        patcher.offload_device = base_offload

    if hasattr(patcher, "register_load_device"):
        patcher.register_load_device(patcher.load_device)
    return patcher


def _prune_multigpu_collision(model: ModelPatcher, primary_device):
    """Drop any multigpu clone whose load_device matches *primary_device*.

    Without pruning, MultiGPU CFG Split would have stacked a clone on
    the same device the primary now occupies (i.e. the workflow places
    MultiGPU CFG Split before Select Model Device). Keeps the clone set
    consistent with the new primary placement.
    """
    multigpu_models = model.get_additional_models_with_key("multigpu")
    if not multigpu_models:
        return
    filtered = [m for m in multigpu_models if m.load_device != primary_device]
    if len(filtered) != len(multigpu_models):
        logging.info(f"Select Model Device: pruning MultiGPU clone on {primary_device} that now collides with the primary model.")
        model.set_additional_models("multigpu", filtered)
        if hasattr(model, "match_multigpu_clones"):
            model.match_multigpu_clones()


class SelectModelDeviceNode(io.ComfyNode):
    """
    Place the diffusion model on a specific device (default / cpu / gpu:N).

    - "default" restores the device assigned by the loader (even after a
      prior Select Model Device call).
    - "cpu" pins both the load and offload device to CPU.
    - "gpu:N" pins the load device to the Nth available GPU; the offload
      device is restored to the loader's original choice.

    If the workflow already has MultiGPU CFG Split applied and the chosen
    GPU collides with one of the existing multigpu clones, that clone is
    dropped so two patchers don't end up bound to the same device.

    When the selected device does not exist on the current machine
    (e.g. a workflow built on a 2-GPU box opened on a 1-GPU box),
    the node passes the model through unchanged and logs a message
    instead of failing.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SelectModelDevice",
            display_name="Select Model Device",
            category="advanced/multigpu",
            description=cleandoc(cls.__doc__),
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("device", options=comfy.model_management.get_gpu_device_options()),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def validate_inputs(cls, device="default"):
        # Allow unknown gpu:N values so portable workflows do not error
        # at validation time; runtime fallback will handle them.
        return True

    @classmethod
    def execute(cls, model: ModelPatcher, device: str = "default") -> io.NodeOutput:
        model = model.clone()
        resolved = comfy.model_management.resolve_gpu_device_option(device)
        if resolved is None and device not in (None, "default"):
            logging.info(f"Select Model Device: requested device '{device}' not available, passing through unchanged.")
            return io.NodeOutput(model)
        model = _apply_patcher_device(model, resolved)
        if resolved is not None:
            _prune_multigpu_collision(model, model.load_device)
        return io.NodeOutput(model)


class SelectCLIPDeviceNode(io.ComfyNode):
    """
    Place the CLIP text encoder on a specific device (default / cpu / gpu:N).

    - "default" restores the device assigned by the loader.
    - "cpu" pins both the load and offload device to CPU.
    - "gpu:N" pins the load device to the Nth available GPU.

    When the selected device does not exist on the current machine
    (e.g. a workflow built on a 2-GPU box opened on a 1-GPU box),
    the node passes the CLIP through unchanged and logs a message
    instead of failing.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SelectCLIPDevice",
            display_name="Select CLIP Device",
            category="advanced/multigpu",
            description=cleandoc(cls.__doc__),
            inputs=[
                io.Clip.Input("clip"),
                io.Combo.Input("device", options=comfy.model_management.get_gpu_device_options()),
            ],
            outputs=[
                io.Clip.Output(),
            ],
        )

    @classmethod
    def validate_inputs(cls, device="default"):
        return True

    @classmethod
    def execute(cls, clip: CLIP, device: str = "default") -> io.NodeOutput:
        clip = clip.clone()
        resolved = comfy.model_management.resolve_gpu_device_option(device)
        if resolved is None and device not in (None, "default"):
            logging.info(f"Select CLIP Device: requested device '{device}' not available, passing through unchanged.")
            return io.NodeOutput(clip)
        clip.patcher = _apply_patcher_device(clip.patcher, resolved)
        return io.NodeOutput(clip)


class SelectVAEDeviceNode(io.ComfyNode):
    """
    Place the VAE on a specific device (default / gpu:N).

    - "default" restores the device assigned by the loader.
    - "gpu:N" pins the load device to the Nth available GPU; the offload
      device is set to the standard VAE offload device.

    CPU is intentionally not exposed in the UI for the VAE; if a workflow
    supplies "cpu" anyway (e.g. opened from another machine), the request
    is dropped with a log message and the VAE is passed through unchanged.

    When the selected device does not exist on the current machine
    (e.g. a workflow built on a 2-GPU box opened on a 1-GPU box),
    the node passes the VAE through unchanged and logs a message
    instead of failing.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SelectVAEDevice",
            display_name="Select VAE Device",
            category="advanced/multigpu",
            description=cleandoc(cls.__doc__),
            inputs=[
                io.Vae.Input("vae"),
                io.Combo.Input("device", options=comfy.model_management.get_gpu_device_options_no_cpu()),
            ],
            outputs=[
                io.Vae.Output(),
            ],
        )

    @classmethod
    def validate_inputs(cls, device="default"):
        return True

    @classmethod
    def execute(cls, vae: VAE, device: str = "default") -> io.NodeOutput:
        # VAE has no .clone(); shallow-copy the wrapper and clone the patcher
        # so we can retarget load/offload device without affecting the input VAE.
        vae = copy.copy(vae)
        vae.patcher = vae.patcher.clone()
        resolved = comfy.model_management.resolve_gpu_device_option(device)
        if resolved is None and device not in (None, "default"):
            logging.info(f"Select VAE Device: requested device '{device}' not available, passing through unchanged.")
            return io.NodeOutput(vae)
        if resolved is not None and resolved.type == "cpu":
            logging.info("Select VAE Device: CPU is not a supported choice, passing through unchanged.")
            return io.NodeOutput(vae)
        vae.patcher = _apply_patcher_device(
            vae.patcher, resolved,
            base_offload_override=comfy.model_management.vae_offload_device(),
        )
        # VAE caches the working device separately from its patcher.
        if not hasattr(vae, "_select_base_device"):
            vae._select_base_device = vae.device
        vae.device = vae._select_base_device if resolved is None else resolved
        return io.NodeOutput(vae)


class MultiGPUOptionsNode(io.ComfyNode):
    """
    Select the relative speed of GPUs in the special case they have significantly different performance from one another.

    NOTE (not registered yet, see MultiGPUExtension.get_node_list below):
    The output GPUOptionsGroup is plumbed through create_multigpu_deepclones() and stored on
    model.model_options['multigpu_options'] via GPUOptionsGroup.register(), but the cond
    scheduler in comfy/samplers.py (calc_cond_batch_outer_multigpu) does NOT yet consult
    relative_speed when distributing conds across devices; it uses a uniform conds_per_device
    round-robin via next_available_device(). Before re-enabling this node, wire its
    relative_speed into the scheduler (e.g. via comfy.multigpu.load_balance_devices(),
    which already implements the proportional split) so the input actually affects work
    distribution.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MultiGPU_Options",
            display_name="MultiGPU Options",
            category="advanced/multigpu",
            description=cleandoc(cls.__doc__),
            inputs=[
                io.Int.Input("device_index", default=0, min=0, max=64),
                io.Float.Input("relative_speed", default=1.0, min=0.0, step=0.01),
                io.Custom("GPU_OPTIONS").Input("gpu_options", optional=True),
            ],
            outputs=[
                io.Custom("GPU_OPTIONS").Output(),
            ],
        )

    @classmethod
    def execute(cls, device_index: int, relative_speed: float, gpu_options: comfy.multigpu.GPUOptionsGroup = None) -> io.NodeOutput:
        if not gpu_options:
            gpu_options = comfy.multigpu.GPUOptionsGroup()
        else:
            gpu_options = gpu_options.clone()

        opt = comfy.multigpu.GPUOptions(device_index=device_index, relative_speed=relative_speed)
        gpu_options.add(opt)

        return io.NodeOutput(gpu_options)


class MultiGPUExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            MultiGPUCFGSplitNode,
            SelectModelDeviceNode,
            SelectCLIPDeviceNode,
            SelectVAEDeviceNode,
            # MultiGPUOptionsNode,
        ]


async def comfy_entrypoint() -> MultiGPUExtension:
    return MultiGPUExtension()
