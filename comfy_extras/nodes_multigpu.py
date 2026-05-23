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


class SelectModelDeviceNode(io.ComfyNode):
    """
    Place the diffusion model on a specific device (default / cpu / gpu:N).

    When the selected device does not exist on the current machine
    (e.g. a workflow built on a 2-GPU box opened on a 1-GPU box),
    the node passes the model through unchanged and logs a message
    instead of failing. This keeps workflows portable across machines
    with different GPU counts.
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
        if resolved is None:
            if device not in (None, "default"):
                logging.info(f"Select Model Device: requested device '{device}' not available, passing through unchanged.")
            return io.NodeOutput(model)
        model.load_device = resolved
        if resolved.type == "cpu":
            model.offload_device = resolved
        if hasattr(model, "register_load_device"):
            model.register_load_device(resolved)
        return io.NodeOutput(model)


class SelectCLIPDeviceNode(io.ComfyNode):
    """
    Place the CLIP text encoder on a specific device (default / cpu / gpu:N).

    When the selected device does not exist on the current machine
    (e.g. a workflow built on a 2-GPU box opened on a 1-GPU box),
    the node passes the CLIP through unchanged and logs a message
    instead of failing. This keeps workflows portable across machines
    with different GPU counts.
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
        if resolved is None:
            if device not in (None, "default"):
                logging.info(f"Select CLIP Device: requested device '{device}' not available, passing through unchanged.")
            return io.NodeOutput(clip)
        clip.patcher.load_device = resolved
        if resolved.type == "cpu":
            clip.patcher.offload_device = resolved
        if hasattr(clip.patcher, "register_load_device"):
            clip.patcher.register_load_device(resolved)
        return io.NodeOutput(clip)


class SelectVAEDeviceNode(io.ComfyNode):
    """
    Place the VAE on a specific device (default / gpu:N).

    CPU is intentionally not offered as a choice; VAE on CPU is impractical.

    When the selected device does not exist on the current machine
    (e.g. a workflow built on a 2-GPU box opened on a 1-GPU box),
    the node passes the VAE through unchanged and logs a message
    instead of failing. This keeps workflows portable across machines
    with different GPU counts.
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
        if resolved is None:
            if device not in (None, "default"):
                logging.info(f"Select VAE Device: requested device '{device}' not available, passing through unchanged.")
            return io.NodeOutput(vae)
        vae.device = resolved
        vae.patcher.load_device = resolved
        vae.patcher.offload_device = comfy.model_management.vae_offload_device()
        if hasattr(vae.patcher, "register_load_device"):
            vae.patcher.register_load_device(resolved)
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
