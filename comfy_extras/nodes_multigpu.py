from __future__ import annotations

from inspect import cleandoc
from typing import TYPE_CHECKING
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

if TYPE_CHECKING:
    from comfy.model_patcher import ModelPatcher
import comfy.multigpu


class MultiGPUCFGSplitNode(io.ComfyNode):
    """
    Attaches per-device deepclones to any connected MODEL, UPSCALE_MODEL, and/or VAE so
    downstream nodes that recognize the attached state dispatch their work across multiple GPUs.

    Place after nodes that modify the model object itself (compile, attention-switch, etc.).
    Otherwise position is not order-sensitive.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MultiGPU_WorkUnits",
            display_name="MultiGPU Work Units",
            category="advanced/multigpu",
            description=cleandoc(cls.__doc__),
            inputs=[
                io.Model.Input("model", optional=True),
                io.UpscaleModel.Input("upscale_model", optional=True),
                io.Vae.Input("vae", optional=True),
                io.Int.Input("max_gpus", default=2, min=1, step=1),
            ],
            outputs=[
                io.Model.Output(),
                io.UpscaleModel.Output(),
                io.Vae.Output(),
            ],
        )

    @classmethod
    def execute(cls, max_gpus: int, model: ModelPatcher = None, upscale_model=None, vae=None) -> io.NodeOutput:
        if model is not None:
            model = comfy.multigpu.create_multigpu_deepclones(model, max_gpus, reuse_loaded=True)
        if upscale_model is not None:
            upscale_model = comfy.multigpu.create_upscale_model_multigpu_deepclones(upscale_model, max_gpus)
        if vae is not None:
            vae = comfy.multigpu.create_vae_multigpu_deepclones(vae, max_gpus)
        return io.NodeOutput(model, upscale_model, vae)


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
            # MultiGPUOptionsNode,
        ]


async def comfy_entrypoint() -> MultiGPUExtension:
    return MultiGPUExtension()
