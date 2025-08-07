from __future__ import annotations
from comfy_api.latest import ComfyExtension, io
import comfy.context_windows
import comfy.patcher_extension
import comfy.samplers
import nodes
import torch


def _prepare_sampling_wrapper(executor, model, noise_shape: torch.Tensor, *args, **kwargs):
    # TODO: handle various dims instead of defaulting to 0th
    # limit noise_shape length to context_length for more accurate vram use estimation
    model_options = kwargs.get("model_options", None)
    if model_options is None:
        raise Exception("model_options not found in prepare_sampling_wrapper; this should never happen, something went wrong.")
    handler: comfy.context_windows.IndexListContextHandler = model_options.get("context_handler", None)
    if handler is not None:
        noise_shape = list(noise_shape)
        noise_shape[handler.dim] = min(noise_shape[handler.dim], handler.context_length)
    return executor(model, noise_shape, *args, **kwargs)


def create_prepare_sampling_wrapper(model_options: dict):
    comfy.patcher_extension.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.PREPARE_SAMPLING,
                                                 "ContextWindows_prepare_sampling",
                                                 _prepare_sampling_wrapper,
                                                 model_options, is_model_options=True)

def _outer_sample_wrapper(executor, *args, **kwargs):
    guider: comfy.samplers.CFGGuider = executor.class_obj
    handler: comfy.context_windows.IndexListContextHandler = guider.model_options.get("context_handler", None)
    if handler is not None:
        args = list(args)
        noise: torch.Tensor = args[0]
        length = noise.shape[handler.dim]
        window = comfy.context_windows.IndexListContextWindow(list(range(handler.context_length)))
        noise = window.get_tensor(noise, dim=handler.dim)
        cat_count = (length // handler.context_length) + 1
        noise = torch.cat([noise] * cat_count, dim=handler.dim)
        if handler.dim == 0:
            noise = noise[:length]
        elif handler.dim == 1:
            noise = noise[:, :length]
        elif handler.dim == 2:
            noise = noise[:, :, :length]
        else:
            pass
        args[0] = noise
        args = tuple(args)
    return executor(*args, **kwargs)

def create_outer_sampler_wrapper(model_options: dict):
    comfy.patcher_extension.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
                                                 "ContextWindows_outer_sample",
                                                 _outer_sample_wrapper,
                                                 model_options, is_model_options=True)


class ContextWindowsNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ContexWindowsTest",
            display_name="Context Windows Test",
            category="context",
            description="Test node for context windows",
            inputs=[
                io.Model.Input("model", tooltip="The model to apply context windows to during sampling."),
                io.Int.Input("context_length", min=1, default=1, tooltip="The length of the context window."),
                io.Int.Input("context_overlap", min=0, default=0, tooltip="The overlap of the context window."),
                io.Combo.Input("context_schedule", options=[
                    comfy.context_windows.ContextSchedules.STATIC_STANDARD,
                    comfy.context_windows.ContextSchedules.UNIFORM_STANDARD,
                    comfy.context_windows.ContextSchedules.UNIFORM_LOOPED,
                    comfy.context_windows.ContextSchedules.BATCHED,
                    ], tooltip="The stride of the context window."),
                io.Int.Input("context_stride", min=1, default=1, tooltip="The stride of the context window."),
                io.Boolean.Input("closed_loop", default=False, tooltip="Whether to close the context window loop."),
                io.Combo.Input("fuse_method", options=comfy.context_windows.ContextFuseMethods.LIST_STATIC, default=comfy.context_windows.ContextFuseMethods.PYRAMID, tooltip="The method to use to fuse the context windows."),
                io.Int.Input("dim", min=0, max=5, default=0, tooltip="The dimension to apply the context windows to."),
            ],
            outputs=[
                io.Model.Output(tooltip="The model with context windows applied during sampling."),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model: io.Model.Type, context_length: int, context_overlap: int, context_schedule: str, context_stride: int, closed_loop: bool, fuse_method: str, dim: int) -> io.Model:
        model = model.clone()
        model.model_options["context_handler"] = comfy.context_windows.IndexListContextHandler(
            context_schedule=comfy.context_windows.get_matching_context_schedule(context_schedule),
            fuse_method=comfy.context_windows.get_matching_fuse_method(fuse_method),
            context_length=context_length,
            context_overlap=context_overlap,
            context_stride=context_stride,
            closed_loop=closed_loop,
            dim=dim)
        create_prepare_sampling_wrapper(model.model_options)
        #create_outer_sampler_wrapper(model.model_options)
        return io.NodeOutput(model)


class WanContextWindowsNode(ContextWindowsNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "WanContextWindowsTest"
        schema.display_name = "Wan Context Windows Test"
        schema.description = "Test node for context windows (WAN)"
        # remove dim input; will always be 2
        schema.inputs = [x for x in schema.inputs if x.id != "dim"]
        # replace context_length input; should be in steps of 4
        context_length_idx = -1
        for idx, x in enumerate(schema.inputs):
            if x.id == "context_length":
                context_length_idx = idx
                break
        if context_length_idx == -1:
            raise Exception("Context length input not found in schema; did something change?")
        schema.inputs[context_length_idx] = io.Int.Input("context_length", min=1, max=nodes.MAX_RESOLUTION, step=4, default=81, tooltip="The length of the context window.")
        return schema

    @classmethod
    def execute(cls, model: io.Model.Type, context_length: int, context_overlap: int, context_schedule: str, context_stride: int, closed_loop: bool, fuse_method: str) -> io.Model:
        context_length = max(((context_length - 1) // 4) + 1, 1)  # at least length 1
        context_overlap = max(((context_overlap - 1) // 4) + 1, 0)  # at least overlap 0
        return super().execute(model, context_length, context_overlap, context_schedule, context_stride, closed_loop, fuse_method, 2)

class ContextWindowsExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            ContextWindowsNode,
            WanContextWindowsNode,
        ]

def comfy_entrypoint():
    return ContextWindowsExtension()
