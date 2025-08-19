from comfy_api.latest import io, ComfyExtension
import comfy.patcher_extension
import logging
import torch
import comfy.model_patcher

def easycache_sample_wrapper(executor, *args, **kwargs):
    try:
        guider = executor.class_obj
        orig_model_options = guider.model_options
        guider.model_options = comfy.model_patcher.create_model_options_clone(orig_model_options)
        if "easycache" in orig_model_options["transformer_options"]:
            guider.model_options["transformer_options"]["easycache"] = guider.model_options["transformer_options"]["easycache"].clone()
            guider.model_options["transformer_options"]["easycache"].dict["start_timestep"] = guider.model_patcher.model.model_sampling.percent_to_sigma(guider.model_options["transformer_options"]["easycache"].dict["start_percent"])
            guider.model_options["transformer_options"]["easycache"].dict["end_timestep"] = guider.model_patcher.model.model_sampling.percent_to_sigma(guider.model_options["transformer_options"]["easycache"].dict["end_percent"])
        return executor(*args, **kwargs)
    finally:
        guider.model_options = orig_model_options

def easycache_forward_wrapper(executor, *args, **kwargs):
    x: torch.Tensor = args[0]
    timestep: torch.Tensor = args[1]
    transformer_options = args[-1]
    do_easycache = timestep < transformer_options["easycache"].dict["start_timestep"] and timestep > transformer_options["easycache"].dict["end_timestep"]
    logging.info(f"easycache_wrapper: do_easycache: {do_easycache}")
    x_prev = None
    input_change = None
    # input_data = x.flatten().abs().mean()
    if do_easycache and "easycache" in transformer_options:
        if "x_prev" in transformer_options["easycache"].dict:
            x_prev = transformer_options["easycache"].dict["x_prev"]
        else:
            transformer_options["easycache"].dict["x_prev"] = x.clone()
    if x_prev is not None:
        input_change = (x_prev - x).flatten().abs().mean()
    if do_easycache and transformer_options["easycache"].dict.get("change_rate", None) is not None:
        change_rate = transformer_options["easycache"].dict["change_rate"]
        output_prev = transformer_options["easycache"].dict["output_prev"]
        pred_change = change_rate * (input_change / output_prev.flatten().abs().mean())
        accumulated_change = transformer_options["easycache"].dict["accumulated_change"] + pred_change
        if transformer_options["easycache"].dict["reuse_threshold"] <= accumulated_change:
            logging.info(f"easycache_wrapper: skipping step; accumulated_change: {accumulated_change}, reuse_threshold: {transformer_options['easycache'].dict['reuse_threshold']}")
            transformer_options["easycache"].dict["accumulated_change"] = 0.0
            return x + transformer_options["easycache"].dict["cache_diff"]
        else:
            transformer_options["easycache"].dict["accumulated_change"] = accumulated_change
            logging.info(f"easycache_wrapper: NOT skipping step; accumulated_change: {accumulated_change}, reuse_threshold: {transformer_options['easycache'].dict['reuse_threshold']}")
        logging.info(f"easycache_wrapper pred_change: {pred_change}")
    output: torch.Tensor = executor(*args, **kwargs)
    if x_prev is not None:
        # output_data = output.flatten().abs().mean()
        output_prev = transformer_options["easycache"].dict["output_prev"]
        output_change = (output_prev - output).flatten().abs().mean()
        k = output_change / input_change
        transformer_options["easycache"].dict["change_rate"] = k
        logging.info(f"easycache_wrapper: {input_change} {output_change} {k}")
    if do_easycache and "easycache" in transformer_options:
        transformer_options["easycache"].dict["output_prev"] = output.clone()
        transformer_options["easycache"].dict["cache_diff"] = output - x
    if not do_easycache:
        transformer_options["easycache"].dict["accumulated_change"] = 0.0
        transformer_options["easycache"].dict["change_rate"] = None
        transformer_options["easycache"].dict["output_prev"] = None
        transformer_options["easycache"].dict["cache_diff"] = None
    return output


class EasyCacheStore:
    def __init__(self, dict: dict):
        self.dict = dict

    def clone(self):
        return EasyCacheStore(self.dict.copy())


class EasyCacheNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EasyCache",
            display_name="Easy Cache",
            description="Easy Cache",
            category="advanced/debug/model",
            inputs=[
                io.Model.Input("model", tooltip="The model to add EasyCache to."),
                io.Float.Input("reuse_threshold", min=0.0, default=0.0, max=100.0, step=0.01, tooltip="The threshold for reusing cached steps."),
                io.Float.Input("start_percent", min=0.0, default=0.0, max=1.0, step=0.01, tooltip="The relative sampling step to begin use of EasyCache."),
                io.Float.Input("end_percent", min=0.0, default=1.0, max=1.0, step=0.01, tooltip="The relative sampling step to end use of EasyCache."),
            ],
            outputs=[
                io.Model.Output(tooltip="The model with EasyCache."),
            ],
        )

    @classmethod
    def execute(cls, model: io.Model.Type, reuse_threshold: float, start_percent: float, end_percent: float) -> io.NodeOutput:
        model = model.clone()
        easycache_dict = {
            "reuse_threshold": reuse_threshold,
            "start_percent": start_percent,
            "end_percent": end_percent,
            "accumulated_change": 0.0,
        }
        model.model_options["transformer_options"]["easycache"] = EasyCacheStore(easycache_dict)
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "easycache", easycache_forward_wrapper)
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, "easycache", easycache_sample_wrapper)
        return io.NodeOutput(model)

class EasyCacheExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            EasyCacheNode,
        ]

def comfy_entrypoint():
    return EasyCacheExtension()
