from comfy_api.latest import io, ComfyExtension
import comfy.patcher_extension
import logging
import torch
import comfy.model_patcher

def easycache_forward_wrapper(executor, *args, **kwargs):
    # get values from args
    x: torch.Tensor = args[0]
    timestep: torch.Tensor = args[1]
    transformer_options: dict[str] = args[-1]
    # x: torch.Tensor = args[0]
    # timestep: torch.Tensor = args[4]
    # transformer_options: dict[str] = args[-2]
    easycache: EasyCacheHolder = transformer_options["easycache"]
    if easycache.is_past_end_timestep(timestep):
        return executor(*args, **kwargs)
    # prepare next x_prev
    next_x_prev = x.clone()
    do_easycache = easycache.should_do_easycache(timestep)
    logging.info(f"easycache_wrapper: do_easycache: {do_easycache}")
    output_prev_norm = None
    input_change = None
    if do_easycache:
        if easycache.has_x_prev():
            input_change = (x - easycache.x_prev).flatten().abs().mean()
        if easycache.has_output_prev() and easycache.has_relative_transformation_rate():
            output_prev_norm = easycache.output_prev.flatten().abs().mean()
            approx_output_change_rate = (easycache.relative_transformation_rate * input_change) / output_prev_norm
            easycache.cumulative_change_rate += approx_output_change_rate
            if easycache.cumulative_change_rate < easycache.reuse_threshold:
                logging.info(f"easycache_wrapper: skipping step; cumulative_change_rate: {easycache.cumulative_change_rate}, reuse_threshold: {easycache.reuse_threshold}")
                return x + easycache.cache_diff
            else:
                easycache.cumulative_change_rate = 0.0
                logging.info(f"easycache_wrapper: NOT skipping step; cumulative_change_rate: {easycache.cumulative_change_rate}, reuse_threshold: {easycache.reuse_threshold}")
                logging.info(f"easycache_wrapper: approx_output_change_rate: {approx_output_change_rate}")

    output: torch.Tensor = executor(*args, **kwargs)
    if easycache.has_output_prev():
        output_change = (output - easycache.output_prev).flatten().abs().mean()
        if output_prev_norm is None:
            output_prev_norm = easycache.output_prev.flatten().abs().mean()
        output_change_rate = output_change / output_prev_norm
        easycache.output_change_rates.append(output_change_rate.item())
        if easycache.has_relative_transformation_rate():
            approx_output_change_rate = (easycache.relative_transformation_rate * input_change) / output_prev_norm
            easycache.approx_output_change_rates.append(approx_output_change_rate.item())
            logging.info(f"easycache_wrapper: approx_output_change_rate: {approx_output_change_rate}")
        if input_change is not None:
            easycache.relative_transformation_rate = output_change / input_change
        logging.info(f"easycache_wrapper: output_change_rate: {output_change_rate}")
    easycache.cache_diff = output - next_x_prev
    easycache.x_prev = next_x_prev
    easycache.output_prev = output.clone()
    return output

def easycache_sample_wrapper(executor, *args, **kwargs):
    try:
        guider = executor.class_obj
        orig_model_options = guider.model_options
        guider.model_options = comfy.model_patcher.create_model_options_clone(orig_model_options)
        # clone and prepare timesteps
        guider.model_options["transformer_options"]["easycache"] = guider.model_options["transformer_options"]["easycache"].clone().prepare_timesteps(guider.model_patcher.model.model_sampling)
        return executor(*args, **kwargs)
    finally:
        output_change_rates = guider.model_options['transformer_options']['easycache'].output_change_rates
        approx_output_change_rates = guider.model_options['transformer_options']['easycache'].approx_output_change_rates
        logging.info(f"easycache_sample_wrapper: output_change_rates {len(output_change_rates)}: {output_change_rates}")
        logging.info(f"easycache_sample_wrapper: approx_output_change_rates {len(approx_output_change_rates)}: {approx_output_change_rates}")
        guider.model_options["transformer_options"]["easycache"].reset()
        guider.model_options = orig_model_options


class EasyCacheHolder:
    def __init__(self, reuse_threshold: float, start_percent: float, end_percent: float):
        self.reuse_threshold = reuse_threshold
        self.start_percent = start_percent
        self.end_percent = end_percent
        # timestep values
        self.start_t = 0.0
        self.end_t = 0.0
        # control values
        self.relative_transformation_rate: float = None
        self.cumulative_change_rate = 0.0
        # cache values
        self.x_prev = None
        self.output_prev = None
        self.cache_diff = None
        self.output_change_rates = []
        self.approx_output_change_rates = []

    def is_past_end_timestep(self, timestep: float) -> bool:
        return not (timestep > self.end_t).item()

    def should_do_easycache(self, timestep: float) -> bool:
        return (timestep <= self.start_t).item()

    def has_x_prev(self) -> bool:
        return self.x_prev is not None

    def has_output_prev(self) -> bool:
        return self.output_prev is not None

    def has_cache_diff(self) -> bool:
        return self.cache_diff is not None

    def has_relative_transformation_rate(self) -> bool:
        return self.relative_transformation_rate is not None

    def prepare_timesteps(self, model_sampling):
        self.start_t = model_sampling.percent_to_sigma(self.start_percent)
        self.end_t = model_sampling.percent_to_sigma(self.end_percent)
        return self

    def apply_cache(self):
        ...

    def accumulate_change(self):
        ...

    def reset(self):
        self.relative_transformation_rate = 0.0
        self.cumulative_change_rate = 0.0
        self.output_change_rates = []
        del self.x_prev
        self.x_prev = None
        del self.output_prev
        self.output_prev = None
        del self.cache_diff
        self.cache_diff = None
        return self

    def clone(self):
        return EasyCacheHolder(self.reuse_threshold, self.start_percent, self.end_percent)


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
                io.Float.Input("reuse_threshold", min=0.0, default=0.0, max=1.0, step=0.01, tooltip="The threshold for reusing cached steps."),
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
        model.model_options["transformer_options"]["easycache"] = EasyCacheHolder(reuse_threshold, start_percent, end_percent)
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
