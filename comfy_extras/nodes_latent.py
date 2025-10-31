import comfy.utils
import comfy_extras.nodes_post_processing
import torch
import nodes
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io


def reshape_latent_to(target_shape, latent, repeat_batch=True):
    if latent.shape[1:] != target_shape[1:]:
        latent = comfy.utils.common_upscale(latent, target_shape[-1], target_shape[-2], "bilinear", "center")
    if repeat_batch:
        return comfy.utils.repeat_to_batch_size(latent, target_shape[0])
    else:
        return latent


class LatentAdd(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LatentAdd",
            category="latent/advanced",
            inputs=[
                io.Latent.Input("samples1"),
                io.Latent.Input("samples2"),
            ],
            outputs=[
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(cls, samples1, samples2) -> io.NodeOutput:
        samples_out = samples1.copy()

        s1 = samples1["samples"]
        s2 = samples2["samples"]

        s2 = reshape_latent_to(s1.shape, s2)
        samples_out["samples"] = s1 + s2
        return io.NodeOutput(samples_out)

class LatentSubtract(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LatentSubtract",
            category="latent/advanced",
            inputs=[
                io.Latent.Input("samples1"),
                io.Latent.Input("samples2"),
            ],
            outputs=[
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(cls, samples1, samples2) -> io.NodeOutput:
        samples_out = samples1.copy()

        s1 = samples1["samples"]
        s2 = samples2["samples"]

        s2 = reshape_latent_to(s1.shape, s2)
        samples_out["samples"] = s1 - s2
        return io.NodeOutput(samples_out)

class LatentMultiply(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LatentMultiply",
            category="latent/advanced",
            inputs=[
                io.Latent.Input("samples"),
                io.Float.Input("multiplier", default=1.0, min=-10.0, max=10.0, step=0.01),
            ],
            outputs=[
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(cls, samples, multiplier) -> io.NodeOutput:
        samples_out = samples.copy()

        s1 = samples["samples"]
        samples_out["samples"] = s1 * multiplier
        return io.NodeOutput(samples_out)

class LatentInterpolate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LatentInterpolate",
            category="latent/advanced",
            inputs=[
                io.Latent.Input("samples1"),
                io.Latent.Input("samples2"),
                io.Float.Input("ratio", default=1.0, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(cls, samples1, samples2, ratio) -> io.NodeOutput:
        samples_out = samples1.copy()

        s1 = samples1["samples"]
        s2 = samples2["samples"]

        s2 = reshape_latent_to(s1.shape, s2)

        m1 = torch.linalg.vector_norm(s1, dim=(1))
        m2 = torch.linalg.vector_norm(s2, dim=(1))

        s1 = torch.nan_to_num(s1 / m1)
        s2 = torch.nan_to_num(s2 / m2)

        t = (s1 * ratio + s2 * (1.0 - ratio))
        mt = torch.linalg.vector_norm(t, dim=(1))
        st = torch.nan_to_num(t / mt)

        samples_out["samples"] = st * (m1 * ratio + m2 * (1.0 - ratio))
        return io.NodeOutput(samples_out)

class LatentConcat(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LatentConcat",
            category="latent/advanced",
            inputs=[
                io.Latent.Input("samples1"),
                io.Latent.Input("samples2"),
                io.Combo.Input("dim", options=["x", "-x", "y", "-y", "t", "-t"]),
            ],
            outputs=[
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(cls, samples1, samples2, dim) -> io.NodeOutput:
        samples_out = samples1.copy()

        s1 = samples1["samples"]
        s2 = samples2["samples"]
        s2 = comfy.utils.repeat_to_batch_size(s2, s1.shape[0])

        if "-" in dim:
            c = (s2, s1)
        else:
            c = (s1, s2)

        if "x" in dim:
            dim = -1
        elif "y" in dim:
            dim = -2
        elif "t" in dim:
            dim = -3

        samples_out["samples"] = torch.cat(c, dim=dim)
        return io.NodeOutput(samples_out)

class LatentCut(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LatentCut",
            category="latent/advanced",
            inputs=[
                io.Latent.Input("samples"),
                io.Combo.Input("dim", options=["x", "y", "t"]),
                io.Int.Input("index", default=0, min=-nodes.MAX_RESOLUTION, max=nodes.MAX_RESOLUTION, step=1),
                io.Int.Input("amount", default=1, min=1, max=nodes.MAX_RESOLUTION, step=1),
            ],
            outputs=[
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(cls, samples, dim, index, amount) -> io.NodeOutput:
        samples_out = samples.copy()

        s1 = samples["samples"]

        if "x" in dim:
            dim = s1.ndim - 1
        elif "y" in dim:
            dim = s1.ndim - 2
        elif "t" in dim:
            dim = s1.ndim - 3

        if index >= 0:
            index = min(index, s1.shape[dim] - 1)
            amount = min(s1.shape[dim] - index, amount)
        else:
            index = max(index, -s1.shape[dim])
            amount = min(-index, amount)

        samples_out["samples"] = torch.narrow(s1, dim, index, amount)
        return io.NodeOutput(samples_out)

class LatentBatch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LatentBatch",
            category="latent/batch",
            inputs=[
                io.Latent.Input("samples1"),
                io.Latent.Input("samples2"),
            ],
            outputs=[
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(cls, samples1, samples2) -> io.NodeOutput:
        samples_out = samples1.copy()
        s1 = samples1["samples"]
        s2 = samples2["samples"]

        s2 = reshape_latent_to(s1.shape, s2, repeat_batch=False)
        s = torch.cat((s1, s2), dim=0)
        samples_out["samples"] = s
        samples_out["batch_index"] = samples1.get("batch_index", [x for x in range(0, s1.shape[0])]) + samples2.get("batch_index", [x for x in range(0, s2.shape[0])])
        return io.NodeOutput(samples_out)

class LatentBatchSeedBehavior(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LatentBatchSeedBehavior",
            category="latent/advanced",
            inputs=[
                io.Latent.Input("samples"),
                io.Combo.Input("seed_behavior", options=["random", "fixed"], default="fixed"),
            ],
            outputs=[
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(cls, samples, seed_behavior) -> io.NodeOutput:
        samples_out = samples.copy()
        latent = samples["samples"]
        if seed_behavior == "random":
            if 'batch_index' in samples_out:
                samples_out.pop('batch_index')
        elif seed_behavior == "fixed":
            batch_number = samples_out.get("batch_index", [0])[0]
            samples_out["batch_index"] = [batch_number] * latent.shape[0]

        return io.NodeOutput(samples_out)

class LatentApplyOperation(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LatentApplyOperation",
            category="latent/advanced/operations",
            is_experimental=True,
            inputs=[
                io.Latent.Input("samples"),
                io.LatentOperation.Input("operation"),
            ],
            outputs=[
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(cls, samples, operation) -> io.NodeOutput:
        samples_out = samples.copy()

        s1 = samples["samples"]
        samples_out["samples"] = operation(latent=s1)
        return io.NodeOutput(samples_out)

class LatentApplyOperationCFG(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LatentApplyOperationCFG",
            category="latent/advanced/operations",
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.LatentOperation.Input("operation"),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model, operation) -> io.NodeOutput:
        m = model.clone()

        def pre_cfg_function(args):
            conds_out = args["conds_out"]
            if len(conds_out) == 2:
                conds_out[0] = operation(latent=(conds_out[0] - conds_out[1])) + conds_out[1]
            else:
                conds_out[0] = operation(latent=conds_out[0])
            return conds_out

        m.set_model_sampler_pre_cfg_function(pre_cfg_function)
        return io.NodeOutput(m)

class LatentOperationTonemapReinhard(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LatentOperationTonemapReinhard",
            category="latent/advanced/operations",
            is_experimental=True,
            inputs=[
                io.Float.Input("multiplier", default=1.0, min=0.0, max=100.0, step=0.01),
            ],
            outputs=[
                io.LatentOperation.Output(),
            ],
        )

    @classmethod
    def execute(cls, multiplier) -> io.NodeOutput:
        def tonemap_reinhard(latent, **kwargs):
            latent_vector_magnitude = (torch.linalg.vector_norm(latent, dim=(1)) + 0.0000000001)[:,None]
            normalized_latent = latent / latent_vector_magnitude

            mean = torch.mean(latent_vector_magnitude, dim=(1,2,3), keepdim=True)
            std = torch.std(latent_vector_magnitude, dim=(1,2,3), keepdim=True)

            top = (std * 5 + mean) * multiplier

            #reinhard
            latent_vector_magnitude *= (1.0 / top)
            new_magnitude = latent_vector_magnitude / (latent_vector_magnitude + 1.0)
            new_magnitude *= top

            return normalized_latent * new_magnitude
        return io.NodeOutput(tonemap_reinhard)

class LatentOperationSharpen(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LatentOperationSharpen",
            category="latent/advanced/operations",
            is_experimental=True,
            inputs=[
                io.Int.Input("sharpen_radius", default=9, min=1, max=31, step=1),
                io.Float.Input("sigma", default=1.0, min=0.1, max=10.0, step=0.1),
                io.Float.Input("alpha", default=0.1, min=0.0, max=5.0, step=0.01),
            ],
            outputs=[
                io.LatentOperation.Output(),
            ],
        )

    @classmethod
    def execute(cls, sharpen_radius, sigma, alpha) -> io.NodeOutput:
        def sharpen(latent, **kwargs):
            luminance = (torch.linalg.vector_norm(latent, dim=(1)) + 1e-6)[:,None]
            normalized_latent = latent / luminance
            channels = latent.shape[1]

            kernel_size = sharpen_radius * 2 + 1
            kernel = comfy_extras.nodes_post_processing.gaussian_kernel(kernel_size, sigma, device=luminance.device)
            center = kernel_size // 2

            kernel *= alpha * -10
            kernel[center, center] = kernel[center, center] - kernel.sum() + 1.0

            padded_image = torch.nn.functional.pad(normalized_latent, (sharpen_radius,sharpen_radius,sharpen_radius,sharpen_radius), 'reflect')
            sharpened = torch.nn.functional.conv2d(padded_image, kernel.repeat(channels, 1, 1).unsqueeze(1), padding=kernel_size // 2, groups=channels)[:,:,sharpen_radius:-sharpen_radius, sharpen_radius:-sharpen_radius]

            return luminance * sharpened
        return io.NodeOutput(sharpen)


class LatentExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LatentAdd,
            LatentSubtract,
            LatentMultiply,
            LatentInterpolate,
            LatentConcat,
            LatentCut,
            LatentBatch,
            LatentBatchSeedBehavior,
            LatentApplyOperation,
            LatentApplyOperationCFG,
            LatentOperationTonemapReinhard,
            LatentOperationSharpen,
        ]


async def comfy_entrypoint() -> LatentExtension:
    return LatentExtension()
