import logging
import os

import numpy as np
import safetensors
import torch
import torch.utils.checkpoint
from tqdm.auto import trange
from PIL import Image, ImageDraw, ImageFont
from typing_extensions import override

import comfy.samplers
import comfy.sd
import comfy.utils
import comfy.model_management
import comfy_extras.nodes_custom_sampler
import folder_paths
import node_helpers
from comfy.weight_adapter import adapters, adapter_maps
from comfy_api.latest import ComfyExtension, io, ui
from comfy.utils import ProgressBar


def make_batch_extra_option_dict(d, indicies, full_size=None):
    new_dict = {}
    for k, v in d.items():
        newv = v
        if isinstance(v, dict):
            newv = make_batch_extra_option_dict(v, indicies, full_size=full_size)
        elif isinstance(v, torch.Tensor):
            if full_size is None or v.size(0) == full_size:
                newv = v[indicies]
        elif isinstance(v, (list, tuple)) and len(v) == full_size:
            newv = [v[i] for i in indicies]
        new_dict[k] = newv
    return new_dict


def process_cond_list(d, prefix=""):
    if hasattr(d, "__iter__") and not hasattr(d, "items"):
        for index, item in enumerate(d):
            process_cond_list(item, f"{prefix}.{index}")
        return d
    elif hasattr(d, "items"):
        for k, v in list(d.items()):
            if isinstance(v, dict):
                process_cond_list(v, f"{prefix}.{k}")
            elif isinstance(v, torch.Tensor):
                d[k] = v.clone()
            elif isinstance(v, (list, tuple)):
                for index, item in enumerate(v):
                    process_cond_list(item, f"{prefix}.{k}.{index}")
    return d


class TrainSampler(comfy.samplers.Sampler):
    def __init__(
        self,
        loss_fn,
        optimizer,
        loss_callback=None,
        batch_size=1,
        grad_acc=1,
        total_steps=1,
        seed=0,
        training_dtype=torch.bfloat16,
        real_dataset=None,
    ):
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.loss_callback = loss_callback
        self.batch_size = batch_size
        self.total_steps = total_steps
        self.grad_acc = grad_acc
        self.seed = seed
        self.training_dtype = training_dtype
        self.real_dataset: list[torch.Tensor] | None = real_dataset

    def fwd_bwd(
        self,
        model_wrap,
        batch_sigmas,
        batch_noise,
        batch_latent,
        cond,
        indicies,
        extra_args,
        dataset_size,
        bwd=True,
    ):
        xt = model_wrap.inner_model.model_sampling.noise_scaling(
            batch_sigmas, batch_noise, batch_latent, False
        )
        x0 = model_wrap.inner_model.model_sampling.noise_scaling(
            torch.zeros_like(batch_sigmas),
            torch.zeros_like(batch_noise),
            batch_latent,
            False,
        )

        model_wrap.conds["positive"] = [cond[i] for i in indicies]
        batch_extra_args = make_batch_extra_option_dict(
            extra_args, indicies, full_size=dataset_size
        )

        with torch.autocast(xt.device.type, dtype=self.training_dtype):
            x0_pred = model_wrap(
                xt.requires_grad_(True),
                batch_sigmas.requires_grad_(True),
                **batch_extra_args,
            )
            loss = self.loss_fn(x0_pred, x0)
        if bwd:
            bwd_loss = loss / self.grad_acc
            bwd_loss.backward()
        return loss

    def sample(
        self,
        model_wrap,
        sigmas,
        extra_args,
        callback,
        noise,
        latent_image=None,
        denoise_mask=None,
        disable_pbar=False,
    ):
        model_wrap.conds = process_cond_list(model_wrap.conds)
        cond = model_wrap.conds["positive"]
        dataset_size = sigmas.size(0)
        torch.cuda.empty_cache()
        ui_pbar = ProgressBar(self.total_steps)
        for i in (
            pbar := trange(
                self.total_steps,
                desc="Training LoRA",
                smoothing=0.01,
                disable=not comfy.utils.PROGRESS_BAR_ENABLED,
            )
        ):
            noisegen = comfy_extras.nodes_custom_sampler.Noise_RandomNoise(
                self.seed + i * 1000
            )
            indicies = torch.randperm(dataset_size)[: self.batch_size].tolist()

            if self.real_dataset is None:
                batch_latent = torch.stack([latent_image[i] for i in indicies])
                batch_noise = noisegen.generate_noise({"samples": batch_latent}).to(
                    batch_latent.device
                )
                batch_sigmas = [
                    model_wrap.inner_model.model_sampling.percent_to_sigma(
                        torch.rand((1,)).item()
                    )
                    for _ in range(min(self.batch_size, dataset_size))
                ]
                batch_sigmas = torch.tensor(batch_sigmas).to(batch_latent.device)

                loss = self.fwd_bwd(
                    model_wrap,
                    batch_sigmas,
                    batch_noise,
                    batch_latent,
                    cond,
                    indicies,
                    extra_args,
                    dataset_size,
                    bwd=True,
                )
                if self.loss_callback:
                    self.loss_callback(loss.item())
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            else:
                total_loss = 0
                for index in indicies:
                    single_latent = self.real_dataset[index].to(latent_image)
                    batch_noise = noisegen.generate_noise(
                        {"samples": single_latent}
                    ).to(single_latent.device)
                    batch_sigmas = (
                        model_wrap.inner_model.model_sampling.percent_to_sigma(
                            torch.rand((1,)).item()
                        )
                    )
                    batch_sigmas = torch.tensor([batch_sigmas]).to(single_latent.device)
                    loss = self.fwd_bwd(
                        model_wrap,
                        batch_sigmas,
                        batch_noise,
                        single_latent,
                        cond,
                        [index],
                        extra_args,
                        dataset_size,
                        bwd=False,
                    )
                    total_loss += loss
                total_loss = total_loss / self.grad_acc / len(indicies)
                total_loss.backward()
                if self.loss_callback:
                    self.loss_callback(total_loss.item())
                pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})

            if (i + 1) % self.grad_acc == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                ui_pbar.update(1)
        torch.cuda.empty_cache()
        return torch.zeros_like(latent_image)


class BiasDiff(torch.nn.Module):
    def __init__(self, bias):
        super().__init__()
        self.bias = bias

    def __call__(self, b):
        org_dtype = b.dtype
        return (b.to(self.bias) + self.bias).to(org_dtype)

    def passive_memory_usage(self):
        return self.bias.nelement() * self.bias.element_size()

    def move_to(self, device):
        self.to(device=device)
        return self.passive_memory_usage()


def draw_loss_graph(loss_map, steps):
    width, height = 500, 300
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    min_loss, max_loss = min(loss_map.values()), max(loss_map.values())
    scaled_loss = [(l - min_loss) / (max_loss - min_loss) for l in loss_map.values()]

    prev_point = (0, height - int(scaled_loss[0] * height))
    for i, l in enumerate(scaled_loss[1:], start=1):
        x = int(i / (steps - 1) * width)
        y = height - int(l * height)
        draw.line([prev_point, (x, y)], fill="blue", width=2)
        prev_point = (x, y)

    return img


def find_all_highest_child_module_with_forward(
    model: torch.nn.Module, result=None, name=None
):
    if result is None:
        result = []
    elif hasattr(model, "forward") and not isinstance(
        model, (torch.nn.ModuleList, torch.nn.Sequential, torch.nn.ModuleDict)
    ):
        result.append(model)
        logging.debug(f"Found module with forward: {name} ({model.__class__.__name__})")
        return result
    name = name or "root"
    for next_name, child in model.named_children():
        find_all_highest_child_module_with_forward(child, result, f"{name}.{next_name}")
    return result


def patch(m):
    if not hasattr(m, "forward"):
        return
    org_forward = m.forward

    def fwd(args, kwargs):
        return org_forward(*args, **kwargs)

    def checkpointing_fwd(*args, **kwargs):
        return torch.utils.checkpoint.checkpoint(fwd, args, kwargs, use_reentrant=False)

    m.org_forward = org_forward
    m.forward = checkpointing_fwd


def unpatch(m):
    if hasattr(m, "org_forward"):
        m.forward = m.org_forward
        del m.org_forward


class TrainLoraNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TrainLoraNode",
            display_name="Train LoRA",
            category="training",
            is_experimental=True,
            is_input_list=True,  # All inputs become lists
            inputs=[
                io.Model.Input("model", tooltip="The model to train the LoRA on."),
                io.Latent.Input(
                    "latents",
                    tooltip="The Latents to use for training, serve as dataset/input of the model.",
                ),
                io.Conditioning.Input(
                    "positive", tooltip="The positive conditioning to use for training."
                ),
                io.Int.Input(
                    "batch_size",
                    default=1,
                    min=1,
                    max=10000,
                    tooltip="The batch size to use for training.",
                ),
                io.Int.Input(
                    "grad_accumulation_steps",
                    default=1,
                    min=1,
                    max=1024,
                    tooltip="The number of gradient accumulation steps to use for training.",
                ),
                io.Int.Input(
                    "steps",
                    default=16,
                    min=1,
                    max=100000,
                    tooltip="The number of steps to train the LoRA for.",
                ),
                io.Float.Input(
                    "learning_rate",
                    default=0.0005,
                    min=0.0000001,
                    max=1.0,
                    step=0.0000001,
                    tooltip="The learning rate to use for training.",
                ),
                io.Int.Input(
                    "rank",
                    default=8,
                    min=1,
                    max=128,
                    tooltip="The rank of the LoRA layers.",
                ),
                io.Combo.Input(
                    "optimizer",
                    options=["AdamW", "Adam", "SGD", "RMSprop"],
                    default="AdamW",
                    tooltip="The optimizer to use for training.",
                ),
                io.Combo.Input(
                    "loss_function",
                    options=["MSE", "L1", "Huber", "SmoothL1"],
                    default="MSE",
                    tooltip="The loss function to use for training.",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    tooltip="The seed to use for training (used in generator for LoRA weight initialization and noise sampling)",
                ),
                io.Combo.Input(
                    "training_dtype",
                    options=["bf16", "fp32"],
                    default="bf16",
                    tooltip="The dtype to use for training.",
                ),
                io.Combo.Input(
                    "lora_dtype",
                    options=["bf16", "fp32"],
                    default="bf16",
                    tooltip="The dtype to use for lora.",
                ),
                io.Combo.Input(
                    "algorithm",
                    options=list(adapter_maps.keys()),
                    default=list(adapter_maps.keys())[0],
                    tooltip="The algorithm to use for training.",
                ),
                io.Boolean.Input(
                    "gradient_checkpointing",
                    default=True,
                    tooltip="Use gradient checkpointing for training.",
                ),
                io.Combo.Input(
                    "existing_lora",
                    options=folder_paths.get_filename_list("loras") + ["[None]"],
                    default="[None]",
                    tooltip="The existing LoRA to append to. Set to None for new LoRA.",
                ),
            ],
            outputs=[
                io.Model.Output(
                    display_name="model", tooltip="Model with LoRA applied"
                ),
                io.Custom("LORA_MODEL").Output(
                    display_name="lora", tooltip="LoRA weights"
                ),
                io.Custom("LOSS_MAP").Output(
                    display_name="loss_map", tooltip="Loss history"
                ),
                io.Int.Output(display_name="steps", tooltip="Total training steps"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        latents,
        positive,
        batch_size,
        steps,
        grad_accumulation_steps,
        learning_rate,
        rank,
        optimizer,
        loss_function,
        seed,
        training_dtype,
        lora_dtype,
        algorithm,
        gradient_checkpointing,
        existing_lora,
    ):
        # Extract scalars from lists (due to is_input_list=True)
        model = model[0]
        batch_size = batch_size[0]
        steps = steps[0]
        grad_accumulation_steps = grad_accumulation_steps[0]
        learning_rate = learning_rate[0]
        rank = rank[0]
        optimizer = optimizer[0]
        loss_function = loss_function[0]
        seed = seed[0]
        training_dtype = training_dtype[0]
        lora_dtype = lora_dtype[0]
        algorithm = algorithm[0]
        gradient_checkpointing = gradient_checkpointing[0]
        existing_lora = existing_lora[0]

        # Handle latents - either single dict or list of dicts
        if len(latents) == 1:
            latents = latents[0]["samples"]  # Single latent dict
        else:
            latent_list = []
            for latent in latents:
                latent = latent["samples"]
                bs = latent.shape[0]
                if bs != 1:
                    for sub_latent in latent:
                        latent_list.append(sub_latent[None])
                else:
                    latent_list.append(latent)
            latents = latent_list

        # Handle conditioning - either single list or list of lists
        if len(positive) == 1:
            positive = positive[0]  # Single conditioning list
        else:
            # Multiple conditioning lists - flatten
            flat_positive = []
            for cond in positive:
                if isinstance(cond, list):
                    flat_positive.extend(cond)
                else:
                    flat_positive.append(cond)
            positive = flat_positive

        mp = model.clone()
        dtype = node_helpers.string_to_torch_dtype(training_dtype)
        lora_dtype = node_helpers.string_to_torch_dtype(lora_dtype)
        mp.set_model_compute_dtype(dtype)

        # latents here can be list of different size latent or one large batch
        if isinstance(latents, list):
            all_shapes = set()
            latents = [t.to(dtype) for t in latents]
            for latent in latents:
                all_shapes.add(latent.shape)
            logging.info(f"Latent shapes: {all_shapes}")
            if len(all_shapes) > 1:
                multi_res = True
            else:
                multi_res = False
                latents = torch.cat(latents, dim=0)
            num_images = len(latents)
        elif isinstance(latents, torch.Tensor):
            latents = latents.to(dtype)
            num_images = latents.shape[0]
        else:
            logging.error(f"Invalid latents type: {type(latents)}")

        logging.info(f"Total Images: {num_images}, Total Captions: {len(positive)}")
        if len(positive) == 1 and num_images > 1:
            positive = positive * num_images
        elif len(positive) != num_images:
            raise ValueError(
                f"Number of positive conditions ({len(positive)}) does not match number of images ({num_images})."
            )

        with torch.inference_mode(False):
            lora_sd = {}
            generator = torch.Generator()
            generator.manual_seed(seed)

            # Load existing LoRA weights if provided
            existing_weights = {}
            existing_steps = 0
            if existing_lora != "[None]":
                lora_path = folder_paths.get_full_path_or_raise("loras", existing_lora)
                # Extract steps from filename like "trained_lora_10_steps_20250225_203716"
                existing_steps = int(existing_lora.split("_steps_")[0].split("_")[-1])
                if lora_path:
                    existing_weights = comfy.utils.load_torch_file(lora_path)

            all_weight_adapters = []
            for n, m in mp.model.named_modules():
                if hasattr(m, "weight_function"):
                    if m.weight is not None:
                        key = "{}.weight".format(n)
                        shape = m.weight.shape
                        if len(shape) >= 2:
                            alpha = float(existing_weights.get(f"{key}.alpha", 1.0))
                            dora_scale = existing_weights.get(f"{key}.dora_scale", None)
                            for adapter_cls in adapters:
                                existing_adapter = adapter_cls.load(
                                    n, existing_weights, alpha, dora_scale
                                )
                                if existing_adapter is not None:
                                    break
                            else:
                                existing_adapter = None
                                adapter_cls = adapter_maps[algorithm]

                            if existing_adapter is not None:
                                train_adapter = existing_adapter.to_train().to(
                                    lora_dtype
                                )
                            else:
                                # Use LoRA with alpha=1.0 by default
                                train_adapter = adapter_cls.create_train(
                                    m.weight, rank=rank, alpha=1.0
                                ).to(lora_dtype)
                            for name, parameter in train_adapter.named_parameters():
                                lora_sd[f"{n}.{name}"] = parameter

                            mp.add_weight_wrapper(key, train_adapter)
                            all_weight_adapters.append(train_adapter)
                        else:
                            diff = torch.nn.Parameter(
                                torch.zeros(
                                    m.weight.shape, dtype=lora_dtype, requires_grad=True
                                )
                            )
                            diff_module = BiasDiff(diff)
                            mp.add_weight_wrapper(key, BiasDiff(diff))
                            all_weight_adapters.append(diff_module)
                            lora_sd["{}.diff".format(n)] = diff
                    if hasattr(m, "bias") and m.bias is not None:
                        key = "{}.bias".format(n)
                        bias = torch.nn.Parameter(
                            torch.zeros(
                                m.bias.shape, dtype=lora_dtype, requires_grad=True
                            )
                        )
                        bias_module = BiasDiff(bias)
                        lora_sd["{}.diff_b".format(n)] = bias
                        mp.add_weight_wrapper(key, BiasDiff(bias))
                        all_weight_adapters.append(bias_module)

            if optimizer == "Adam":
                optimizer = torch.optim.Adam(lora_sd.values(), lr=learning_rate)
            elif optimizer == "AdamW":
                optimizer = torch.optim.AdamW(lora_sd.values(), lr=learning_rate)
            elif optimizer == "SGD":
                optimizer = torch.optim.SGD(lora_sd.values(), lr=learning_rate)
            elif optimizer == "RMSprop":
                optimizer = torch.optim.RMSprop(lora_sd.values(), lr=learning_rate)

            # Setup loss function based on selection
            if loss_function == "MSE":
                criterion = torch.nn.MSELoss()
            elif loss_function == "L1":
                criterion = torch.nn.L1Loss()
            elif loss_function == "Huber":
                criterion = torch.nn.HuberLoss()
            elif loss_function == "SmoothL1":
                criterion = torch.nn.SmoothL1Loss()

            # setup models
            if gradient_checkpointing:
                for m in find_all_highest_child_module_with_forward(
                    mp.model.diffusion_model
                ):
                    patch(m)
            mp.model.requires_grad_(False)
            comfy.model_management.load_models_gpu(
                [mp], memory_required=1e20, force_full_load=True
            )

            # Setup sampler and guider like in test script
            loss_map = {"loss": []}

            def loss_callback(loss):
                loss_map["loss"].append(loss)

            train_sampler = TrainSampler(
                criterion,
                optimizer,
                loss_callback=loss_callback,
                batch_size=batch_size,
                grad_acc=grad_accumulation_steps,
                total_steps=steps * grad_accumulation_steps,
                seed=seed,
                training_dtype=dtype,
                real_dataset=latents if multi_res else None,
            )
            guider = comfy_extras.nodes_custom_sampler.Guider_Basic(mp)
            guider.set_conds(positive)  # Set conditioning from input

            # Training loop
            try:
                # Generate dummy sigmas and noise
                sigmas = torch.tensor(range(num_images))
                noise = comfy_extras.nodes_custom_sampler.Noise_RandomNoise(seed)
                if multi_res:
                    # use first latent as dummy latent if multi_res
                    latents = latents[0].repeat(num_images, 1, 1, 1)
                guider.sample(
                    noise.generate_noise({"samples": latents}),
                    latents,
                    train_sampler,
                    sigmas,
                    seed=noise.seed,
                )
            finally:
                for m in mp.model.modules():
                    unpatch(m)
            del train_sampler, optimizer

            for adapter in all_weight_adapters:
                adapter.requires_grad_(False)

            for param in lora_sd:
                lora_sd[param] = lora_sd[param].to(lora_dtype)

            return io.NodeOutput(mp, lora_sd, loss_map, steps + existing_steps)


class LoraModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoraModelLoader",
            display_name="Load LoRA Model",
            category="loaders",
            is_experimental=True,
            inputs=[
                io.Model.Input(
                    "model", tooltip="The diffusion model the LoRA will be applied to."
                ),
                io.Custom("LORA_MODEL").Input(
                    "lora", tooltip="The LoRA model to apply to the diffusion model."
                ),
                io.Float.Input(
                    "strength_model",
                    default=1.0,
                    min=-100.0,
                    max=100.0,
                    tooltip="How strongly to modify the diffusion model. This value can be negative.",
                ),
            ],
            outputs=[
                io.Model.Output(
                    display_name="model", tooltip="The modified diffusion model."
                ),
            ],
        )

    @classmethod
    def execute(cls, model, lora, strength_model):
        if strength_model == 0:
            return io.NodeOutput(model)

        model_lora, _ = comfy.sd.load_lora_for_models(
            model, None, lora, strength_model, 0
        )
        return io.NodeOutput(model_lora)


class SaveLoRA(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveLoRA",
            display_name="Save LoRA Weights",
            category="loaders",
            is_experimental=True,
            is_output_node=True,
            inputs=[
                io.Custom("LORA_MODEL").Input(
                    "lora",
                    tooltip="The LoRA model to save. Do not use the model with LoRA layers.",
                ),
                io.String.Input(
                    "prefix",
                    default="loras/ComfyUI_trained_lora",
                    tooltip="The prefix to use for the saved LoRA file.",
                ),
                io.Int.Input(
                    "steps",
                    optional=True,
                    tooltip="Optional: The number of steps to LoRA has been trained for, used to name the saved file.",
                ),
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, lora, prefix, steps=None):
        output_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(prefix, output_dir)
        )
        if steps is None:
            output_checkpoint = f"{filename}_{counter:05}_.safetensors"
        else:
            output_checkpoint = f"{filename}_{steps}_steps_{counter:05}_.safetensors"
        output_checkpoint = os.path.join(full_output_folder, output_checkpoint)
        safetensors.torch.save_file(lora, output_checkpoint)
        return io.NodeOutput()


class LossGraphNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LossGraphNode",
            display_name="Plot Loss Graph",
            category="training",
            is_experimental=True,
            is_output_node=True,
            inputs=[
                io.Custom("LOSS_MAP").Input(
                    "loss", tooltip="Loss map from training node."
                ),
                io.String.Input(
                    "filename_prefix",
                    default="loss_graph",
                    tooltip="Prefix for the saved loss graph image.",
                ),
            ],
            outputs=[],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
        )

    @classmethod
    def execute(cls, loss, filename_prefix, prompt=None, extra_pnginfo=None):
        loss_values = loss["loss"]
        width, height = 800, 480
        margin = 40

        img = Image.new(
            "RGB", (width + margin, height + margin), "white"
        )  # Extend canvas
        draw = ImageDraw.Draw(img)

        min_loss, max_loss = min(loss_values), max(loss_values)
        scaled_loss = [(l - min_loss) / (max_loss - min_loss) for l in loss_values]

        steps = len(loss_values)

        prev_point = (margin, height - int(scaled_loss[0] * height))
        for i, l in enumerate(scaled_loss[1:], start=1):
            x = margin + int(i / steps * width)  # Scale X properly
            y = height - int(l * height)
            draw.line([prev_point, (x, y)], fill="blue", width=2)
            prev_point = (x, y)

        draw.line([(margin, 0), (margin, height)], fill="black", width=2)  # Y-axis
        draw.line(
            [(margin, height), (width + margin, height)], fill="black", width=2
        )  # X-axis

        font = None
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except IOError:
            font = ImageFont.load_default()

        # Add axis labels
        draw.text((5, height // 2), "Loss", font=font, fill="black")
        draw.text((width // 2, height + 10), "Steps", font=font, fill="black")

        # Add min/max loss values
        draw.text((margin - 30, 0), f"{max_loss:.2f}", font=font, fill="black")
        draw.text(
            (margin - 30, height - 10), f"{min_loss:.2f}", font=font, fill="black"
        )

        # Convert PIL image to tensor for PreviewImage
        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)[None,]  # [1, H, W, 3]

        # Return preview UI
        return io.NodeOutput(ui=ui.PreviewImage(img_tensor, cls=cls))


# ========== Extension Setup ==========


class TrainingExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            TrainLoraNode,
            LoraModelLoader,
            SaveLoRA,
            LossGraphNode,
        ]


async def comfy_entrypoint() -> TrainingExtension:
    return TrainingExtension()
