from __future__ import annotations

import logging
import os

import numpy as np
import safetensors
import torch
import torch.utils.checkpoint
import tqdm
from PIL import Image, ImageDraw, ImageFont

import comfy.model_management
import comfy.samplers
import comfy.sd
import comfy.utils
import comfy_extras.nodes_custom_sampler
import folder_paths
import node_helpers
from comfy.weight_adapter import adapter_maps, adapters
from comfy_api.latest import io, ui


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


class TrainSampler(comfy.samplers.Sampler):

    def __init__(self, loss_fn, optimizer, loss_callback=None, batch_size=1, grad_acc=1, total_steps=1, seed=0, training_dtype=torch.bfloat16):
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.loss_callback = loss_callback
        self.batch_size = batch_size
        self.total_steps = total_steps
        self.grad_acc = grad_acc
        self.seed = seed
        self.training_dtype = training_dtype

    def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
        cond = model_wrap.conds["positive"]
        dataset_size = sigmas.size(0)
        torch.cuda.empty_cache()
        for i in (pbar:=tqdm.trange(self.total_steps, desc="Training LoRA", smoothing=0.01, disable=not comfy.utils.PROGRESS_BAR_ENABLED)):
            noisegen = comfy_extras.nodes_custom_sampler.Noise_RandomNoise(self.seed + i * 1000)
            indicies = torch.randperm(dataset_size)[:self.batch_size].tolist()

            batch_latent = torch.stack([latent_image[i] for i in indicies])
            batch_noise = noisegen.generate_noise({"samples": batch_latent}).to(batch_latent.device)
            batch_sigmas = [
                model_wrap.inner_model.model_sampling.percent_to_sigma(
                    torch.rand((1,)).item()
                ) for _ in range(min(self.batch_size, dataset_size))
            ]
            batch_sigmas = torch.tensor(batch_sigmas).to(batch_latent.device)

            xt = model_wrap.inner_model.model_sampling.noise_scaling(
                batch_sigmas,
                batch_noise,
                batch_latent,
                False
            )
            x0 = model_wrap.inner_model.model_sampling.noise_scaling(
                torch.zeros_like(batch_sigmas),
                torch.zeros_like(batch_noise),
                batch_latent,
                False
            )

            model_wrap.conds["positive"] = [
                cond[i] for i in indicies
            ]
            batch_extra_args = make_batch_extra_option_dict(extra_args, indicies, full_size=dataset_size)

            with torch.autocast(xt.device.type, dtype=self.training_dtype):
                x0_pred = model_wrap(xt, batch_sigmas, **batch_extra_args)
                loss = self.loss_fn(x0_pred, x0)
            loss.backward()
            if self.loss_callback:
                self.loss_callback(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if (i + 1) % self.grad_acc == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
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


def load_and_process_images(image_files, input_dir, resize_method="None", w=None, h=None):
    """Utility function to load and process a list of images.

    Args:
        image_files: List of image filenames
        input_dir: Base directory containing the images
        resize_method: How to handle images of different sizes ("None", "Stretch", "Crop", "Pad")

    Returns:
        torch.Tensor: Batch of processed images
    """
    if not image_files:
        raise ValueError("No valid images found in input")

    output_images = []

    for file in image_files:
        image_path = os.path.join(input_dir, file)
        img = node_helpers.pillow(Image.open, image_path)

        if img.mode == "I":
            img = img.point(lambda i: i * (1 / 255))
        img = img.convert("RGB")

        if w is None and h is None:
            w, h = img.size[0], img.size[1]

        # Resize image to first image
        if img.size[0] != w or img.size[1] != h:
            if resize_method == "Stretch":
                img = img.resize((w, h), Image.Resampling.LANCZOS)
            elif resize_method == "Crop":
                img = img.crop((0, 0, w, h))
            elif resize_method == "Pad":
                img = img.resize((w, h), Image.Resampling.LANCZOS)
            elif resize_method == "None":
                raise ValueError(
                    "Your input image size does not match the first image in the dataset. Either select a valid resize method or use the same size for all images."
                )

        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)[None,]
        output_images.append(img_tensor)

    return torch.cat(output_images, dim=0)


def draw_loss_graph(loss_map, steps):
    width, height = 500, 300
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    min_loss, max_loss = min(loss_map.values()), max(loss_map.values())
    scaled_loss = [(l_v - min_loss) / (max_loss - min_loss) for l_v in loss_map.values()]

    prev_point = (0, height - int(scaled_loss[0] * height))
    for i, l_v in enumerate(scaled_loss[1:], start=1):
        x = int(i / (steps - 1) * width)
        y = height - int(l_v * height)
        draw.line([prev_point, (x, y)], fill="blue", width=2)
        prev_point = (x, y)

    return img


def find_all_highest_child_module_with_forward(model: torch.nn.Module, result = None, name = None):
    if result is None:
        result = []
    elif hasattr(model, "forward") and not isinstance(model, (torch.nn.ModuleList, torch.nn.Sequential, torch.nn.ModuleDict)):
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
        return torch.utils.checkpoint.checkpoint(
            fwd, args, kwargs, use_reentrant=False
        )
    m.org_forward = org_forward
    m.forward = checkpointing_fwd


def unpatch(m):
    if hasattr(m, "org_forward"):
        m.forward = m.org_forward
        del m.org_forward


class LoadImageSetFromFolderNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadImageSetFromFolderNode_V3",
            display_name="Load Image Dataset from Folder _V3",
            category="loaders",
            description="Loads a batch of images from a directory for training.",
            is_experimental=True,
            inputs=[
                io.Combo.Input(
                    "folder", options=folder_paths.get_input_subfolders(), tooltip="The folder to load images from."
                ),
                io.Combo.Input(
                    "resize_method", options=["None", "Stretch", "Crop", "Pad"], default="None", optional=True
                ),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, folder, resize_method="None"):
        sub_input_dir = os.path.join(folder_paths.get_input_directory(), folder)
        valid_extensions = [".png", ".jpg", ".jpeg", ".webp"]
        image_files = [
            f
            for f in os.listdir(sub_input_dir)
            if any(f.lower().endswith(ext) for ext in valid_extensions)
        ]
        return io.NodeOutput(load_and_process_images(image_files, sub_input_dir, resize_method))


class LoadImageTextSetFromFolderNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadImageTextSetFromFolderNode_V3",
            display_name="Load Image and Text Dataset from Folder _V3",
            category="loaders",
            description="Loads a batch of images and caption from a directory for training.",
            is_experimental=True,
            inputs=[
                io.Combo.Input("folder", options=folder_paths.get_input_subfolders(), tooltip="The folder to load images from."),
                io.Clip.Input("clip", tooltip="The CLIP model used for encoding the text."),
                io.Combo.Input("resize_method", options=["None", "Stretch", "Crop", "Pad"], default="None", optional=True),
                io.Int.Input("width", default=-1, min=-1, max=10000, step=1, tooltip="The width to resize the images to. -1 means use the original width.", optional=True),
                io.Int.Input("height", default=-1, min=-1, max=10000, step=1, tooltip="The height to resize the images to. -1 means use the original height.", optional=True),
            ],
            outputs=[
                io.Image.Output(),
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, folder, clip, resize_method="None", width=None, height=None):
        if clip is None:
            raise RuntimeError(
                "ERROR: clip input is invalid: None\n\n"
                "If the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model."
            )

        logging.info(f"Loading images from folder: {folder}")

        sub_input_dir = os.path.join(folder_paths.get_input_directory(), folder)
        valid_extensions = [".png", ".jpg", ".jpeg", ".webp"]

        image_files = []
        for item in os.listdir(sub_input_dir):
            path = os.path.join(sub_input_dir, item)
            if any(item.lower().endswith(ext) for ext in valid_extensions):
                image_files.append(path)
            elif os.path.isdir(path):
                # Support kohya-ss/sd-scripts folder structure
                repeat = 1
                if item.split("_")[0].isdigit():
                    repeat = int(item.split("_")[0])
                image_files.extend([
                    os.path.join(path, f) for f in os.listdir(path) if any(f.lower().endswith(ext) for ext in valid_extensions)
                ] * repeat)

        caption_file_path = [
            f.replace(os.path.splitext(f)[1], ".txt")
            for f in image_files
        ]
        captions = []
        for caption_file in caption_file_path:
            caption_path = os.path.join(sub_input_dir, caption_file)
            if os.path.exists(caption_path):
                with open(caption_path, "r", encoding="utf-8") as f:
                    caption = f.read().strip()
                    captions.append(caption)
            else:
                captions.append("")

        width = width if width != -1 else None
        height = height if height != -1 else None
        output_tensor = load_and_process_images(image_files, sub_input_dir, resize_method, width, height)

        logging.info(f"Loaded {len(output_tensor)} images from {sub_input_dir}.")

        logging.info(f"Encoding captions from {sub_input_dir}.")
        conditions = []
        empty_cond = clip.encode_from_tokens_scheduled(clip.tokenize(""))
        for text in captions:
            if text == "":
                conditions.append(empty_cond)
            tokens = clip.tokenize(text)
            conditions.extend(clip.encode_from_tokens_scheduled(tokens))
        logging.info(f"Encoded {len(conditions)} captions from {sub_input_dir}.")
        return io.NodeOutput(output_tensor, conditions)


class LoraModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoraModelLoader_V3",
            display_name="Load LoRA Model _V3",
            category="loaders",
            description="Load Trained LoRA weights from Train LoRA node.",
            is_experimental=True,
            inputs=[
                io.Model.Input("model", tooltip="The diffusion model the LoRA will be applied to."),
                io.LoraModel.Input("lora", tooltip="The LoRA model to apply to the diffusion model."),
                io.Float.Input("strength_model", default=1.0, min=-100.0, max=100.0, step=0.01, tooltip="How strongly to modify the diffusion model. This value can be negative."),
            ],
            outputs=[
                io.Model.Output(tooltip="The modified diffusion model."),
            ],
        )

    @classmethod
    def execute(cls, model, lora, strength_model):
        if strength_model == 0:
            return io.NodeOutput(model)

        model_lora, _ = comfy.sd.load_lora_for_models(model, None, lora, strength_model, 0)
        return io.NodeOutput(model_lora)


class LossGraphNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LossGraphNode_V3",
            display_name="Plot Loss Graph _V3",
            category="training",
            description="Plots the loss graph and saves it to the output directory.",
            is_experimental=True,
            is_output_node=True,
            inputs=[
                io.LossMap.Input("loss"),  # TODO: original V1 node has also `default={}` parameter
                io.String.Input("filename_prefix", default="loss_graph"),
            ],
            outputs=[],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
        )

    @classmethod
    def execute(cls, loss, filename_prefix):
        loss_values = loss["loss"]
        width, height = 800, 480
        margin = 40

        img = Image.new(
            "RGB", (width + margin, height + margin), "white"
        )  # Extend canvas
        draw = ImageDraw.Draw(img)

        min_loss, max_loss = min(loss_values), max(loss_values)
        scaled_loss = [(l_v - min_loss) / (max_loss - min_loss) for l_v in loss_values]

        steps = len(loss_values)

        prev_point = (margin, height - int(scaled_loss[0] * height))
        for i, l_v in enumerate(scaled_loss[1:], start=1):
            x = margin + int(i / steps * width)  # Scale X properly
            y = height - int(l_v * height)
            draw.line([prev_point, (x, y)], fill="blue", width=2)
            prev_point = (x, y)

        draw.line([(margin, 0), (margin, height)], fill="black", width=2)  # Y-axis
        draw.line(
            [(margin, height), (width + margin, height)], fill="black", width=2
        )  # X-axis

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
        return io.NodeOutput(ui=ui.PreviewImage(img, cls=cls))


class SaveLoRA(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveLoRA_V3",
            display_name="Save LoRA Weights _V3",
            category="loaders",
            is_experimental=True,
            is_output_node=True,
            inputs=[
                io.LoraModel.Input("lora", tooltip="The LoRA model to save. Do not use the model with LoRA layers."),
                io.String.Input("prefix", default="loras/ComfyUI_trained_lora", tooltip="The prefix to use for the saved LoRA file."),
                io.Int.Input("steps", tooltip="Optional: The number of steps to LoRA has been trained for, used to name the saved file.", optional=True),
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, lora, prefix, steps=None):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            prefix, folder_paths.get_output_directory()
        )
        if steps is None:
            output_checkpoint = f"{filename}_{counter:05}_.safetensors"
        else:
            output_checkpoint = f"{filename}_{steps}_steps_{counter:05}_.safetensors"
        output_checkpoint = os.path.join(full_output_folder, output_checkpoint)
        safetensors.torch.save_file(lora, output_checkpoint)
        return io.NodeOutput()


class TrainLoraNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TrainLoraNode_V3",
            display_name="Train LoRA _V3",
            category="training",
            is_experimental=True,
            inputs=[
                io.Model.Input("model", tooltip="The model to train the LoRA on."),
                io.Latent.Input("latents", tooltip="The Latents to use for training, serve as dataset/input of the model."),
                io.Conditioning.Input("positive", tooltip="The positive conditioning to use for training."),
                io.Int.Input("batch_size", default=1, min=1, max=10000, step=1, tooltip="The batch size to use for training."),
                io.Int.Input("grad_accumulation_steps", default=1, min=1, max=1024, step=1, tooltip="The number of gradient accumulation steps to use for training."),
                io.Int.Input("steps", default=16, min=1, max=100000, tooltip="The number of steps to train the LoRA for."),
                io.Float.Input("learning_rate", default=0.0005, min=0.0000001, max=1.0, step=0.000001, tooltip="The learning rate to use for training."),
                io.Int.Input("rank", default=8, min=1, max=128, tooltip="The rank of the LoRA layers."),
                io.Combo.Input("optimizer", options=["AdamW", "Adam", "SGD", "RMSprop"], default="AdamW", tooltip="The optimizer to use for training."),
                io.Combo.Input("loss_function", options=["MSE", "L1", "Huber", "SmoothL1"], default="MSE", tooltip="The loss function to use for training."),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, tooltip="The seed to use for training (used in generator for LoRA weight initialization and noise sampling)"),
                io.Combo.Input("training_dtype", options=["bf16", "fp32"], default="bf16", tooltip="The dtype to use for training."),
                io.Combo.Input("lora_dtype", options=["bf16", "fp32"], default="bf16", tooltip="The dtype to use for lora."),
                io.Combo.Input("algorithm", options=list(adapter_maps.keys()), default=list(adapter_maps.keys())[0], tooltip="The algorithm to use for training."),
                io.Boolean.Input("gradient_checkpointing", default=True, tooltip="Use gradient checkpointing for training."),
                io.Combo.Input("existing_lora", options=folder_paths.get_filename_list("loras") + ["[None]"], default="[None]", tooltip="The existing LoRA to append to. Set to None for new LoRA."),
            ],
            outputs=[
                io.Model.Output(display_name="model_with_lora"),
                io.LoraModel.Output(display_name="lora"),
                io.LossMap.Output(display_name="loss"),
                io.Int.Output(display_name="steps"),
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
        mp = model.clone()
        dtype = node_helpers.string_to_torch_dtype(training_dtype)
        lora_dtype = node_helpers.string_to_torch_dtype(lora_dtype)
        mp.set_model_compute_dtype(dtype)

        latents = latents["samples"].to(dtype)
        num_images = latents.shape[0]
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
                            dora_scale = existing_weights.get(
                                f"{key}.dora_scale", None
                            )
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
                                train_adapter = existing_adapter.to_train().to(lora_dtype)
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
                            torch.zeros(m.bias.shape, dtype=lora_dtype, requires_grad=True)
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
                for m in find_all_highest_child_module_with_forward(mp.model.diffusion_model):
                    patch(m)
            mp.model.requires_grad_(False)
            comfy.model_management.load_models_gpu([mp], memory_required=1e20, force_full_load=True)

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
                training_dtype=dtype
            )
            guider = comfy_extras.nodes_custom_sampler.Guider_Basic(mp)
            guider.set_conds(positive)  # Set conditioning from input

            # Training loop
            try:
                # Generate dummy sigmas and noise
                sigmas = torch.tensor(range(num_images))
                noise = comfy_extras.nodes_custom_sampler.Noise_RandomNoise(seed)
                guider.sample(
                    noise.generate_noise({"samples": latents}),
                    latents,
                    train_sampler,
                    sigmas,
                    seed=noise.seed
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


NODES_LIST = [
    LoadImageSetFromFolderNode,
    LoadImageTextSetFromFolderNode,
    LoraModelLoader,
    LossGraphNode,
    SaveLoRA,
    TrainLoraNode,
]
