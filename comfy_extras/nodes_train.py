import datetime
import json
import logging
import math
import os

import numpy as np
import safetensors
import torch
from PIL import Image, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo

import comfy.samplers
import comfy.utils
import comfy_extras.nodes_custom_sampler
import folder_paths
import node_helpers
from comfy.cli_args import args
from comfy.comfy_types.node_typing import IO


class TrainSampler(comfy.samplers.Sampler):

    def __init__(self, loss_fn, optimizer, loss_callback=None):
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.loss_callback = loss_callback

    def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
        self.optimizer.zero_grad()
        noise = model_wrap.inner_model.model_sampling.noise_scaling(sigmas, noise, latent_image, False)
        latent = model_wrap.inner_model.model_sampling.noise_scaling(
            torch.zeros_like(sigmas),
            torch.zeros_like(noise, requires_grad=True),
            latent_image,
            False
        )

        # Ensure model is in training mode and computing gradients
        denoised = model_wrap(noise, sigmas, **extra_args)
        try:
            loss = self.loss_fn(denoised, latent.clone())
        except RuntimeError as e:
            if "does not require grad and does not have a grad_fn" in str(e):
                logging.info("WARNING: This is likely due to the model is loaded in inference mode.")
        loss.backward()
        logging.info(f"Current Training Loss: {loss.item():.6f}")
        if self.loss_callback:
            self.loss_callback(loss.item())

        self.optimizer.step()
        # torch.cuda.memory._dump_snapshot("trainn.pickle")
        # torch.cuda.memory._record_memory_history(enabled=None)
        return torch.zeros_like(latent_image)


class BiasDiff(torch.nn.Module):
    def __init__(self, bias):
        super().__init__()
        self.bias = bias

    def __call__(self, b):
        return b + self.bias

    def passive_memory_usage(self):
        return self.bias.nelement() * self.bias.element_size()

    def move_to(self, device):
        self.to(device=device)
        return self.passive_memory_usage()


class LoraDiff(torch.nn.Module):
    def __init__(self, lora_down, lora_up):
        super().__init__()
        self.lora_down = lora_down
        self.lora_up = lora_up

    def __call__(self, w):
        return w + (self.lora_up @ self.lora_down).reshape(w.shape)

    def passive_memory_usage(self):
        return self.lora_down.nelement() * self.lora_down.element_size() + self.lora_up.nelement() * self.lora_up.element_size()

    def move_to(self, device):
        self.to(device=device)
        return self.passive_memory_usage()


def load_and_process_images(image_files, input_dir, resize_method="None"):
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
    w, h = None, None

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


class LoadImageSetNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": (
                    [
                        f
                        for f in os.listdir(folder_paths.get_input_directory())
                        if f.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".jpe", ".apng", ".tif", ".tiff"))
                    ],
                    {"image_upload": True, "allow_batch": True},
                )
            },
            "optional": {
                "resize_method": (
                    ["None", "Stretch", "Crop", "Pad"],
                    {"default": "None"},
                ),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_images"
    CATEGORY = "loaders"
    EXPERIMENTAL = True
    DESCRIPTION = "Loads a batch of images from a directory for training."

    @classmethod
    def VALIDATE_INPUTS(s, images, resize_method):
        filenames = images[0] if isinstance(images[0], list) else images

        for image in filenames:
            if not folder_paths.exists_annotated_filepath(image):
                return "Invalid image file: {}".format(image)
        return True

    def load_images(self, input_files, resize_method):
        input_dir = folder_paths.get_input_directory()
        valid_extensions = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".jpe", ".apng", ".tif", ".tiff"]
        image_files = [
            f
            for f in input_files
            if any(f.lower().endswith(ext) for ext in valid_extensions)
        ]
        output_tensor = load_and_process_images(image_files, input_dir, resize_method)
        return (output_tensor,)


class LoadImageSetFromFolderNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder": (folder_paths.get_input_subfolders(), {"tooltip": "The folder to load images from."})
            },
            "optional": {
                "resize_method": (
                    ["None", "Stretch", "Crop", "Pad"],
                    {"default": "None"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_images"
    CATEGORY = "loaders"
    EXPERIMENTAL = True
    DESCRIPTION = "Loads a batch of images from a directory for training."

    def load_images(self, folder, resize_method):
        sub_input_dir = os.path.join(folder_paths.get_input_directory(), folder)
        valid_extensions = [".png", ".jpg", ".jpeg", ".webp"]
        image_files = [
            f
            for f in os.listdir(sub_input_dir)
            if any(f.lower().endswith(ext) for ext in valid_extensions)
        ]
        output_tensor = load_and_process_images(image_files, sub_input_dir, resize_method)
        return (output_tensor,)


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


class TrainLoraNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (IO.MODEL, {"tooltip": "The model to train the LoRA on."}),
                "vae": (
                    IO.VAE,
                    {
                        "tooltip": "The VAE model to use for encoding images for training."
                    },
                ),
                "positive": (
                    IO.CONDITIONING,
                    {"tooltip": "The positive conditioning to use for training."},
                ),
                "image": (
                    IO.IMAGE,
                    {"tooltip": "The image or image batch to train the LoRA on."},
                ),
                "batch_size": (
                    IO.INT,
                    {
                        "default": 1,
                        "min": 1,
                        "max": 10000,
                        "step": 1,
                        "tooltip": "The batch size to use for training.",
                    },
                ),
                "steps": (
                    IO.INT,
                    {
                        "default": 50,
                        "min": 1,
                        "max": 1000,
                        "tooltip": "The number of steps to train the LoRA for.",
                    },
                ),
                "learning_rate": (
                    IO.FLOAT,
                    {
                        "default": 0.0003,
                        "min": 0.0000001,
                        "max": 1.0,
                        "step": 0.00001,
                        "tooltip": "The learning rate to use for training.",
                    },
                ),
                "rank": (
                    IO.INT,
                    {
                        "default": 8,
                        "min": 1,
                        "max": 128,
                        "tooltip": "The rank of the LoRA layers.",
                    },
                ),
                "optimizer": (
                    ["Adam", "AdamW", "SGD", "RMSprop"],
                    {
                        "default": "Adam",
                        "tooltip": "The optimizer to use for training.",
                    },
                ),
                "loss_function": (
                    ["MSE", "L1", "Huber", "SmoothL1"],
                    {
                        "default": "MSE",
                        "tooltip": "The loss function to use for training.",
                    },
                ),
                "seed": (
                    IO.INT,
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "The seed to use for training (used in generator for LoRA weight initialization and noise sampling)",
                    },
                ),
                "training_dtype": (
                    ["bf16", "fp32"],
                    {"default": "bf16", "tooltip": "The dtype to use for training."},
                ),
                "existing_lora": (
                    folder_paths.get_filename_list("loras") + ["[None]"],
                    {
                        "default": "[None]",
                        "tooltip": "The existing LoRA to append to. Set to None for new LoRA.",
                    },
                ),
            },
        }

    RETURN_TYPES = (IO.MODEL, IO.LORA_MODEL, IO.LOSS_MAP, IO.INT)
    RETURN_NAMES = ("model_with_lora", "lora", "loss", "steps")
    FUNCTION = "train"
    CATEGORY = "training"
    EXPERIMENTAL = True

    def train(
        self,
        model,
        vae,
        positive,
        image,
        batch_size,
        steps,
        learning_rate,
        rank,
        optimizer,
        loss_function,
        seed,
        training_dtype,
        existing_lora,
    ):
        num_images = image.shape[0]
        indices = torch.randperm(num_images)[:batch_size]
        batch_tensor = image[indices]

        # Ensure we're not in inference mode when encoding
        encoded = vae.encode(batch_tensor)
        mp = model.clone()
        dtype = node_helpers.string_to_torch_dtype(training_dtype)
        mp.set_model_compute_dtype(dtype)

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

            for n, m in mp.model.named_modules():
                if hasattr(m, "weight_function"):
                    if m.weight is not None:
                        key = "{}.weight".format(n)
                        shape = m.weight.shape
                        if len(shape) >= 2:
                            in_dim = math.prod(shape[1:])
                            out_dim = shape[0]

                            # Check if we have existing weights for this layer
                            lora_up_key = "{}.lora_up.weight".format(n)
                            lora_down_key = "{}.lora_down.weight".format(n)

                            if existing_lora != "[None]" and (
                                    lora_up_key in existing_weights
                                    and lora_down_key in existing_weights
                            ):
                                # Initialize with existing weights
                                lora_up = torch.nn.Parameter(
                                        existing_weights[lora_up_key].to(dtype=dtype),
                                        requires_grad=True,
                                    )
                                lora_down = torch.nn.Parameter(
                                        existing_weights[lora_down_key].to(dtype=dtype),
                                        requires_grad=True,
                                    )
                            else:
                                if existing_lora != "[None]":
                                    logging.info(f"Warning: No existing weights found for {lora_up_key} or {lora_down_key}")
                                # Initialize new weights
                                lora_down = torch.nn.Parameter(
                                    torch.zeros(
                                        (
                                            rank,
                                            in_dim,
                                        ),
                                        dtype=dtype,
                                    ),
                                    requires_grad=True,
                                )
                                lora_up = torch.nn.Parameter(
                                    torch.zeros((out_dim, rank), dtype=dtype),
                                    requires_grad=True,
                                )
                                torch.nn.init.zeros_(lora_up)
                                torch.nn.init.kaiming_uniform_(
                                    lora_down, a=math.sqrt(5), generator=generator
                                )

                            lora_sd[lora_up_key] = lora_up
                            lora_sd[lora_down_key] = lora_down
                            mp.add_weight_wrapper(key, LoraDiff(lora_down, lora_up))
                        else:
                            diff = torch.nn.Parameter(
                                torch.zeros(
                                    m.weight.shape, dtype=dtype, requires_grad=True
                                )
                            )
                            mp.add_weight_wrapper(key, BiasDiff(diff))
                            lora_sd["{}.diff".format(n)] = diff
                    if hasattr(m, "bias") and m.bias is not None:
                        key = "{}.bias".format(n)
                        bias = torch.nn.Parameter(
                            torch.zeros(m.bias.shape, dtype=dtype, requires_grad=True)
                        )
                        lora_sd["{}.diff_b".format(n)] = bias
                        mp.add_weight_wrapper(key, BiasDiff(bias))

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

            # Setup sampler and guider like in test script
            loss_map = {"loss": []}
            loss_callback = lambda loss: loss_map["loss"].append(loss)
            train_sampler = TrainSampler(
                criterion, optimizer, loss_callback=loss_callback
            )
            guider = comfy_extras.nodes_custom_sampler.Guider_Basic(mp)
            guider.set_conds(positive)  # Set conditioning from input
            ss = comfy_extras.nodes_custom_sampler.SamplerCustomAdvanced()

            # yoland: this currently resize to the first image in the dataset

            # Training loop
            for step in range(steps):
                # Generate random sigma
                sigma = mp.model.model_sampling.percent_to_sigma(
                    torch.rand((1,)).item()
                )
                sigma = torch.tensor([sigma])

                noise = comfy_extras.nodes_custom_sampler.Noise_RandomNoise(step * 1000 + seed)

                ss.sample(
                    noise, guider, train_sampler, sigma, {"samples": encoded.clone()}
                )

            return (mp, lora_sd, loss_map, steps + existing_steps)


class SaveLoRA:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora": (
                    IO.LORA_MODEL,
                    {
                        "tooltip": "The LoRA model to save. Do not use the model with LoRA layers."
                    },
                ),
                "prefix": (
                    "STRING",
                    {
                        "default": "trained_lora",
                        "tooltip": "The prefix to use for the saved LoRA file.",
                    },
                ),
            },
            "optional": {
                "steps": (
                    IO.INT,
                    {
                        "forceInput": True,
                        "tooltip": "Optional: The number of steps to LoRA has been trained for, used to name the saved file.",
                    },
                ),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    CATEGORY = "loaders"
    EXPERIMENTAL = True
    OUTPUT_NODE = True

    def save(self, lora, prefix, steps=None):
        date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if steps is None:
            output_file = f"models/loras/{prefix}_{date}_lora.safetensors"
        else:
            output_file = f"models/loras/{prefix}_{steps}_steps_{date}_lora.safetensors"
        safetensors.torch.save_file(lora, output_file)
        return {}


class LossGraphNode:
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "loss": (IO.LOSS_MAP, {"default": {}}),
                "filename_prefix": (IO.STRING, {"default": "loss_graph"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "plot_loss"
    OUTPUT_NODE = True
    CATEGORY = "training"
    EXPERIMENTAL = True
    DESCRIPTION = "Plots the loss graph and saves it to the output directory."

    def plot_loss(self, loss, filename_prefix, prompt=None, extra_pnginfo=None):
        loss_values = loss["loss"]
        width, height = 500, 300
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

        metadata = None
        if not args.disable_metadata:
            metadata = PngInfo()
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata.add_text(x, json.dumps(extra_pnginfo[x]))

        date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        img.save(
            os.path.join(self.output_dir, f"{filename_prefix}_{date}.png"),
            pnginfo=metadata,
        )
        return {
            "ui": {
                "images": [
                    {
                        "filename": f"{filename_prefix}_{date}.png",
                        "subfolder": "",
                        "type": "temp",
                    }
                ]
            }
        }


NODE_CLASS_MAPPINGS = {
    "TrainLoraNode": TrainLoraNode,
    "SaveLoRANode": SaveLoRA,
    "LoadImageSetFromFolderNode": LoadImageSetFromFolderNode,
    "LossGraphNode": LossGraphNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TrainLoraNode": "Train LoRA",
    "SaveLoRANode": "Save LoRA Weights",
    "LoadImageSetFromFolderNode": "Load Image Dataset from Folder",
    "LossGraphNode": "Plot Loss Graph",
}
