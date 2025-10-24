import node_helpers
import comfy.utils
import comfy.conds
import math
import torch
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io


class TextEncodeQwenImageEdit(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodeQwenImageEdit",
            category="advanced/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Vae.Input("vae", optional=True),
                io.Image.Input("image", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, vae=None, image=None) -> io.NodeOutput:
        ref_latent = None
        if image is None:
            images = []
        else:
            samples = image.movedim(-1, 1)
            total = int(1024 * 1024)

            scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
            width = round(samples.shape[3] * scale_by)
            height = round(samples.shape[2] * scale_by)

            s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
            image = s.movedim(1, -1)
            images = [image[:, :, :, :3]]
            if vae is not None:
                ref_latent = vae.encode(image[:, :, :, :3])

        tokens = clip.tokenize(prompt, images=images)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if ref_latent is not None:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": [ref_latent]}, append=True)
        return io.NodeOutput(conditioning)


class TextEncodeQwenImageEditPlus(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodeQwenImageEditPlus",
            category="advanced/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Vae.Input("vae", optional=True),
                io.Image.Input("image1", optional=True),
                io.Image.Input("image2", optional=True),
                io.Image.Input("image3", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, vae=None, image1=None, image2=None, image3=None) -> io.NodeOutput:
        ref_latents = []
        images = [image1, image2, image3]
        images_vl = []
        llama_template = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        image_prompt = ""

        for i, image in enumerate(images):
            if image is not None:
                samples = image.movedim(-1, 1)
                total = int(384 * 384)

                scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                width = round(samples.shape[3] * scale_by)
                height = round(samples.shape[2] * scale_by)

                s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                images_vl.append(s.movedim(1, -1))
                if vae is not None:
                    total = int(1024 * 1024)
                    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                    width = round(samples.shape[3] * scale_by / 8.0) * 8
                    height = round(samples.shape[2] * scale_by / 8.0) * 8

                    s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                    ref_latents.append(vae.encode(s.movedim(1, -1)[:, :, :, :3]))

                image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)

        tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
        return io.NodeOutput(conditioning)


class TextEncodeQwenImageEliGen(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodeQwenImageEliGen",
            category="advanced/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.Conditioning.Input("global_conditioning"),
                io.Latent.Input("latent"),
                io.Image.Input("entity_mask_1", optional=True),
                io.String.Input("entity_prompt_1", multiline=True, dynamic_prompts=True, default=""),
                io.Image.Input("entity_mask_2", optional=True),
                io.String.Input("entity_prompt_2", multiline=True, dynamic_prompts=True, default=""),
                io.Image.Input("entity_mask_3", optional=True),
                io.String.Input("entity_prompt_3", multiline=True, dynamic_prompts=True, default=""),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, global_conditioning, latent, entity_prompt_1="", entity_mask_1=None,
                entity_prompt_2="", entity_mask_2=None, entity_prompt_3="", entity_mask_3=None) -> io.NodeOutput:

        # Extract dimensions from latent tensor
        # latent["samples"] shape: [batch, channels, latent_h, latent_w]
        latent_samples = latent["samples"]
        unpadded_latent_height = latent_samples.shape[2]  # Unpadded latent space
        unpadded_latent_width = latent_samples.shape[3]   # Unpadded latent space

        # Calculate padded dimensions (same logic as model's pad_to_patch_size with patch_size=2)
        # The model pads latents to be multiples of patch_size (2 for Qwen)
        patch_size = 2
        pad_h = (patch_size - unpadded_latent_height % patch_size) % patch_size
        pad_w = (patch_size - unpadded_latent_width % patch_size) % patch_size
        latent_height = unpadded_latent_height + pad_h  # Padded latent dimensions
        latent_width = unpadded_latent_width + pad_w     # Padded latent dimensions

        height = latent_height * 8  # Convert to pixel space for logging
        width = latent_width * 8

        if pad_h > 0 or pad_w > 0:
            print(f"[EliGen] Latent padding detected: {unpadded_latent_height}x{unpadded_latent_width} → {latent_height}x{latent_width}")
        print(f"[EliGen] Target generation dimensions: {height}x{width} pixels ({latent_height}x{latent_width} latent)")

        # Collect entity prompts and masks
        entity_prompts = [entity_prompt_1, entity_prompt_2, entity_prompt_3]
        entity_masks_raw = [entity_mask_1, entity_mask_2, entity_mask_3]

        # Filter out entities with empty prompts or missing masks
        valid_entities = []
        for prompt, mask in zip(entity_prompts, entity_masks_raw):
            if prompt.strip() and mask is not None:
                valid_entities.append((prompt, mask))

        # Log warning if some entities were skipped
        total_prompts_provided = len([p for p in entity_prompts if p.strip()])
        if len(valid_entities) < total_prompts_provided:
            print(f"[EliGen] Warning: Only {len(valid_entities)} of {total_prompts_provided} entity prompts have valid masks")

        # If no valid entities, return standard conditioning
        if len(valid_entities) == 0:
            return io.NodeOutput(global_conditioning)

        # Encode each entity prompt separately
        entity_prompt_emb_list = []
        entity_prompt_emb_mask_list = []

        for entity_prompt, _ in valid_entities: # mask not used at this point
            entity_tokens = clip.tokenize(entity_prompt)
            entity_cond_dict = clip.encode_from_tokens(entity_tokens, return_pooled=True, return_dict=True)
            
            entity_prompt_emb = entity_cond_dict["cond"]
            entity_prompt_emb_mask = entity_cond_dict.get("attention_mask", None)

            # If no attention mask in extra_dict, create one (all True)
            if entity_prompt_emb_mask is None:
                seq_len = entity_prompt_emb.shape[1]
                entity_prompt_emb_mask = torch.ones((entity_prompt_emb.shape[0], seq_len),
                                                   dtype=torch.bool, device=entity_prompt_emb.device)


            entity_prompt_emb_list.append(entity_prompt_emb)
            entity_prompt_emb_mask_list.append(entity_prompt_emb_mask)

        # Process spatial masks to latent space
        processed_entity_masks = []
        for i, (_, mask) in enumerate(valid_entities):
            # mask is expected to be [batch, height, width, channels] or [batch, height, width]
            mask_tensor = mask

            # Log original mask dimensions
            original_shape = mask_tensor.shape
            if len(original_shape) == 3:
                orig_h, orig_w = original_shape[0], original_shape[1]
            elif len(original_shape) == 4:
                orig_h, orig_w = original_shape[1], original_shape[2]
            else:
                orig_h, orig_w = original_shape[-2], original_shape[-1]

            print(f"[EliGen] Entity {i+1} mask: {orig_h}x{orig_w} → will resize to {latent_height}x{latent_width} latent")

            # Ensure mask is in [batch, channels, height, width] format for upscale
            if len(mask_tensor.shape) == 3:
                # [height, width, channels] -> [1, height, width, channels] (add batch dimension)
                mask_tensor = mask_tensor.unsqueeze(0)
            elif len(mask_tensor.shape) == 4 and mask_tensor.shape[-1] in [1, 3, 4]:
                # [batch, height, width, channels] -> [batch, channels, height, width]
                mask_tensor = mask_tensor.movedim(-1, 1)

            # Take only first channel if multiple channels
            if mask_tensor.shape[1] > 1:
                mask_tensor = mask_tensor[:, 0:1, :, :]

            # Resize to latent space dimensions using nearest neighbor
            resized_mask = comfy.utils.common_upscale(
                mask_tensor,
                latent_width,
                latent_height,
                upscale_method="nearest-exact",
                crop="disabled"
            )

            # Threshold to binary (0 or 1)
            # Use > 0 instead of > 0.5 to preserve edge pixels from nearest-neighbor downsampling
            resized_mask = (resized_mask > 0).float()

            # Log how many pixels are active in the mask
            active_pixels = (resized_mask > 0).sum().item()
            total_pixels = resized_mask.numel()
            print(f"[EliGen] Entity {i+1} mask coverage: {active_pixels}/{total_pixels} pixels ({100*active_pixels/total_pixels:.1f}%)")

            processed_entity_masks.append(resized_mask)

        # Stack masks: [batch, num_entities, 1, latent_height, latent_width] (1 is selected channel)
        # No padding - handle dynamic number of entities
        entity_masks_tensor = torch.stack(processed_entity_masks, dim=1)

        # Extract global prompt embedding and mask from conditioning
        # Conditioning format: [[cond_tensor, extra_dict]]
        global_prompt_emb = global_conditioning[0][0]  # The embedding tensor directly
        global_extra_dict = global_conditioning[0][1]  # Metadata dict

        global_prompt_emb_mask = global_extra_dict.get("attention_mask", None)

        # If no attention mask, create one (all True)
        if global_prompt_emb_mask is None:
            global_prompt_emb_mask = torch.ones((global_prompt_emb.shape[0], global_prompt_emb.shape[1]),
                                                dtype=torch.bool, device=global_prompt_emb.device)

        # Attach entity data to conditioning using conditioning_set_values
        entity_data = {
            "entity_prompt_emb": entity_prompt_emb_list,
            "entity_prompt_emb_mask": entity_prompt_emb_mask_list,
            "entity_masks": entity_masks_tensor,
        }

        conditioning_with_entities = node_helpers.conditioning_set_values(
            global_conditioning,
            entity_data,
            append=True
        )

        return io.NodeOutput(conditioning_with_entities)


class QwenExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            TextEncodeQwenImageEdit,
            TextEncodeQwenImageEditPlus,
            TextEncodeQwenImageEliGen,
        ]


async def comfy_entrypoint() -> QwenExtension:
    return QwenExtension()
