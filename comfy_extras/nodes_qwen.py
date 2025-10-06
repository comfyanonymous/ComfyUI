import node_helpers
import comfy.utils
import math
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
            # Konsistente Bildverarbeitung - nur einmal skalieren
            samples = image.movedim(-1, 1)
            
            # Für Qwen-Image-Edit: Zielauflösung 1024x1024 (wie in der offiziellen Pipeline)
            target_pixels = int(1024 * 1024)
            current_pixels = samples.shape[3] * samples.shape[2]
            
            if current_pixels != target_pixels:
                scale_by = math.sqrt(target_pixels / current_pixels)
                width = round(samples.shape[3] * scale_by)
                height = round(samples.shape[2] * scale_by)
                
                # Skalierung mit area interpolation für bessere Qualität
                s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                image = s.movedim(1, -1)
            else:
                image = samples.movedim(1, -1)
            
            images = [image[:, :, :, :3]]
            
            # VAE-Encoding für reference latents
            if vae is not None:
                ref_latent = vae.encode(image[:, :, :, :3])

        # Tokenisierung mit korrektem Template für Qwen-Image-Edit
        tokens = clip.tokenize(prompt, images=images)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        
        # Reference latents hinzufügen (wie in der offiziellen Pipeline)
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
        
        # Korrigierter Template für Qwen-Image-Edit (basierend auf offizieller Pipeline)
        llama_template = "<|im_start|>system\nYou are a helpful assistant that can edit images based on user instructions. Analyze the input image and provide detailed guidance on how to modify it according to the user's request.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        image_prompt = ""

        for i, image in enumerate(images):
            if image is not None:
                # KONSISTENTE Bildverarbeitung - nur einmal skalieren
                samples = image.movedim(-1, 1)
                
                # Einheitliche Zielauflösung für alle Verarbeitungsschritte
                target_pixels = int(1024 * 1024)
                current_pixels = samples.shape[3] * samples.shape[2]
                
                if current_pixels != target_pixels:
                    scale_by = math.sqrt(target_pixels / current_pixels)
                    width = round(samples.shape[3] * scale_by)
                    height = round(samples.shape[2] * scale_by)
                    
                    # Skalierung mit area interpolation
                    s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                    processed_image = s.movedim(1, -1)
                else:
                    processed_image = samples.movedim(1, -1)
                
                # Gleiche Bildverarbeitung für Vision und VAE
                images_vl.append(processed_image)
                
                # VAE-Encoding für reference latents (gleiche Auflösung!)
                if vae is not None:
                    ref_latents.append(vae.encode(processed_image[:, :, :, :3]))

                image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)

        # Tokenisierung mit korrigiertem Template
        tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        
        # Reference latents hinzufügen
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
        
        return io.NodeOutput(conditioning)


class TextEncodeQwenNegative(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodeQwenNegative",
            category="advanced/conditioning",
            description="Creates proper negative conditioning for Qwen-Image-Edit models (uses empty string instead of zero conditioning)",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("negative_prompt", multiline=True, default=" ", tooltip="Negative prompt (use empty string ' ' for best results)"),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, negative_prompt=" ") -> io.NodeOutput:
        # Für Qwen-Image-Edit: Leerstring als negative prompt (wie in der offiziellen Pipeline)
        if not negative_prompt or negative_prompt.strip() == "":
            negative_prompt = " "
        
        tokens = clip.tokenize(negative_prompt)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return io.NodeOutput(conditioning)


class QwenExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            TextEncodeQwenImageEdit,
            TextEncodeQwenImageEditPlus,
            TextEncodeQwenNegative,
        ]


async def comfy_entrypoint() -> QwenExtension:
    return QwenExtension()
