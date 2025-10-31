import torch
import comfy.model_management
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

COMPUTED_RESO_GROUPS = ['512x2048', '512x1984', '512x1920', '512x1856', '512x1792', '512x1728', '512x1664', '512x1600', '512x1536', '576x1472', '640x1408', '704x1344', '768x1280', '832x1216', '896x1152', '960x1088', '1024x1024', '1088x960', '1152x896', '1216x832', '1280x768', '1344x704', '1408x640', '1472x576', '1536x512', '1600x512', '1664x512', '1728x512', '1792x512', '1856x512', '1920x512', '1984x512', '2048x512']
RATIOS = [torch.tensor(int(r.split("x")[0]) / int(r.split("x")[1])) for r in COMPUTED_RESO_GROUPS]
def get_target_size(height, width):
    ratio = height / width
    idx = torch.argmin(torch.abs(torch.tensor(RATIOS) - ratio))
    reso = COMPUTED_RESO_GROUPS[idx]
    return reso.split("x")

class EmptyLatentHunyuanImage3(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EmptyLatentHunyuanImage3",
            display_name="EmptyLatentHunyuanImage3",
            category="image/latent",
            inputs = [
                io.Int.Input("height", min = 1, default = 512),
                io.Int.Input("width", min = 1, default = 512),
                io.Int.Input("batch_size", min = 1, max = 48_000, default = 1),
                io.Clip.Input("clip")
            ],
            outputs=[io.Latent.Output(display_name="latent")]
        )
    @classmethod
    def execute(cls, height, width, batch_size, clip):
        encode_fn = clip.tokenizer.tokenizer.convert_tokens_to_ids
        special_fn = clip.tokenizer.tokenizer.added_tokens_encoder
        def fn(string, func = encode_fn):
            return torch.tensor(func(string), device=comfy.model_management.intermediate_device()).unsqueeze(0)

        height, width = get_target_size(height, width)
        latent = torch.randn(batch_size, 32, height // 16, width // 16, device=comfy.model_management.intermediate_device())
        latent = torch.cat([fn("<boi>"), fn("<all_img>_start"), fn("<img_size_1024>", special_fn), fn(f"<img_ratio_{height / width}", special_fn), fn("<timestep>", special_fn),
                            latent, fn("<eoi>"), fn("<img>_start"), fn("<img>_end"), fn("<all_img>_end")], dim = 1)
        return io.NodeOutput({"samples": latent, "type": "hunyuan_image_3"}, )

class HunyuanImage3Conditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HunyuanImage3Conditioning",
            display_name="HunyuanImage3Conditioning",
            category="conditioning/video_models",
            inputs = [
                io.Conditioning.Input("vae_encoding"),
                io.Conditioning.Input("vit_encoding"),
                io.Conditioning.Input("text_encoding_positive"),
                io.Conditioning.Input("text_encoding_negative", optional = True),
                io.Clip.Input("clip")
            ],
            outputs=[io.Conditioning.Output(display_name= "positive"), io.Conditioning.Output(display_name="negative")]
        )

    @classmethod
    def execute(cls, vae_encoding, vit_encoding, text_encoding, clip, text_encoding_negative=None):
        encode_fn = clip.tokenizer.tokenizer.convert_tokens_to_ids
        special_fn = clip.tokenizer.tokenizer.added_tokens_encoder
        def fn(string, func = encode_fn):
            return torch.tensor(func(string), device=text_encoding.device).unsqueeze(0)

        text_encoding = text_encoding[0][0]

        text_tokens = torch.cat([fn("<text>_start"), text_encoding, fn("<text>_end")], dim = 1)
        vae_tokens = torch.cat([fn("<vae_img>_start"), fn("<joint_img>_start"), fn("<all_img>_start"), vae_encoding, fn("<vae_img>_end"), fn("<all_img>_end"), fn("<joint_img_sep>")], dim = 1)
        vit_tokens = torch.cat([fn("<vit_img>_start"), fn("<all_img>_start"), vit_encoding, fn("<vit_img>_end"), fn("<joint_img>_end"), fn("<all_img>_end")], dim = 1)
        n, seq_len, dim = vit_tokens.shape
        vit_tokens = vit_tokens.reshape(n * seq_len, dim)
        #                                                                                                     should dynamically change in model logic
        joint_image = torch.cat([fn("<boi>"), fn("<img_size_1024>", special_fn), fn("<img_ratio_3>", special_fn), fn("<timestep>", special_fn), vae_tokens, vit_tokens, fn("<eoi>")], dim = 1)

        seq_len_total = joint_image.shape[1]
        mask = torch.zeros(seq_len_total, dtype=torch.bool, device=joint_image.device)
        positions = {}
        current = 4

        def mark_region(name, tensor):
            nonlocal current
            start = current
            current += tensor.shape[1]
            end = current - 1
            positions[f"<{name}>_start"] = start
            positions[f"<{name}>_end"] = end
            mask[start:end + 1] = True
            return start, end

        mark_region("vae_img", vae_tokens)

        mask_list = []
        for prefix in ["text", "vae_img", "vit_img"]:
            start = positions[f"<{prefix}>_start"]
            end = positions[f"<{prefix}>_end"]
            
            section_mask = torch.arange(start, end + 1, device=mask.device)
            mask_list.append(section_mask)

        mask_list.insert(0, joint_image)
        mask_list.append(text_tokens)
        ragged_tensors = torch.nested.nested_tensor(mask_list, dtype=torch.long)

        if text_encoding_negative is not None:
            uncond_ragged_tensors = cls.execute(vae_encoding, vit_encoding, text_encoding_negative, clip=clip, text_encoding_negative = None)
        else:
            uncond_ragged_tensors = torch.nested.nested_tensor([torch.zeros_like(t) for t in ragged_tensors.unbind()])

        return ragged_tensors, uncond_ragged_tensors

class Image3Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            HunyuanImage3Conditioning,
            EmptyLatentHunyuanImage3
        ]

async def comfy_entrypoint() -> Image3Extension:
    return Image3Extension()
