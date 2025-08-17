#credit to shakker-labs and instantX for this module
#from https://github.com/Shakker-Labs/ComfyUI-IPAdapter-Flux
import torch
from PIL import Image
import numpy as np
from .attention_processor import IPAFluxAttnProcessor2_0
from .utils import is_model_pathched, FluxUpdateModules
from .sd3.resampler import TimeResampler
from .sd3.joinblock import JointBlockIPWrapper, IPAttnProcessor

image_proj_model = None
class MLPProjModel(torch.nn.Module):
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, num_tokens=4):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(id_embeddings_dim, id_embeddings_dim * 2),
            torch.nn.GELU(),
            torch.nn.Linear(id_embeddings_dim * 2, cross_attention_dim * num_tokens),
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, id_embeds):
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        return x

class InstantXFluxIpadapterApply:
    def __init__(self, num_tokens=128):
        self.device = None
        self.dtype = torch.float16
        self.num_tokens = num_tokens
        self.ip_ckpt = None
        self.clip_vision = None
        self.image_encoder = None
        self.clip_image_processor = None
        # state_dict
        self.state_dict = None
        self.joint_attention_dim = 4096
        self.hidden_size = 3072

    def set_ip_adapter(self, flux_model, weight, timestep_percent_range=(0.0, 1.0)):
        s = flux_model.model_sampling
        percent_to_timestep_function = lambda a: s.percent_to_sigma(a)
        timestep_range = (percent_to_timestep_function(timestep_percent_range[0]),
                          percent_to_timestep_function(timestep_percent_range[1]))
        ip_attn_procs = {}  # 19+38=57
        dsb_count = len(flux_model.diffusion_model.double_blocks)
        for i in range(dsb_count):
            name = f"double_blocks.{i}"
            ip_attn_procs[name] = IPAFluxAttnProcessor2_0(
                hidden_size=self.hidden_size,
                cross_attention_dim=self.joint_attention_dim,
                num_tokens=self.num_tokens,
                scale=weight,
                timestep_range=timestep_range
            ).to(self.device, dtype=self.dtype)
        ssb_count = len(flux_model.diffusion_model.single_blocks)
        for i in range(ssb_count):
            name = f"single_blocks.{i}"
            ip_attn_procs[name] = IPAFluxAttnProcessor2_0(
                hidden_size=self.hidden_size,
                cross_attention_dim=self.joint_attention_dim,
                num_tokens=self.num_tokens,
                scale=weight,
                timestep_range=timestep_range
            ).to(self.device, dtype=self.dtype)
        return ip_attn_procs

    def load_ip_adapter(self, flux_model, weight, timestep_percent_range=(0.0, 1.0)):
        global image_proj_model
        image_proj_model.load_state_dict(self.state_dict["image_proj"], strict=True)
        ip_attn_procs = self.set_ip_adapter(flux_model, weight, timestep_percent_range)
        ip_layers = torch.nn.ModuleList(ip_attn_procs.values())
        ip_layers.load_state_dict(self.state_dict["ip_adapter"], strict=True)
        return ip_attn_procs

    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        # outputs = self.clip_vision.encode_image(pil_image)
        # clip_image_embeds = outputs['image_embeds']
        # clip_image_embeds = clip_image_embeds.to(self.device, dtype=self.dtype)
        # image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image_embeds = self.image_encoder(
                clip_image.to(self.device, dtype=self.image_encoder.dtype)).pooler_output
            clip_image_embeds = clip_image_embeds.to(dtype=self.dtype)
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=self.dtype)
        global image_proj_model
        image_prompt_embeds = image_proj_model(clip_image_embeds)
        return image_prompt_embeds

    def apply_ipadapter(self, model, ipadapter, image, weight, start_at, end_at, provider=None, use_tiled=False):
        self.device = provider.lower()
        if "clipvision" in ipadapter:
            # self.clip_vision = ipadapter["clipvision"]['model']
            self.image_encoder = ipadapter["clipvision"]['model']['image_encoder'].to(self.device, dtype=self.dtype)
            self.clip_image_processor = ipadapter["clipvision"]['model']['clip_image_processor']
        if "ipadapter" in ipadapter:
            self.ip_ckpt = ipadapter["ipadapter"]['file']
            self.state_dict = ipadapter["ipadapter"]['model']

        # process image
        pil_image = image.numpy()[0] * 255.0
        pil_image = Image.fromarray(pil_image.astype(np.uint8))
        # initialize ipadapter
        global image_proj_model
        if image_proj_model is None:
            image_proj_model = MLPProjModel(
                cross_attention_dim=self.joint_attention_dim,  # 4096
                id_embeddings_dim=1152,
                num_tokens=self.num_tokens,
            )
        image_proj_model.to(self.device, dtype=self.dtype)
        ip_attn_procs = self.load_ip_adapter(model.model, weight, (start_at, end_at))
        # process control image
        image_prompt_embeds = self.get_image_embeds(pil_image=pil_image, clip_image_embeds=None)
        # set model
        # is_patched = is_model_pathched(model.model)
        bi = model.clone()
        FluxUpdateModules(bi, ip_attn_procs, image_prompt_embeds)

        return (bi, image)


def patch_sd3(
    patcher,
    ip_procs,
    resampler: TimeResampler,
    clip_embeds,
    weight=1.0,
    start=0.0,
    end=1.0,
):
    """
    Patches a model_sampler to add the ipadapter
    """
    mmdit = patcher.model.diffusion_model
    timestep_schedule_max = patcher.model.model_config.sampling_settings.get(
        "timesteps", 1000
    )
    # hook the model's forward function
    # so that when it gets called, we can grab the timestep and send it to the resampler
    ip_options = {
        "hidden_states": None,
        "t_emb": None,
        "weight": weight,
    }

    def ddit_wrapper(forward, args):
        # this is between 0 and 1, so the adapters can calculate start_point and end_point
        # actually, do we need to get the sigma value instead?
        t_percent = 1 - args["timestep"].flatten()[0].cpu().item()
        if start <= t_percent <= end:
            batch_size = args["input"].shape[0] // len(args["cond_or_uncond"])
            # if we're only doing cond or only doing uncond, only pass one of them through the resampler
            embeds = clip_embeds[args["cond_or_uncond"]]
            # slight efficiency optimization todo: pass the embeds through and then afterwards
            # repeat to the batch size
            embeds = torch.repeat_interleave(embeds, batch_size, dim=0)
            # the resampler wants between 0 and MAX_STEPS
            timestep = args["timestep"] * timestep_schedule_max
            image_emb, t_emb = resampler(embeds, timestep, need_temb=True)
            # these will need to be accessible to the IPAdapters
            ip_options["hidden_states"] = image_emb
            ip_options["t_emb"] = t_emb
        else:
            ip_options["hidden_states"] = None
            ip_options["t_emb"] = None

        return forward(args["input"], args["timestep"], **args["c"])

    patcher.set_model_unet_function_wrapper(ddit_wrapper)
    # patch each dit block
    for i, block in enumerate(mmdit.joint_blocks):
        wrapper = JointBlockIPWrapper(block, ip_procs[i], ip_options)
        patcher.set_model_patch_replace(wrapper, "dit", "double_block", i)

class InstantXSD3IpadapterApply:
    def __init__(self):
        self.device = None
        self.dtype = torch.float16
        self.clip_image_processor = None
        self.image_encoder = None
        self.resampler = None
        self.procs = None

    @torch.inference_mode()
    def encode(self, image):
        clip_image = self.clip_image_processor.image_processor(image, return_tensors="pt", do_rescale=False).pixel_values
        clip_image_embeds = self.image_encoder(
            clip_image.to(self.device, dtype=self.image_encoder.dtype),
            output_hidden_states=True,
        ).hidden_states[-2]
        clip_image_embeds = torch.cat(
            [clip_image_embeds, torch.zeros_like(clip_image_embeds)], dim=0
        )
        clip_image_embeds = clip_image_embeds.to(dtype=torch.float16)
        return clip_image_embeds

    def apply_ipadapter(self, model, ipadapter, image, weight, start_at, end_at, provider=None, use_tiled=False):
        self.device = provider.lower()
        if "clipvision" in ipadapter:
            self.image_encoder = ipadapter["clipvision"]['model']['image_encoder'].to(self.device, dtype=self.dtype)
            self.clip_image_processor = ipadapter["clipvision"]['model']['clip_image_processor']
        if "ipadapter" in ipadapter:
            self.ip_ckpt = ipadapter["ipadapter"]['file']
            self.state_dict = ipadapter["ipadapter"]['model']

        self.resampler = TimeResampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=64,
            embedding_dim=1152,
            output_dim=2432,
            ff_mult=4,
            timestep_in_dim=320,
            timestep_flip_sin_to_cos=True,
            timestep_freq_shift=0,
        )
        self.resampler.eval()
        self.resampler.to(self.device, dtype=self.dtype)
        self.resampler.load_state_dict(self.state_dict["image_proj"])

        # now we'll create the attention processors
        # ip_adapter.keys looks like [0.proj, 0.to_k, ..., 1.proj, 1.to_k, ...]
        n_procs = len(
            set(x.split(".")[0] for x in self.state_dict["ip_adapter"].keys())
        )
        self.procs = torch.nn.ModuleList(
            [
                # this is hardcoded for SD3.5L
                IPAttnProcessor(
                    hidden_size=2432,
                    cross_attention_dim=2432,
                    ip_hidden_states_dim=2432,
                    ip_encoder_hidden_states_dim=2432,
                    head_dim=64,
                    timesteps_emb_dim=1280,
                ).to(self.device, dtype=torch.float16)
                for _ in range(n_procs)
            ]
        )
        self.procs.load_state_dict(self.state_dict["ip_adapter"])

        work_model = model.clone()
        embeds = self.encode(image)

        patch_sd3(
            work_model,
            self.procs,
            self.resampler,
            embeds,
            weight,
            start_at,
            end_at,
        )

        return (work_model, image)