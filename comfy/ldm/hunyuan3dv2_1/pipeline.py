import torch
import torch.nn as nn
from PIL import Image
from typing import List, Union
from torch.utils._pytree import tree_map
from torch.utils.data._utils.collate import default_collate

import sys
import os

def find_project_root(target_folder_name="ComfyUI"):
    """ Walks directory until it finds ComfyUI base directroy """
    current = os.path.abspath(os.path.dirname(__file__))
    while True:
        if os.path.basename(current) == target_folder_name:
            return current
        parent = os.path.dirname(current)
        if parent == current:
            raise RuntimeError(f"Could not find folder named '{target_folder_name}' in parent paths.")
        current = parent

comfyui_root = find_project_root()
sys.path.append(comfyui_root)

from comfy_extras.nodes_hunyuan3d import save_glb

class Hunyuan3DDiTFlowMatchingPipeline(nn.Module):
    def __init__(self, model, vae, conditioner, image_processor, scheduler, device, dtype):
        super().__init__()
        
        self.vae = vae
        self.model = model
        self.conditioner = conditioner
        self.image_processor = image_processor
        self.scheduler = scheduler
        self.device = device
        self.dtype = dtype

    def compile(self):
        self.vae = torch.compile(self.vae)
        self.model = torch.compile(self.model)
        self.conditioner = torch.compile(self.conditioner)

    def load_ckpt(self, checkpoint_path: str):

        checkpoint = torch.load(checkpoint_path, weights_only = True)
        self.model.load_state_dict(checkpoint["model"])
        self.vae.load_state_dict(checkpoint["vae"])
        self.conditioner.load_state_dict(checkpoint["conditioner"])
        

    def encode_cond(self, image, additional_cond_inputs, do_classifier_free_guidance):

        bsz = image.shape[0]
        cond = self.conditioner(image=image, **additional_cond_inputs)

        if do_classifier_free_guidance:
            un_cond = self.conditioner.unconditional_embedding(bsz, **additional_cond_inputs)

            # avoid python recursion by using tree_map
            _fn = lambda x, y: torch.cat([x, y], dim=0).to(self.dtype)

            cond = tree_map(_fn, cond, un_cond)

        return cond
    
    def to(self, device=None, dtype=None):
        if dtype is not None:
            self.dtype = dtype
            self.vae.to(dtype=dtype)
            self.model.to(dtype=dtype)
            self.conditioner.to(dtype=dtype)
        if device is not None:
            self.device = torch.device(device)
            self.vae.to(device)
            self.model.to(device)
            self.conditioner.to(device)
    
    def prepare_images(self, images):

        if isinstance(images, (str, Image.Image)):
            return self.image_processor(images)

        outputs = []
        for image in images:
            output = self.image_processor(image)
            outputs.append(output)

        return default_collate(outputs)
    
    def get_guidance_scale_embedding(self, w, embedding_dim=512, dtype=torch.float32):

        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)

        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]

        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))

        assert emb.shape == (w.shape[0], embedding_dim)

        return emb
    
    def prepare_latents(self, batch_size, dtype, device):

        shape = (batch_size, *self.vae.latent_shape)
        latents = torch.randn(shape, dtype = dtype, device = device)

        return latents

    @torch.inference_mode()
    def __call__(
        self,
        image: Union[str, List[str], Image.Image, dict, List[dict], torch.Tensor] = None,
        guidance_scale: float = 5.0,
        bounds = 1.01,
        octree_res = 384,
        num_chunks = 8000,
        save_file = None,
        **kwargs,
    ):
        
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        device = self.device
        dtype = self.dtype
        do_classifier_free_guidance = guidance_scale >= 0 and not (
            hasattr(self.model, 'guidance_embed') and
            self.model.guidance_embed is True
        )

        cond_inputs = self.prepare_images(image)
        image = cond_inputs.pop('image')

        cond = self.encode_cond(
            image = image,
            additional_cond_inputs = cond_inputs,
            do_classifier_free_guidance = do_classifier_free_guidance,
        ) 

        guidance = None
        batch_size = image.shape[0]

        latents = self.prepare_latents(batch_size, dtype, device)

        if hasattr(self.model, 'guidance_embed') and \
            self.model.guidance_embed is True:
            guidance = torch.tensor([guidance_scale] * batch_size, device=device, dtype=dtype)

        timesteps = self.scheduler.timesteps

        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            if do_classifier_free_guidance:
                latent_model_input = torch.cat([latents] * 2)
            else:
                latent_model_input = latents

            timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype)
            timestep = timestep / self.scheduler.num_training_timesteps
            noise_pred = self.model(latent_model_input, timestep, cond, guidance=guidance)

            if do_classifier_free_guidance:
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, latents)

            if callback is not None and i % callback_steps == 0:
                step_idx = i // getattr(self.scheduler, "order", 1)
                callback(step_idx, t, latents)

        latents = 1. / self.vae.scale_factor * latents
        mesh = self.vae.decode(latents, bounds = bounds, octree_res = octree_res, num_chunks = num_chunks)

        try:
            if save_file is not None:
                for i, output in enumerate(mesh):
                    output_file = f"{save_file}_{i}" if len(mesh) > 1 else save_file
                    save_glb(output.mesh_v, output.mesh_f, output_file, numpy_ready = True)
        except Exception as e:
            print(e)
            
        return mesh