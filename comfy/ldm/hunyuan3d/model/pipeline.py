import torch
import trimesh
import torch.nn as nn
from PIL import Image
from typing import List, Union
from torch.utils._pytree import tree_map
from torch.utils.data._utils.collate import default_collate

def export_to_trimesh(mesh_output):
    if isinstance(mesh_output, list):
        outputs = []
        for mesh in mesh_output:
            if mesh is None:
                outputs.append(None)
            else:
                mesh.mesh_f = mesh.mesh_f[:, ::-1]
                mesh_output = trimesh.Trimesh(mesh.mesh_v, mesh.mesh_f)
                outputs.append(mesh_output)
        return outputs
    else:
        mesh_output.mesh_f = mesh_output.mesh_f[:, ::-1]
        mesh_output = trimesh.Trimesh(mesh_output.mesh_v, mesh_output.mesh_f)
        return mesh_output

class Hunyuan3DDiTFlowMatchingPipeline(nn.Module):
    def __init__(self, model, vae, conditioner, image_processor, scheduler):

        self.vae = vae
        self.model = model
        self.conditioner = conditioner
        self.image_processor = image_processor
        self.scheduler = scheduler

    def compile(self):
        self.vae = torch.compile(self.vae)
        self.model = torch.compile(self.model)
        self.conditioner = torch.compile(self.conditioner)

    def encode_cond(self, image, additional_cond_inputs, do_classifier_free_guidance):

        bsz = image.shape[0]
        cond = self.conditioner(image=image, **additional_cond_inputs)

        if do_classifier_free_guidance:
            un_cond = self.conditioner.unconditional_embedding(bsz, **additional_cond_inputs)

            # avoid python recursion by using tree_map
            _fn = lambda x, y: torch.cat([x, y], dim=0).to(self.dtype)

            cond = tree_map(_fn, cond, un_cond)

        return cond
    
    def prepare_images(self, images):

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
        **kwargs,
    ) -> List[List[trimesh.Trimesh]]:
        
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        device = self.device
        dtype = self.dtype
        do_classifier_free_guidance = guidance_scale >= 0 and not (
            hasattr(self.model, 'guidance_embed') and
            self.model.guidance_embed is True
        )

        cond_inputs = self.prepare_image(image)
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
            timestep = timestep / self.scheduler.num_train_timesteps
            noise_pred = self.model(latent_model_input, timestep, cond, guidance=guidance)

            if do_classifier_free_guidance:
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents)

            if callback is not None and i % callback_steps == 0:
                step_idx = i // getattr(self.scheduler, "order", 1)
                callback(step_idx, t, latents)

        latents = 1. / self.vae.scale_factor * latents
        mesh = self.vae.decode(latents, bounds = bounds, octree_res = octree_res, num_chunks = num_chunks)

        return export_to_trimesh(mesh)
    