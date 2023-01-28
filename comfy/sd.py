import torch

import sd1_clip
import sd2_clip
from ldm.util import instantiate_from_config
from ldm.models.autoencoder import AutoencoderKL
from omegaconf import OmegaConf


def load_model_from_config(config, ckpt, verbose=False, load_state_dict_to=[]):
    print(f"Loading model from {ckpt}")

    if ckpt.lower().endswith(".safetensors"):
        import safetensors.torch
        sd = safetensors.torch.load_file(ckpt, device="cpu")
    else:
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        else:
            sd = pl_sd
    model = instantiate_from_config(config.model)

    m, u = model.load_state_dict(sd, strict=False)

    k = list(sd.keys())
    for x in k:
        # print(x)
        if x.startswith("cond_stage_model.transformer.") and not x.startswith("cond_stage_model.transformer.text_model."):
            y = x.replace("cond_stage_model.transformer.", "cond_stage_model.transformer.text_model.")
            sd[y] = sd.pop(x)

    if 'cond_stage_model.transformer.text_model.embeddings.position_ids' in sd:
        sd['cond_stage_model.transformer.text_model.embeddings.position_ids'] = sd['cond_stage_model.transformer.text_model.embeddings.position_ids'].round()

    for x in load_state_dict_to:
        x.load_state_dict(sd, strict=False)

    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.eval()
    return model



class CLIP:
    def __init__(self, config):
        self.target_clip = config["target"]
        if self.target_clip == "ldm.modules.encoders.modules.FrozenOpenCLIPEmbedder":
            clip = sd2_clip.SD2ClipModel
            tokenizer = sd2_clip.SD2Tokenizer
        elif self.target_clip == "ldm.modules.encoders.modules.FrozenCLIPEmbedder":
            clip = sd1_clip.SD1ClipModel
            tokenizer = sd1_clip.SD1Tokenizer
        if "params" in config:
            self.cond_stage_model = clip(**(config["params"]))
        else:
            self.cond_stage_model = clip()
        self.tokenizer = tokenizer()

    def encode(self, text):
        tokens = self.tokenizer.tokenize_with_weights(text)
        cond = self.cond_stage_model.encode_token_weights(tokens)
        return cond


class VAE:
    def __init__(self, ckpt_path=None, scale_factor=0.18215, device="cuda", config=None):
        if config is None:
            #default SD1.x/SD2.x VAE parameters
            ddconfig = {'double_z': True, 'z_channels': 4, 'resolution': 256, 'in_channels': 3, 'out_ch': 3, 'ch': 128, 'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}
            self.first_stage_model = AutoencoderKL(ddconfig, {'target': 'torch.nn.Identity'}, 4, monitor="val/rec_loss", ckpt_path=ckpt_path)
        else:
            self.first_stage_model = AutoencoderKL(**(config['params']), ckpt_path=ckpt_path)
        self.first_stage_model = self.first_stage_model.eval()
        self.scale_factor = scale_factor
        self.device = device

    def decode(self, samples):
        self.first_stage_model = self.first_stage_model.to(self.device)
        samples = samples.to(self.device)
        pixel_samples = self.first_stage_model.decode(1. / self.scale_factor * samples)
        pixel_samples = torch.clamp((pixel_samples + 1.0) / 2.0, min=0.0, max=1.0)
        self.first_stage_model = self.first_stage_model.cpu()
        pixel_samples = pixel_samples.cpu().movedim(1,-1)
        return pixel_samples

    def encode(self, pixel_samples):
        self.first_stage_model = self.first_stage_model.to(self.device)
        pixel_samples = pixel_samples.movedim(-1,1).to(self.device)
        samples = self.first_stage_model.encode(2. * pixel_samples - 1.).sample() * self.scale_factor
        self.first_stage_model = self.first_stage_model.cpu()
        samples = samples.cpu()
        return samples


def load_checkpoint(config_path, ckpt_path, output_vae=True, output_clip=True):
    config = OmegaConf.load(config_path)
    model_config_params = config['model']['params']
    clip_config = model_config_params['cond_stage_config']
    scale_factor = model_config_params['scale_factor']
    vae_config = model_config_params['first_stage_config']

    clip = None
    vae = None

    class WeightsLoader(torch.nn.Module):
        pass

    w = WeightsLoader()
    load_state_dict_to = []
    if output_vae:
        vae = VAE(scale_factor=scale_factor, config=vae_config)
        w.first_stage_model = vae.first_stage_model
        load_state_dict_to = [w]

    if output_clip:
        clip = CLIP(config=clip_config)
        w.cond_stage_model = clip.cond_stage_model
        load_state_dict_to = [w]

    model = load_model_from_config(config, ckpt_path, verbose=False, load_state_dict_to=load_state_dict_to)
    return (model, clip, vae)
