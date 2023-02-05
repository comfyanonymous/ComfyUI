import torch

import sd1_clip
import sd2_clip
from ldm.util import instantiate_from_config
from ldm.models.autoencoder import AutoencoderKL
from omegaconf import OmegaConf

def load_torch_file(ckpt):
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
    return sd

def load_model_from_config(config, ckpt, verbose=False, load_state_dict_to=[]):
    print(f"Loading model from {ckpt}")

    sd = load_torch_file(ckpt)
    model = instantiate_from_config(config.model)

    m, u = model.load_state_dict(sd, strict=False)

    k = list(sd.keys())
    for x in k:
        # print(x)
        if x.startswith("cond_stage_model.transformer.") and not x.startswith("cond_stage_model.transformer.text_model."):
            y = x.replace("cond_stage_model.transformer.", "cond_stage_model.transformer.text_model.")
            sd[y] = sd.pop(x)

    if 'cond_stage_model.transformer.text_model.embeddings.position_ids' in sd:
        ids = sd['cond_stage_model.transformer.text_model.embeddings.position_ids']
        if ids.dtype == torch.float32:
            sd['cond_stage_model.transformer.text_model.embeddings.position_ids'] = ids.round()

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

LORA_CLIP_MAP = {
    "mlp.fc1": "mlp_fc1",
    "mlp.fc2": "mlp_fc2",
    "self_attn.k_proj": "self_attn_k_proj",
    "self_attn.q_proj": "self_attn_q_proj",
    "self_attn.v_proj": "self_attn_v_proj",
    "self_attn.out_proj": "self_attn_out_proj",
}

LORA_CLIP2_MAP = {
    "mlp.c_fc": "mlp_fc1",
    "mlp.c_proj": "mlp_fc2",
    "attn.out_proj": "self_attn_out_proj",
}

LORA_UNET_MAP = {
    "proj_in": "proj_in",
    "proj_out": "proj_out",
    "transformer_blocks.0.attn1.to_q": "transformer_blocks_0_attn1_to_q",
    "transformer_blocks.0.attn1.to_k": "transformer_blocks_0_attn1_to_k",
    "transformer_blocks.0.attn1.to_v": "transformer_blocks_0_attn1_to_v",
    "transformer_blocks.0.attn1.to_out.0": "transformer_blocks_0_attn1_to_out_0",
    "transformer_blocks.0.attn2.to_q": "transformer_blocks_0_attn2_to_q",
    "transformer_blocks.0.attn2.to_k": "transformer_blocks_0_attn2_to_k",
    "transformer_blocks.0.attn2.to_v": "transformer_blocks_0_attn2_to_v",
    "transformer_blocks.0.attn2.to_out.0": "transformer_blocks_0_attn2_to_out_0",
    "transformer_blocks.0.ff.net.0.proj": "transformer_blocks_0_ff_net_0_proj",
    "transformer_blocks.0.ff.net.2": "transformer_blocks_0_ff_net_2",
}


def load_lora(path, to_load):
    lora = load_torch_file(path)
    patch_dict = {}
    loaded_keys = set()
    for x in to_load:
        A_name = "{}.lora_up.weight".format(x)
        B_name = "{}.lora_down.weight".format(x)
        alpha_name = "{}.alpha".format(x)
        if A_name in lora.keys():
            alpha = None
            if alpha_name in lora.keys():
                alpha = lora[alpha_name].item()
                loaded_keys.add(alpha_name)
            patch_dict[to_load[x]] = (lora[A_name], lora[B_name], alpha)
            loaded_keys.add(A_name)
            loaded_keys.add(B_name)
    for x in lora.keys():
        if x not in loaded_keys:
            print("lora key not loaded", x)
    return patch_dict

def model_lora_keys(model, key_map={}):
    sdk = model.state_dict().keys()

    counter = 0
    for b in range(12):
        tk = "model.diffusion_model.input_blocks.{}.1".format(b)
        up_counter = 0
        for c in LORA_UNET_MAP:
            k = "{}.{}.weight".format(tk, c)
            if k in sdk:
                lora_key = "lora_unet_down_blocks_{}_attentions_{}_{}".format(counter // 2, counter % 2, LORA_UNET_MAP[c])
                key_map[lora_key] = (k, 0)
                up_counter += 1
        if up_counter >= 4:
            counter += 1
    for c in LORA_UNET_MAP:
        k = "model.diffusion_model.middle_block.1.{}.weight".format(c)
        if k in sdk:
            lora_key = "lora_unet_mid_block_attentions_0_{}".format(LORA_UNET_MAP[c])
            key_map[lora_key] = (k, 0)
    counter = 3
    for b in range(12):
        tk = "model.diffusion_model.output_blocks.{}.1".format(b)
        up_counter = 0
        for c in LORA_UNET_MAP:
            k = "{}.{}.weight".format(tk, c)
            if k in sdk:
                lora_key = "lora_unet_up_blocks_{}_attentions_{}_{}".format(counter // 3, counter % 3, LORA_UNET_MAP[c])
                key_map[lora_key] = (k, 0)
                up_counter += 1
        if up_counter >= 4:
            counter += 1
    counter = 0
    text_model_lora_key = "lora_te_text_model_encoder_layers_{}_{}"
    for b in range(12):
        for c in LORA_CLIP_MAP:
            k = "transformer.text_model.encoder.layers.{}.{}.weight".format(b, c)
            if k in sdk:
                lora_key = text_model_lora_key.format(b, LORA_CLIP_MAP[c])
                key_map[lora_key] = (k, 0)
    for b in range(24):
        for c in LORA_CLIP2_MAP:
            k = "model.transformer.resblocks.{}.{}.weight".format(b, c)
            if k in sdk:
                lora_key = text_model_lora_key.format(b, LORA_CLIP2_MAP[c])
                key_map[lora_key] = (k, 0)
        k = "model.transformer.resblocks.{}.attn.in_proj_weight".format(b)
        if k in sdk:
            key_map[text_model_lora_key.format(b, "self_attn_q_proj")] = (k, 0)
            key_map[text_model_lora_key.format(b, "self_attn_k_proj")] = (k, 1)
            key_map[text_model_lora_key.format(b, "self_attn_v_proj")] = (k, 2)

    return key_map

class ModelPatcher:
    def __init__(self, model):
        self.model = model
        self.patches = []
        self.backup = {}

    def clone(self):
        n = ModelPatcher(self.model)
        n.patches = self.patches[:]
        return n

    def add_patches(self, patches, strength=1.0):
        p = {}
        model_sd = self.model.state_dict()
        for k in patches:
            if k[0] in model_sd:
                p[k] = patches[k]
        self.patches += [(strength, p)]
        return p.keys()

    def patch_model(self):
        model_sd = self.model.state_dict()
        for p in self.patches:
            for k in p[1]:
                v = p[1][k]
                key = k[0]
                index = k[1]
                if key not in model_sd:
                    print("could not patch. key doesn't exist in model:", k)
                    continue

                weight = model_sd[key]
                if key not in self.backup:
                    self.backup[key] = weight.clone()

                alpha = p[0]
                mat1 = v[0]
                mat2 = v[1]
                if v[2] is not None:
                    alpha *= v[2] / mat2.shape[0]
                calc = (alpha * torch.mm(mat1.flatten(start_dim=1).float(), mat2.flatten(start_dim=1).float()))
                if len(weight.shape) > 2:
                    calc = calc.reshape(weight.shape)
                weight[index * mat1.shape[0]:(index + 1) * mat1.shape[0]] += calc.type(weight.dtype).to(weight.device)
        return self.model
    def unpatch_model(self):
        model_sd = self.model.state_dict()
        for k in self.backup:
            model_sd[k][:] = self.backup[k]
        self.backup = {}

def load_lora_for_models(model, clip, lora_path, strength_model, strength_clip):
    key_map = model_lora_keys(model.model)
    key_map = model_lora_keys(clip.cond_stage_model, key_map)
    loaded = load_lora(lora_path, key_map)
    new_modelpatcher = model.clone()
    k = new_modelpatcher.add_patches(loaded, strength_model)
    new_clip = clip.clone()
    k1 = new_clip.add_patches(loaded, strength_clip)
    k = set(k)
    k1 = set(k1)
    for x in loaded:
        if (x not in k) and (x not in k1):
            print("NOT LOADED", x)

    return (new_modelpatcher, new_clip)


class CLIP:
    def __init__(self, config={}, embedding_directory=None, no_init=False):
        if no_init:
            return
        self.target_clip = config["target"]
        if "params" in config:
            params = config["params"]
        else:
            params = {}

        tokenizer_params = {}

        if self.target_clip == "ldm.modules.encoders.modules.FrozenOpenCLIPEmbedder":
            clip = sd2_clip.SD2ClipModel
            tokenizer = sd2_clip.SD2Tokenizer
        elif self.target_clip == "ldm.modules.encoders.modules.FrozenCLIPEmbedder":
            clip = sd1_clip.SD1ClipModel
            tokenizer = sd1_clip.SD1Tokenizer
            tokenizer_params['embedding_directory'] = embedding_directory

        self.cond_stage_model = clip(**(params))
        self.tokenizer = tokenizer(**(tokenizer_params))
        self.patcher = ModelPatcher(self.cond_stage_model)

    def clone(self):
        n = CLIP(no_init=True)
        n.target_clip = self.target_clip
        n.patcher = self.patcher.clone()
        n.cond_stage_model = self.cond_stage_model
        n.tokenizer = self.tokenizer
        return n

    def add_patches(self, patches, strength=1.0):
        return self.patcher.add_patches(patches, strength)

    def encode(self, text):
        tokens = self.tokenizer.tokenize_with_weights(text)
        try:
            self.patcher.patch_model()
            cond = self.cond_stage_model.encode_token_weights(tokens)
            self.patcher.unpatch_model()
        except Exception as e:
            self.patcher.unpatch_model()
            raise e
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


def load_checkpoint(config_path, ckpt_path, output_vae=True, output_clip=True, embedding_directory=None):
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
        clip = CLIP(config=clip_config, embedding_directory=embedding_directory)
        w.cond_stage_model = clip.cond_stage_model
        load_state_dict_to = [w]

    model = load_model_from_config(config, ckpt_path, verbose=False, load_state_dict_to=load_state_dict_to)
    return (ModelPatcher(model), clip, vae)
