import torch
import contextlib
import copy

import sd1_clip
import sd2_clip
import model_management
from .ldm.util import instantiate_from_config
from .ldm.models.autoencoder import AutoencoderKL
import yaml
from .cldm import cldm
from .t2i_adapter import adapter

from . import utils
from . import clip_vision

def load_model_weights(model, sd, verbose=False, load_state_dict_to=[]):
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

    keys_to_replace = {
        "cond_stage_model.model.positional_embedding": "cond_stage_model.transformer.text_model.embeddings.position_embedding.weight",
        "cond_stage_model.model.token_embedding.weight": "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight",
        "cond_stage_model.model.ln_final.weight": "cond_stage_model.transformer.text_model.final_layer_norm.weight",
        "cond_stage_model.model.ln_final.bias": "cond_stage_model.transformer.text_model.final_layer_norm.bias",
    }

    for x in keys_to_replace:
        if x in sd:
            sd[keys_to_replace[x]] = sd.pop(x)

    sd = utils.transformers_convert(sd, "cond_stage_model.model", "cond_stage_model.transformer.text_model", 24)

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

LORA_UNET_MAP_ATTENTIONS = {
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

LORA_UNET_MAP_RESNET = {
    "in_layers.2": "resnets_{}_conv1",
    "emb_layers.1": "resnets_{}_time_emb_proj",
    "out_layers.3": "resnets_{}_conv2",
    "skip_connection": "resnets_{}_conv_shortcut"
}

def load_lora(path, to_load):
    lora = utils.load_torch_file(path)
    patch_dict = {}
    loaded_keys = set()
    for x in to_load:
        alpha_name = "{}.alpha".format(x)
        alpha = None
        if alpha_name in lora.keys():
            alpha = lora[alpha_name].item()
            loaded_keys.add(alpha_name)

        A_name = "{}.lora_up.weight".format(x)
        B_name = "{}.lora_down.weight".format(x)
        mid_name = "{}.lora_mid.weight".format(x)

        if A_name in lora.keys():
            mid = None
            if mid_name in lora.keys():
                mid = lora[mid_name]
                loaded_keys.add(mid_name)
            patch_dict[to_load[x]] = (lora[A_name], lora[B_name], alpha, mid)
            loaded_keys.add(A_name)
            loaded_keys.add(B_name)

        hada_w1_a_name = "{}.hada_w1_a".format(x)
        hada_w1_b_name = "{}.hada_w1_b".format(x)
        hada_w2_a_name = "{}.hada_w2_a".format(x)
        hada_w2_b_name = "{}.hada_w2_b".format(x)
        hada_t1_name = "{}.hada_t1".format(x)
        hada_t2_name = "{}.hada_t2".format(x)
        if hada_w1_a_name in lora.keys():
            hada_t1 = None
            hada_t2 = None
            if hada_t1_name in lora.keys():
                hada_t1 = lora[hada_t1_name]
                hada_t2 = lora[hada_t2_name]
                loaded_keys.add(hada_t1_name)
                loaded_keys.add(hada_t2_name)

            patch_dict[to_load[x]] = (lora[hada_w1_a_name], lora[hada_w1_b_name], alpha, lora[hada_w2_a_name], lora[hada_w2_b_name], hada_t1, hada_t2)
            loaded_keys.add(hada_w1_a_name)
            loaded_keys.add(hada_w1_b_name)
            loaded_keys.add(hada_w2_a_name)
            loaded_keys.add(hada_w2_b_name)

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
        for c in LORA_UNET_MAP_ATTENTIONS:
            k = "{}.{}.weight".format(tk, c)
            if k in sdk:
                lora_key = "lora_unet_down_blocks_{}_attentions_{}_{}".format(counter // 2, counter % 2, LORA_UNET_MAP_ATTENTIONS[c])
                key_map[lora_key] = k
                up_counter += 1
        if up_counter >= 4:
            counter += 1
    for c in LORA_UNET_MAP_ATTENTIONS:
        k = "model.diffusion_model.middle_block.1.{}.weight".format(c)
        if k in sdk:
            lora_key = "lora_unet_mid_block_attentions_0_{}".format(LORA_UNET_MAP_ATTENTIONS[c])
            key_map[lora_key] = k
    counter = 3
    for b in range(12):
        tk = "model.diffusion_model.output_blocks.{}.1".format(b)
        up_counter = 0
        for c in LORA_UNET_MAP_ATTENTIONS:
            k = "{}.{}.weight".format(tk, c)
            if k in sdk:
                lora_key = "lora_unet_up_blocks_{}_attentions_{}_{}".format(counter // 3, counter % 3, LORA_UNET_MAP_ATTENTIONS[c])
                key_map[lora_key] = k
                up_counter += 1
        if up_counter >= 4:
            counter += 1
    counter = 0
    text_model_lora_key = "lora_te_text_model_encoder_layers_{}_{}"
    for b in range(24):
        for c in LORA_CLIP_MAP:
            k = "transformer.text_model.encoder.layers.{}.{}.weight".format(b, c)
            if k in sdk:
                lora_key = text_model_lora_key.format(b, LORA_CLIP_MAP[c])
                key_map[lora_key] = k


    #Locon stuff
    ds_counter = 0
    counter = 0
    for b in range(12):
        tk = "model.diffusion_model.input_blocks.{}.0".format(b)
        key_in = False
        for c in LORA_UNET_MAP_RESNET:
            k = "{}.{}.weight".format(tk, c)
            if k in sdk:
                lora_key = "lora_unet_down_blocks_{}_{}".format(counter // 2, LORA_UNET_MAP_RESNET[c].format(counter % 2))
                key_map[lora_key] = k
                key_in = True
        for bb in range(3):
            k = "{}.{}.op.weight".format(tk[:-2], bb)
            if k in sdk:
                lora_key = "lora_unet_down_blocks_{}_downsamplers_0_conv".format(ds_counter)
                key_map[lora_key] = k
                ds_counter += 1
        if key_in:
            counter += 1

    counter = 0
    for b in range(3):
        tk = "model.diffusion_model.middle_block.{}".format(b)
        key_in = False
        for c in LORA_UNET_MAP_RESNET:
            k = "{}.{}.weight".format(tk, c)
            if k in sdk:
                lora_key = "lora_unet_mid_block_{}".format(LORA_UNET_MAP_RESNET[c].format(counter))
                key_map[lora_key] = k
                key_in = True
        if key_in:
            counter += 1

    counter = 0
    us_counter = 0
    for b in range(12):
        tk = "model.diffusion_model.output_blocks.{}.0".format(b)
        key_in = False
        for c in LORA_UNET_MAP_RESNET:
            k = "{}.{}.weight".format(tk, c)
            if k in sdk:
                lora_key = "lora_unet_up_blocks_{}_{}".format(counter // 3, LORA_UNET_MAP_RESNET[c].format(counter % 3))
                key_map[lora_key] = k
                key_in = True
        for bb in range(3):
            k = "{}.{}.conv.weight".format(tk[:-2], bb)
            if k in sdk:
                lora_key = "lora_unet_up_blocks_{}_upsamplers_0_conv".format(us_counter)
                key_map[lora_key] = k
                us_counter += 1
        if key_in:
            counter += 1

    return key_map

class ModelPatcher:
    def __init__(self, model):
        self.model = model
        self.patches = []
        self.backup = {}
        self.model_options = {"transformer_options":{}}

    def clone(self):
        n = ModelPatcher(self.model)
        n.patches = self.patches[:]
        n.model_options = copy.deepcopy(self.model_options)
        return n

    def set_model_tomesd(self, ratio):
        self.model_options["transformer_options"]["tomesd"] = {"ratio": ratio}

    def model_dtype(self):
        return self.model.diffusion_model.dtype

    def add_patches(self, patches, strength=1.0):
        p = {}
        model_sd = self.model.state_dict()
        for k in patches:
            if k in model_sd:
                p[k] = patches[k]
        self.patches += [(strength, p)]
        return p.keys()

    def patch_model(self):
        model_sd = self.model.state_dict()
        for p in self.patches:
            for k in p[1]:
                v = p[1][k]
                key = k
                if key not in model_sd:
                    print("could not patch. key doesn't exist in model:", k)
                    continue

                weight = model_sd[key]
                if key not in self.backup:
                    self.backup[key] = weight.clone()

                alpha = p[0]

                if len(v) == 4: #lora/locon
                    mat1 = v[0]
                    mat2 = v[1]
                    if v[2] is not None:
                        alpha *= v[2] / mat2.shape[0]
                    if v[3] is not None:
                        #locon mid weights, hopefully the math is fine because I didn't properly test it
                        final_shape = [mat2.shape[1], mat2.shape[0], v[3].shape[2], v[3].shape[3]]
                        mat2 = torch.mm(mat2.transpose(0, 1).flatten(start_dim=1).float(), v[3].transpose(0, 1).flatten(start_dim=1).float()).reshape(final_shape).transpose(0, 1)
                    weight += (alpha * torch.mm(mat1.flatten(start_dim=1).float(), mat2.flatten(start_dim=1).float())).reshape(weight.shape).type(weight.dtype).to(weight.device)
                else: #loha
                    w1a = v[0]
                    w1b = v[1]
                    if v[2] is not None:
                        alpha *= v[2] / w1b.shape[0]
                    w2a = v[3]
                    w2b = v[4]
                    if v[5] is not None: #cp decomposition
                        t1 = v[5]
                        t2 = v[6]
                        m1 = torch.einsum('i j k l, j r, i p -> p r k l', t1.float(), w1b.float(), w1a.float())
                        m2 = torch.einsum('i j k l, j r, i p -> p r k l', t2.float(), w2b.float(), w2a.float())
                    else:
                        m1 = torch.mm(w1a.float(), w1b.float())
                        m2 = torch.mm(w2a.float(), w2b.float())

                    weight += (alpha * m1 * m2).reshape(weight.shape).type(weight.dtype).to(weight.device)
        return self.model
    def unpatch_model(self):
        model_sd = self.model.state_dict()
        keys = list(self.backup.keys())
        for k in keys:
            model_sd[k][:] = self.backup[k]
            del self.backup[k]

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

        if self.target_clip == "ldm.modules.encoders.modules.FrozenOpenCLIPEmbedder":
            clip = sd2_clip.SD2ClipModel
            tokenizer = sd2_clip.SD2Tokenizer
        elif self.target_clip == "ldm.modules.encoders.modules.FrozenCLIPEmbedder":
            clip = sd1_clip.SD1ClipModel
            tokenizer = sd1_clip.SD1Tokenizer

        self.cond_stage_model = clip(**(params))
        self.tokenizer = tokenizer(embedding_directory=embedding_directory)
        self.patcher = ModelPatcher(self.cond_stage_model)
        self.layer_idx = None

    def clone(self):
        n = CLIP(no_init=True)
        n.target_clip = self.target_clip
        n.patcher = self.patcher.clone()
        n.cond_stage_model = self.cond_stage_model
        n.tokenizer = self.tokenizer
        n.layer_idx = self.layer_idx
        return n

    def load_from_state_dict(self, sd):
        self.cond_stage_model.transformer.load_state_dict(sd, strict=False)

    def add_patches(self, patches, strength=1.0):
        return self.patcher.add_patches(patches, strength)

    def clip_layer(self, layer_idx):
        self.layer_idx = layer_idx

    def encode(self, text):
        if self.layer_idx is not None:
            self.cond_stage_model.clip_layer(self.layer_idx)
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
    def __init__(self, ckpt_path=None, scale_factor=0.18215, device=None, config=None):
        if config is None:
            #default SD1.x/SD2.x VAE parameters
            ddconfig = {'double_z': True, 'z_channels': 4, 'resolution': 256, 'in_channels': 3, 'out_ch': 3, 'ch': 128, 'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}
            self.first_stage_model = AutoencoderKL(ddconfig, {'target': 'torch.nn.Identity'}, 4, monitor="val/rec_loss", ckpt_path=ckpt_path)
        else:
            self.first_stage_model = AutoencoderKL(**(config['params']), ckpt_path=ckpt_path)
        self.first_stage_model = self.first_stage_model.eval()
        self.scale_factor = scale_factor
        if device is None:
            device = model_management.get_torch_device()
        self.device = device

    def decode_tiled_(self, samples, tile_x=64, tile_y=64, overlap = 16):
        decode_fn = lambda a: (self.first_stage_model.decode(1. / self.scale_factor * a.to(self.device)) + 1.0)
        output = torch.clamp((
            (utils.tiled_scale(samples, decode_fn, tile_x // 2, tile_y * 2, overlap, upscale_amount = 8) +
            utils.tiled_scale(samples, decode_fn, tile_x * 2, tile_y // 2, overlap, upscale_amount = 8) +
             utils.tiled_scale(samples, decode_fn, tile_x, tile_y, overlap, upscale_amount = 8))
            / 3.0) / 2.0, min=0.0, max=1.0)
        return output

    def decode(self, samples_in):
        model_management.unload_model()
        self.first_stage_model = self.first_stage_model.to(self.device)
        try:
            free_memory = model_management.get_free_memory(self.device)
            batch_number = int((free_memory * 0.7) / (2562 * samples_in.shape[2] * samples_in.shape[3] * 64))
            batch_number = max(1, batch_number)

            pixel_samples = torch.empty((samples_in.shape[0], 3, round(samples_in.shape[2] * 8), round(samples_in.shape[3] * 8)), device="cpu")
            for x in range(0, samples_in.shape[0], batch_number):
                samples = samples_in[x:x+batch_number].to(self.device)
                pixel_samples[x:x+batch_number] = torch.clamp((self.first_stage_model.decode(1. / self.scale_factor * samples) + 1.0) / 2.0, min=0.0, max=1.0).cpu()
        except model_management.OOM_EXCEPTION as e:
            print("Warning: Ran out of memory when regular VAE decoding, retrying with tiled VAE decoding.")
            pixel_samples = self.decode_tiled_(samples_in)

        self.first_stage_model = self.first_stage_model.cpu()
        pixel_samples = pixel_samples.cpu().movedim(1,-1)
        return pixel_samples

    def decode_tiled(self, samples, tile_x=64, tile_y=64, overlap = 16):
        model_management.unload_model()
        self.first_stage_model = self.first_stage_model.to(self.device)
        output = self.decode_tiled_(samples, tile_x, tile_y, overlap)
        self.first_stage_model = self.first_stage_model.cpu()
        return output.movedim(1,-1)

    def encode(self, pixel_samples):
        model_management.unload_model()
        self.first_stage_model = self.first_stage_model.to(self.device)
        pixel_samples = pixel_samples.movedim(-1,1).to(self.device)
        samples = self.first_stage_model.encode(2. * pixel_samples - 1.).sample() * self.scale_factor
        self.first_stage_model = self.first_stage_model.cpu()
        samples = samples.cpu()
        return samples

    def encode_tiled(self, pixel_samples, tile_x=512, tile_y=512, overlap = 64):
        model_management.unload_model()
        self.first_stage_model = self.first_stage_model.to(self.device)
        pixel_samples = pixel_samples.movedim(-1,1).to(self.device)
        samples = utils.tiled_scale(pixel_samples, lambda a: self.first_stage_model.encode(2. * a - 1.).sample() * self.scale_factor, tile_x, tile_y, overlap, upscale_amount = (1/8), out_channels=4)
        samples += utils.tiled_scale(pixel_samples, lambda a: self.first_stage_model.encode(2. * a - 1.).sample() * self.scale_factor, tile_x * 2, tile_y // 2, overlap, upscale_amount = (1/8), out_channels=4)
        samples += utils.tiled_scale(pixel_samples, lambda a: self.first_stage_model.encode(2. * a - 1.).sample() * self.scale_factor, tile_x // 2, tile_y * 2, overlap, upscale_amount = (1/8), out_channels=4)
        samples /= 3.0
        self.first_stage_model = self.first_stage_model.cpu()
        samples = samples.cpu()
        return samples

def resize_image_to(tensor, target_latent_tensor, batched_number):
    tensor = utils.common_upscale(tensor, target_latent_tensor.shape[3] * 8, target_latent_tensor.shape[2] * 8, 'nearest-exact', "center")
    target_batch_size = target_latent_tensor.shape[0]

    current_batch_size = tensor.shape[0]
    print(current_batch_size, target_batch_size)
    if current_batch_size == 1:
        return tensor

    per_batch = target_batch_size // batched_number
    tensor = tensor[:per_batch]

    if per_batch > tensor.shape[0]:
        tensor = torch.cat([tensor] * (per_batch // tensor.shape[0]) + [tensor[:(per_batch % tensor.shape[0])]], dim=0)

    current_batch_size = tensor.shape[0]
    if current_batch_size == target_batch_size:
        return tensor
    else:
        return torch.cat([tensor] * batched_number, dim=0)

class ControlNet:
    def __init__(self, control_model, device=None):
        self.control_model = control_model
        self.cond_hint_original = None
        self.cond_hint = None
        self.strength = 1.0
        if device is None:
            device = model_management.get_torch_device()
        self.device = device
        self.previous_controlnet = None

    def get_control(self, x_noisy, t, cond_txt, batched_number):
        control_prev = None
        if self.previous_controlnet is not None:
            control_prev = self.previous_controlnet.get_control(x_noisy, t, cond_txt, batched_number)

        output_dtype = x_noisy.dtype
        if self.cond_hint is None or x_noisy.shape[2] * 8 != self.cond_hint.shape[2] or x_noisy.shape[3] * 8 != self.cond_hint.shape[3]:
            if self.cond_hint is not None:
                del self.cond_hint
            self.cond_hint = None
            self.cond_hint = resize_image_to(self.cond_hint_original, x_noisy, batched_number).to(self.control_model.dtype).to(self.device)

        if self.control_model.dtype == torch.float16:
            precision_scope = torch.autocast
        else:
            precision_scope = contextlib.nullcontext

        with precision_scope(model_management.get_autocast_device(self.device)):
            self.control_model = model_management.load_if_low_vram(self.control_model)
            control = self.control_model(x=x_noisy, hint=self.cond_hint, timesteps=t, context=cond_txt)
            self.control_model = model_management.unload_if_low_vram(self.control_model)
        out = {'middle':[], 'output': []}
        autocast_enabled = torch.is_autocast_enabled()

        for i in range(len(control)):
            if i == (len(control) - 1):
                key = 'middle'
                index = 0
            else:
                key = 'output'
                index = i
            x = control[i]
            x *= self.strength
            if x.dtype != output_dtype and not autocast_enabled:
                x = x.to(output_dtype)

            if control_prev is not None and key in control_prev:
                prev = control_prev[key][index]
                if prev is not None:
                    x += prev
            out[key].append(x)
        if control_prev is not None and 'input' in control_prev:
            out['input'] = control_prev['input']
        return out

    def set_cond_hint(self, cond_hint, strength=1.0):
        self.cond_hint_original = cond_hint
        self.strength = strength
        return self

    def set_previous_controlnet(self, controlnet):
        self.previous_controlnet = controlnet
        return self

    def cleanup(self):
        if self.previous_controlnet is not None:
            self.previous_controlnet.cleanup()
        if self.cond_hint is not None:
            del self.cond_hint
            self.cond_hint = None

    def copy(self):
        c = ControlNet(self.control_model)
        c.cond_hint_original = self.cond_hint_original
        c.strength = self.strength
        return c

    def get_control_models(self):
        out = []
        if self.previous_controlnet is not None:
            out += self.previous_controlnet.get_control_models()
        out.append(self.control_model)
        return out

def load_controlnet(ckpt_path, model=None):
    controlnet_data = utils.load_torch_file(ckpt_path)
    pth_key = 'control_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight'
    pth = False
    sd2 = False
    key = 'input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight'
    if pth_key in controlnet_data:
        pth = True
        key = pth_key
    elif key in controlnet_data:
        pass
    else:
        net = load_t2i_adapter(controlnet_data)
        if net is None:
            print("error checkpoint does not contain controlnet or t2i adapter data", ckpt_path)
        return net

    context_dim = controlnet_data[key].shape[1]

    use_fp16 = False
    if model_management.should_use_fp16() and controlnet_data[key].dtype == torch.float16:
        use_fp16 = True

    if context_dim == 768:
        #SD1.x
        control_model = cldm.ControlNet(image_size=32,
                                        in_channels=4,
                                        hint_channels=3,
                                        model_channels=320,
                                        attention_resolutions=[ 4, 2, 1 ],
                                        num_res_blocks=2,
                                        channel_mult=[ 1, 2, 4, 4 ],
                                        num_heads=8,
                                        use_spatial_transformer=True,
                                        transformer_depth=1,
                                        context_dim=context_dim,
                                        use_checkpoint=True,
                                        legacy=False,
                                        use_fp16=use_fp16)
    else:
        #SD2.x
        control_model = cldm.ControlNet(image_size=32,
                                        in_channels=4,
                                        hint_channels=3,
                                        model_channels=320,
                                        attention_resolutions=[ 4, 2, 1 ],
                                        num_res_blocks=2,
                                        channel_mult=[ 1, 2, 4, 4 ],
                                        num_head_channels=64,
                                        use_spatial_transformer=True,
                                        use_linear_in_transformer=True,
                                        transformer_depth=1,
                                        context_dim=context_dim,
                                        use_checkpoint=True,
                                        legacy=False,
                                        use_fp16=use_fp16)
    if pth:
        if 'difference' in controlnet_data:
            if model is not None:
                m = model.patch_model()
                model_sd = m.state_dict()
                for x in controlnet_data:
                    c_m = "control_model."
                    if x.startswith(c_m):
                        sd_key = "model.diffusion_model.{}".format(x[len(c_m):])
                        if sd_key in model_sd:
                            cd = controlnet_data[x]
                            cd += model_sd[sd_key].type(cd.dtype).to(cd.device)
                model.unpatch_model()
            else:
                print("WARNING: Loaded a diff controlnet without a model. It will very likely not work.")

        class WeightsLoader(torch.nn.Module):
            pass
        w = WeightsLoader()
        w.control_model = control_model
        w.load_state_dict(controlnet_data, strict=False)
    else:
        control_model.load_state_dict(controlnet_data, strict=False)

    if use_fp16:
        control_model = control_model.half()

    control = ControlNet(control_model)
    return control

class T2IAdapter:
    def __init__(self, t2i_model, channels_in, device=None):
        self.t2i_model = t2i_model
        self.channels_in = channels_in
        self.strength = 1.0
        if device is None:
            device = model_management.get_torch_device()
        self.device = device
        self.previous_controlnet = None
        self.control_input = None
        self.cond_hint_original = None
        self.cond_hint = None

    def get_control(self, x_noisy, t, cond_txt, batched_number):
        control_prev = None
        if self.previous_controlnet is not None:
            control_prev = self.previous_controlnet.get_control(x_noisy, t, cond_txt, batched_number)

        if self.cond_hint is None or x_noisy.shape[2] * 8 != self.cond_hint.shape[2] or x_noisy.shape[3] * 8 != self.cond_hint.shape[3]:
            if self.cond_hint is not None:
                del self.cond_hint
            self.cond_hint = None
            self.cond_hint = resize_image_to(self.cond_hint_original, x_noisy, batched_number).float().to(self.device)
            if self.channels_in == 1 and self.cond_hint.shape[1] > 1:
                self.cond_hint = torch.mean(self.cond_hint, 1, keepdim=True)
            self.t2i_model.to(self.device)
            self.control_input = self.t2i_model(self.cond_hint)
            self.t2i_model.cpu()

        output_dtype = x_noisy.dtype
        out = {'input':[]}

        autocast_enabled = torch.is_autocast_enabled()
        for i in range(len(self.control_input)):
            key = 'input'
            x = self.control_input[i] * self.strength
            if x.dtype != output_dtype and not autocast_enabled:
                x = x.to(output_dtype)

            if control_prev is not None and key in control_prev:
                index = len(control_prev[key]) - i * 3 - 3
                prev = control_prev[key][index]
                if prev is not None:
                    x += prev
            out[key].insert(0, None)
            out[key].insert(0, None)
            out[key].insert(0, x)

        if control_prev is not None and 'input' in control_prev:
            for i in range(len(out['input'])):
                if out['input'][i] is None:
                    out['input'][i] = control_prev['input'][i]
        if control_prev is not None and 'middle' in control_prev:
            out['middle'] = control_prev['middle']
        if control_prev is not None and 'output' in control_prev:
            out['output'] = control_prev['output']
        return out

    def set_cond_hint(self, cond_hint, strength=1.0):
        self.cond_hint_original = cond_hint
        self.strength = strength
        return self

    def set_previous_controlnet(self, controlnet):
        self.previous_controlnet = controlnet
        return self

    def copy(self):
        c = T2IAdapter(self.t2i_model, self.channels_in)
        c.cond_hint_original = self.cond_hint_original
        c.strength = self.strength
        return c

    def cleanup(self):
        if self.previous_controlnet is not None:
            self.previous_controlnet.cleanup()
        if self.cond_hint is not None:
            del self.cond_hint
            self.cond_hint = None

    def get_control_models(self):
        out = []
        if self.previous_controlnet is not None:
            out += self.previous_controlnet.get_control_models()
        return out

def load_t2i_adapter(t2i_data):
    keys = t2i_data.keys()
    if "body.0.in_conv.weight" in keys:
        cin = t2i_data['body.0.in_conv.weight'].shape[1]
        model_ad = adapter.Adapter_light(cin=cin, channels=[320, 640, 1280, 1280], nums_rb=4)
    elif 'conv_in.weight' in keys:
        cin = t2i_data['conv_in.weight'].shape[1]
        model_ad = adapter.Adapter(cin=cin, channels=[320, 640, 1280, 1280][:4], nums_rb=2, ksize=1, sk=True, use_conv=False)
    else:
        return None
    model_ad.load_state_dict(t2i_data)
    return T2IAdapter(model_ad, cin // 64)


class StyleModel:
    def __init__(self, model, device="cpu"):
        self.model = model

    def get_cond(self, input):
        return self.model(input.last_hidden_state)


def load_style_model(ckpt_path):
    model_data = utils.load_torch_file(ckpt_path)
    keys = model_data.keys()
    if "style_embedding" in keys:
        model = adapter.StyleAdapter(width=1024, context_dim=768, num_head=8, n_layes=3, num_token=8)
    else:
        raise Exception("invalid style model {}".format(ckpt_path))
    model.load_state_dict(model_data)
    return StyleModel(model)


def load_clip(ckpt_path, embedding_directory=None):
    clip_data = utils.load_torch_file(ckpt_path)
    config = {}
    if "text_model.encoder.layers.22.mlp.fc1.weight" in clip_data:
        config['target'] = 'ldm.modules.encoders.modules.FrozenOpenCLIPEmbedder'
    else:
        config['target'] = 'ldm.modules.encoders.modules.FrozenCLIPEmbedder'
    clip = CLIP(config=config, embedding_directory=embedding_directory)
    clip.load_from_state_dict(clip_data)
    return clip

def load_checkpoint(config_path, ckpt_path, output_vae=True, output_clip=True, embedding_directory=None):
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    model_config_params = config['model']['params']
    clip_config = model_config_params['cond_stage_config']
    scale_factor = model_config_params['scale_factor']
    vae_config = model_config_params['first_stage_config']

    fp16 = False
    if "unet_config" in model_config_params:
        if "params" in model_config_params["unet_config"]:
            if "use_fp16" in model_config_params["unet_config"]["params"]:
                fp16 = model_config_params["unet_config"]["params"]["use_fp16"]

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

    model = instantiate_from_config(config["model"])
    sd = utils.load_torch_file(ckpt_path)
    model = load_model_weights(model, sd, verbose=False, load_state_dict_to=load_state_dict_to)

    if fp16:
        model = model.half()

    return (ModelPatcher(model), clip, vae)


def load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, output_clipvision=False, embedding_directory=None):
    sd = utils.load_torch_file(ckpt_path)
    sd_keys = sd.keys()
    clip = None
    clipvision = None
    vae = None

    fp16 = model_management.should_use_fp16()

    class WeightsLoader(torch.nn.Module):
        pass

    w = WeightsLoader()
    load_state_dict_to = []
    if output_vae:
        vae = VAE()
        w.first_stage_model = vae.first_stage_model
        load_state_dict_to = [w]

    if output_clip:
        clip_config = {}
        if "cond_stage_model.model.transformer.resblocks.22.attn.out_proj.weight" in sd_keys:
            clip_config['target'] = 'ldm.modules.encoders.modules.FrozenOpenCLIPEmbedder'
        else:
            clip_config['target'] = 'ldm.modules.encoders.modules.FrozenCLIPEmbedder'
        clip = CLIP(config=clip_config, embedding_directory=embedding_directory)
        w.cond_stage_model = clip.cond_stage_model
        load_state_dict_to = [w]

    clipvision_key = "embedder.model.visual.transformer.resblocks.0.attn.in_proj_weight"
    noise_aug_config = None
    if clipvision_key in sd_keys:
        size = sd[clipvision_key].shape[1]

        if output_clipvision:
            clipvision = clip_vision.load_clipvision_from_sd(sd)

        noise_aug_key = "noise_augmentor.betas"
        if noise_aug_key in sd_keys:
            noise_aug_config = {}
            params = {}
            noise_schedule_config = {}
            noise_schedule_config["timesteps"] = sd[noise_aug_key].shape[0]
            noise_schedule_config["beta_schedule"] = "squaredcos_cap_v2"
            params["noise_schedule_config"] = noise_schedule_config
            noise_aug_config['target'] = "ldm.modules.encoders.noise_aug_modules.CLIPEmbeddingNoiseAugmentation"
            if size == 1280: #h
                params["timestep_dim"] = 1024
            elif size == 1024: #l
                params["timestep_dim"] = 768
            noise_aug_config['params'] = params

    sd_config = {
        "linear_start": 0.00085,
        "linear_end": 0.012,
        "num_timesteps_cond": 1,
        "log_every_t": 200,
        "timesteps": 1000,
        "first_stage_key": "jpg",
        "cond_stage_key": "txt",
        "image_size": 64,
        "channels": 4,
        "cond_stage_trainable": False,
        "monitor": "val/loss_simple_ema",
        "scale_factor": 0.18215,
        "use_ema": False,
    }

    unet_config = {
        "use_checkpoint": True,
        "image_size": 32,
        "out_channels": 4,
        "attention_resolutions": [
            4,
            2,
            1
        ],
        "num_res_blocks": 2,
        "channel_mult": [
            1,
            2,
            4,
            4
        ],
        "use_spatial_transformer": True,
        "transformer_depth": 1,
        "legacy": False
    }

    if len(sd['model.diffusion_model.input_blocks.1.1.proj_in.weight'].shape) == 2:
        unet_config['use_linear_in_transformer'] = True

    unet_config["use_fp16"] = fp16
    unet_config["model_channels"] = sd['model.diffusion_model.input_blocks.0.0.weight'].shape[0]
    unet_config["in_channels"] = sd['model.diffusion_model.input_blocks.0.0.weight'].shape[1]
    unet_config["context_dim"] = sd['model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight'].shape[1]

    sd_config["unet_config"] = {"target": "ldm.modules.diffusionmodules.openaimodel.UNetModel", "params": unet_config}
    model_config = {"target": "ldm.models.diffusion.ddpm.LatentDiffusion", "params": sd_config}

    if noise_aug_config is not None: #SD2.x unclip model
        sd_config["noise_aug_config"] = noise_aug_config
        sd_config["image_size"] = 96
        sd_config["embedding_dropout"] = 0.25
        sd_config["conditioning_key"] = 'crossattn-adm'
        model_config["target"] = "ldm.models.diffusion.ddpm.ImageEmbeddingConditionedLatentDiffusion"
    elif unet_config["in_channels"] > 4: #inpainting model
        sd_config["conditioning_key"] = "hybrid"
        sd_config["finetune_keys"] = None
        model_config["target"] = "ldm.models.diffusion.ddpm.LatentInpaintDiffusion"
    else:
        sd_config["conditioning_key"] = "crossattn"

    if unet_config["context_dim"] == 1024:
        unet_config["num_head_channels"] = 64 #SD2.x
    else:
        unet_config["num_heads"] = 8 #SD1.x

    unclip = 'model.diffusion_model.label_emb.0.0.weight'
    if unclip in sd_keys:
        unet_config["num_classes"] = "sequential"
        unet_config["adm_in_channels"] = sd[unclip].shape[1]

    if unet_config["context_dim"] == 1024 and unet_config["in_channels"] == 4: #only SD2.x non inpainting models are v prediction
        k = "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.norm1.bias"
        out = sd[k]
        if torch.std(out, unbiased=False) > 0.09: # not sure how well this will actually work. I guess we will find out.
            sd_config["parameterization"] = 'v'

    model = instantiate_from_config(model_config)
    model = load_model_weights(model, sd, verbose=False, load_state_dict_to=load_state_dict_to)

    if fp16:
        model = model.half()

    return (ModelPatcher(model), clip, vae, clipvision)
