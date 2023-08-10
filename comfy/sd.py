import torch
import contextlib
import copy
import inspect

from comfy import model_management
from .ldm.util import instantiate_from_config
from .ldm.models.autoencoder import AutoencoderKL
import yaml
from .cldm import cldm
from .t2i_adapter import adapter

from . import utils
from . import clip_vision
from . import gligen
from . import diffusers_convert
from . import model_base
from . import model_detection

from . import sd1_clip
from . import sd2_clip
from . import sdxl_clip

def load_model_weights(model, sd):
    m, u = model.load_state_dict(sd, strict=False)
    m = set(m)
    unexpected_keys = set(u)

    k = list(sd.keys())
    for x in k:
        if x not in unexpected_keys:
            w = sd.pop(x)
            del w
    if len(m) > 0:
        print("missing", m)
    return model

def load_clip_weights(model, sd):
    k = list(sd.keys())
    for x in k:
        if x.startswith("cond_stage_model.transformer.") and not x.startswith("cond_stage_model.transformer.text_model."):
            y = x.replace("cond_stage_model.transformer.", "cond_stage_model.transformer.text_model.")
            sd[y] = sd.pop(x)

    if 'cond_stage_model.transformer.text_model.embeddings.position_ids' in sd:
        ids = sd['cond_stage_model.transformer.text_model.embeddings.position_ids']
        if ids.dtype == torch.float32:
            sd['cond_stage_model.transformer.text_model.embeddings.position_ids'] = ids.round()

    sd = utils.transformers_convert(sd, "cond_stage_model.model.", "cond_stage_model.transformer.text_model.", 24)
    return load_model_weights(model, sd)

LORA_CLIP_MAP = {
    "mlp.fc1": "mlp_fc1",
    "mlp.fc2": "mlp_fc2",
    "self_attn.k_proj": "self_attn_k_proj",
    "self_attn.q_proj": "self_attn_q_proj",
    "self_attn.v_proj": "self_attn_v_proj",
    "self_attn.out_proj": "self_attn_out_proj",
}


def load_lora(lora, to_load):
    patch_dict = {}
    loaded_keys = set()
    for x in to_load:
        alpha_name = "{}.alpha".format(x)
        alpha = None
        if alpha_name in lora.keys():
            alpha = lora[alpha_name].item()
            loaded_keys.add(alpha_name)

        regular_lora = "{}.lora_up.weight".format(x)
        diffusers_lora = "{}_lora.up.weight".format(x)
        transformers_lora = "{}.lora_linear_layer.up.weight".format(x)
        A_name = None

        if regular_lora in lora.keys():
            A_name = regular_lora
            B_name = "{}.lora_down.weight".format(x)
            mid_name = "{}.lora_mid.weight".format(x)
        elif diffusers_lora in lora.keys():
            A_name = diffusers_lora
            B_name = "{}_lora.down.weight".format(x)
            mid_name = None
        elif transformers_lora in lora.keys():
            A_name = transformers_lora
            B_name ="{}.lora_linear_layer.down.weight".format(x)
            mid_name = None

        if A_name is not None:
            mid = None
            if mid_name is not None and mid_name in lora.keys():
                mid = lora[mid_name]
                loaded_keys.add(mid_name)
            patch_dict[to_load[x]] = (lora[A_name], lora[B_name], alpha, mid)
            loaded_keys.add(A_name)
            loaded_keys.add(B_name)


        ######## loha
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


        ######## lokr
        lokr_w1_name = "{}.lokr_w1".format(x)
        lokr_w2_name = "{}.lokr_w2".format(x)
        lokr_w1_a_name = "{}.lokr_w1_a".format(x)
        lokr_w1_b_name = "{}.lokr_w1_b".format(x)
        lokr_t2_name = "{}.lokr_t2".format(x)
        lokr_w2_a_name = "{}.lokr_w2_a".format(x)
        lokr_w2_b_name = "{}.lokr_w2_b".format(x)

        lokr_w1 = None
        if lokr_w1_name in lora.keys():
            lokr_w1 = lora[lokr_w1_name]
            loaded_keys.add(lokr_w1_name)

        lokr_w2 = None
        if lokr_w2_name in lora.keys():
            lokr_w2 = lora[lokr_w2_name]
            loaded_keys.add(lokr_w2_name)

        lokr_w1_a = None
        if lokr_w1_a_name in lora.keys():
            lokr_w1_a = lora[lokr_w1_a_name]
            loaded_keys.add(lokr_w1_a_name)

        lokr_w1_b = None
        if lokr_w1_b_name in lora.keys():
            lokr_w1_b = lora[lokr_w1_b_name]
            loaded_keys.add(lokr_w1_b_name)

        lokr_w2_a = None
        if lokr_w2_a_name in lora.keys():
            lokr_w2_a = lora[lokr_w2_a_name]
            loaded_keys.add(lokr_w2_a_name)

        lokr_w2_b = None
        if lokr_w2_b_name in lora.keys():
            lokr_w2_b = lora[lokr_w2_b_name]
            loaded_keys.add(lokr_w2_b_name)

        lokr_t2 = None
        if lokr_t2_name in lora.keys():
            lokr_t2 = lora[lokr_t2_name]
            loaded_keys.add(lokr_t2_name)

        if (lokr_w1 is not None) or (lokr_w2 is not None) or (lokr_w1_a is not None) or (lokr_w2_a is not None):
            patch_dict[to_load[x]] = (lokr_w1, lokr_w2, alpha, lokr_w1_a, lokr_w1_b, lokr_w2_a, lokr_w2_b, lokr_t2)

    for x in lora.keys():
        if x not in loaded_keys:
            print("lora key not loaded", x)
    return patch_dict

def model_lora_keys_clip(model, key_map={}):
    sdk = model.state_dict().keys()

    text_model_lora_key = "lora_te_text_model_encoder_layers_{}_{}"
    clip_l_present = False
    for b in range(32):
        for c in LORA_CLIP_MAP:
            k = "transformer.text_model.encoder.layers.{}.{}.weight".format(b, c)
            if k in sdk:
                lora_key = text_model_lora_key.format(b, LORA_CLIP_MAP[c])
                key_map[lora_key] = k
                lora_key = "lora_te1_text_model_encoder_layers_{}_{}".format(b, LORA_CLIP_MAP[c])
                key_map[lora_key] = k
                lora_key = "text_encoder.text_model.encoder.layers.{}.{}".format(b, c) #diffusers lora
                key_map[lora_key] = k

            k = "clip_l.transformer.text_model.encoder.layers.{}.{}.weight".format(b, c)
            if k in sdk:
                lora_key = "lora_te1_text_model_encoder_layers_{}_{}".format(b, LORA_CLIP_MAP[c]) #SDXL base
                key_map[lora_key] = k
                clip_l_present = True
                lora_key = "text_encoder.text_model.encoder.layers.{}.{}".format(b, c) #diffusers lora
                key_map[lora_key] = k

            k = "clip_g.transformer.text_model.encoder.layers.{}.{}.weight".format(b, c)
            if k in sdk:
                if clip_l_present:
                    lora_key = "lora_te2_text_model_encoder_layers_{}_{}".format(b, LORA_CLIP_MAP[c]) #SDXL base
                    key_map[lora_key] = k
                    lora_key = "text_encoder_2.text_model.encoder.layers.{}.{}".format(b, c) #diffusers lora
                    key_map[lora_key] = k
                else:
                    lora_key = "lora_te_text_model_encoder_layers_{}_{}".format(b, LORA_CLIP_MAP[c]) #TODO: test if this is correct for SDXL-Refiner
                    key_map[lora_key] = k
                    lora_key = "text_encoder.text_model.encoder.layers.{}.{}".format(b, c) #diffusers lora
                    key_map[lora_key] = k

    return key_map

def model_lora_keys_unet(model, key_map={}):
    sdk = model.state_dict().keys()

    for k in sdk:
        if k.startswith("diffusion_model.") and k.endswith(".weight"):
            key_lora = k[len("diffusion_model."):-len(".weight")].replace(".", "_")
            key_map["lora_unet_{}".format(key_lora)] = k

    diffusers_keys = utils.unet_to_diffusers(model.model_config.unet_config)
    for k in diffusers_keys:
        if k.endswith(".weight"):
            key_lora = k[:-len(".weight")].replace(".", "_")
            key_map["lora_unet_{}".format(key_lora)] = "diffusion_model.{}".format(diffusers_keys[k])

            diffusers_lora_key = "unet.{}".format(k[:-len(".weight")].replace(".to_", ".processor.to_"))
            if diffusers_lora_key.endswith(".to_out.0"):
                diffusers_lora_key = diffusers_lora_key[:-2]
            key_map[diffusers_lora_key] = "diffusion_model.{}".format(diffusers_keys[k])
    return key_map

def set_attr(obj, attr, value):
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    prev = getattr(obj, attrs[-1])
    setattr(obj, attrs[-1], torch.nn.Parameter(value))
    del prev

class ModelPatcher:
    def __init__(self, model, load_device, offload_device, size=0):
        self.size = size
        self.model = model
        self.patches = {}
        self.backup = {}
        self.model_options = {"transformer_options":{}}
        self.model_size()
        self.load_device = load_device
        self.offload_device = offload_device

    def model_size(self):
        if self.size > 0:
            return self.size
        model_sd = self.model.state_dict()
        size = 0
        for k in model_sd:
            t = model_sd[k]
            size += t.nelement() * t.element_size()
        self.size = size
        self.model_keys = set(model_sd.keys())
        return size

    def clone(self):
        n = ModelPatcher(self.model, self.load_device, self.offload_device, self.size)
        n.patches = {}
        for k in self.patches:
            n.patches[k] = self.patches[k][:]

        n.model_options = copy.deepcopy(self.model_options)
        n.model_keys = self.model_keys
        return n

    def set_model_sampler_cfg_function(self, sampler_cfg_function):
        if len(inspect.signature(sampler_cfg_function).parameters) == 3:
            self.model_options["sampler_cfg_function"] = lambda args: sampler_cfg_function(args["cond"], args["uncond"], args["cond_scale"]) #Old way
        else:
            self.model_options["sampler_cfg_function"] = sampler_cfg_function

    def set_model_unet_function_wrapper(self, unet_wrapper_function):
        self.model_options["model_function_wrapper"] = unet_wrapper_function

    def set_model_patch(self, patch, name):
        to = self.model_options["transformer_options"]
        if "patches" not in to:
            to["patches"] = {}
        to["patches"][name] = to["patches"].get(name, []) + [patch]

    def set_model_patch_replace(self, patch, name, block_name, number):
        to = self.model_options["transformer_options"]
        if "patches_replace" not in to:
            to["patches_replace"] = {}
        if name not in to["patches_replace"]:
            to["patches_replace"][name] = {}
        to["patches_replace"][name][(block_name, number)] = patch

    def set_model_attn1_patch(self, patch):
        self.set_model_patch(patch, "attn1_patch")

    def set_model_attn2_patch(self, patch):
        self.set_model_patch(patch, "attn2_patch")

    def set_model_attn1_replace(self, patch, block_name, number):
        self.set_model_patch_replace(patch, "attn1", block_name, number)

    def set_model_attn2_replace(self, patch, block_name, number):
        self.set_model_patch_replace(patch, "attn2", block_name, number)

    def set_model_attn1_output_patch(self, patch):
        self.set_model_patch(patch, "attn1_output_patch")

    def set_model_attn2_output_patch(self, patch):
        self.set_model_patch(patch, "attn2_output_patch")

    def model_patches_to(self, device):
        to = self.model_options["transformer_options"]
        if "patches" in to:
            patches = to["patches"]
            for name in patches:
                patch_list = patches[name]
                for i in range(len(patch_list)):
                    if hasattr(patch_list[i], "to"):
                        patch_list[i] = patch_list[i].to(device)
        if "patches_replace" in to:
            patches = to["patches_replace"]
            for name in patches:
                patch_list = patches[name]
                for k in patch_list:
                    if hasattr(patch_list[k], "to"):
                        patch_list[k] = patch_list[k].to(device)

    def model_dtype(self):
        if hasattr(self.model, "get_dtype"):
            return self.model.get_dtype()

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        p = set()
        for k in patches:
            if k in self.model_keys:
                p.add(k)
                current_patches = self.patches.get(k, [])
                current_patches.append((strength_patch, patches[k], strength_model))
                self.patches[k] = current_patches

        return list(p)

    def get_key_patches(self, filter_prefix=None):
        model_sd = self.model_state_dict()
        p = {}
        for k in model_sd:
            if filter_prefix is not None:
                if not k.startswith(filter_prefix):
                    continue
            if k in self.patches:
                p[k] = [model_sd[k]] + self.patches[k]
            else:
                p[k] = (model_sd[k],)
        return p

    def model_state_dict(self, filter_prefix=None):
        sd = self.model.state_dict()
        keys = list(sd.keys())
        if filter_prefix is not None:
            for k in keys:
                if not k.startswith(filter_prefix):
                    sd.pop(k)
        return sd

    def patch_model(self, device_to=None):
        model_sd = self.model_state_dict()
        for key in self.patches:
            if key not in model_sd:
                print("could not patch. key doesn't exist in model:", k)
                continue

            weight = model_sd[key]

            if key not in self.backup:
                self.backup[key] = weight.to(self.offload_device)

            if device_to is not None:
                temp_weight = weight.float().to(device_to, copy=True)
            else:
                temp_weight = weight.to(torch.float32, copy=True)
            out_weight = self.calculate_weight(self.patches[key], temp_weight, key).to(weight.dtype)
            set_attr(self.model, key, out_weight)
            del temp_weight
        return self.model

    def calculate_weight(self, patches, weight, key):
        for p in patches:
            alpha = p[0]
            v = p[1]
            strength_model = p[2]

            if strength_model != 1.0:
                weight *= strength_model

            if isinstance(v, list):
                v = (self.calculate_weight(v[1:], v[0].clone(), key), )

            if len(v) == 1:
                w1 = v[0]
                if alpha != 0.0:
                    if w1.shape != weight.shape:
                        print("WARNING SHAPE MISMATCH {} WEIGHT NOT MERGED {} != {}".format(key, w1.shape, weight.shape))
                    else:
                        weight += alpha * w1.type(weight.dtype).to(weight.device)
            elif len(v) == 4: #lora/locon
                mat1 = v[0].float().to(weight.device)
                mat2 = v[1].float().to(weight.device)
                if v[2] is not None:
                    alpha *= v[2] / mat2.shape[0]
                if v[3] is not None:
                    #locon mid weights, hopefully the math is fine because I didn't properly test it
                    mat3 = v[3].float().to(weight.device)
                    final_shape = [mat2.shape[1], mat2.shape[0], mat3.shape[2], mat3.shape[3]]
                    mat2 = torch.mm(mat2.transpose(0, 1).flatten(start_dim=1), mat3.transpose(0, 1).flatten(start_dim=1)).reshape(final_shape).transpose(0, 1)
                try:
                    weight += (alpha * torch.mm(mat1.flatten(start_dim=1), mat2.flatten(start_dim=1))).reshape(weight.shape).type(weight.dtype)
                except Exception as e:
                    print("ERROR", key, e)
            elif len(v) == 8: #lokr
                w1 = v[0]
                w2 = v[1]
                w1_a = v[3]
                w1_b = v[4]
                w2_a = v[5]
                w2_b = v[6]
                t2 = v[7]
                dim = None

                if w1 is None:
                    dim = w1_b.shape[0]
                    w1 = torch.mm(w1_a.float(), w1_b.float())
                else:
                    w1 = w1.float().to(weight.device)

                if w2 is None:
                    dim = w2_b.shape[0]
                    if t2 is None:
                        w2 = torch.mm(w2_a.float().to(weight.device), w2_b.float().to(weight.device))
                    else:
                        w2 = torch.einsum('i j k l, j r, i p -> p r k l', t2.float().to(weight.device), w2_b.float().to(weight.device), w2_a.float().to(weight.device))
                else:
                    w2 = w2.float().to(weight.device)

                if len(w2.shape) == 4:
                    w1 = w1.unsqueeze(2).unsqueeze(2)
                if v[2] is not None and dim is not None:
                    alpha *= v[2] / dim

                try:
                    weight += alpha * torch.kron(w1, w2).reshape(weight.shape).type(weight.dtype)
                except Exception as e:
                    print("ERROR", key, e)
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
                    m1 = torch.einsum('i j k l, j r, i p -> p r k l', t1.float().to(weight.device), w1b.float().to(weight.device), w1a.float().to(weight.device))
                    m2 = torch.einsum('i j k l, j r, i p -> p r k l', t2.float().to(weight.device), w2b.float().to(weight.device), w2a.float().to(weight.device))
                else:
                    m1 = torch.mm(w1a.float().to(weight.device), w1b.float().to(weight.device))
                    m2 = torch.mm(w2a.float().to(weight.device), w2b.float().to(weight.device))

                try:
                    weight += (alpha * m1 * m2).reshape(weight.shape).type(weight.dtype)
                except Exception as e:
                    print("ERROR", key, e)

        return weight

    def unpatch_model(self):
        keys = list(self.backup.keys())

        for k in keys:
            set_attr(self.model, k, self.backup[k])

        self.backup = {}

def load_lora_for_models(model, clip, lora, strength_model, strength_clip):
    key_map = model_lora_keys_unet(model.model)
    key_map = model_lora_keys_clip(clip.cond_stage_model, key_map)
    loaded = load_lora(lora, key_map)
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
    def __init__(self, target=None, embedding_directory=None, no_init=False):
        if no_init:
            return
        params = target.params.copy()
        clip = target.clip
        tokenizer = target.tokenizer

        load_device = model_management.text_encoder_device()
        offload_device = model_management.text_encoder_offload_device()
        params['device'] = load_device
        self.cond_stage_model = clip(**(params))
        #TODO: make sure this doesn't have a quality loss before enabling.
        # if model_management.should_use_fp16(load_device):
        #     self.cond_stage_model.half()

        self.cond_stage_model = self.cond_stage_model.to()

        self.tokenizer = tokenizer(embedding_directory=embedding_directory)
        self.patcher = ModelPatcher(self.cond_stage_model, load_device=load_device, offload_device=offload_device)
        self.layer_idx = None

    def clone(self):
        n = CLIP(no_init=True)
        n.patcher = self.patcher.clone()
        n.cond_stage_model = self.cond_stage_model
        n.tokenizer = self.tokenizer
        n.layer_idx = self.layer_idx
        return n

    def load_from_state_dict(self, sd):
        self.cond_stage_model.load_sd(sd)

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        return self.patcher.add_patches(patches, strength_patch, strength_model)

    def clip_layer(self, layer_idx):
        self.layer_idx = layer_idx

    def tokenize(self, text, return_word_ids=False):
        return self.tokenizer.tokenize_with_weights(text, return_word_ids)

    def encode_from_tokens(self, tokens, return_pooled=False):
        if self.layer_idx is not None:
            self.cond_stage_model.clip_layer(self.layer_idx)
        else:
            self.cond_stage_model.reset_clip_layer()

        model_management.load_model_gpu(self.patcher)
        cond, pooled = self.cond_stage_model.encode_token_weights(tokens)
        if return_pooled:
            return cond, pooled
        return cond

    def encode(self, text):
        tokens = self.tokenize(text)
        return self.encode_from_tokens(tokens)

    def load_sd(self, sd):
        return self.cond_stage_model.load_sd(sd)

    def get_sd(self):
        return self.cond_stage_model.state_dict()

    def patch_model(self):
        self.patcher.patch_model()

    def unpatch_model(self):
        self.patcher.unpatch_model()

    def get_key_patches(self):
        return self.patcher.get_key_patches()

class VAE:
    def __init__(self, ckpt_path=None, device=None, config=None):
        if config is None:
            #default SD1.x/SD2.x VAE parameters
            ddconfig = {'double_z': True, 'z_channels': 4, 'resolution': 256, 'in_channels': 3, 'out_ch': 3, 'ch': 128, 'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}
            self.first_stage_model = AutoencoderKL(ddconfig, {'target': 'torch.nn.Identity'}, 4, monitor="val/rec_loss")
        else:
            self.first_stage_model = AutoencoderKL(**(config['params']))
        self.first_stage_model = self.first_stage_model.eval()
        if ckpt_path is not None:
            sd = utils.load_torch_file(ckpt_path)
            if 'decoder.up_blocks.0.resnets.0.norm1.weight' in sd.keys(): #diffusers format
                sd = diffusers_convert.convert_vae_state_dict(sd)
            self.first_stage_model.load_state_dict(sd, strict=False)

        if device is None:
            device = model_management.vae_device()
        self.device = device
        self.offload_device = model_management.vae_offload_device()
        self.vae_dtype = model_management.vae_dtype()
        self.first_stage_model.to(self.vae_dtype)

    def decode_tiled_(self, samples, tile_x=64, tile_y=64, overlap = 16):
        steps = samples.shape[0] * utils.get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x, tile_y, overlap)
        steps += samples.shape[0] * utils.get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x // 2, tile_y * 2, overlap)
        steps += samples.shape[0] * utils.get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x * 2, tile_y // 2, overlap)
        pbar = utils.ProgressBar(steps)

        decode_fn = lambda a: (self.first_stage_model.decode(a.to(self.vae_dtype).to(self.device)) + 1.0).float()
        output = torch.clamp((
            (utils.tiled_scale(samples, decode_fn, tile_x // 2, tile_y * 2, overlap, upscale_amount = 8, pbar = pbar) +
            utils.tiled_scale(samples, decode_fn, tile_x * 2, tile_y // 2, overlap, upscale_amount = 8, pbar = pbar) +
             utils.tiled_scale(samples, decode_fn, tile_x, tile_y, overlap, upscale_amount = 8, pbar = pbar))
            / 3.0) / 2.0, min=0.0, max=1.0)
        return output

    def encode_tiled_(self, pixel_samples, tile_x=512, tile_y=512, overlap = 64):
        steps = pixel_samples.shape[0] * utils.get_tiled_scale_steps(pixel_samples.shape[3], pixel_samples.shape[2], tile_x, tile_y, overlap)
        steps += pixel_samples.shape[0] * utils.get_tiled_scale_steps(pixel_samples.shape[3], pixel_samples.shape[2], tile_x // 2, tile_y * 2, overlap)
        steps += pixel_samples.shape[0] * utils.get_tiled_scale_steps(pixel_samples.shape[3], pixel_samples.shape[2], tile_x * 2, tile_y // 2, overlap)
        pbar = utils.ProgressBar(steps)

        encode_fn = lambda a: self.first_stage_model.encode(2. * a.to(self.vae_dtype).to(self.device) - 1.).sample().float()
        samples = utils.tiled_scale(pixel_samples, encode_fn, tile_x, tile_y, overlap, upscale_amount = (1/8), out_channels=4, pbar=pbar)
        samples += utils.tiled_scale(pixel_samples, encode_fn, tile_x * 2, tile_y // 2, overlap, upscale_amount = (1/8), out_channels=4, pbar=pbar)
        samples += utils.tiled_scale(pixel_samples, encode_fn, tile_x // 2, tile_y * 2, overlap, upscale_amount = (1/8), out_channels=4, pbar=pbar)
        samples /= 3.0
        return samples

    def decode(self, samples_in):
        model_management.unload_model()
        self.first_stage_model = self.first_stage_model.to(self.device)
        try:
            free_memory = model_management.get_free_memory(self.device)
            batch_number = int((free_memory * 0.7) / (2562 * samples_in.shape[2] * samples_in.shape[3] * 64))
            batch_number = max(1, batch_number)

            pixel_samples = torch.empty((samples_in.shape[0], 3, round(samples_in.shape[2] * 8), round(samples_in.shape[3] * 8)), device="cpu")
            for x in range(0, samples_in.shape[0], batch_number):
                samples = samples_in[x:x+batch_number].to(self.vae_dtype).to(self.device)
                pixel_samples[x:x+batch_number] = torch.clamp((self.first_stage_model.decode(samples) + 1.0) / 2.0, min=0.0, max=1.0).cpu().float()
        except model_management.OOM_EXCEPTION as e:
            print("Warning: Ran out of memory when regular VAE decoding, retrying with tiled VAE decoding.")
            pixel_samples = self.decode_tiled_(samples_in)

        self.first_stage_model = self.first_stage_model.to(self.offload_device)
        pixel_samples = pixel_samples.cpu().movedim(1,-1)
        return pixel_samples

    def decode_tiled(self, samples, tile_x=64, tile_y=64, overlap = 16):
        model_management.unload_model()
        self.first_stage_model = self.first_stage_model.to(self.device)
        output = self.decode_tiled_(samples, tile_x, tile_y, overlap)
        self.first_stage_model = self.first_stage_model.to(self.offload_device)
        return output.movedim(1,-1)

    def encode(self, pixel_samples):
        model_management.unload_model()
        self.first_stage_model = self.first_stage_model.to(self.device)
        pixel_samples = pixel_samples.movedim(-1,1)
        try:
            free_memory = model_management.get_free_memory(self.device)
            batch_number = int((free_memory * 0.7) / (2078 * pixel_samples.shape[2] * pixel_samples.shape[3])) #NOTE: this constant along with the one in the decode above are estimated from the mem usage for the VAE and could change.
            batch_number = max(1, batch_number)
            samples = torch.empty((pixel_samples.shape[0], 4, round(pixel_samples.shape[2] // 8), round(pixel_samples.shape[3] // 8)), device="cpu")
            for x in range(0, pixel_samples.shape[0], batch_number):
                pixels_in = (2. * pixel_samples[x:x+batch_number] - 1.).to(self.vae_dtype).to(self.device)
                samples[x:x+batch_number] = self.first_stage_model.encode(pixels_in).sample().cpu().float()

        except model_management.OOM_EXCEPTION as e:
            print("Warning: Ran out of memory when regular VAE encoding, retrying with tiled VAE encoding.")
            samples = self.encode_tiled_(pixel_samples)

        self.first_stage_model = self.first_stage_model.to(self.offload_device)
        return samples

    def encode_tiled(self, pixel_samples, tile_x=512, tile_y=512, overlap = 64):
        model_management.unload_model()
        self.first_stage_model = self.first_stage_model.to(self.device)
        pixel_samples = pixel_samples.movedim(-1,1)
        samples = self.encode_tiled_(pixel_samples, tile_x=tile_x, tile_y=tile_y, overlap=overlap)
        self.first_stage_model = self.first_stage_model.to(self.offload_device)
        return samples

    def get_sd(self):
        return self.first_stage_model.state_dict()


def broadcast_image_to(tensor, target_batch_size, batched_number):
    current_batch_size = tensor.shape[0]
    #print(current_batch_size, target_batch_size)
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

class ControlBase:
    def __init__(self, device=None):
        self.cond_hint_original = None
        self.cond_hint = None
        self.strength = 1.0
        self.timestep_percent_range = (1.0, 0.0)
        self.timestep_range = None

        if device is None:
            device = model_management.get_torch_device()
        self.device = device
        self.previous_controlnet = None

    def set_cond_hint(self, cond_hint, strength=1.0, timestep_percent_range=(1.0, 0.0)):
        self.cond_hint_original = cond_hint
        self.strength = strength
        self.timestep_percent_range = timestep_percent_range
        return self

    def pre_run(self, model, percent_to_timestep_function):
        self.timestep_range = (percent_to_timestep_function(self.timestep_percent_range[0]), percent_to_timestep_function(self.timestep_percent_range[1]))
        if self.previous_controlnet is not None:
            self.previous_controlnet.pre_run(model, percent_to_timestep_function)

    def set_previous_controlnet(self, controlnet):
        self.previous_controlnet = controlnet
        return self

    def cleanup(self):
        if self.previous_controlnet is not None:
            self.previous_controlnet.cleanup()
        if self.cond_hint is not None:
            del self.cond_hint
            self.cond_hint = None
        self.timestep_range = None

    def get_models(self):
        out = []
        if self.previous_controlnet is not None:
            out += self.previous_controlnet.get_models()
        return out

    def copy_to(self, c):
        c.cond_hint_original = self.cond_hint_original
        c.strength = self.strength
        c.timestep_percent_range = self.timestep_percent_range

class ControlNet(ControlBase):
    def __init__(self, control_model, global_average_pooling=False, device=None):
        super().__init__(device)
        self.control_model = control_model
        self.global_average_pooling = global_average_pooling

    def get_control(self, x_noisy, t, cond, batched_number):
        control_prev = None
        if self.previous_controlnet is not None:
            control_prev = self.previous_controlnet.get_control(x_noisy, t, cond, batched_number)

        if self.timestep_range is not None:
            if t[0] > self.timestep_range[0] or t[0] < self.timestep_range[1]:
                if control_prev is not None:
                    return control_prev
                else:
                    return {}

        output_dtype = x_noisy.dtype
        if self.cond_hint is None or x_noisy.shape[2] * 8 != self.cond_hint.shape[2] or x_noisy.shape[3] * 8 != self.cond_hint.shape[3]:
            if self.cond_hint is not None:
                del self.cond_hint
            self.cond_hint = None
            self.cond_hint = utils.common_upscale(self.cond_hint_original, x_noisy.shape[3] * 8, x_noisy.shape[2] * 8, 'nearest-exact', "center").to(self.control_model.dtype).to(self.device)
        if x_noisy.shape[0] != self.cond_hint.shape[0]:
            self.cond_hint = broadcast_image_to(self.cond_hint, x_noisy.shape[0], batched_number)

        if self.control_model.dtype == torch.float16:
            precision_scope = torch.autocast
        else:
            precision_scope = contextlib.nullcontext

        with precision_scope(model_management.get_autocast_device(self.device)):
            self.control_model = model_management.load_if_low_vram(self.control_model)
            context = torch.cat(cond['c_crossattn'], 1)
            y = cond.get('c_adm', None)
            control = self.control_model(x=x_noisy, hint=self.cond_hint, timesteps=t, context=context, y=y)
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
            if self.global_average_pooling:
                x = torch.mean(x, dim=(2, 3), keepdim=True).repeat(1, 1, x.shape[2], x.shape[3])

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

    def copy(self):
        c = ControlNet(self.control_model, global_average_pooling=self.global_average_pooling)
        self.copy_to(c)
        return c

    def get_models(self):
        out = super().get_models()
        out.append(self.control_model)
        return out


def load_controlnet(ckpt_path, model=None):
    controlnet_data = utils.load_torch_file(ckpt_path, safe_load=True)

    controlnet_config = None
    if "controlnet_cond_embedding.conv_in.weight" in controlnet_data: #diffusers format
        use_fp16 = model_management.should_use_fp16()
        controlnet_config = model_detection.model_config_from_diffusers_unet(controlnet_data, use_fp16).unet_config
        diffusers_keys = utils.unet_to_diffusers(controlnet_config)
        diffusers_keys["controlnet_mid_block.weight"] = "middle_block_out.0.weight"
        diffusers_keys["controlnet_mid_block.bias"] = "middle_block_out.0.bias"

        count = 0
        loop = True
        while loop:
            suffix = [".weight", ".bias"]
            for s in suffix:
                k_in = "controlnet_down_blocks.{}{}".format(count, s)
                k_out = "zero_convs.{}.0{}".format(count, s)
                if k_in not in controlnet_data:
                    loop = False
                    break
                diffusers_keys[k_in] = k_out
            count += 1

        count = 0
        loop = True
        while loop:
            suffix = [".weight", ".bias"]
            for s in suffix:
                if count == 0:
                    k_in = "controlnet_cond_embedding.conv_in{}".format(s)
                else:
                    k_in = "controlnet_cond_embedding.blocks.{}{}".format(count - 1, s)
                k_out = "input_hint_block.{}{}".format(count * 2, s)
                if k_in not in controlnet_data:
                    k_in = "controlnet_cond_embedding.conv_out{}".format(s)
                    loop = False
                diffusers_keys[k_in] = k_out
            count += 1

        new_sd = {}
        for k in diffusers_keys:
            if k in controlnet_data:
                new_sd[diffusers_keys[k]] = controlnet_data.pop(k)

        controlnet_data = new_sd

    pth_key = 'control_model.zero_convs.0.0.weight'
    pth = False
    key = 'zero_convs.0.0.weight'
    if pth_key in controlnet_data:
        pth = True
        key = pth_key
        prefix = "control_model."
    elif key in controlnet_data:
        prefix = ""
    else:
        net = load_t2i_adapter(controlnet_data)
        if net is None:
            print("error checkpoint does not contain controlnet or t2i adapter data", ckpt_path)
        return net

    if controlnet_config is None:
        use_fp16 = model_management.should_use_fp16()
        controlnet_config = model_detection.model_config_from_unet(controlnet_data, prefix, use_fp16).unet_config
    controlnet_config.pop("out_channels")
    controlnet_config["hint_channels"] = controlnet_data["{}input_hint_block.0.weight".format(prefix)].shape[1]
    control_model = cldm.ControlNet(**controlnet_config)

    if pth:
        if 'difference' in controlnet_data:
            if model is not None:
                m = model.patch_model()
                model_sd = m.state_dict()
                for x in controlnet_data:
                    c_m = "control_model."
                    if x.startswith(c_m):
                        sd_key = "diffusion_model.{}".format(x[len(c_m):])
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
        missing, unexpected = w.load_state_dict(controlnet_data, strict=False)
    else:
        missing, unexpected = control_model.load_state_dict(controlnet_data, strict=False)
    print(missing, unexpected)

    if use_fp16:
        control_model = control_model.half()

    global_average_pooling = False
    if ckpt_path.endswith("_shuffle.pth") or ckpt_path.endswith("_shuffle.safetensors") or ckpt_path.endswith("_shuffle_fp16.safetensors"): #TODO: smarter way of enabling global_average_pooling
        global_average_pooling = True

    control = ControlNet(control_model, global_average_pooling=global_average_pooling)
    return control

class T2IAdapter(ControlBase):
    def __init__(self, t2i_model, channels_in, device=None):
        super().__init__(device)
        self.t2i_model = t2i_model
        self.channels_in = channels_in
        self.control_input = None

    def get_control(self, x_noisy, t, cond, batched_number):
        control_prev = None
        if self.previous_controlnet is not None:
            control_prev = self.previous_controlnet.get_control(x_noisy, t, cond, batched_number)

        if self.timestep_range is not None:
            if t[0] > self.timestep_range[0] or t[0] < self.timestep_range[1]:
                if control_prev is not None:
                    return control_prev
                else:
                    return {}

        if self.cond_hint is None or x_noisy.shape[2] * 8 != self.cond_hint.shape[2] or x_noisy.shape[3] * 8 != self.cond_hint.shape[3]:
            if self.cond_hint is not None:
                del self.cond_hint
            self.control_input = None
            self.cond_hint = None
            self.cond_hint = utils.common_upscale(self.cond_hint_original, x_noisy.shape[3] * 8, x_noisy.shape[2] * 8, 'nearest-exact', "center").float().to(self.device)
            if self.channels_in == 1 and self.cond_hint.shape[1] > 1:
                self.cond_hint = torch.mean(self.cond_hint, 1, keepdim=True)
        if x_noisy.shape[0] != self.cond_hint.shape[0]:
            self.cond_hint = broadcast_image_to(self.cond_hint, x_noisy.shape[0], batched_number)
        if self.control_input is None:
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

    def copy(self):
        c = T2IAdapter(self.t2i_model, self.channels_in)
        self.copy_to(c)
        return c


def load_t2i_adapter(t2i_data):
    keys = t2i_data.keys()
    if 'adapter' in keys:
        t2i_data = t2i_data['adapter']
        keys = t2i_data.keys()
    if "body.0.in_conv.weight" in keys:
        cin = t2i_data['body.0.in_conv.weight'].shape[1]
        model_ad = adapter.Adapter_light(cin=cin, channels=[320, 640, 1280, 1280], nums_rb=4)
    elif 'conv_in.weight' in keys:
        cin = t2i_data['conv_in.weight'].shape[1]
        channel = t2i_data['conv_in.weight'].shape[0]
        ksize = t2i_data['body.0.block2.weight'].shape[2]
        use_conv = False
        down_opts = list(filter(lambda a: a.endswith("down_opt.op.weight"), keys))
        if len(down_opts) > 0:
            use_conv = True
        model_ad = adapter.Adapter(cin=cin, channels=[channel, channel*2, channel*4, channel*4][:4], nums_rb=2, ksize=ksize, sk=True, use_conv=use_conv)
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
    model_data = utils.load_torch_file(ckpt_path, safe_load=True)
    keys = model_data.keys()
    if "style_embedding" in keys:
        model = adapter.StyleAdapter(width=1024, context_dim=768, num_head=8, n_layes=3, num_token=8)
    else:
        raise Exception("invalid style model {}".format(ckpt_path))
    model.load_state_dict(model_data)
    return StyleModel(model)


def load_clip(ckpt_paths, embedding_directory=None):
    clip_data = []
    for p in ckpt_paths:
        clip_data.append(utils.load_torch_file(p, safe_load=True))

    class EmptyClass:
        pass

    for i in range(len(clip_data)):
        if "transformer.resblocks.0.ln_1.weight" in clip_data[i]:
            clip_data[i] = utils.transformers_convert(clip_data[i], "", "text_model.", 32)

    clip_target = EmptyClass()
    clip_target.params = {}
    if len(clip_data) == 1:
        if "text_model.encoder.layers.30.mlp.fc1.weight" in clip_data[0]:
            clip_target.clip = sdxl_clip.SDXLRefinerClipModel
            clip_target.tokenizer = sdxl_clip.SDXLTokenizer
        elif "text_model.encoder.layers.22.mlp.fc1.weight" in clip_data[0]:
            clip_target.clip = sd2_clip.SD2ClipModel
            clip_target.tokenizer = sd2_clip.SD2Tokenizer
        else:
            clip_target.clip = sd1_clip.SD1ClipModel
            clip_target.tokenizer = sd1_clip.SD1Tokenizer
    else:
        clip_target.clip = sdxl_clip.SDXLClipModel
        clip_target.tokenizer = sdxl_clip.SDXLTokenizer

    clip = CLIP(clip_target, embedding_directory=embedding_directory)
    for c in clip_data:
        m, u = clip.load_sd(c)
        if len(m) > 0:
            print("clip missing:", m)

        if len(u) > 0:
            print("clip unexpected:", u)
    return clip

def load_gligen(ckpt_path):
    data = utils.load_torch_file(ckpt_path, safe_load=True)
    model = gligen.load_gligen(data)
    if model_management.should_use_fp16():
        model = model.half()
    return model

def load_checkpoint(config_path=None, ckpt_path=None, output_vae=True, output_clip=True, embedding_directory=None, state_dict=None, config=None):
    #TODO: this function is a mess and should be removed eventually
    if config is None:
        with open(config_path, 'r') as stream:
            config = yaml.safe_load(stream)
    model_config_params = config['model']['params']
    clip_config = model_config_params['cond_stage_config']
    scale_factor = model_config_params['scale_factor']
    vae_config = model_config_params['first_stage_config']

    fp16 = False
    if "unet_config" in model_config_params:
        if "params" in model_config_params["unet_config"]:
            unet_config = model_config_params["unet_config"]["params"]
            if "use_fp16" in unet_config:
                fp16 = unet_config["use_fp16"]

    noise_aug_config = None
    if "noise_aug_config" in model_config_params:
        noise_aug_config = model_config_params["noise_aug_config"]

    model_type = model_base.ModelType.EPS

    if "parameterization" in model_config_params:
        if model_config_params["parameterization"] == "v":
            model_type = model_base.ModelType.V_PREDICTION

    clip = None
    vae = None

    class WeightsLoader(torch.nn.Module):
        pass

    if state_dict is None:
        state_dict = utils.load_torch_file(ckpt_path)

    class EmptyClass:
        pass

    model_config = EmptyClass()
    model_config.unet_config = unet_config
    from . import latent_formats
    model_config.latent_format = latent_formats.SD15(scale_factor=scale_factor)

    if config['model']["target"].endswith("LatentInpaintDiffusion"):
        model = model_base.SDInpaint(model_config, model_type=model_type)
    elif config['model']["target"].endswith("ImageEmbeddingConditionedLatentDiffusion"):
        model = model_base.SD21UNCLIP(model_config, noise_aug_config["params"], model_type=model_type)
    else:
        model = model_base.BaseModel(model_config, model_type=model_type)

    if fp16:
        model = model.half()

    offload_device = model_management.unet_offload_device()
    model = model.to(offload_device)
    model.load_model_weights(state_dict, "model.diffusion_model.")

    if output_vae:
        w = WeightsLoader()
        vae = VAE(config=vae_config)
        w.first_stage_model = vae.first_stage_model
        load_model_weights(w, state_dict)

    if output_clip:
        w = WeightsLoader()
        clip_target = EmptyClass()
        clip_target.params = clip_config.get("params", {})
        if clip_config["target"].endswith("FrozenOpenCLIPEmbedder"):
            clip_target.clip = sd2_clip.SD2ClipModel
            clip_target.tokenizer = sd2_clip.SD2Tokenizer
        elif clip_config["target"].endswith("FrozenCLIPEmbedder"):
            clip_target.clip = sd1_clip.SD1ClipModel
            clip_target.tokenizer = sd1_clip.SD1Tokenizer
        clip = CLIP(clip_target, embedding_directory=embedding_directory)
        w.cond_stage_model = clip.cond_stage_model
        load_clip_weights(w, state_dict)

    return (ModelPatcher(model, load_device=model_management.get_torch_device(), offload_device=offload_device), clip, vae)

def calculate_parameters(sd, prefix):
    params = 0
    for k in sd.keys():
        if k.startswith(prefix):
            params += sd[k].nelement()
    return params

def load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, output_clipvision=False, embedding_directory=None):
    sd = utils.load_torch_file(ckpt_path)
    sd_keys = sd.keys()
    clip = None
    clipvision = None
    vae = None
    model = None
    clip_target = None

    parameters = calculate_parameters(sd, "model.diffusion_model.")
    fp16 = model_management.should_use_fp16(model_params=parameters)

    class WeightsLoader(torch.nn.Module):
        pass

    model_config = model_detection.model_config_from_unet(sd, "model.diffusion_model.", fp16)
    if model_config is None:
        raise RuntimeError("ERROR: Could not detect model type of: {}".format(ckpt_path))

    if model_config.clip_vision_prefix is not None:
        if output_clipvision:
            clipvision = clip_vision.load_clipvision_from_sd(sd, model_config.clip_vision_prefix, True)

    offload_device = model_management.unet_offload_device()
    model = model_config.get_model(sd, "model.diffusion_model.", device=offload_device)
    model.load_model_weights(sd, "model.diffusion_model.")

    if output_vae:
        vae = VAE()
        w = WeightsLoader()
        w.first_stage_model = vae.first_stage_model
        load_model_weights(w, sd)

    if output_clip:
        w = WeightsLoader()
        clip_target = model_config.clip_target()
        clip = CLIP(clip_target, embedding_directory=embedding_directory)
        w.cond_stage_model = clip.cond_stage_model
        sd = model_config.process_clip_state_dict(sd)
        load_model_weights(w, sd)

    left_over = sd.keys()
    if len(left_over) > 0:
        print("left over keys:", left_over)

    return (ModelPatcher(model, load_device=model_management.get_torch_device(), offload_device=offload_device), clip, vae, clipvision)


def load_unet(unet_path): #load unet in diffusers format
    sd = utils.load_torch_file(unet_path)
    parameters = calculate_parameters(sd, "")
    fp16 = model_management.should_use_fp16(model_params=parameters)

    model_config = model_detection.model_config_from_diffusers_unet(sd, fp16)
    if model_config is None:
        print("ERROR UNSUPPORTED UNET", unet_path)
        return None

    diffusers_keys = utils.unet_to_diffusers(model_config.unet_config)

    new_sd = {}
    for k in diffusers_keys:
        if k in sd:
            new_sd[diffusers_keys[k]] = sd.pop(k)
        else:
            print(diffusers_keys[k], k)
    offload_device = model_management.unet_offload_device()
    model = model_config.get_model(new_sd, "")
    model = model.to(offload_device)
    model.load_model_weights(new_sd, "")
    return ModelPatcher(model, load_device=model_management.get_torch_device(), offload_device=offload_device)

def save_checkpoint(output_path, model, clip, vae, metadata=None):
    try:
        model.patch_model()
        clip.patch_model()
        sd = model.model.state_dict_for_saving(clip.get_sd(), vae.get_sd())
        utils.save_torch_file(sd, output_path, metadata=metadata)
        model.unpatch_model()
        clip.unpatch_model()
    except Exception as e:
        model.unpatch_model()
        clip.unpatch_model()
        raise e
