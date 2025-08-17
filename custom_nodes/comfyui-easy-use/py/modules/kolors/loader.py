import json
import os
import torch
import subprocess
import sys
import comfy.supported_models
import comfy.model_patcher
import comfy.model_management
import comfy.model_detection as model_detection
import comfy.model_base as model_base
from comfy.model_base import sdxl_pooled, CLIPEmbeddingNoiseAugmentation, Timestep, ModelType
from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel
from comfy.clip_vision import ClipVisionModel, Output
from comfy.utils import load_torch_file
from .chatglm.modeling_chatglm import ChatGLMModel, ChatGLMConfig
from .chatglm.tokenization_chatglm import ChatGLMTokenizer

class KolorsUNetModel(UNetModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder_hid_proj = torch.nn.Linear(4096, 2048, bias=True)

    def forward(self, *args, **kwargs):
        with torch.cuda.amp.autocast(enabled=True):
            if "context" in kwargs:
                kwargs["context"] = self.encoder_hid_proj(kwargs["context"])
            result = super().forward(*args, **kwargs)
            return result

class KolorsSDXL(model_base.SDXL):
    def __init__(self, model_config, model_type=ModelType.EPS, device=None):
        model_base.BaseModel.__init__(self, model_config, model_type, device=device, unet_model=KolorsUNetModel)
        self.embedder = Timestep(256)
        self.noise_augmentor = CLIPEmbeddingNoiseAugmentation(**{"noise_schedule_config": {"timesteps": 1000, "beta_schedule": "squaredcos_cap_v2"}, "timestep_dim": 1280})

    def encode_adm(self, **kwargs):
        clip_pooled = sdxl_pooled(kwargs, self.noise_augmentor)
        width = kwargs.get("width", 768)
        height = kwargs.get("height", 768)
        crop_w = kwargs.get("crop_w", 0)
        crop_h = kwargs.get("crop_h", 0)
        target_width = kwargs.get("target_width", width)
        target_height = kwargs.get("target_height", height)

        out = []
        out.append(self.embedder(torch.Tensor([height])))
        out.append(self.embedder(torch.Tensor([width])))
        out.append(self.embedder(torch.Tensor([crop_h])))
        out.append(self.embedder(torch.Tensor([crop_w])))
        out.append(self.embedder(torch.Tensor([target_height])))
        out.append(self.embedder(torch.Tensor([target_width])))
        flat = torch.flatten(torch.cat(out)).unsqueeze(
            dim=0).repeat(clip_pooled.shape[0], 1)
        return torch.cat((clip_pooled.to(flat.device), flat), dim=1)

class Kolors(comfy.supported_models.SDXL):
    unet_config = {
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "transformer_depth": [0, 0, 2, 2, 10, 10],
        "context_dim": 2048,
        "adm_in_channels": 5632,
        "use_temporal_attention": False,
    }

    def get_model(self, state_dict, prefix="", device=None):
        out = KolorsSDXL(self, model_type=self.model_type(state_dict, prefix), device=device, )
        out.__class__ = model_base.SDXL
        if self.inpaint_model():
            out.set_inpaint()
        return out

def kolors_unet_config_from_diffusers_unet(state_dict, dtype=None):
    match = {}
    transformer_depth = []

    attn_res = 1
    count_blocks = model_detection.count_blocks
    down_blocks = count_blocks(state_dict, "down_blocks.{}")
    for i in range(down_blocks):
        attn_blocks = count_blocks(
            state_dict, "down_blocks.{}.attentions.".format(i) + '{}')
        res_blocks = count_blocks(
            state_dict, "down_blocks.{}.resnets.".format(i) + '{}')
        for ab in range(attn_blocks):
            transformer_count = count_blocks(
                state_dict, "down_blocks.{}.attentions.{}.transformer_blocks.".format(i, ab) + '{}')
            transformer_depth.append(transformer_count)
            if transformer_count > 0:
                match["context_dim"] = state_dict["down_blocks.{}.attentions.{}.transformer_blocks.0.attn2.to_k.weight".format(
                    i, ab)].shape[1]

        attn_res *= 2
        if attn_blocks == 0:
            for i in range(res_blocks):
                transformer_depth.append(0)

    match["transformer_depth"] = transformer_depth

    match["model_channels"] = state_dict["conv_in.weight"].shape[0]
    match["in_channels"] = state_dict["conv_in.weight"].shape[1]
    match["adm_in_channels"] = None
    if "class_embedding.linear_1.weight" in state_dict:
        match["adm_in_channels"] = state_dict["class_embedding.linear_1.weight"].shape[1]
    elif "add_embedding.linear_1.weight" in state_dict:
        match["adm_in_channels"] = state_dict["add_embedding.linear_1.weight"].shape[1]

    Kolors = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
              'num_classes': 'sequential', 'adm_in_channels': 5632, 'dtype': dtype, 'in_channels': 4, 'model_channels': 320,
              'num_res_blocks': [2, 2, 2], 'transformer_depth': [0, 0, 2, 2, 10, 10], 'channel_mult': [1, 2, 4], 'transformer_depth_middle': 10,
              'use_linear_in_transformer': True, 'context_dim': 2048, 'num_head_channels': 64, 'transformer_depth_output': [0, 0, 0, 2, 2, 2, 10, 10, 10],
              'use_temporal_attention': False, 'use_temporal_resblock': False}

    Kolors_inpaint = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True,
                      'legacy': False,
                      'num_classes': 'sequential', 'adm_in_channels': 5632, 'dtype': dtype, 'in_channels': 9,
                      'model_channels': 320,
                      'num_res_blocks': [2, 2, 2], 'transformer_depth': [0, 0, 2, 2, 10, 10], 'channel_mult': [1, 2, 4],
                      'transformer_depth_middle': 10,
                      'use_linear_in_transformer': True, 'context_dim': 2048, 'num_head_channels': 64,
                      'transformer_depth_output': [0, 0, 0, 2, 2, 2, 10, 10, 10],
                      'use_temporal_attention': False, 'use_temporal_resblock': False}

    Kolors_ip2p = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True,
                   'legacy': False,
                   'num_classes': 'sequential', 'adm_in_channels': 5632, 'dtype': dtype, 'in_channels': 8,
                   'model_channels': 320,
                   'num_res_blocks': [2, 2, 2], 'transformer_depth': [0, 0, 2, 2, 10, 10], 'channel_mult': [1, 2, 4],
                   'transformer_depth_middle': 10,
                   'use_linear_in_transformer': True, 'context_dim': 2048, 'num_head_channels': 64,
                   'transformer_depth_output': [0, 0, 0, 2, 2, 2, 10, 10, 10],
                   'use_temporal_attention': False, 'use_temporal_resblock': False}

    SDXL = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True,
            'legacy': False,
            'num_classes': 'sequential', 'adm_in_channels': 2816, 'dtype': dtype, 'in_channels': 4,
            'model_channels': 320,
            'num_res_blocks': [2, 2, 2], 'transformer_depth': [0, 0, 2, 2, 10, 10], 'channel_mult': [1, 2, 4],
            'transformer_depth_middle': 10,
            'use_linear_in_transformer': True, 'context_dim': 2048, 'num_head_channels': 64,
            'transformer_depth_output': [0, 0, 0, 2, 2, 2, 10, 10, 10],
            'use_temporal_attention': False, 'use_temporal_resblock': False}

    SDXL_mid_cnet = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True,
                     'legacy': False,
                     'num_classes': 'sequential', 'adm_in_channels': 2816, 'dtype': dtype, 'in_channels': 4,
                     'model_channels': 320,
                     'num_res_blocks': [2, 2, 2], 'transformer_depth': [0, 0, 0, 0, 1, 1], 'channel_mult': [1, 2, 4],
                     'transformer_depth_middle': 1,
                     'use_linear_in_transformer': True, 'context_dim': 2048, 'num_head_channels': 64,
                     'transformer_depth_output': [0, 0, 0, 0, 0, 0, 1, 1, 1],
                     'use_temporal_attention': False, 'use_temporal_resblock': False}

    SDXL_small_cnet = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True,
                       'legacy': False,
                       'num_classes': 'sequential', 'adm_in_channels': 2816, 'dtype': dtype, 'in_channels': 4,
                       'model_channels': 320,
                       'num_res_blocks': [2, 2, 2], 'transformer_depth': [0, 0, 0, 0, 0, 0], 'channel_mult': [1, 2, 4],
                       'transformer_depth_middle': 0,
                       'use_linear_in_transformer': True, 'num_head_channels': 64, 'context_dim': 1,
                       'transformer_depth_output': [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       'use_temporal_attention': False, 'use_temporal_resblock': False}

    supported_models = [Kolors, Kolors_inpaint,
                        Kolors_ip2p, SDXL, SDXL_mid_cnet, SDXL_small_cnet]


    for unet_config in supported_models:
        matches = True
        for k in match:
            if match[k] != unet_config[k]:
                # print("key {} does not match".format(k), match[k], "||", unet_config[k])
                matches = False
                break
        if matches:
            return model_detection.convert_config(unet_config)
    return None

# chatglm3 model
class chatGLM3Model(torch.nn.Module):
    def __init__(self, textmodel_json_config=None, device='cpu', offload_device='cpu', model_path=None):
        super().__init__()
        if model_path is None:
            raise ValueError("model_path is required")
        self.device = device
        if textmodel_json_config is None:
            textmodel_json_config = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "chatglm",
                "config_chatglm.json"
            )
        with open(textmodel_json_config, 'r') as file:
            config = json.load(file)
        textmodel_json_config = ChatGLMConfig(**config)
        is_accelerate_available = False
        try:
            from accelerate import init_empty_weights
            from accelerate.utils import set_module_tensor_to_device
            is_accelerate_available = True
        except:
            pass

        from contextlib import nullcontext
        with (init_empty_weights() if is_accelerate_available else nullcontext()):
            with torch.no_grad():
                print('torch version:', torch.__version__)
                self.text_encoder = ChatGLMModel(textmodel_json_config).eval()
                if '4bit' in model_path:
                    try:
                        import cpm_kernels
                    except ImportError:
                        print("Installing cpm_kernels...")
                        subprocess.run([sys.executable, "-m", "pip", "install", "cpm_kernels"], check=True)
                        pass
                    self.text_encoder.quantize(4)
                elif '8bit' in model_path:
                    self.text_encoder.quantize(8)

        sd = load_torch_file(model_path)
        if is_accelerate_available:
            for key in sd:
                set_module_tensor_to_device(self.text_encoder, key, device=offload_device, value=sd[key])
        else:
            print("WARNING: Accelerate not available, use load_state_dict load model")
            self.text_encoder.load_state_dict()

def load_chatglm3(model_path=None):
    if model_path is None:
        return

    load_device = comfy.model_management.text_encoder_device()
    offload_device = comfy.model_management.text_encoder_offload_device()

    glm3model = chatGLM3Model(
        device=load_device,
        offload_device=offload_device,
        model_path=model_path
    )
    tokenizer_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'chatglm', "tokenizer")
    tokenizer = ChatGLMTokenizer.from_pretrained(tokenizer_path)
    text_encoder = glm3model.text_encoder
    return {"text_encoder":text_encoder, "tokenizer":tokenizer}


# clipvision model
def load_clipvision_vitl_336(path):
    sd = load_torch_file(path)
    if "vision_model.encoder.layers.22.layer_norm1.weight" in sd:
        json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "clip_vision_config_vitl_336.json")
    else:
        raise Exception("Unsupported clip vision model")
    clip = ClipVisionModel(json_config)
    m, u = clip.load_sd(sd)
    if len(m) > 0:
        print("missing clip vision: {}".format(m))
    u = set(u)
    keys = list(sd.keys())
    for k in keys:
        if k not in u:
            t = sd.pop(k)
            del t
    return clip

class applyKolorsUnet:
    def __enter__(self):
        import comfy.ldm.modules.diffusionmodules.openaimodel
        import comfy.utils
        import comfy.clip_vision

        self.original_UNET_MAP_BASIC = comfy.utils.UNET_MAP_BASIC.copy()
        comfy.utils.UNET_MAP_BASIC.add(("encoder_hid_proj.weight", "encoder_hid_proj.weight"),)
        comfy.utils.UNET_MAP_BASIC.add(("encoder_hid_proj.bias", "encoder_hid_proj.bias"),)

        self.original_unet_config_from_diffusers_unet = model_detection.unet_config_from_diffusers_unet
        model_detection.unet_config_from_diffusers_unet = kolors_unet_config_from_diffusers_unet

        import comfy.supported_models
        self.original_supported_models = comfy.supported_models.models
        comfy.supported_models.models = [Kolors]

        self.original_load_clipvision_from_sd = comfy.clip_vision.load_clipvision_from_sd
        comfy.clip_vision.load_clipvision_from_sd = load_clipvision_vitl_336

    def __exit__(self, type, value, traceback):
        import comfy.ldm.modules.diffusionmodules.openaimodel
        import comfy.utils
        import comfy.supported_models
        import comfy.clip_vision

        comfy.utils.UNET_MAP_BASIC = self.original_UNET_MAP_BASIC

        model_detection.unet_config_from_diffusers_unet = self.original_unet_config_from_diffusers_unet
        comfy.supported_models.models = self.original_supported_models

        comfy.clip_vision.load_clipvision_from_sd = self.original_load_clipvision_from_sd


def is_kolors_model(model):
    unet_config = model.model.model_config.unet_config if hasattr(model, 'model') else None
    if unet_config and "adm_in_channels" in unet_config and unet_config["adm_in_channels"] == 5632:
        return True
    else:
        return False