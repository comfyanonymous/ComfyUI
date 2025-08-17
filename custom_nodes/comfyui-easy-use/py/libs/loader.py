import re, time, os, psutil
import folder_paths
import comfy.utils
import comfy.sd
import comfy.controlnet

from comfy.model_patcher import ModelPatcher
from nodes import NODE_CLASS_MAPPINGS
from collections import defaultdict
from .log import log_node_info, log_node_error
from ..modules.dit.pixArt.loader import load_pixart

diffusion_loaders = ["easy fullLoader", "easy a1111Loader", "easy fluxLoader", "easy comfyLoader", "easy hunyuanDiTLoader", "easy zero123Loader", "easy svdLoader"]
stable_cascade_loaders = ["easy cascadeLoader"]
dit_loaders = ['easy pixArtLoader']
controlnet_loaders = ["easy controlnetLoader", "easy controlnetLoaderADV", "easy controlnetLoader++"]
instant_loaders = ["easy instantIDApply", "easy instantIDApplyADV"]
cascade_vae_node = ["easy preSamplingCascade", "easy fullCascadeKSampler"]
model_merge_node = ["easy XYInputs: ModelMergeBlocks"]
lora_widget = ["easy fullLoader", "easy a1111Loader", "easy comfyLoader", "easy fluxLoader"]

class easyLoader:
    def __init__(self):
        self.loaded_objects = {
            "ckpt": defaultdict(tuple),  # {ckpt_name: (model, ...)}
            "unet": defaultdict(tuple),
            "clip": defaultdict(tuple),
            "clip_vision": defaultdict(tuple),
            "bvae": defaultdict(tuple),
            "vae": defaultdict(object),
            "lora": defaultdict(dict),  # {lora_name: {UID: (model_lora, clip_lora)}}
            "controlnet": defaultdict(dict),
            "t5": defaultdict(tuple),
            "chatglm3": defaultdict(tuple),
        }
        self.memory_threshold = self.determine_memory_threshold(1)
        self.lora_name_cache = []

    def clean_values(self, values: str):
        original_values = values.split("; ")
        cleaned_values = []

        for value in original_values:
            cleaned_value = value.strip(';').strip()
            if cleaned_value == "":
                continue
            try:
                cleaned_value = int(cleaned_value)
            except ValueError:
                try:
                    cleaned_value = float(cleaned_value)
                except ValueError:
                    pass
            cleaned_values.append(cleaned_value)

        return cleaned_values

    def clear_unused_objects(self, desired_names: set, object_type: str):
        keys = set(self.loaded_objects[object_type].keys())
        for key in keys - desired_names:
            del self.loaded_objects[object_type][key]

    def get_input_value(self, entry, key, prompt=None):
        val = entry["inputs"][key]
        if isinstance(val, str):
            return val
        elif isinstance(val, list):
            if prompt is not None and val[0]:
                return prompt[val[0]]['inputs'][key]
            else:
                return val[0]
        else:
            return str(val)

    def process_pipe_loader(self, entry, desired_ckpt_names, desired_vae_names, desired_lora_names, desired_lora_settings, num_loras=3, suffix=""):
        for idx in range(1, num_loras + 1):
            lora_name_key = f"{suffix}lora{idx}_name"
            desired_lora_names.add(self.get_input_value(entry, lora_name_key))
            setting = f'{self.get_input_value(entry, lora_name_key)};{entry["inputs"][f"{suffix}lora{idx}_model_strength"]};{entry["inputs"][f"{suffix}lora{idx}_clip_strength"]}'
            desired_lora_settings.add(setting)

        desired_ckpt_names.add(self.get_input_value(entry, f"{suffix}ckpt_name"))
        desired_vae_names.add(self.get_input_value(entry, f"{suffix}vae_name"))

    def update_loaded_objects(self, prompt):
        desired_ckpt_names = set()
        desired_unet_names = set()
        desired_clip_names = set()
        desired_vae_names = set()
        desired_lora_names = set()
        desired_lora_settings = set()
        desired_controlnet_names = set()
        desired_t5_names = set()
        desired_glm3_names = set()

        for entry in prompt.values():
            class_type = entry["class_type"]
            if class_type in lora_widget:
                lora_name = self.get_input_value(entry, "lora_name")
                desired_lora_names.add(lora_name)
                setting = f'{lora_name};{entry["inputs"]["lora_model_strength"]};{entry["inputs"]["lora_clip_strength"]}'
                desired_lora_settings.add(setting)

            if class_type in diffusion_loaders:
                desired_ckpt_names.add(self.get_input_value(entry, "ckpt_name", prompt))
                desired_vae_names.add(self.get_input_value(entry, "vae_name"))

            elif class_type in ['easy kolorsLoader']:
                desired_unet_names.add(self.get_input_value(entry, "unet_name"))
                desired_vae_names.add(self.get_input_value(entry, "vae_name"))
                desired_glm3_names.add(self.get_input_value(entry, "chatglm3_name"))

            elif class_type in dit_loaders:
                t5_name = self.get_input_value(entry, "mt5_name") if "mt5_name" in entry["inputs"] else  None
                clip_name = self.get_input_value(entry, "clip_name") if "clip_name" in entry["inputs"] else None
                model_name = self.get_input_value(entry, "model_name")
                ckpt_name = self.get_input_value(entry, "ckpt_name", prompt)
                if t5_name:
                    desired_t5_names.add(t5_name)
                if clip_name:
                    desired_clip_names.add(clip_name)
                desired_ckpt_names.add(ckpt_name+'_'+model_name)

            elif class_type in stable_cascade_loaders:
                desired_unet_names.add(self.get_input_value(entry, "stage_c"))
                desired_unet_names.add(self.get_input_value(entry, "stage_b"))
                desired_clip_names.add(self.get_input_value(entry, "clip_name"))
                desired_vae_names.add(self.get_input_value(entry, "stage_a"))

            elif class_type in cascade_vae_node:
                encode_vae_name = self.get_input_value(entry, "encode_vae_name")
                decode_vae_name = self.get_input_value(entry, "decode_vae_name")
                if encode_vae_name and encode_vae_name != 'None':
                    desired_vae_names.add(encode_vae_name)
                if decode_vae_name and decode_vae_name != 'None':
                    desired_vae_names.add(decode_vae_name)

            elif class_type in controlnet_loaders:
                control_net_name = self.get_input_value(entry, "control_net_name", prompt)
                scale_soft_weights = self.get_input_value(entry, "scale_soft_weights")
                desired_controlnet_names.add(f'{control_net_name};{scale_soft_weights}')

            elif class_type in instant_loaders:
                control_net_name = self.get_input_value(entry, "control_net_name", prompt)
                scale_soft_weights = self.get_input_value(entry, "cn_soft_weights")
                desired_controlnet_names.add(f'{control_net_name};{scale_soft_weights}')

            elif class_type in model_merge_node:
                desired_ckpt_names.add(self.get_input_value(entry, "ckpt_name_1"))
                desired_ckpt_names.add(self.get_input_value(entry, "ckpt_name_2"))
                vae_use = self.get_input_value(entry, "vae_use")
                if vae_use != 'Use Model 1' and vae_use != 'Use Model 2':
                    desired_vae_names.add(vae_use)

        object_types = ["ckpt", "unet", "clip", "bvae", "vae", "lora", "controlnet", "t5"]
        for object_type in object_types:
            if object_type == 'unet':
                desired_names = desired_unet_names
            elif object_type in ["ckpt", "clip", "bvae"]:
                if object_type == 'clip':
                    desired_names = desired_ckpt_names.union(desired_clip_names)
                else:
                    desired_names = desired_ckpt_names
            elif object_type == "vae":
                desired_names = desired_vae_names
            elif object_type == "controlnet":
                desired_names = desired_controlnet_names
            elif object_type == "t5":
                desired_names = desired_t5_names
            elif object_type == "chatglm3":
                desired_names = desired_glm3_names
            else:
                desired_names = desired_lora_names
            self.clear_unused_objects(desired_names, object_type)

    def add_to_cache(self, obj_type, key, value):
        """
        Add an item to the cache with the current timestamp.
        """
        timestamped_value = (value, time.time())
        self.loaded_objects[obj_type][key] = timestamped_value

    def determine_memory_threshold(self, percentage=0.8):
        """
        Determines the memory threshold as a percentage of the total available memory.
        Args:
        - percentage (float): The fraction of total memory to use as the threshold.
                              Should be a value between 0 and 1. Default is 0.8 (80%).
        Returns:
        - memory_threshold (int): Memory threshold in bytes.
        """
        total_memory = psutil.virtual_memory().total
        memory_threshold = total_memory * percentage
        return memory_threshold

    def get_memory_usage(self):
        """
        Returns the memory usage of the current process in bytes.
        """
        process = psutil.Process(os.getpid())
        return process.memory_info().rss

    def eviction_based_on_memory(self):
        """
        Evicts objects from cache based on memory usage and priority.
        """
        current_memory = self.get_memory_usage()
        if current_memory < self.memory_threshold:
            return
        eviction_order = ["vae", "lora", "bvae", "clip", "ckpt", "controlnet", "unet", "t5", "chatglm3"]
        for obj_type in eviction_order:
            if current_memory < self.memory_threshold:
                break
            # Sort items based on age (using the timestamp)
            items = list(self.loaded_objects[obj_type].items())
            items.sort(key=lambda x: x[1][1])  # Sorting by timestamp

            for item in items:
                if current_memory < self.memory_threshold:
                    break
                del self.loaded_objects[obj_type][item[0]]
                current_memory = self.get_memory_usage()

    def load_checkpoint(self, ckpt_name, config_name=None, load_vision=False):
        cache_name = ckpt_name
        if config_name not in [None, "Default"]:
            cache_name = ckpt_name + "_" + config_name
        if cache_name in self.loaded_objects["ckpt"]:
            clip_vision = self.loaded_objects["clip_vision"][cache_name][0] if load_vision else None
            clip = self.loaded_objects["clip"][cache_name][0] if not load_vision else None
            return self.loaded_objects["ckpt"][cache_name][0], clip, self.loaded_objects["bvae"][cache_name][0], clip_vision

        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)

        output_clip = False if load_vision else True
        output_clipvision = True if load_vision else False
        if config_name not in [None, "Default"]:
            config_path = folder_paths.get_full_path("configs", config_name)
            loaded_ckpt = comfy.sd.load_checkpoint(config_path, ckpt_path, output_vae=True, output_clip=output_clip, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        else:
            model_options = {}
            if re.search("nf4", ckpt_name):
                from ..modules.bitsandbytes_NF4 import OPS
                model_options = {"custom_operations": OPS}
            loaded_ckpt = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=output_clip, output_clipvision=output_clipvision, embedding_directory=folder_paths.get_folder_paths("embeddings"), model_options=model_options)

        self.add_to_cache("ckpt", cache_name, loaded_ckpt[0])
        self.add_to_cache("bvae", cache_name, loaded_ckpt[2])

        clip = loaded_ckpt[1]
        clip_vision = loaded_ckpt[3]
        if clip:
            self.add_to_cache("clip", cache_name, clip)
        if clip_vision:
            self.add_to_cache("clip_vision", cache_name, clip_vision)

        self.eviction_based_on_memory()

        return loaded_ckpt[0], clip, loaded_ckpt[2], clip_vision

    def load_vae(self, vae_name):
        if vae_name in self.loaded_objects["vae"]:
            return self.loaded_objects["vae"][vae_name][0]

        vae_path = folder_paths.get_full_path("vae", vae_name)
        sd = comfy.utils.load_torch_file(vae_path)
        loaded_vae = comfy.sd.VAE(sd=sd)
        self.add_to_cache("vae", vae_name, loaded_vae)
        self.eviction_based_on_memory()

        return loaded_vae

    def load_unet(self, unet_name):
        if unet_name in self.loaded_objects["unet"]:
            log_node_info("Load UNet", f"{unet_name} cached")
            return self.loaded_objects["unet"][unet_name][0]

        unet_path = folder_paths.get_full_path("unet", unet_name)
        model = comfy.sd.load_unet(unet_path)
        self.add_to_cache("unet", unet_name, model)
        self.eviction_based_on_memory()

        return model

    def load_controlnet(self, control_net_name, scale_soft_weights=1, use_cache=True):
        unique_id = f'{control_net_name};{str(scale_soft_weights)}'
        if use_cache and unique_id in self.loaded_objects["controlnet"]:
            return self.loaded_objects["controlnet"][unique_id][0]
        if scale_soft_weights < 1:
            if "ScaledSoftControlNetWeights" in NODE_CLASS_MAPPINGS:
                soft_weight_cls = NODE_CLASS_MAPPINGS['ScaledSoftControlNetWeights']
                (weights, timestep_keyframe) = soft_weight_cls().load_weights(scale_soft_weights, False)
                cn_adv_cls = NODE_CLASS_MAPPINGS['ControlNetLoaderAdvanced']
                control_net, = cn_adv_cls().load_controlnet(control_net_name, timestep_keyframe)
            else:
                raise Exception(f"[Advanced-ControlNet Not Found] you need to install 'COMFYUI-Advanced-ControlNet'")
        else:
            controlnet_path = folder_paths.get_full_path("controlnet", control_net_name)
            control_net = comfy.controlnet.load_controlnet(controlnet_path)
        if use_cache:
            self.add_to_cache("controlnet", unique_id, control_net)
            self.eviction_based_on_memory()

        return control_net
    def load_clip(self, clip_name, type='stable_diffusion', load_clip=None):
        if clip_name in self.loaded_objects["clip"]:
            return self.loaded_objects["clip"][clip_name][0]

        if type == 'stable_diffusion':
            clip_type = comfy.sd.CLIPType.STABLE_DIFFUSION
        elif type == 'stable_cascade':
            clip_type = comfy.sd.CLIPType.STABLE_CASCADE
        elif type == 'sd3':
            clip_type = comfy.sd.CLIPType.SD3
        elif type == 'flux':
            clip_type = comfy.sd.CLIPType.FLUX
        elif type == 'stable_audio':
            clip_type = comfy.sd.CLIPType.STABLE_AUDIO
        clip_path = folder_paths.get_full_path("clip", clip_name)
        load_clip = comfy.sd.load_clip(ckpt_paths=[clip_path], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=clip_type)
        self.add_to_cache("clip", clip_name, load_clip)
        self.eviction_based_on_memory()

        return load_clip

    def load_lora(self, lora, model=None, clip=None, type=None , use_cache=True):
        lora_name = lora["lora_name"]
        model = model if model is not None else lora["model"]
        clip = clip if clip is not None else lora["clip"]
        model_strength = lora["model_strength"]
        clip_strength = lora["clip_strength"]
        lbw = lora["lbw"] if "lbw" in lora else None
        lbw_a = lora["lbw_a"] if "lbw_a" in lora else None
        lbw_b = lora["lbw_b"] if "lbw_b" in lora else None

        model_hash = str(model)[44:-1]
        clip_hash = str(clip)[25:-1] if clip else ''

        unique_id = f'{model_hash};{clip_hash};{lora_name};{model_strength};{clip_strength}'

        if use_cache and unique_id in self.loaded_objects["lora"]:
            log_node_info("Load LORA",f"{lora_name} cached")
            return self.loaded_objects["lora"][unique_id][0]

        orig_lora_name = lora_name
        lora_name = self.resolve_lora_name(lora_name)

        if lora_name is not None:
            lora_path = folder_paths.get_full_path("loras", lora_name)
        else:
            lora_path = None

        if lora_path is not None:
            log_node_info("Load LORA",f"{lora_name}: model={model_strength:.3f}, clip={clip_strength:.3f}, LBW={lbw}, A={lbw_a}, B={lbw_b}")
            if lbw:
                lbw = lora["lbw"]
                lbw_a = lora["lbw_a"]
                lbw_b = lora["lbw_b"]
                if 'LoraLoaderBlockWeight //Inspire' not in NODE_CLASS_MAPPINGS:
                    raise Exception('[InspirePack Not Found] you need to install ComfyUI-Inspire-Pack')
                cls = NODE_CLASS_MAPPINGS['LoraLoaderBlockWeight //Inspire']
                model, clip, _ = cls().doit(model, clip, lora_name, model_strength, clip_strength, False, 0,
                                            lbw_a, lbw_b, "", lbw)
            else:
                _lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                keys = _lora.keys()
                if "down_blocks.0.resnets.0.norm1.bias" in keys:
                    print('Using LORA for Resadapter')
                    key_map = {}
                    key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
                    mapping_norm = {}

                    for key in keys:
                        if ".weight" in key:
                            key_name_in_ori_sd = key_map[key.replace(".weight", "")]
                            mapping_norm[key_name_in_ori_sd] = _lora[key]
                        elif ".bias" in key:
                            key_name_in_ori_sd = key_map[key.replace(".bias", "")]
                            mapping_norm[key_name_in_ori_sd.replace(".weight", ".bias")] = _lora[
                                key
                            ]
                        else:
                            print("===>Unexpected key", key)
                            mapping_norm[key] = _lora[key]

                    for k in mapping_norm.keys():
                        if k not in model.model.state_dict():
                            print("===>Missing key:", k)
                    model.model.load_state_dict(mapping_norm, strict=False)
                    return (model, clip)

                # PixArt
                if type is not None and type == 'PixArt':
                    from ..modules.dit.pixArt.loader import load_pixart_lora
                    model = load_pixart_lora(model, _lora, lora_path, model_strength)
                else:
                    model, clip = comfy.sd.load_lora_for_models(model, clip, _lora, model_strength, clip_strength)

            if use_cache:
                self.add_to_cache("lora", unique_id, (model, clip))
                self.eviction_based_on_memory()
        else:
            log_node_error(f"LORA NOT FOUND", orig_lora_name)

        return model, clip

    def resolve_lora_name(self, name):
        if os.path.exists(name):
            return name
        else:
            if len(self.lora_name_cache) == 0:
                loras = folder_paths.get_filename_list("loras")
                self.lora_name_cache.extend(loras)
            for x in self.lora_name_cache:
                if x.endswith(name):
                    return x

            # 如果刷新网页后新添加的lora走这个逻辑
            log_node_info("LORA NOT IN CACHE", f"{name}")
            loras = folder_paths.get_filename_list("loras")
            for x in loras:
                if x.endswith(name):
                    self.lora_name_cache.append(x)
                    return x

            return None

    def load_main(self, ckpt_name, config_name, vae_name, lora_name, lora_model_strength, lora_clip_strength, optional_lora_stack, model_override, clip_override, vae_override, prompt, nf4=False):
        model: ModelPatcher | None = None
        clip: comfy.sd.CLIP | None = None
        vae: comfy.sd.VAE | None = None
        clip_vision = None
        lora_stack = []

        # Check for model override
        can_load_lora = True
        # 判断是否存在 模型或Lora叠加xyplot, 若存在优先缓存第一个模型
        # Determine whether there is a model or Lora overlapping xyplot, and if there is, prioritize caching the first model.
        xy_model_id = next((x for x in prompt if str(prompt[x]["class_type"]) in ["easy XYInputs: ModelMergeBlocks",
                                                                                  "easy XYInputs: Checkpoint"]), None)
        # This will find nodes that aren't actively connected to anything, and skip loading lora's for them.
        xy_lora_id = next((x for x in prompt if str(prompt[x]["class_type"]) == "easy XYInputs: Lora"), None)
        if xy_lora_id is not None:
            can_load_lora = False
        if xy_model_id is not None:
            node = prompt[xy_model_id]
            if "ckpt_name_1" in node["inputs"]:
                ckpt_name_1 = node["inputs"]["ckpt_name_1"]
                model, clip, vae, clip_vision = self.load_checkpoint(ckpt_name_1)
                can_load_lora = False
        elif model_override is not None and clip_override is not None and vae_override is not None:
            model = model_override
            clip = clip_override
            vae = vae_override
        else:
            model, clip, vae, clip_vision = self.load_checkpoint(ckpt_name, config_name)
            if model_override is not None:
                model = model_override
            if vae_override is not None:
                vae = vae_override
            elif clip_override is not None:
                clip = clip_override


        if optional_lora_stack is not None and can_load_lora:
            for lora in optional_lora_stack:
                # This is a subtle bit of code because it uses the model created by the last call, and passes it to the next call.
                lora = {"lora_name": lora[0], "model": model, "clip": clip, "model_strength": lora[1],
                        "clip_strength": lora[2]}
                model, clip = self.load_lora(lora)
                lora['model'] = model
                lora['clip'] = clip
                lora_stack.append(lora)

        if lora_name != "None" and can_load_lora:
            lora = {"lora_name": lora_name, "model": model, "clip": clip, "model_strength": lora_model_strength,
                    "clip_strength": lora_clip_strength}
            model, clip = self.load_lora(lora)
            lora_stack.append(lora)

        # Check for custom VAE
        if vae_name not in ["Baked VAE", "Baked-VAE"]:
            vae = self.load_vae(vae_name)
        # CLIP skip
        if not clip:
            raise Exception("No CLIP found")

        return model, clip, vae, clip_vision, lora_stack

    # Kolors
    def load_kolors_unet(self, unet_name):
        if unet_name in self.loaded_objects["unet"]:
            log_node_info("Load Kolors UNet", f"{unet_name} cached")
            return self.loaded_objects["unet"][unet_name][0]
        else:
            from ..modules.kolors.loader import applyKolorsUnet
            with applyKolorsUnet():
                unet_path = folder_paths.get_full_path("unet", unet_name)
                sd = comfy.utils.load_torch_file(unet_path)
                model = comfy.sd.load_unet_state_dict(sd)
                if model is None:
                    raise RuntimeError("ERROR: Could not detect model type of: {}".format(unet_path))

                self.add_to_cache("unet", unet_name, model)
                self.eviction_based_on_memory()

                return model

    def load_chatglm3(self, chatglm3_name):
        from ..modules.kolors.loader import load_chatglm3
        if chatglm3_name in self.loaded_objects["chatglm3"]:
            log_node_info("Load ChatGLM3", f"{chatglm3_name} cached")
            return self.loaded_objects["chatglm3"][chatglm3_name][0]

        chatglm_model = load_chatglm3(model_path=folder_paths.get_full_path("llm", chatglm3_name))
        self.add_to_cache("chatglm3", chatglm3_name, chatglm_model)
        self.eviction_based_on_memory()

        return chatglm_model


    # DiT
    def load_dit_ckpt(self, ckpt_name, model_name, **kwargs):
        if (ckpt_name+'_'+model_name) in self.loaded_objects["ckpt"]:
            return self.loaded_objects["ckpt"][ckpt_name+'_'+model_name][0]
        model = None
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        model_type = kwargs['model_type'] if "model_type" in kwargs else 'PixArt'
        if model_type == 'PixArt':
            pixart_conf = kwargs['pixart_conf']
            model_conf = pixart_conf[model_name]
            model = load_pixart(ckpt_path, model_conf)
        if model:
            self.add_to_cache("ckpt", ckpt_name + '_' + model_name, model)
            self.eviction_based_on_memory()
        return model

    def load_t5_from_sd3_clip(self, sd3_clip, padding):
        try:
            from comfy.text_encoders.sd3_clip import SD3Tokenizer, SD3ClipModel
        except:
            from comfy.sd3_clip import SD3Tokenizer, SD3ClipModel
        import copy

        clip = sd3_clip.clone()
        assert clip.cond_stage_model.t5xxl is not None, "CLIP must have T5 loaded!"

        # remove transformer
        transformer = clip.cond_stage_model.t5xxl.transformer
        clip.cond_stage_model.t5xxl.transformer = None

        # clone object
        tmp = SD3ClipModel(clip_l=False, clip_g=False, t5=False)
        tmp.t5xxl = copy.deepcopy(clip.cond_stage_model.t5xxl)
        # put transformer back
        clip.cond_stage_model.t5xxl.transformer = transformer
        tmp.t5xxl.transformer = transformer

        # override special tokens
        tmp.t5xxl.special_tokens = copy.deepcopy(clip.cond_stage_model.t5xxl.special_tokens)
        tmp.t5xxl.special_tokens.pop("end")  # make sure empty tokens match

        # tokenizer
        tok = SD3Tokenizer()
        tok.t5xxl.min_length = padding

        clip.cond_stage_model = tmp
        clip.tokenizer = tok

        return clip
