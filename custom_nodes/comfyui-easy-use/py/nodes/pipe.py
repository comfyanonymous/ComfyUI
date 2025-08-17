import os
import folder_paths
import comfy.samplers, comfy.supported_models

from nodes import LatentFromBatch, RepeatLatentBatch
from ..config import MAX_SEED_NUM

from ..libs.log import log_node_warn
from ..libs.utils import get_sd_version
from ..libs.conditioning import prompt_to_cond, set_cond

from .. import easyCache

# 节点束输入
class pipeIn:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
             "required": {},
             "optional": {
                "pipe": ("PIPE_LINE",),
                "model": ("MODEL",),
                "pos": ("CONDITIONING",),
                "neg": ("CONDITIONING",),
                "latent": ("LATENT",),
                "vae": ("VAE",),
                "clip": ("CLIP",),
                "image": ("IMAGE",),
                "xyPlot": ("XYPLOT",),
            },
            "hidden": {"my_unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "flush"

    CATEGORY = "EasyUse/Pipe"

    def flush(self, pipe=None, model=None, pos=None, neg=None, latent=None, vae=None, clip=None, image=None, xyplot=None, my_unique_id=None):

        model = model if model is not None else pipe.get("model")
        if model is None:
            log_node_warn(f'pipeIn[{my_unique_id}]', "Model missing from pipeLine")
        pos = pos if pos is not None else pipe.get("positive")
        if pos is None:
            log_node_warn(f'pipeIn[{my_unique_id}]', "Pos Conditioning missing from pipeLine")
        neg = neg if neg is not None else pipe.get("negative")
        if neg is None:
            log_node_warn(f'pipeIn[{my_unique_id}]', "Neg Conditioning missing from pipeLine")
        vae = vae if vae is not None else pipe.get("vae")
        if vae is None:
            log_node_warn(f'pipeIn[{my_unique_id}]', "VAE missing from pipeLine")
        clip = clip if clip is not None else pipe.get("clip") if pipe is not None and "clip" in pipe else None
        # if clip is None:
        #     log_node_warn(f'pipeIn[{my_unique_id}]', "Clip missing from pipeLine")
        if latent is not None:
            samples = latent
        elif image is None:
            samples = pipe.get("samples") if pipe is not None else None
            image = pipe.get("images") if pipe is not None else None
        elif image is not None:
            if pipe is None:
                batch_size = 1
            else:
                batch_size = pipe["loader_settings"]["batch_size"] if "batch_size" in pipe["loader_settings"] else 1
            samples = {"samples": vae.encode(image[:, :, :, :3])}
            samples = RepeatLatentBatch().repeat(samples, batch_size)[0]

        if pipe is None:
            pipe = {"loader_settings": {"positive": "", "negative": "", "xyplot": None}}

        xyplot = xyplot if xyplot is not None else pipe['loader_settings']['xyplot'] if xyplot in pipe['loader_settings'] else None

        new_pipe = {
            **pipe,
            "model": model,
            "positive": pos,
            "negative": neg,
            "vae": vae,
            "clip": clip,

            "samples": samples,
            "images": image,
            "seed": pipe.get('seed') if pipe is not None and "seed" in pipe else None,

            "loader_settings": {
                **pipe["loader_settings"],
                "xyplot": xyplot
            }
        }
        del pipe

        return (new_pipe,)

# 节点束输出
class pipeOut:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
             "required": {
                "pipe": ("PIPE_LINE",),
            },
            "hidden": {"my_unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "CLIP", "IMAGE", "INT",)
    RETURN_NAMES = ("pipe", "model", "pos", "neg", "latent", "vae", "clip", "image", "seed",)
    FUNCTION = "flush"

    CATEGORY = "EasyUse/Pipe"

    def flush(self, pipe, my_unique_id=None):
        model = pipe.get("model")
        pos = pipe.get("positive")
        neg = pipe.get("negative")
        latent = pipe.get("samples")
        vae = pipe.get("vae")
        clip = pipe.get("clip")
        image = pipe.get("images")
        seed = pipe.get("seed")

        return pipe, model, pos, neg, latent, vae, clip, image, seed

# 编辑节点束
class pipeEdit:
    @classmethod
    def INPUT_TYPES(s):
        return {
             "required": {
                 "clip_skip": ("INT", {"default": -1, "min": -24, "max": 0, "step": 1}),

                 "optional_positive": ("STRING", {"default": "", "multiline": True}),
                 "positive_token_normalization": (["none", "mean", "length", "length+mean"],),
                 "positive_weight_interpretation": (["comfy", "A1111", "comfy++", "compel", "fixed attention"],),

                 "optional_negative": ("STRING", {"default": "", "multiline": True}),
                 "negative_token_normalization": (["none", "mean", "length", "length+mean"],),
                 "negative_weight_interpretation": (["comfy", "A1111", "comfy++", "compel", "fixed attention"],),

                 "a1111_prompt_style": ("BOOLEAN", {"default": False}),
                 "conditioning_mode": (['replace', 'concat', 'combine', 'average', 'timestep'], {"default": "replace"}),
                 "average_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                 "old_cond_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                 "old_cond_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                 "new_cond_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                 "new_cond_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
             },
             "optional": {
                "pipe": ("PIPE_LINE",),
                "model": ("MODEL",),
                "pos": ("CONDITIONING",),
                "neg": ("CONDITIONING",),
                "latent": ("LATENT",),
                "vae": ("VAE",),
                "clip": ("CLIP",),
                "image": ("IMAGE",),
             },
            "hidden": {"my_unique_id": "UNIQUE_ID", "prompt":"PROMPT"},
        }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "CLIP", "IMAGE")
    RETURN_NAMES = ("pipe", "model", "pos", "neg", "latent", "vae", "clip", "image")
    FUNCTION = "edit"

    CATEGORY = "EasyUse/Pipe"

    def edit(self, clip_skip, optional_positive, positive_token_normalization, positive_weight_interpretation, optional_negative, negative_token_normalization, negative_weight_interpretation, a1111_prompt_style, conditioning_mode, average_strength, old_cond_start, old_cond_end, new_cond_start, new_cond_end, pipe=None, model=None, pos=None, neg=None, latent=None, vae=None, clip=None, image=None, my_unique_id=None, prompt=None):

        model = model if model is not None else pipe.get("model")
        if model is None:
            log_node_warn(f'pipeIn[{my_unique_id}]', "Model missing from pipeLine")
        vae = vae if vae is not None else pipe.get("vae")
        if vae is None:
            log_node_warn(f'pipeIn[{my_unique_id}]', "VAE missing from pipeLine")
        clip = clip if clip is not None else pipe.get("clip")
        if clip is None:
            log_node_warn(f'pipeIn[{my_unique_id}]', "Clip missing from pipeLine")
        if image is None:
            image = pipe.get("images") if pipe is not None else None
            samples = latent if latent is not None else pipe.get("samples")
            if samples is None:
                log_node_warn(f'pipeIn[{my_unique_id}]', "Latent missing from pipeLine")
        else:
            batch_size = pipe["loader_settings"]["batch_size"] if "batch_size" in pipe["loader_settings"] else 1
            samples = {"samples": vae.encode(image[:, :, :, :3])}
            samples = RepeatLatentBatch().repeat(samples, batch_size)[0]

        pipe_lora_stack = pipe.get("lora_stack") if pipe is not None and "lora_stack" in pipe else []

        steps = pipe["loader_settings"]["steps"] if "steps" in pipe["loader_settings"] else 1
        if pos is None and optional_positive != '':
            pos, positive_wildcard_prompt, model, clip = prompt_to_cond('positive', model, clip, clip_skip,
                                                                        pipe_lora_stack, optional_positive, positive_token_normalization,positive_weight_interpretation,
                                                                        a1111_prompt_style, my_unique_id, prompt, easyCache, True, steps)
            pos = set_cond(pipe['positive'], pos, conditioning_mode, average_strength, old_cond_start, old_cond_end, new_cond_start, new_cond_end)
            pipe['loader_settings']['positive'] = positive_wildcard_prompt
            pipe['loader_settings']['positive_token_normalization'] = positive_token_normalization
            pipe['loader_settings']['positive_weight_interpretation'] = positive_weight_interpretation
            if a1111_prompt_style:
                pipe['loader_settings']['a1111_prompt_style'] = True
        else:
            pos = pipe.get("positive")
            if pos is None:
                log_node_warn(f'pipeIn[{my_unique_id}]', "Pos Conditioning missing from pipeLine")

        if neg is None and optional_negative != '':
            neg, negative_wildcard_prompt, model, clip = prompt_to_cond("negative", model, clip, clip_skip, pipe_lora_stack, optional_negative,
                                                      negative_token_normalization, negative_weight_interpretation,
                                                      a1111_prompt_style, my_unique_id, prompt, easyCache, True, steps)
            neg = set_cond(pipe['negative'], neg, conditioning_mode, average_strength, old_cond_start, old_cond_end, new_cond_start, new_cond_end)
            pipe['loader_settings']['negative'] = negative_wildcard_prompt
            pipe['loader_settings']['negative_token_normalization'] = negative_token_normalization
            pipe['loader_settings']['negative_weight_interpretation'] = negative_weight_interpretation
            if a1111_prompt_style:
                pipe['loader_settings']['a1111_prompt_style'] = True
        else:
            neg = pipe.get("negative")
            if neg is None:
                log_node_warn(f'pipeIn[{my_unique_id}]', "Neg Conditioning missing from pipeLine")
        if pipe is None:
            pipe = {"loader_settings": {"positive": "", "negative": "", "xyplot": None}}

        new_pipe = {
            **pipe,
            "model": model,
            "positive": pos,
            "negative": neg,
            "vae": vae,
            "clip": clip,

            "samples": samples,
            "images": image,
            "seed": pipe.get('seed') if pipe is not None and "seed" in pipe else None,
            "loader_settings":{
                **pipe["loader_settings"]
            }
        }
        del pipe

        return (new_pipe, model,pos, neg, latent, vae, clip, image)

# 编辑节点束提示词
class pipeEditPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("PIPE_LINE",),
                "positive": ("STRING", {"default": "", "multiline": True}),
                "negative": ("STRING", {"default": "", "multiline": True}),
            },
            "hidden": {"my_unique_id": "UNIQUE_ID", "prompt": "PROMPT"},
        }

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "edit"

    CATEGORY = "EasyUse/Pipe"

    def edit(self, pipe, positive, negative, my_unique_id=None, prompt=None):
        model = pipe.get("model")
        if model is None:
            log_node_warn(f'pipeEdit[{my_unique_id}]', "Model missing from pipeLine")

        from ..modules.kolors.loader import is_kolors_model
        model_type = get_sd_version(model)
        if model_type == 'sdxl' and is_kolors_model(model):
            from ..modules.kolors.text_encode import chatglm3_adv_text_encode
            auto_clean_gpu = pipe["loader_settings"]["auto_clean_gpu"] if "auto_clean_gpu" in pipe["loader_settings"] else False
            chatglm3_model = pipe["chatglm3_model"] if "chatglm3_model" in pipe else None
            # text encode
            log_node_warn("Positive encoding...")
            positive_embeddings_final = chatglm3_adv_text_encode(chatglm3_model, positive, auto_clean_gpu)
            log_node_warn("Negative encoding...")
            negative_embeddings_final = chatglm3_adv_text_encode(chatglm3_model, negative, auto_clean_gpu)
        else:
            clip_skip = pipe["loader_settings"]["clip_skip"] if "clip_skip" in pipe["loader_settings"] else -1
            lora_stack = pipe.get("lora_stack") if pipe is not None and "lora_stack" in pipe else []
            clip = pipe.get("clip") if pipe is not None and "clip" in pipe else None
            positive_token_normalization = pipe["loader_settings"]["positive_token_normalization"] if "positive_token_normalization" in pipe["loader_settings"] else "none"
            positive_weight_interpretation = pipe["loader_settings"]["positive_weight_interpretation"] if "positive_weight_interpretation" in pipe["loader_settings"] else "comfy"
            negative_token_normalization = pipe["loader_settings"]["negative_token_normalization"] if "negative_token_normalization" in pipe["loader_settings"] else "none"
            negative_weight_interpretation = pipe["loader_settings"]["negative_weight_interpretation"] if "negative_weight_interpretation" in pipe["loader_settings"] else "comfy"
            a1111_prompt_style = pipe["loader_settings"]["a1111_prompt_style"] if "a1111_prompt_style" in pipe["loader_settings"] else False
            # Prompt to Conditioning
            positive_embeddings_final, positive_wildcard_prompt, model, clip = prompt_to_cond('positive', model, clip,
                                                                                              clip_skip, lora_stack,
                                                                                              positive,
                                                                                              positive_token_normalization,
                                                                                              positive_weight_interpretation,
                                                                                              a1111_prompt_style,
                                                                                              my_unique_id, prompt,
                                                                                              easyCache,
                                                                                              model_type=model_type)
            negative_embeddings_final, negative_wildcard_prompt, model, clip = prompt_to_cond('negative', model, clip,
                                                                                              clip_skip, lora_stack,
                                                                                              negative,
                                                                                              negative_token_normalization,
                                                                                              negative_weight_interpretation,
                                                                                              a1111_prompt_style,
                                                                                              my_unique_id, prompt,
                                                                                              easyCache,
                                                                                              model_type=model_type)
        new_pipe = {
            **pipe,
            "model": model,
            "positive": positive_embeddings_final,
            "negative": negative_embeddings_final,
        }
        del pipe

        return (new_pipe,)


# 节点束到基础节点束（pipe to ComfyUI-Impack-pack's basic_pipe）
class pipeToBasicPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("PIPE_LINE",),
            },
            "hidden": {"my_unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("BASIC_PIPE",)
    RETURN_NAMES = ("basic_pipe",)
    FUNCTION = "doit"

    CATEGORY = "EasyUse/Pipe"

    def doit(self, pipe, my_unique_id=None):
        new_pipe = (pipe.get('model'), pipe.get('clip'), pipe.get('vae'), pipe.get('positive'), pipe.get('negative'))
        del pipe
        return (new_pipe,)

# 批次索引
class pipeBatchIndex:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"pipe": ("PIPE_LINE",),
                             "batch_index": ("INT", {"default": 0, "min": 0, "max": 63}),
                             "length": ("INT", {"default": 1, "min": 1, "max": 64}),
                             },
                "hidden": {"my_unique_id": "UNIQUE_ID"},}

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "doit"

    CATEGORY = "EasyUse/Pipe"

    def doit(self, pipe, batch_index, length, my_unique_id=None):
        samples = pipe["samples"]
        new_samples, = LatentFromBatch().frombatch(samples, batch_index, length)
        new_pipe = {
            **pipe,
            "samples": new_samples
        }
        del pipe
        return (new_pipe,)

# pipeXYPlot
class pipeXYPlot:
    lora_list = ["None"] + folder_paths.get_filename_list("loras")
    lora_strengths = {"min": -4.0, "max": 4.0, "step": 0.01}
    token_normalization = ["none", "mean", "length", "length+mean"]
    weight_interpretation = ["comfy", "A1111", "compel", "comfy++"]

    loader_dict = {
        "ckpt_name": folder_paths.get_filename_list("checkpoints"),
        "vae_name": ["Baked-VAE"] + folder_paths.get_filename_list("vae"),
        "clip_skip": {"min": -24, "max": -1, "step": 1},
        "lora_name": lora_list,
        "lora_model_strength": lora_strengths,
        "lora_clip_strength": lora_strengths,
        "positive": [],
        "negative": [],
    }

    sampler_dict = {
        "steps": {"min": 1, "max": 100, "step": 1},
        "cfg": {"min": 0.0, "max": 100.0, "step": 1.0},
        "sampler_name": comfy.samplers.KSampler.SAMPLERS,
        "scheduler": comfy.samplers.KSampler.SCHEDULERS,
        "denoise": {"min": 0.0, "max": 1.0, "step": 0.01},
        "seed": {"min": 0, "max": MAX_SEED_NUM},
    }

    plot_dict = {**sampler_dict, **loader_dict}

    plot_values = ["None", ]
    plot_values.append("---------------------")
    for k in sampler_dict:
        plot_values.append(f'preSampling: {k}')
    plot_values.append("---------------------")
    for k in loader_dict:
        plot_values.append(f'loader: {k}')

    def __init__(self):
        pass

    rejected = ["None", "---------------------", "Nothing"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "grid_spacing": ("INT", {"min": 0, "max": 500, "step": 5, "default": 0, }),
                "output_individuals": (["False", "True"], {"default": "False"}),
                "flip_xy": (["False", "True"], {"default": "False"}),
                "x_axis": (pipeXYPlot.plot_values, {"default": 'None'}),
                "x_values": (
                "STRING", {"default": '', "multiline": True, "placeholder": 'insert values seperated by "; "'}),
                "y_axis": (pipeXYPlot.plot_values, {"default": 'None'}),
                "y_values": (
                "STRING", {"default": '', "multiline": True, "placeholder": 'insert values seperated by "; "'}),
            },
            "optional": {
              "pipe": ("PIPE_LINE",)
            },
            "hidden": {
                "plot_dict": (pipeXYPlot.plot_dict,),
            },
        }

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "plot"

    CATEGORY = "EasyUse/Pipe"

    def plot(self, grid_spacing, output_individuals, flip_xy, x_axis, x_values, y_axis, y_values, pipe=None, font_path=None):
        def clean_values(values):
            original_values = values.split("; ")
            cleaned_values = []

            for value in original_values:
                # Strip the semi-colon
                cleaned_value = value.strip(';').strip()

                if cleaned_value == "":
                    continue

                # Try to convert the cleaned_value back to int or float if possible
                try:
                    cleaned_value = int(cleaned_value)
                except ValueError:
                    try:
                        cleaned_value = float(cleaned_value)
                    except ValueError:
                        pass

                # Append the cleaned_value to the list
                cleaned_values.append(cleaned_value)

            return cleaned_values

        if x_axis in self.rejected:
            x_axis = "None"
            x_values = []
        else:
            x_values = clean_values(x_values)

        if y_axis in self.rejected:
            y_axis = "None"
            y_values = []
        else:
            y_values = clean_values(y_values)

        if flip_xy == "True":
            x_axis, y_axis = y_axis, x_axis
            x_values, y_values = y_values, x_values


        xy_plot = {"x_axis": x_axis,
                   "x_vals": x_values,
                   "y_axis": y_axis,
                   "y_vals": y_values,
                   "custom_font": font_path,
                   "grid_spacing": grid_spacing,
                   "output_individuals": output_individuals}

        if pipe is not None:
            new_pipe = pipe.copy()
            new_pipe['loader_settings'] = {
                **pipe['loader_settings'],
                "xyplot": xy_plot
            }
            del pipe
        return (new_pipe, xy_plot,)

# pipeXYPlotAdvanced
import platform
class pipeXYPlotAdvanced:
    if platform.system() == "Windows":
        system_root = os.environ.get("SystemRoot")
        user_root = os.environ.get("USERPROFILE")
        font_dir = os.path.join(system_root, "Fonts") if system_root else None
        user_font_dir = os.path.join(user_root, "AppData","Local","Microsoft","Windows", "Fonts") if user_root else None

    # Default debian-based Linux & MacOS font dirs
    elif platform.system() == "Linux":
        font_dir = "/usr/share/fonts/truetype"
        user_font_dir = None
    elif platform.system() == "Darwin":
        font_dir = "/System/Library/Fonts"
        user_font_dir = None
    else:
        font_dir = None
        user_font_dir = None

    @classmethod
    def INPUT_TYPES(s):
        files_list = []
        if s.font_dir and os.path.exists(s.font_dir):
            font_dir = s.font_dir
            files_list = files_list + [f for f in os.listdir(font_dir) if os.path.isfile(os.path.join(font_dir, f)) and f.lower().endswith(".ttf")]

        if s.user_font_dir and os.path.exists(s.user_font_dir):
            files_list = files_list + [f for f in os.listdir(s.user_font_dir) if os.path.isfile(os.path.join(s.user_font_dir, f)) and f.lower().endswith(".ttf")]

        return {
            "required": {
                "pipe": ("PIPE_LINE",),
                "grid_spacing": ("INT", {"min": 0, "max": 500, "step": 5, "default": 0, }),
                "output_individuals": (["False", "True"], {"default": "False"}),
                "flip_xy": (["False", "True"], {"default": "False"}),
            },
            "optional": {
                "X": ("X_Y",),
                "Y": ("X_Y",),
                "font": (["None"] + files_list,)
            },
            "hidden": {"my_unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "plot"

    CATEGORY = "EasyUse/Pipe"

    def plot(self, pipe, grid_spacing, output_individuals, flip_xy, X=None, Y=None, font=None, my_unique_id=None):
        font_path = os.path.join(self.font_dir, font) if font != "None" else None
        if font_path and not os.path.exists(font_path):
            font_path = os.path.join(self.user_font_dir, font)

        if X != None:
            x_axis = X.get('axis')
            x_values = X.get('values')
        else:
            x_axis = "Nothing"
            x_values = [""]
        if Y != None:
            y_axis = Y.get('axis')
            y_values = Y.get('values')
        else:
            y_axis = "Nothing"
            y_values = [""]

        if pipe is not None:
            new_pipe = pipe.copy()
            positive = pipe["loader_settings"]["positive"] if "positive" in pipe["loader_settings"] else ""
            negative = pipe["loader_settings"]["negative"] if "negative" in pipe["loader_settings"] else ""

            if x_axis == 'advanced: ModelMergeBlocks':
                models = X.get('models')
                vae_use = X.get('vae_use')
                if models is None:
                    raise Exception("models is not found")
                new_pipe['loader_settings'] = {
                    **pipe['loader_settings'],
                    "models": models,
                    "vae_use": vae_use
                }
            if y_axis == 'advanced: ModelMergeBlocks':
                models = Y.get('models')
                vae_use = Y.get('vae_use')
                if models is None:
                    raise Exception("models is not found")
                new_pipe['loader_settings'] = {
                    **pipe['loader_settings'],
                    "models": models,
                    "vae_use": vae_use
                }

            if x_axis in ['advanced: Lora', 'advanced: Checkpoint']:
                lora_stack = X.get('lora_stack')
                _lora_stack = []
                if lora_stack is not None:
                    for lora in lora_stack:
                        _lora_stack.append(
                            {"lora_name": lora[0], "model": pipe['model'], "clip": pipe['clip'], "model_strength": lora[1],
                             "clip_strength": lora[2]})
                del lora_stack
                x_values = "; ".join(x_values)
                lora_stack = pipe['lora_stack'] + _lora_stack if 'lora_stack' in pipe else _lora_stack
                new_pipe['loader_settings'] = {
                    **pipe['loader_settings'],
                    "lora_stack": lora_stack,
                }

            if y_axis in ['advanced: Lora', 'advanced: Checkpoint']:
                lora_stack = Y.get('lora_stack')
                _lora_stack = []
                if lora_stack is not None:
                    for lora in lora_stack:
                        _lora_stack.append(
                            {"lora_name": lora[0], "model": pipe['model'], "clip": pipe['clip'], "model_strength": lora[1],
                             "clip_strength": lora[2]})
                del lora_stack
                y_values = "; ".join(y_values)
                lora_stack = pipe['lora_stack'] + _lora_stack if 'lora_stack' in pipe else _lora_stack
                new_pipe['loader_settings'] = {
                    **pipe['loader_settings'],
                    "lora_stack": lora_stack,
                }

            if x_axis == 'advanced: Seeds++ Batch':
                if new_pipe['seed']:
                    value = x_values
                    x_values = []
                    for index in range(value):
                        x_values.append(str(new_pipe['seed'] + index))
                    x_values = "; ".join(x_values)
            if y_axis == 'advanced: Seeds++ Batch':
                if new_pipe['seed']:
                    value = y_values
                    y_values = []
                    for index in range(value):
                        y_values.append(str(new_pipe['seed'] + index))
                    y_values = "; ".join(y_values)

            if x_axis == 'advanced: Positive Prompt S/R':
                if positive:
                    x_value = x_values
                    x_values = []
                    for index, value in enumerate(x_value):
                        search_txt, replace_txt, replace_all = value
                        if replace_all:
                            txt = replace_txt if replace_txt is not None else positive
                            x_values.append(txt)
                        else:
                            txt = positive.replace(search_txt, replace_txt, 1) if replace_txt is not None else positive
                            x_values.append(txt)
                    x_values = "; ".join(x_values)
            if y_axis == 'advanced: Positive Prompt S/R':
                if positive:
                    y_value = y_values
                    y_values = []
                    for index, value in enumerate(y_value):
                        search_txt, replace_txt, replace_all = value
                        if replace_all:
                            txt = replace_txt if replace_txt is not None else positive
                            y_values.append(txt)
                        else:
                            txt = positive.replace(search_txt, replace_txt, 1) if replace_txt is not None else positive
                            y_values.append(txt)
                    y_values = "; ".join(y_values)

            if x_axis == 'advanced: Negative Prompt S/R':
                if negative:
                    x_value = x_values
                    x_values = []
                    for index, value in enumerate(x_value):
                        search_txt, replace_txt, replace_all = value
                        if replace_all:
                            txt = replace_txt if replace_txt is not None else negative
                            x_values.append(txt)
                        else:
                            txt = negative.replace(search_txt, replace_txt, 1) if replace_txt is not None else negative
                            x_values.append(txt)
                    x_values = "; ".join(x_values)
            if y_axis == 'advanced: Negative Prompt S/R':
                if negative:
                    y_value = y_values
                    y_values = []
                    for index, value in enumerate(y_value):
                        search_txt, replace_txt, replace_all = value
                        if replace_all:
                            txt = replace_txt if replace_txt is not None else negative
                            y_values.append(txt)
                        else:
                            txt = negative.replace(search_txt, replace_txt, 1) if replace_txt is not None else negative
                            y_values.append(txt)
                    y_values = "; ".join(y_values)

            if "advanced: ControlNet" in x_axis:
                x_value = x_values
                x_values = []
                cnet = []
                for index, value in enumerate(x_value):
                    cnet.append(value)
                    x_values.append(str(index))
                x_values = "; ".join(x_values)
                new_pipe['loader_settings'] = {
                    **pipe['loader_settings'],
                    "cnet_stack": cnet,
                }

            if "advanced: ControlNet" in y_axis:
                y_value = y_values
                y_values = []
                cnet = []
                for index, value in enumerate(y_value):
                    cnet.append(value)
                    y_values.append(str(index))
                y_values = "; ".join(y_values)
                new_pipe['loader_settings'] = {
                    **pipe['loader_settings'],
                    "cnet_stack": cnet,
                }

            if "advanced: Pos Condition" in x_axis:
                x_values = "; ".join(x_values)
                cond = X.get('cond')
                new_pipe['loader_settings'] = {
                    **pipe['loader_settings'],
                    "positive_cond_stack": cond,
                }
            if "advanced: Pos Condition" in y_axis:
                y_values = "; ".join(y_values)
                cond = Y.get('cond')
                new_pipe['loader_settings'] = {
                    **pipe['loader_settings'],
                    "positive_cond_stack": cond,
                }

            if "advanced: Neg Condition" in x_axis:
                x_values = "; ".join(x_values)
                cond = X.get('cond')
                new_pipe['loader_settings'] = {
                    **pipe['loader_settings'],
                    "negative_cond_stack": cond,
                }
            if "advanced: Neg Condition" in y_axis:
                y_values = "; ".join(y_values)
                cond = Y.get('cond')
                new_pipe['loader_settings'] = {
                    **pipe['loader_settings'],
                    "negative_cond_stack": cond,
                }

            del pipe

        return pipeXYPlot().plot(grid_spacing, output_individuals, flip_xy, x_axis, x_values, y_axis, y_values, new_pipe, font_path)


NODE_CLASS_MAPPINGS = {
    "easy pipeIn": pipeIn,
    "easy pipeOut": pipeOut,
    "easy pipeEdit": pipeEdit,
    "easy pipeEditPrompt": pipeEditPrompt,
    "easy pipeToBasicPipe": pipeToBasicPipe,
    "easy pipeBatchIndex": pipeBatchIndex,
    "easy XYPlot": pipeXYPlot,
    "easy XYPlotAdvanced": pipeXYPlotAdvanced
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "easy pipeIn": "Pipe In",
    "easy pipeOut": "Pipe Out",
    "easy pipeEdit": "Pipe Edit",
    "easy pipeEditPrompt": "Pipe Edit Prompt",
    "easy pipeBatchIndex": "Pipe Batch Index",
    "easy pipeToBasicPipe": "Pipe -> BasicPipe",
    "easy XYPlot": "XY Plot",
    "easy XYPlotAdvanced": "XY Plot Advanced"
}