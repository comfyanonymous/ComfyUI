import os, torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from .utils import easySave, get_sd_version
from .adv_encode import advanced_encode
from .controlnet import easyControlnet
from .log import log_node_warn
from ..modules.layer_diffuse import LayerDiffuse
from ..config import RESOURCES_DIR
from nodes import CLIPTextEncode
import pprint
try:
    from comfy_extras.nodes_flux import FluxGuidance
except:
    FluxGuidance = None

class easyXYPlot():

    def __init__(self, xyPlotData, save_prefix, image_output, prompt, extra_pnginfo, my_unique_id, sampler, easyCache):
        self.x_node_type, self.x_type = sampler.safe_split(xyPlotData.get("x_axis"), ': ')
        self.y_node_type, self.y_type = sampler.safe_split(xyPlotData.get("y_axis"), ': ')
        self.x_values = xyPlotData.get("x_vals") if self.x_type != "None" else []
        self.y_values = xyPlotData.get("y_vals") if self.y_type != "None" else []
        self.custom_font = xyPlotData.get("custom_font")

        self.grid_spacing = xyPlotData.get("grid_spacing")
        self.latent_id = 0
        self.output_individuals = xyPlotData.get("output_individuals")

        self.x_label, self.y_label = [], []
        self.max_width, self.max_height = 0, 0
        self.latents_plot = []
        self.image_list = []

        self.num_cols = len(self.x_values) if len(self.x_values) > 0 else 1
        self.num_rows = len(self.y_values) if len(self.y_values) > 0 else 1

        self.total = self.num_cols * self.num_rows
        self.num = 0

        self.save_prefix = save_prefix
        self.image_output = image_output
        self.prompt = prompt
        self.extra_pnginfo = extra_pnginfo
        self.my_unique_id = my_unique_id

        self.sampler = sampler
        self.easyCache = easyCache

    # Helper Functions
    @staticmethod
    def define_variable(plot_image_vars, value_type, value, index):

        plot_image_vars[value_type] = value
        if value_type in ["seed", "Seeds++ Batch"]:
            value_label = f"seed: {value}"
        else:
            value_label = f"{value_type}: {value}"

        if "ControlNet" in value_type:
            value_label = f"ControlNet {index + 1}"

        if value_type in ['Lora', 'Checkpoint']:
            arr = value.split(',')
            model_name = os.path.basename(os.path.splitext(arr[0])[0])
            trigger_words = ' ' + arr[3] if value_type == 'Lora' and len(arr[3]) > 2 else ''
            lora_weight = float(arr[1]) if value_type == 'Lora' and len(arr) > 1 else 0
            lora_weight_desc = f"({lora_weight:.2f})" if lora_weight > 0 else ''
            value_label = f"{model_name[:30]}{lora_weight_desc} {trigger_words}"

        if value_type in ["ModelMergeBlocks"]:
            if ":" in value:
                line = value.split(':')
                value_label = f"{line[0]}"
            elif len(value) > 16:
                value_label = f"ModelMergeBlocks {index + 1}"
            else:
                value_label = f"MMB: {value}"

        if value_type in ["Pos Condition"]:
            value_label = f"pos cond {index + 1}" if index>0 else f"pos cond"
        if value_type in ["Neg Condition"]:
            value_label = f"neg cond {index + 1}" if index>0 else f"neg cond"

        if value_type in ["Positive Prompt S/R"]:
            value_label = f"pos prompt {index + 1}" if index>0 else f"pos prompt"
        if value_type in ["Negative Prompt S/R"]:
            value_label = f"neg prompt {index + 1}" if index>0 else f"neg prompt"

        if value_type in ["steps", "cfg", "denoise", "clip_skip",
                          "lora_model_strength", "lora_clip_strength"]:
            value_label = f"{value_type}: {value}"

        if value_type == "positive":
            value_label = f"pos prompt {index + 1}"
        elif value_type == "negative":
            value_label = f"neg prompt {index + 1}"

        return plot_image_vars, value_label

    @staticmethod
    def get_font(font_size, font_path=None):
        if font_path is None:
            font_path = str(Path(os.path.join(RESOURCES_DIR, 'OpenSans-Medium.ttf')))
        return ImageFont.truetype(font_path, font_size)

    @staticmethod
    def update_label(label, value, num_items):
        if len(label) < num_items:
            return [*label, value]
        return label

    @staticmethod
    def rearrange_tensors(latent, num_cols, num_rows):
        new_latent = []
        for i in range(num_rows):
            for j in range(num_cols):
                index = j * num_rows + i
                new_latent.append(latent[index])
        return new_latent

    def calculate_background_dimensions(self):
        border_size = int((self.max_width // 8) * 1.5) if self.y_type != "None" or self.x_type != "None" else 0

        bg_width = self.num_cols * (self.max_width + self.grid_spacing) - self.grid_spacing + border_size * (
                    self.y_type != "None")
        bg_height = self.num_rows * (self.max_height + self.grid_spacing) - self.grid_spacing + border_size * (
                    self.x_type != "None")
        
        # Add space at the bottom of the image for common informaiton about the image
        bg_height = bg_height + (border_size*2)
#        print(f"Grid Size: width = {bg_width} height = {bg_height} border_size = {border_size}")

        x_offset_initial = border_size if self.y_type != "None" else 0
        y_offset = border_size if self.x_type != "None" else 0

        return bg_width, bg_height, x_offset_initial, y_offset


    def adjust_font_size(self, text, initial_font_size, label_width):
        font = self.get_font(initial_font_size, self.custom_font)
        text_width = font.getbbox(text)
#        pprint.pp(f"Initial font size: {initial_font_size}, text: {text}, text_width: {text_width}")
        if text_width and text_width[2]:
            text_width = text_width[2]

        scaling_factor = 0.9
        if text_width > (label_width * scaling_factor):
#            print(f"Adjusting font size from {initial_font_size} to fit text width {text_width} into label width {label_width} scaling_factor {scaling_factor}")
            return int(initial_font_size * (label_width / text_width) * scaling_factor)
        else:
            return initial_font_size

    def textsize(self, d, text, font):
        _, _, width, height = d.textbbox((0, 0), text=text, font=font)
        return width, height

    def create_label(self, img, text, initial_font_size, is_x_label=True, max_font_size=70, min_font_size=10, label_width=0, label_height=0):

        # if the label_width is specified, leave it along.  Otherwise do the old logic.
        if label_width == 0:          
            label_width = img.width if is_x_label else img.height

        text_lines = text.split('\n')
        longest_line = max(text_lines, key=len)
                
        # Adjust font size
        font_size = self.adjust_font_size(longest_line, initial_font_size, label_width)
        font_size = min(max_font_size, font_size)  # Ensure font isn't too large
        font_size = max(min_font_size, font_size)  # Ensure font isn't too small

        if label_height == 0:
            label_height = int(font_size * 1.5) if is_x_label else font_size

        label_bg = Image.new('RGBA', (label_width, label_height), color=(255, 255, 255, 0))
        d = ImageDraw.Draw(label_bg)

        font = self.get_font(font_size, self.custom_font)

        # Check if text will fit, if not insert ellipsis and reduce text
        if self.textsize(d, text, font=font)[0] > label_width:
            while self.textsize(d, text + '...', font=font)[0] > label_width and len(text) > 0:
                text = text[:-1]
            text = text + '...'

        # Compute text width and height for multi-line text
  
        text_widths, text_heights = zip(*[self.textsize(d, line, font=font) for line in text_lines])
        max_text_width = max(text_widths)
        total_text_height = sum(text_heights)

        # Compute position for each line of text
        lines_positions = []
        current_y = 0
        for line, line_width, line_height in zip(text_lines, text_widths, text_heights):
            text_x = (label_width - line_width) // 2
            text_y = current_y + (label_height - total_text_height) // 2
            current_y += line_height
            lines_positions.append((line, (text_x, text_y)))

        # Draw each line of text
        for line, (text_x, text_y) in lines_positions:
            d.text((text_x, text_y), line, fill='black', font=font)

        return label_bg

    def sample_plot_image(self, plot_image_vars, samples, preview_latent, latents_plot, image_list, disable_noise,
                          start_step, last_step, force_full_denoise, x_value=None, y_value=None):
        model, clip, vae, positive, negative, seed, steps, cfg = None, None, None, None, None, None, None, None
        sampler_name, scheduler, denoise = None, None, None

        a1111_prompt_style = plot_image_vars['a1111_prompt_style'] if "a1111_prompt_style" in plot_image_vars else False
        clip = clip if clip is not None else plot_image_vars["clip"]
        steps = plot_image_vars['steps'] if "steps" in plot_image_vars else 1

        sd_version = get_sd_version(plot_image_vars['model'])          
        # 高级用法
        if plot_image_vars["x_node_type"] == "advanced" or plot_image_vars["y_node_type"] == "advanced":
            if self.x_type == "Seeds++ Batch" or self.y_type == "Seeds++ Batch":
                seed = int(x_value) if self.x_type == "Seeds++ Batch" else int(y_value)
            if self.x_type == "Steps" or self.y_type == "Steps":
                steps = int(x_value) if self.x_type == "Steps" else int(y_value)
            if self.x_type == "StartStep" or self.y_type == "StartStep":
                start_step = int(x_value) if self.x_type == "StartStep" else int(y_value)
            if self.x_type == "EndStep" or self.y_type == "EndStep":
                last_step = int(x_value) if self.x_type == "EndStep" else int(y_value)
            if self.x_type == "CFG Scale" or self.y_type == "CFG Scale":
                cfg = float(x_value) if self.x_type == "CFG Scale" else float(y_value)
            if self.x_type == "Sampler" or self.y_type == "Sampler":
                sampler_name = x_value if self.x_type == "Sampler" else y_value
            if self.x_type == "Scheduler" or self.y_type == "Scheduler":
                scheduler = x_value if self.x_type == "Scheduler" else y_value
            if self.x_type == "Sampler&Scheduler" or self.y_type == "Sampler&Scheduler":
                arr = x_value.split(',') if self.x_type == "Sampler&Scheduler" else y_value.split(',')
                if arr[0] and arr[0]!= 'None':
                    sampler_name = arr[0]
                if arr[1] and arr[1]!= 'None':
                    scheduler = arr[1]
            if self.x_type == "Denoise" or self.y_type == "Denoise":
                denoise = float(x_value) if self.x_type == "Denoise" else float(y_value)
            if self.x_type == "Pos Condition" or self.y_type == "Pos Condition":
                positive = plot_image_vars['positive_cond_stack'][int(x_value)] if self.x_type == "Pos Condition" else plot_image_vars['positive_cond_stack'][int(y_value)]
            if self.x_type == "Neg Condition" or self.y_type == "Neg Condition":
                negative = plot_image_vars['negative_cond_stack'][int(x_value)] if self.x_type == "Neg Condition" else plot_image_vars['negative_cond_stack'][int(y_value)]
            # 模型叠加
            if self.x_type == "ModelMergeBlocks" or self.y_type == "ModelMergeBlocks":
                ckpt_name_1, ckpt_name_2 = plot_image_vars['models']
                model1, clip1, vae1, clip_vision = self.easyCache.load_checkpoint(ckpt_name_1)
                model2, clip2, vae2, clip_vision = self.easyCache.load_checkpoint(ckpt_name_2)
                xy_values = x_value if self.x_type == "ModelMergeBlocks" else y_value
                if ":" in xy_values:
                    xy_line = xy_values.split(':')
                    xy_values = xy_line[1]

                xy_arrs = xy_values.split(',')
                # ModelMergeBlocks
                if len(xy_arrs) == 3:
                    input, middle, out = xy_arrs
                    kwargs = {
                        "input": input,
                        "middle": middle,
                        "out": out
                    }
                elif len(xy_arrs) == 30:
                    kwargs = {}
                    kwargs["time_embed."] = xy_arrs[0]
                    kwargs["label_emb."] = xy_arrs[1]

                    for i in range(12):
                        kwargs["input_blocks.{}.".format(i)] = xy_arrs[2+i]

                    for i in range(3):
                        kwargs["middle_block.{}.".format(i)] = xy_arrs[14+i]

                    for i in range(12):
                        kwargs["output_blocks.{}.".format(i)] = xy_arrs[17+i]

                    kwargs["out."] = xy_arrs[29]
                else:
                    raise Exception("ModelMergeBlocks weight length error")
                default_ratio = next(iter(kwargs.values()))

                m = model1.clone()
                kp = model2.get_key_patches("diffusion_model.")

                for k in kp:
                    ratio = float(default_ratio)
                    k_unet = k[len("diffusion_model."):]

                    last_arg_size = 0
                    for arg in kwargs:
                        if k_unet.startswith(arg) and last_arg_size < len(arg):
                            ratio = float(kwargs[arg])
                            last_arg_size = len(arg)

                    m.add_patches({k: kp[k]}, 1.0 - ratio, ratio)

                vae_use = plot_image_vars['vae_use']

                clip = clip2 if vae_use == 'Use Model 2' else clip1
                if vae_use == 'Use Model 2':
                    vae = vae2
                elif vae_use == 'Use Model 1':
                    vae = vae1
                else:
                    vae = self.easyCache.load_vae(vae_use)
                model = m

                # 如果存在lora_stack叠加lora
                optional_lora_stack = plot_image_vars['lora_stack']
                if optional_lora_stack is not None and optional_lora_stack != []:
                    for lora in optional_lora_stack:
                        model, clip = self.easyCache.load_lora(lora)

                # 处理clip
                clip = clip.clone()
                if plot_image_vars['clip_skip'] != 0:
                    clip.clip_layer(plot_image_vars['clip_skip'])

            # CheckPoint
            if self.x_type == "Checkpoint" or self.y_type == "Checkpoint":
                xy_values = x_value if self.x_type == "Checkpoint" else y_value
                ckpt_name, clip_skip, vae_name = xy_values.split(",")
                ckpt_name = ckpt_name.replace('*', ',')
                vae_name = vae_name.replace('*', ',')
                model, clip, vae, clip_vision = self.easyCache.load_checkpoint(ckpt_name)
                if vae_name != 'None':
                    vae = self.easyCache.load_vae(vae_name)

                # 如果存在lora_stack叠加lora
                optional_lora_stack = plot_image_vars['lora_stack']
                if optional_lora_stack is not None and optional_lora_stack != []:
                    for lora in optional_lora_stack:
                        lora['model'] = model
                        lora['clip'] = clip
                        model, clip = self.easyCache.load_lora(lora)

                # 处理clip
                clip = clip.clone()
                if clip_skip != 'None':
                    clip.clip_layer(int(clip_skip))
                    positive = plot_image_vars['positive']
                    negative = plot_image_vars['negative']
                    a1111_prompt_style = plot_image_vars['a1111_prompt_style']
                    steps = plot_image_vars['steps']
                    clip = clip if clip is not None else plot_image_vars["clip"]
                    positive = advanced_encode(clip, positive,
                                               plot_image_vars['positive_token_normalization'],
                                               plot_image_vars['positive_weight_interpretation'],
                                               w_max=1.0,
                                               apply_to_pooled="enable",
                                               a1111_prompt_style=a1111_prompt_style, steps=steps)

                    negative = advanced_encode(clip, negative,
                                               plot_image_vars['negative_token_normalization'],
                                               plot_image_vars['negative_weight_interpretation'],
                                               w_max=1.0,
                                               apply_to_pooled="enable",
                                               a1111_prompt_style=a1111_prompt_style, steps=steps)
                    if "positive_cond" in plot_image_vars:
                        positive = positive + plot_image_vars["positive_cond"]
                    if "negative_cond" in plot_image_vars:
                        negative = negative + plot_image_vars["negative_cond"]

            # Lora
            if self.x_type == "Lora" or self.y_type == "Lora":
#                print(f"Lora: {x_value} {y_value}")
                model = model if model is not None else plot_image_vars["model"]
                clip = clip if clip is not None else plot_image_vars["clip"]
                xy_values = x_value if self.x_type == "Lora" else y_value
                lora_name, lora_model_strength, lora_clip_strength, _ = xy_values.split(",")
                lora_stack = [{"lora_name": lora_name, "model": model, "clip" :clip, "model_strength": float(lora_model_strength), "clip_strength": float(lora_clip_strength)}]
                
#                print(f"new_lora_stack: {new_lora_stack}")

                
                if 'lora_stack' in plot_image_vars:
                    lora_stack = lora_stack + plot_image_vars['lora_stack']
                
                if lora_stack is not None and lora_stack != []:
                    for lora in lora_stack:
                        # Each generation of the model, must use the reference to previously created model / clip objects.
                        lora['model'] = model
                        lora['clip'] = clip
                        model, clip = self.easyCache.load_lora(lora)

            # 提示词
            if "Positive" in self.x_type or "Positive" in self.y_type:
                if self.x_type == 'Positive Prompt S/R' or self.y_type == 'Positive Prompt S/R':
                    positive = x_value if self.x_type == "Positive Prompt S/R" else y_value

                if sd_version == 'flux':
                    positive, = CLIPTextEncode().encode(clip, positive)
                else:
                    positive = advanced_encode(clip, positive,
                                                plot_image_vars['positive_token_normalization'],
                                                plot_image_vars['positive_weight_interpretation'],
                                                w_max=1.0,
                                                apply_to_pooled="enable", a1111_prompt_style=a1111_prompt_style, steps=steps)

                # if "positive_cond" in plot_image_vars:
                #     positive = positive + plot_image_vars["positive_cond"]

            if "Negative" in self.x_type or "Negative" in self.y_type:
                if self.x_type == 'Negative Prompt S/R' or self.y_type == 'Negative Prompt S/R':
                    negative = x_value if self.x_type == "Negative Prompt S/R" else y_value

                if sd_version == 'flux':
                    negative, = CLIPTextEncode().encode(clip, negative)
                else:
                    negative = advanced_encode(clip, negative,
                                                plot_image_vars['negative_token_normalization'],
                                                plot_image_vars['negative_weight_interpretation'],
                                                w_max=1.0,
                                                apply_to_pooled="enable", a1111_prompt_style=a1111_prompt_style, steps=steps)
                # if "negative_cond" in plot_image_vars:
                #     negative = negative + plot_image_vars["negative_cond"]

            # ControlNet
            if "ControlNet" in self.x_type or "ControlNet" in self.y_type:
                cnet = plot_image_vars["cnet"] if "cnet" in plot_image_vars else None
                positive = plot_image_vars["positive_cond"] if "positive" in plot_image_vars else None
                negative = plot_image_vars["negative_cond"] if "negative" in plot_image_vars else None
                if cnet:
                    index = x_value if "ControlNet" in self.x_type else y_value
                    controlnet = cnet[index]
                    for index, item in enumerate(controlnet):
                        control_net_name = item[0]
                        image = item[1]
                        strength = item[2]
                        start_percent = item[3]
                        end_percent = item[4]
                        positive, negative = easyControlnet().apply(control_net_name, image, positive, negative, strength, start_percent, end_percent, None, 1)
            # Flux guidance
            if self.x_type == "Flux Guidance" or self.y_type == "Flux Guidance":
                positive = plot_image_vars["positive_cond"] if "positive" in plot_image_vars else None
                flux_guidance = float(x_value) if self.x_type == "Flux Guidance" else float(y_value)
                positive, = FluxGuidance().append(positive, flux_guidance)

        # 简单用法
        if plot_image_vars["x_node_type"] == "loader" or plot_image_vars["y_node_type"] == "loader":
            if self.x_type == 'ckpt_name' or self.y_type == 'ckpt_name':
                ckpt_name = x_value if self.x_type == "ckpt_name" else y_value
                model, clip, vae, clip_vision = self.easyCache.load_checkpoint(ckpt_name)

            if self.x_type == 'lora_name' or self.y_type == 'lora_name':
                model, clip, vae, clip_vision = self.easyCache.load_checkpoint(plot_image_vars['ckpt_name'])
                lora_name = x_value if self.x_type == "lora_name" else y_value
                lora = {"lora_name": lora_name, "model": model, "clip": clip, "model_strength": 1, "clip_strength": 1}
                model, clip = self.easyCache.load_lora(lora)

            if self.x_type == 'lora_model_strength' or self.y_type == 'lora_model_strength':
                model, clip, vae, clip_vision = self.easyCache.load_checkpoint(plot_image_vars['ckpt_name'])
                lora_model_strength = float(x_value) if self.x_type == "lora_model_strength" else float(y_value)
                lora = {"lora_name": plot_image_vars['lora_name'], "model": model, "clip": clip, "model_strength": lora_model_strength, "clip_strength": plot_image_vars['lora_clip_strength']}
                model, clip = self.easyCache.load_lora(lora)

            if self.x_type == 'lora_clip_strength' or self.y_type == 'lora_clip_strength':
                model, clip, vae, clip_vision = self.easyCache.load_checkpoint(plot_image_vars['ckpt_name'])
                lora_clip_strength = float(x_value) if self.x_type == "lora_clip_strength" else float(y_value)
                lora = {"lora_name": plot_image_vars['lora_name'], "model": model, "clip": clip, "model_strength": plot_image_vars['lora_model_strength'], "clip_strength": lora_clip_strength}
                model, clip = self.easyCache.load_lora(lora)

            # Check for custom VAE
            if self.x_type == 'vae_name' or self.y_type == 'vae_name':
                vae_name = x_value if self.x_type == "vae_name" else y_value
                vae = self.easyCache.load_vae(vae_name)

            # CLIP skip
            if not clip:
                raise Exception("No CLIP found")
            clip = clip.clone()
            clip.clip_layer(plot_image_vars['clip_skip'])

            if sd_version == 'flux':
                positive, = CLIPTextEncode().encode(clip, positive)
            else:
                positive = advanced_encode(clip, plot_image_vars['positive'],
                                                            plot_image_vars['positive_token_normalization'],
                                                            plot_image_vars['positive_weight_interpretation'], w_max=1.0,
                                                            apply_to_pooled="enable",a1111_prompt_style=a1111_prompt_style, steps=steps)

            if sd_version == 'flux':
                negative, = CLIPTextEncode().encode(clip, negative)
            else:
                negative = advanced_encode(clip, plot_image_vars['negative'],
                                                            plot_image_vars['negative_token_normalization'],
                                                            plot_image_vars['negative_weight_interpretation'], w_max=1.0,
                                                            apply_to_pooled="enable", a1111_prompt_style=a1111_prompt_style, steps=steps)


        model = model if model is not None else plot_image_vars["model"]
        vae = vae if vae is not None else plot_image_vars["vae"]
        positive = positive if positive is not None else plot_image_vars["positive_cond"]
        negative = negative if negative is not None else plot_image_vars["negative_cond"]

        seed = seed if seed is not None else plot_image_vars["seed"]
        steps = steps if steps is not None else plot_image_vars["steps"]
        cfg = cfg if cfg is not None else plot_image_vars["cfg"]
        sampler_name = sampler_name if sampler_name is not None else plot_image_vars["sampler_name"]
        scheduler = scheduler if scheduler is not None else plot_image_vars["scheduler"]
        denoise = denoise if denoise is not None else plot_image_vars["denoise"]

        noise_device = plot_image_vars["noise_device"] if "noise_device" in plot_image_vars else 'cpu'

        # LayerDiffuse
        layer_diffusion_method = plot_image_vars["layer_diffusion_method"] if "layer_diffusion_method" in plot_image_vars else None
        empty_samples = plot_image_vars["empty_samples"] if "empty_samples" in plot_image_vars else None

        if layer_diffusion_method:
            samp_blend_samples = plot_image_vars["blend_samples"] if "blend_samples" in plot_image_vars else None
            additional_cond = plot_image_vars["layer_diffusion_cond"] if "layer_diffusion_cond" in plot_image_vars else None

            images = plot_image_vars["images"].movedim(-1, 1) if "images" in plot_image_vars else None
            weight = plot_image_vars['layer_diffusion_weight'] if 'layer_diffusion_weight' in plot_image_vars else 1.0
            model, positive, negative = LayerDiffuse().apply_layer_diffusion(model, layer_diffusion_method, weight, samples,
                                                                                  samp_blend_samples, positive,
                                                                                  negative, images, additional_cond)

        samples = empty_samples if layer_diffusion_method is not None and empty_samples is not None else samples
        # Sample
        samples = self.sampler.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, samples,
                                          denoise=denoise, disable_noise=disable_noise, preview_latent=preview_latent,
                                          start_step=start_step, last_step=last_step,
                                          force_full_denoise=force_full_denoise, noise_device=noise_device)

        # Decode images and store
        latent = samples["samples"]

        # Add the latent tensor to the tensors list
        latents_plot.append(latent)

        # Decode the image
        image = vae.decode(latent).cpu()

        if self.output_individuals in [True, "True"]:
            easySave(image, self.save_prefix, self.image_output)

        # Convert the image from tensor to PIL Image and add it to the list
        pil_image = self.sampler.tensor2pil(image)
        image_list.append(pil_image)

        # Update max dimensions
        self.max_width = max(self.max_width, pil_image.width)
        self.max_height = max(self.max_height, pil_image.height)

        # Return the touched variables
        return image_list, self.max_width, self.max_height, latents_plot

    # Process Functions
    def validate_xy_plot(self):
        if self.x_type == 'None' and self.y_type == 'None':
            log_node_warn(f'#{self.my_unique_id}','No Valid Plot Types - Reverting to default sampling...')
            return False
        else:
            return True

    def get_latent(self, samples):
        # Extract the 'samples' tensor from the dictionary
        latent_image_tensor = samples["samples"]

        # Split the tensor into individual image tensors
        image_tensors = torch.split(latent_image_tensor, 1, dim=0)

        # Create a list of dictionaries containing the individual image tensors
        latent_list = [{'samples': image} for image in image_tensors]

        # Set latent only to the first latent of batch
        if self.latent_id >= len(latent_list):
            log_node_warn(f'#{self.my_unique_id}',f'The selected latent_id ({self.latent_id}) is out of range.')
            log_node_warn(f'#{self.my_unique_id}', f'Automatically setting the latent_id to the last image in the list (index: {len(latent_list) - 1}).')

            self.latent_id = len(latent_list) - 1

        return latent_list[self.latent_id]

    def get_labels_and_sample(self, plot_image_vars, latent_image, preview_latent, start_step, last_step,
                              force_full_denoise, disable_noise):
        for x_index, x_value in enumerate(self.x_values):
            plot_image_vars, x_value_label = self.define_variable(plot_image_vars, self.x_type, x_value,
                                                                  x_index)
            self.x_label = self.update_label(self.x_label, x_value_label, len(self.x_values))
            if self.y_type != 'None':
                for y_index, y_value in enumerate(self.y_values):
                    plot_image_vars, y_value_label = self.define_variable(plot_image_vars, self.y_type, y_value,
                                                                          y_index)
                    self.y_label = self.update_label(self.y_label, y_value_label, len(self.y_values))
                    # ttNl(f'{CC.GREY}X: {x_value_label}, Y: {y_value_label}').t(
                    #     f'Plot Values {self.num}/{self.total} ->').p()

                    self.image_list, self.max_width, self.max_height, self.latents_plot = self.sample_plot_image(
                        plot_image_vars, latent_image, preview_latent, self.latents_plot, self.image_list,
                        disable_noise, start_step, last_step, force_full_denoise, x_value, y_value)
                    self.num += 1
            else:
                # ttNl(f'{CC.GREY}X: {x_value_label}').t(f'Plot Values {self.num}/{self.total} ->').p()
                self.image_list, self.max_width, self.max_height, self.latents_plot = self.sample_plot_image(
                    plot_image_vars, latent_image, preview_latent, self.latents_plot, self.image_list, disable_noise,
                    start_step, last_step, force_full_denoise, x_value)
                self.num += 1

        # Rearrange latent array to match preview image grid
        self.latents_plot = self.rearrange_tensors(self.latents_plot, self.num_cols, self.num_rows)

        # Concatenate the tensors along the first dimension (dim=0)
        self.latents_plot = torch.cat(self.latents_plot, dim=0)

        return self.latents_plot

    def plot_images_and_labels(self, plot_image_vars):
                    
        bg_width, bg_height, x_offset_initial, y_offset = self.calculate_background_dimensions()

        background = Image.new('RGBA', (int(bg_width), int(bg_height)), color=(255, 255, 255, 255))

        output_image = []
        for row_index in range(self.num_rows):
            x_offset = x_offset_initial

            for col_index in range(self.num_cols):
                index = col_index * self.num_rows + row_index
                img = self.image_list[index]
                output_image.append(self.sampler.pil2tensor(img))
                background.paste(img, (x_offset, y_offset))

                # Handle X label
                if row_index == 0 and self.x_type != "None":
                    label_bg = self.create_label(img, self.x_label[col_index], int(48 * img.width / 512))
                    label_y = (y_offset - label_bg.height) // 2
                    background.alpha_composite(label_bg, (x_offset, label_y))

                # Handle Y label
                if col_index == 0 and self.y_type != "None":
                    label_bg = self.create_label(img, self.y_label[row_index], int(48 * img.height / 512), False)
                    label_bg = label_bg.rotate(90, expand=True)

                    label_x = (x_offset - label_bg.width) // 2
                    label_y = y_offset + (img.height - label_bg.height) // 2
                    background.alpha_composite(label_bg, (label_x, label_y))

                x_offset += img.width + self.grid_spacing

            y_offset += img.height + self.grid_spacing

        # lookup used models in the image
        common_label = ""
        # Update to add a function to do the heavy lifting. Parameters are plot_image_vars name, label to use, names of the axis, 

        # pprint.pp(plot_image_vars)

        # We don't process LORAs here because there can be multiple of them.
        labels = [
            {"id": "ckpt_name", "id_desc": "ckpt", "axis_type" : "Checkpoint"},
            {"id": "vae_name", "id_desc": '', "axis_type" : "vae_name"},
            {"id": "sampler_name", "id_desc": "sampler", "axis_type" : "Sampler"},
            {"id": "scheduler", "id_desc": '', "axis_type" : "Scheduler"},
            {"id": "steps", "id_desc": '', "axis_type" : "Steps"},
            {"id": "Flux Guidance", "id_desc": 'guidance', "axis_type" : "Flux Guidance"},     
            {"id": "seed", "id_desc": '', "axis_type" : "Seeds++ Batch"}
        ]
        
        for item in labels:
            # Only add the label if it's not one of the axis
            # print(f"Checking item: {item['id']} axis_type {item['axis_type']} x_type: {self.x_type} y_type: {self.y_type}")
            if self.x_type != item['axis_type'] and self.y_type != item['axis_type']:
                common_label += self.add_common_label(item['id'], plot_image_vars, item['id_desc'])
        common_label += f"\n"
                
        if plot_image_vars['lora_stack'] is not None and plot_image_vars['lora_stack'] != []:
#            print(f"lora_stack: {plot_image_vars['lora_stack']}")
            for lora in plot_image_vars['lora_stack']:

                lora_name = lora['lora_name']
                lora_weight = lora['model_strength']
                if lora_name is not None and len(lora_name) > 0 and lora_weight > 0:
                    common_label += f"LORA: {lora_name} weight: {lora_weight:.2f} \n"
                    
        common_label = common_label.strip()
        
        if len(common_label) > 0:
            label_height = background.height - y_offset
            label_bg = self.create_label(background, common_label, int(48 * background.width / 512), label_width=background.width, label_height=label_height)
            label_x = (background.width - label_bg.width) // 2
            label_y = y_offset
#            print(f"Adding common label: {common_label} x = {label_x} y = {label_y}")
            background.alpha_composite(label_bg, (label_x, label_y))

        return (self.sampler.pil2tensor(background), output_image)

    def add_common_label(self, tag, plot_image_vars, description = ''):
        label = ''
        if description == '': description = tag
        if tag in plot_image_vars and plot_image_vars[tag] is not None and plot_image_vars[tag] != 'None':
            label += f"{description}: {plot_image_vars[tag]} "
#        print(f"add_common_label: {tag} description: {description} label: {label}" )
        return label
