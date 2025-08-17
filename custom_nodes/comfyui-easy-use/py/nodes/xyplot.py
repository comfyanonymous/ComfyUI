import os
import json
import comfy
import folder_paths
from ..config import RESOURCES_DIR
from ..libs.utils import getMetadata
def load_preset(filename):
    path = os.path.join(RESOURCES_DIR, filename)
    path = os.path.abspath(path)
    preset_list = []

    if os.path.exists(path):
        with open(path, 'r') as file:
            for line in file:
                preset_list.append(line.strip())

        return preset_list
    else:
        return []
def generate_floats(batch_count, first_float, last_float):
    if batch_count > 1:
        interval = (last_float - first_float) / (batch_count - 1)
        values = [str(round(first_float + i * interval, 3)) for i in range(batch_count)]
    else:
        values = [str(first_float)] if batch_count == 1 else []
    return "; ".join(values)

def generate_ints(batch_count, first_int, last_int):
    if batch_count > 1:
        interval = (last_int - first_int) / (batch_count - 1)
        values = [str(int(first_int + i * interval)) for i in range(batch_count)]
    else:
        values = [str(first_int)] if batch_count == 1 else []
    # values = list(set(values))  # Remove duplicates
    # values.sort()  # Sort in ascending order
    return "; ".join(values)

# Seed++ Batch
class XYplot_SeedsBatch:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "batch_count": ("INT", {"default": 3, "min": 1, "max": 50}), },
        }

    RETURN_TYPES = ("X_Y",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "EasyUse/XY Inputs"

    def xy_value(self, batch_count):

        axis = "advanced: Seeds++ Batch"
        xy_values = {"axis": axis, "values": batch_count}
        return (xy_values,)

# Step Values
class XYplot_Steps:
    parameters = ["steps", "start_at_step", "end_at_step",]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_parameter": (cls.parameters,),
                "batch_count": ("INT", {"default": 3, "min": 0, "max": 50}),
                "first_step": ("INT", {"default": 10, "min": 1, "max": 10000}),
                "last_step": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "first_start_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "last_start_step": ("INT", {"default": 10, "min": 0, "max": 10000}),
                "first_end_step": ("INT", {"default": 10, "min": 0, "max": 10000}),
                "last_end_step": ("INT", {"default": 20, "min": 0, "max": 10000}),
            }
        }

    RETURN_TYPES = ("X_Y",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "EasyUse/XY Inputs"

    def xy_value(self, target_parameter, batch_count, first_step, last_step, first_start_step, last_start_step,
                 first_end_step, last_end_step,):

        axis, xy_first, xy_last = None, None, None

        if target_parameter == "steps":
            axis = "advanced: Steps"
            xy_first = first_step
            xy_last = last_step
        elif target_parameter == "start_at_step":
            axis = "advanced: StartStep"
            xy_first = first_start_step
            xy_last = last_start_step
        elif target_parameter == "end_at_step":
            axis = "advanced: EndStep"
            xy_first = first_end_step
            xy_last = last_end_step

        values = generate_ints(batch_count, xy_first, xy_last)
        return ({"axis": axis, "values": values},) if values is not None else (None,)

class XYplot_CFG:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_count": ("INT", {"default": 3, "min": 0, "max": 50}),
                "first_cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
                "last_cfg": ("FLOAT", {"default": 9.0, "min": 0.0, "max": 100.0}),
            }
        }

    RETURN_TYPES = ("X_Y",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "EasyUse/XY Inputs"

    def xy_value(self, batch_count, first_cfg, last_cfg):
        axis = "advanced: CFG Scale"
        values = generate_floats(batch_count, first_cfg, last_cfg)
        return ({"axis": axis, "values": values},) if values else (None,)

class XYplot_FluxGuidance:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_count": ("INT", {"default": 3, "min": 0, "max": 50}),
                "first_guidance": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0}),
                "last_guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0}),
            }
        }

    RETURN_TYPES = ("X_Y",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "EasyUse/XY Inputs"

    def xy_value(self, batch_count, first_guidance, last_guidance):
        axis = "advanced: Flux Guidance"
        values = generate_floats(batch_count, first_guidance, last_guidance)
        return ({"axis": axis, "values": values},) if values else (None,)

# Step Values
class XYplot_Sampler_Scheduler:
    parameters = ["sampler", "scheduler", "sampler & scheduler"]

    @classmethod
    def INPUT_TYPES(cls):
        samplers = ["None"] + comfy.samplers.KSampler.SAMPLERS
        schedulers = ["None"] + comfy.samplers.KSampler.SCHEDULERS
        inputs = {
            "required": {
                "target_parameter": (cls.parameters,),
                "input_count": ("INT", {"default": 1, "min": 1, "max": 30, "step": 1})
            }
        }
        for i in range(1, 30 + 1):
            inputs["required"][f"sampler_{i}"] = (samplers,)
            inputs["required"][f"scheduler_{i}"] = (schedulers,)

        return inputs

    RETURN_TYPES = ("X_Y",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "EasyUse/XY Inputs"

    def xy_value(self, target_parameter, input_count, **kwargs):
        axis, values, = None, None,
        if target_parameter == "scheduler":
            axis = "advanced: Scheduler"
            schedulers = [kwargs.get(f"scheduler_{i}") for i in range(1, input_count + 1)]
            values = [scheduler for scheduler in schedulers if scheduler != "None"]
        elif target_parameter == "sampler":
            axis = "advanced: Sampler"
            samplers = [kwargs.get(f"sampler_{i}") for i in range(1, input_count + 1)]
            values = [sampler for sampler in samplers if sampler != "None"]
        else:
            axis = "advanced: Sampler&Scheduler"
            samplers = [kwargs.get(f"sampler_{i}") for i in range(1, input_count + 1)]
            schedulers = [kwargs.get(f"scheduler_{i}") for i in range(1, input_count + 1)]
            values = []
            for sampler, scheduler in zip(samplers, schedulers):
                sampler = sampler if sampler else 'None'
                scheduler = scheduler if scheduler else 'None'
                values.append(sampler +','+ scheduler)
        values = "; ".join(values)
        return ({"axis": axis, "values": values},) if values else (None,)

class XYplot_Denoise:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_count": ("INT", {"default": 3, "min": 0, "max": 50}),
                "first_denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                "last_denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("X_Y",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "EasyUse/XY Inputs"

    def xy_value(self, batch_count, first_denoise, last_denoise):
        axis = "advanced: Denoise"
        values = generate_floats(batch_count, first_denoise, last_denoise)
        return ({"axis": axis, "values": values},) if values else (None,)

# PromptSR
class XYplot_PromptSR:

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "target_prompt": (["positive", "negative"],),
                "search_txt": ("STRING", {"default": "", "multiline": False}),
                "replace_all_text": ("BOOLEAN", {"default": False}),
                "replace_count": ("INT", {"default": 3, "min": 1, "max": 30 - 1}),
            }
        }

        # Dynamically add replace_X inputs
        for i in range(1, 30):
            replace_key = f"replace_{i}"
            inputs["required"][replace_key] = ("STRING", {"default": "", "multiline": False, "placeholder": replace_key})

        return inputs

    RETURN_TYPES = ("X_Y",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "EasyUse/XY Inputs"

    def xy_value(self, target_prompt, search_txt, replace_all_text, replace_count, **kwargs):
        axis = None

        if target_prompt == "positive":
            axis = "advanced: Positive Prompt S/R"
        elif target_prompt == "negative":
            axis = "advanced: Negative Prompt S/R"

        # Create base entry
        values = [(search_txt, None, replace_all_text)]

        if replace_count > 0:
            # Append additional entries based on replace_count
            values.extend([(search_txt, kwargs.get(f"replace_{i+1}"), replace_all_text) for i in range(replace_count)])
        return ({"axis": axis, "values": values},) if values is not None else (None,)

# XYPlot Pos Condition
class XYplot_Positive_Cond:

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "optional": {
                "positive_1": ("CONDITIONING",),
                "positive_2": ("CONDITIONING",),
                "positive_3": ("CONDITIONING",),
                "positive_4": ("CONDITIONING",),
            }
        }

        return inputs

    RETURN_TYPES = ("X_Y",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "EasyUse/XY Inputs"

    def xy_value(self, positive_1=None, positive_2=None, positive_3=None, positive_4=None):
        axis = "advanced: Pos Condition"
        values = []
        cond = []
        # Create base entry
        if positive_1 is not None:
            values.append("0")
            cond.append(positive_1)
        if positive_2 is not None:
            values.append("1")
            cond.append(positive_2)
        if positive_3 is not None:
            values.append("2")
            cond.append(positive_3)
        if positive_4 is not None:
            values.append("3")
            cond.append(positive_4)

        return ({"axis": axis, "values": values, "cond": cond},) if values is not None else (None,)

# XYPlot Neg Condition
class XYplot_Negative_Cond:

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "optional": {
                "negative_1": ("CONDITIONING",),
                "negative_2": ("CONDITIONING",),
                "negative_3": ("CONDITIONING",),
                "negative_4": ("CONDITIONING",),
            }
        }

        return inputs

    RETURN_TYPES = ("X_Y",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "EasyUse/XY Inputs"

    def xy_value(self, negative_1=None, negative_2=None, negative_3=None, negative_4=None):
        axis = "advanced: Neg Condition"
        values = []
        cond = []
        # Create base entry
        if negative_1 is not None:
            values.append(0)
            cond.append(negative_1)
        if negative_2 is not None:
            values.append(1)
            cond.append(negative_2)
        if negative_3 is not None:
            values.append(2)
            cond.append(negative_3)
        if negative_4 is not None:
            values.append(3)
            cond.append(negative_4)

        return ({"axis": axis, "values": values, "cond": cond},) if values is not None else (None,)

# XYPlot Pos Condition List
class XYplot_Positive_Cond_List:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("X_Y",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "EasyUse/XY Inputs"

    def xy_value(self, positive):
        axis = "advanced: Pos Condition"
        values = []
        cond = []
        for index, c in enumerate(positive):
            values.append(str(index))
            cond.append(c)

        return ({"axis": axis, "values": values, "cond": cond},) if values is not None else (None,)

# XYPlot Neg Condition List
class XYplot_Negative_Cond_List:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "negative": ("CONDITIONING",),
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("X_Y",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "EasyUse/XY Inputs"

    def xy_value(self, negative):
        axis = "advanced: Neg Condition"
        values = []
        cond = []
        for index, c in enumerate(negative):
            values.append(index)
            cond.append(c)

        return ({"axis": axis, "values": values, "cond": cond},) if values is not None else (None,)

# XY Plot: ControlNet
class XYplot_Control_Net:
    parameters = ["strength", "start_percent", "end_percent"]
    @classmethod
    def INPUT_TYPES(cls):
        def get_file_list(filenames):
            return [file for file in filenames if file != "put_models_here.txt" and "lllite" not in file]

        return {
            "required": {
                "control_net_name": (get_file_list(folder_paths.get_filename_list("controlnet")),),
                "image": ("IMAGE",),
                "target_parameter": (cls.parameters,),
                "batch_count": ("INT", {"default": 3, "min": 1, "max": 30}),
                "first_strength": ("FLOAT", {"default": 0.0, "min": 0.00, "max": 10.0, "step": 0.01}),
                "last_strength": ("FLOAT", {"default": 1.0, "min": 0.00, "max": 10.0, "step": 0.01}),
                "first_start_percent": ("FLOAT", {"default": 0.0, "min": 0.00, "max": 1.0, "step": 0.01}),
                "last_start_percent": ("FLOAT", {"default": 1.0, "min": 0.00, "max": 1.0, "step": 0.01}),
                "first_end_percent": ("FLOAT", {"default": 0.0, "min": 0.00, "max": 1.0, "step": 0.01}),
                "last_end_percent": ("FLOAT", {"default": 1.0, "min": 0.00, "max": 1.0, "step": 0.01}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.00, "max": 10.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.00, "max": 1.0, "step": 0.01}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.00, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("X_Y",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "EasyUse/XY Inputs"

    def xy_value(self, control_net_name, image, target_parameter, batch_count, first_strength, last_strength, first_start_percent,
                 last_start_percent, first_end_percent, last_end_percent, strength, start_percent, end_percent):

        axis, = None,

        values = []

        if target_parameter == "strength":
            axis = "advanced: ControlNetStrength"

            values.append([(control_net_name, image, first_strength, start_percent, end_percent)])
            strength_increment = (last_strength - first_strength) / (batch_count - 1) if batch_count > 1 else 0
            for i in range(1, batch_count - 1):
                values.append([(control_net_name, image, first_strength + i * strength_increment, start_percent,
                                end_percent)])
            if batch_count > 1:
                values.append([(control_net_name, image, last_strength, start_percent, end_percent)])

        elif target_parameter == "start_percent":
            axis = "advanced: ControlNetStart%"

            percent_increment = (last_start_percent - first_start_percent) / (batch_count - 1) if batch_count > 1 else 0
            values.append([(control_net_name, image, strength, first_start_percent, end_percent)])
            for i in range(1, batch_count - 1):
                values.append([(control_net_name, image, strength, first_start_percent + i * percent_increment,
                                  end_percent)])

            # Always add the last start_percent if batch_count is more than 1.
            if batch_count > 1:
                values.append((control_net_name, image, strength, last_start_percent, end_percent))

        elif target_parameter == "end_percent":
            axis = "advanced: ControlNetEnd%"

            percent_increment = (last_end_percent - first_end_percent) / (batch_count - 1) if batch_count > 1 else 0
            values.append([(control_net_name, image, image, strength, start_percent, first_end_percent)])
            for i in range(1, batch_count - 1):
                values.append([(control_net_name, image, strength, start_percent,
                                  first_end_percent + i * percent_increment)])

            if batch_count > 1:
                values.append([(control_net_name, image, strength, start_percent, last_end_percent)])


        return ({"axis": axis, "values": values},)


#Checkpoints
class XYplot_Checkpoint:

    modes = ["Ckpt Names", "Ckpt Names+ClipSkip", "Ckpt Names+ClipSkip+VAE"]

    @classmethod
    def INPUT_TYPES(cls):

        checkpoints = ["None"] + folder_paths.get_filename_list("checkpoints")
        vaes = ["Baked VAE"] + folder_paths.get_filename_list("vae")

        inputs = {
            "required": {
                "input_mode": (cls.modes,),
                "ckpt_count": ("INT", {"default": 3, "min": 0, "max": 10, "step": 1}),
            }
        }

        for i in range(1, 10 + 1):
            inputs["required"][f"ckpt_name_{i}"] = (checkpoints,)
            inputs["required"][f"clip_skip_{i}"] = ("INT", {"default": -1, "min": -24, "max": -1, "step": 1})
            inputs["required"][f"vae_name_{i}"] = (vaes,)

        inputs["optional"] = {
            "optional_lora_stack": ("LORA_STACK",)
        }
        return inputs

    RETURN_TYPES = ("X_Y",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"

    CATEGORY = "EasyUse/XY Inputs"

    def xy_value(self, input_mode, ckpt_count, **kwargs):

        axis = "advanced: Checkpoint"

        checkpoints = [kwargs.get(f"ckpt_name_{i}") for i in range(1, ckpt_count + 1)]
        clip_skips = [kwargs.get(f"clip_skip_{i}") for i in range(1, ckpt_count + 1)]
        vaes = [kwargs.get(f"vae_name_{i}") for i in range(1, ckpt_count + 1)]

        # Set None for Clip Skip and/or VAE if not correct modes
        for i in range(ckpt_count):
            if "ClipSkip" not in input_mode:
                clip_skips[i] = 'None'
            if "VAE" not in input_mode:
                vaes[i] = 'None'

        # Extend each sub-array with lora_stack if it's not None
        values = [checkpoint.replace(',', '*')+','+str(clip_skip)+','+vae.replace(',', '*') for checkpoint, clip_skip, vae in zip(checkpoints, clip_skips, vaes) if
                        checkpoint != "None"]

        optional_lora_stack = kwargs.get("optional_lora_stack") if "optional_lora_stack" in kwargs else []

        xy_values = {"axis": axis, "values": values, "lora_stack": optional_lora_stack}
        return (xy_values,)

#Loras
class XYplot_Lora:

    modes = ["Lora Names", "Lora Names+Weights"]

    @classmethod
    def INPUT_TYPES(cls):
        loras = ["None"] + folder_paths.get_filename_list("loras")

        inputs = {
            "required": {
                "input_mode": (cls.modes,),
                "lora_count": ("INT", {"default": 3, "min": 0, "max": 10, "step": 1}),
                "model_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "clip_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            }
        }

        for i in range(1, 10 + 1):
            inputs["required"][f"lora_name_{i}"] = (loras,)
            inputs["required"][f"model_str_{i}"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})
            inputs["required"][f"clip_str_{i}"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})

        inputs["optional"] = {
            "optional_lora_stack": ("LORA_STACK",),
            "display_trigger_word": ("BOOLEAN", {"display_trigger_word": True, "tooltip": "Trigger words showing lora model pass through the model's metadata, but not necessarily accurately."}),
        }
        return inputs

    RETURN_TYPES = ("X_Y",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"

    CATEGORY = "EasyUse/XY Inputs"

    def sort_tags_by_frequency(self, meta_tags):
        if meta_tags is None:
            return []
        if "ss_tag_frequency" in meta_tags:
            meta_tags = meta_tags["ss_tag_frequency"]
            meta_tags = json.loads(meta_tags)
            sorted_tags = {}
            for _, dataset in meta_tags.items():
                for tag, count in dataset.items():
                    tag = str(tag).strip()
                    if tag in sorted_tags:
                        sorted_tags[tag] = sorted_tags[tag] + count
                    else:
                        sorted_tags[tag] = count
            # sort tags by training frequency. Most seen tags firsts
            sorted_tags = dict(sorted(sorted_tags.items(), key=lambda item: item[1], reverse=True))
            return list(sorted_tags.keys())
        else:
            return []

    def get_trigger_words(self, lora_name, display=False):
        if not display:
            return ""

        file_path = folder_paths.get_full_path('loras', lora_name)
        if not file_path:
            return ''
        header = getMetadata(file_path)
        header_json = json.loads(header)
        meta = header_json["__metadata__"] if "__metadata__" in header_json else None
        tags = self.sort_tags_by_frequency(meta)
        return ' '+ tags[0] if len(tags) > 0 else ''
    def xy_value(self, input_mode, lora_count, model_strength, clip_strength, display_trigger_words=True, **kwargs):

        axis = "advanced: Lora"
        # Extract values from kwargs
        loras = [kwargs.get(f"lora_name_{i}") for i in range(1, lora_count + 1)]
        model_strs = [kwargs.get(f"model_str_{i}", model_strength) for i in range(1, lora_count + 1)]
        clip_strs = [kwargs.get(f"clip_str_{i}", clip_strength) for i in range(1, lora_count + 1)]

        # Use model_strength and clip_strength for the loras where values are not provided
        if "Weights" not in input_mode:
            for i in range(lora_count):
                model_strs[i] = model_strength
                clip_strs[i] = clip_strength

        # Extend each sub-array with lora_stack if it's not None
        values = [lora.replace(',', '*')+','+str(model_str)+','+str(clip_str) +',' + self.get_trigger_words(lora, display_trigger_words) for lora, model_str, clip_str
                    in zip(loras, model_strs, clip_strs) if lora != "None"]

        optional_lora_stack = kwargs.get("optional_lora_stack") if "optional_lora_stack" in kwargs else []

        xy_values = {"axis": axis, "values": values, "lora_stack": optional_lora_stack}
        return (xy_values,)

# 模型叠加
class XYplot_ModelMergeBlocks:

    @classmethod
    def INPUT_TYPES(s):
        checkpoints = folder_paths.get_filename_list("checkpoints")
        vae = ["Use Model 1", "Use Model 2"] + folder_paths.get_filename_list("vae")

        preset = ["Preset"]  # 20
        preset += load_preset("mmb-preset.txt")
        preset += load_preset("mmb-preset.custom.txt")

        default_vectors = "1,0,0; \n0,1,0; \n0,0,1; \n1,1,0; \n1,0,1; \n0,1,1; "
        return {
            "required": {
                "ckpt_name_1": (checkpoints,),
                "ckpt_name_2": (checkpoints,),
                "vae_use": (vae, {"default": "Use Model 1"}),
                "preset": (preset, {"default": "preset"}),
                "values": ("STRING", {"default": default_vectors, "multiline": True, "placeholder": 'Support 2 methods:\n\n1.input, middle, out in same line and insert values seperated by "; "\n\n2.model merge block number seperated by ", " in same line and insert values seperated by "; "'}),
            },
            "hidden": {"my_unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("X_Y",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"

    CATEGORY = "EasyUse/XY Inputs"

    def xy_value(self, ckpt_name_1, ckpt_name_2, vae_use, preset, values, my_unique_id=None):

        axis = "advanced: ModelMergeBlocks"
        if ckpt_name_1 is None:
            raise Exception("ckpt_name_1 is not found")
        if ckpt_name_2 is None:
            raise Exception("ckpt_name_2 is not found")

        models = (ckpt_name_1, ckpt_name_2)

        xy_values = {"axis":axis, "values":values, "models":models, "vae_use": vae_use}
        return (xy_values,)


NODE_CLASS_MAPPINGS = {
    "easy XYInputs: Seeds++ Batch": XYplot_SeedsBatch,
    "easy XYInputs: Steps": XYplot_Steps,
    "easy XYInputs: CFG Scale": XYplot_CFG,
    "easy XYInputs: FluxGuidance": XYplot_FluxGuidance,
    "easy XYInputs: Sampler/Scheduler": XYplot_Sampler_Scheduler,
    "easy XYInputs: Denoise": XYplot_Denoise,
    "easy XYInputs: Checkpoint": XYplot_Checkpoint,
    "easy XYInputs: Lora": XYplot_Lora,
    "easy XYInputs: ModelMergeBlocks": XYplot_ModelMergeBlocks,
    "easy XYInputs: PromptSR": XYplot_PromptSR,
    "easy XYInputs: ControlNet": XYplot_Control_Net,
    "easy XYInputs: PositiveCond": XYplot_Positive_Cond,
    "easy XYInputs: PositiveCondList": XYplot_Positive_Cond_List,
    "easy XYInputs: NegativeCond": XYplot_Negative_Cond,
    "easy XYInputs: NegativeCondList": XYplot_Negative_Cond_List,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "easy XYInputs: Seeds++ Batch": "XY Inputs: Seeds++ Batch //EasyUse",
    "easy XYInputs: Steps": "XY Inputs: Steps //EasyUse",
    "easy XYInputs: CFG Scale": "XY Inputs: CFG Scale //EasyUse",
    "easy XYInputs: FluxGuidance": "XY Inputs: Flux Guidance //EasyUse",
    "easy XYInputs: Sampler/Scheduler": "XY Inputs: Sampler/Scheduler //EasyUse",
    "easy XYInputs: Denoise": "XY Inputs: Denoise //EasyUse",
    "easy XYInputs: Checkpoint": "XY Inputs: Checkpoint //EasyUse",
    "easy XYInputs: Lora": "XY Inputs: Lora //EasyUse",
    "easy XYInputs: ModelMergeBlocks": "XY Inputs: ModelMergeBlocks //EasyUse",
    "easy XYInputs: PromptSR": "XY Inputs: PromptSR //EasyUse",
    "easy XYInputs: ControlNet": "XY Inputs: Controlnet //EasyUse",
    "easy XYInputs: PositiveCond": "XY Inputs: PosCond //EasyUse",
    "easy XYInputs: PositiveCondList": "XY Inputs: PosCondList //EasyUse",
    "easy XYInputs: NegativeCond": "XY Inputs: NegCond //EasyUse",
    "easy XYInputs: NegativeCondList": "XY Inputs: NegCondList //EasyUse",
}