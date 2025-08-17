from impact.utils import any_typ, ByPassTypeTuple, make_3d_mask
import comfy_extras.nodes_mask
from nodes import MAX_RESOLUTION
import torch
import comfy
import sys
import nodes
import re
import impact.core as core
from server import PromptServer
import inspect
import logging


class GeneralSwitch:
    @classmethod
    def INPUT_TYPES(s):
        dyn_inputs = {"input1": (any_typ, {"lazy": True, "tooltip": "Any input. When connected, one more input slot is added."}), }
        if core.is_execution_model_version_supported():
            stack = inspect.stack()
            if stack[2].function == 'get_input_info':
                # bypass validation
                class AllContainer:
                    def __contains__(self, item):
                        return True

                    def __getitem__(self, key):
                        return any_typ, {"lazy": True}

                dyn_inputs = AllContainer()

        inputs = {"required": {
                    "select": ("INT", {"default": 1, "min": 1, "max": 999999, "step": 1, "tooltip": "The input number you want to output among the inputs"}),
                    "sel_mode": ("BOOLEAN", {"default": False, "label_on": "select_on_prompt", "label_off": "select_on_execution", "forceInput": False,
                                             "tooltip": "In the case of 'select_on_execution', the selection is dynamically determined at the time of workflow execution. 'select_on_prompt' is an option that exists for older versions of ComfyUI, and it makes the decision before the workflow execution."}),
                    },
                "optional": dyn_inputs,
                "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO"}
                }

        return inputs

    RETURN_TYPES = (any_typ, "STRING", "INT")
    RETURN_NAMES = ("selected_value", "selected_label", "selected_index")
    OUTPUT_TOOLTIPS = ("Output is generated only from the input chosen by the 'select' value.", "Slot label of the selected input slot", "Outputs the select value as is")
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def check_lazy_status(self, *args, **kwargs):
        selected_index = int(kwargs['select'])
        input_name = f"input{selected_index}"

        logging.info(f"SELECTED: {input_name}")

        if input_name in kwargs:
            return [input_name]
        else:
            return []

    @staticmethod
    def doit(*args, **kwargs):
        selected_index = int(kwargs['select'])
        input_name = f"input{selected_index}"

        selected_label = input_name
        node_id = kwargs['unique_id']

        if 'extra_pnginfo' in kwargs and kwargs['extra_pnginfo'] is not None:
            nodelist = kwargs['extra_pnginfo']['workflow']['nodes']
            for node in nodelist:
                if str(node['id']) == node_id:
                    inputs = node['inputs']

                    for slot in inputs:
                        if slot['name'] == input_name and 'label' in slot:
                            selected_label = slot['label']

                    break
        else:
            logging.info("[Impact-Pack] The switch node does not guarantee proper functioning in API mode.")

        if input_name in kwargs:
            return kwargs[input_name], selected_label, selected_index
        else:
            logging.info("ImpactSwitch: invalid select index (ignored)")
            return None, "", selected_index

class LatentSwitch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "select": ("INT", {"default": 1, "min": 1, "max": 99999, "step": 1}),
                    "latent1": ("LATENT",),
                    },
                }

    RETURN_TYPES = ("LATENT", )

    OUTPUT_NODE = True

    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, *args, **kwargs):
        input_name = f"latent{int(kwargs['select'])}"

        if input_name in kwargs:
            return (kwargs[input_name],)
        else:
            logging.info("LatentSwitch: invalid select index ('latent1' is selected)")
            return (kwargs['latent1'],)


class ImageMaskSwitch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "select": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
            "images1": ("IMAGE",),
        },

            "optional": {
                "mask1_opt": ("MASK",),
                "images2_opt": ("IMAGE",),
                "mask2_opt": ("MASK",),
                "images3_opt": ("IMAGE",),
                "mask3_opt": ("MASK",),
                "images4_opt": ("IMAGE",),
                "mask4_opt": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK",)

    OUTPUT_NODE = True

    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, select, images1, mask1_opt=None, images2_opt=None, mask2_opt=None, images3_opt=None, mask3_opt=None,
             images4_opt=None, mask4_opt=None):
        if select == 1:
            return images1, mask1_opt,
        elif select == 2:
            return images2_opt, mask2_opt,
        elif select == 3:
            return images3_opt, mask3_opt,
        else:
            return images4_opt, mask4_opt,


class GeneralInversedSwitch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "select": ("INT", {"default": 1, "min": 1, "max": 999999, "step": 1, "tooltip": "The output number you want to send from the input"}),
                    "input": (any_typ, {"tooltip": "Any input. When connected, one more input slot is added."}),

                    },
                "optional": {
                    "sel_mode": ("BOOLEAN", {"default": False, "label_on": "select_on_prompt", "label_off": "select_on_execution", "forceInput": False,
                                             "tooltip": "In the case of 'select_on_execution', the selection is dynamically determined at the time of workflow execution. 'select_on_prompt' is an option that exists for older versions of ComfyUI, and it makes the decision before the workflow execution."}),
                    },
                "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
                }

    RETURN_TYPES = ByPassTypeTuple((any_typ, ))
    OUTPUT_TOOLTIPS = ("Output occurs only from the output selected by the 'select' value.\nWhen slots are connected, additional slots are created.", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, select, prompt, unique_id, input, **kwargs):
        if core.is_execution_model_version_supported():
            from comfy_execution.graph import ExecutionBlocker
        else:
            logging.warning("[Impact Pack] InversedSwitch: ComfyUI is outdated. The 'select_on_execution' mode cannot function properly.")

        res = []

        # search max output count in prompt
        cnt = 0
        for x in prompt.values():
            for y in x.get('inputs', {}).values():
                if isinstance(y, list) and len(y) == 2:
                    if y[0] == unique_id:
                        cnt = max(cnt, y[1])

        for i in range(0, cnt + 1):
            if select == i+1:
                res.append(input)
            elif core.is_execution_model_version_supported():
                res.append(ExecutionBlocker(None))
            else:
                res.append(None)

        return res


class RemoveNoiseMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"samples": ("LATENT",)}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, samples):
        res = {key: value for key, value in samples.items() if key != 'noise_mask'}
        return (res, )


class ImagePasteMasked:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "destination": ("IMAGE",),
                "source": ("IMAGE",),
                "x": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "y": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "resize_source": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "composite"

    CATEGORY = "image"

    def composite(self, destination, source, x, y, resize_source, mask = None):
        destination = destination.clone().movedim(-1, 1)
        output = comfy_extras.nodes_mask.composite(destination, source.movedim(-1, 1), x, y, mask, 1, resize_source).movedim(1, -1)
        return (output,)


from impact.utils import any_typ

class ImpactLogger:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "data": (any_typ,),
                        "text": ("STRING", {"multiline": True}),
                    },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID"},
                }

    CATEGORY = "ImpactPack/Debug"

    OUTPUT_NODE = True

    RETURN_TYPES = ()
    FUNCTION = "doit"

    def doit(self, data, text, prompt, extra_pnginfo, unique_id):
        shape = ""
        if hasattr(data, "shape"):
            shape = f"{data.shape} / "

        logging.info(f"[IMPACT LOGGER]: {shape}{data}")

        logging.info(f"         PROMPT: {prompt}")

        # for x in prompt:
        #     if 'inputs' in x and 'populated_text' in x['inputs']:
        #         print(f"PROMPT: {x['10']['inputs']['populated_text']}")
        #
        # for x in extra_pnginfo['workflow']['nodes']:
        #     if x['type'] == 'ImpactWildcardProcessor':
        #         print(f" WV : {x['widgets_values'][1]}\n")

        PromptServer.instance.send_sync("impact-node-feedback", {"node_id": unique_id, "widget_name": "text", "type": "TEXT", "value": f"{data}"})
        return {}


class ImpactDummyInput:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}

    CATEGORY = "ImpactPack/Debug"

    RETURN_TYPES = (any_typ,)
    FUNCTION = "doit"

    def doit(self):
        return ("DUMMY",)


class MasksToMaskList:
    @classmethod
    def INPUT_TYPES(s):
        return {"optional": {
                        "masks": ("MASK", ),
                      }
                }

    RETURN_TYPES = ("MASK", )
    OUTPUT_IS_LIST = (True, )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Operation"

    def doit(self, masks):
        if masks is None:
            empty_mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            return ([empty_mask], )

        res = []

        for mask in masks:
            res.append(mask)

        res = [make_3d_mask(x) for x in res]

        return (res, )


class MaskListToMaskBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "mask": ("MASK", ),
                      }
                }

    INPUT_IS_LIST = True

    RETURN_TYPES = ("MASK", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Operation"

    def doit(self, mask):
        if len(mask) == 1:
            mask = make_3d_mask(mask[0])
            return (mask,)
        elif len(mask) > 1:
            mask1 = make_3d_mask(mask[0])

            for mask2 in mask[1:]:
                mask2 = make_3d_mask(mask2)
                if mask1.shape[1:] != mask2.shape[1:]:
                    mask2 = comfy.utils.common_upscale(mask2.movedim(-1, 1), mask1.shape[2], mask1.shape[1], "lanczos", "center").movedim(1, -1)
                mask1 = torch.cat((mask1, mask2), dim=0)

            return (mask1,)
        else:
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32, device="cpu").unsqueeze(0)
            return (empty_mask,)


class ImageListToImageBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "images": ("IMAGE", ),
                      }
                }

    INPUT_IS_LIST = True

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Operation"

    def doit(self, images):
        if len(images) <= 1:
            return (images[0],)
        else:
            image1 = images[0]
            for image2 in images[1:]:
                if image1.shape[1:] != image2.shape[1:]:
                    image2 = comfy.utils.common_upscale(image2.movedim(-1, 1), image1.shape[2], image1.shape[1], "lanczos", "center").movedim(1, -1)
                image1 = torch.cat((image1, image2), dim=0)
            return (image1,)


class ImageBatchToImageList:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",), }}

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, image):
        images = [image[i:i + 1, ...] for i in range(image.shape[0])]
        return (images, )


class MakeAnyList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {"value1": (any_typ,), }
        }

    RETURN_TYPES = (any_typ,)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, **kwargs):
        values = []

        for k, v in kwargs.items():
            if v is not None:
                values.append(v)

        return (values, )


class MakeMaskList:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"mask1": ("MASK",), }}

    RETURN_TYPES = ("MASK",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, **kwargs):
        masks = []

        for k, v in kwargs.items():
            masks.append(v)

        return (masks, )


class NthItemOfAnyList:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":  {
                    "any_list": (any_typ,),
                    "index": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1, "tooltip": "The index of the item you want to select from the list."}),
                    }
        }

    RETURN_TYPES = (any_typ,)
    INPUT_IS_LIST = True
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    DESCRIPTION = "Selects the Nth item from a list. If the index is out of range, it returns the last item in the list."

    def doit(self, any_list, index):
        i = index[0]
        if i >= len(any_list):
            return (any_list[-1],)
        else:
            return (any_list[i],)


class MakeImageList:
    @classmethod
    def INPUT_TYPES(s):
        return {"optional": {"image1": ("IMAGE",), }}

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, **kwargs):
        images = []

        for k, v in kwargs.items():
            images.append(v)

        return (images, )


class MakeImageBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"optional": {"image1": ("IMAGE",), }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, **kwargs):
        images = [value for value in kwargs.values()]

        if len(images) == 1:
            return (images[0],)
        else:
            image1 = images[0]
            for image2 in images[1:]:
                if image1.shape[1:] != image2.shape[1:]:
                    image2 = comfy.utils.common_upscale(image2.movedim(-1, 1), image1.shape[2], image1.shape[1], "lanczos", "center").movedim(1, -1)
                image1 = torch.cat((image1, image2), dim=0)
            return (image1,)


class MakeMaskBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"optional": {"mask1": ("MASK",), }}

    RETURN_TYPES = ("MASK",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, **kwargs):
        masks = [make_3d_mask(value) for value in kwargs.values()]

        if len(masks) == 1:
            return (masks[0],)
        else:
            mask1 = masks[0]
            for mask2 in masks[1:]:
                if mask1.shape[1:] != mask2.shape[1:]:
                    mask2 = comfy.utils.common_upscale(mask2.movedim(-1, 1), mask1.shape[2], mask1.shape[1], "lanczos", "center").movedim(1, -1)
                mask1 = torch.cat((mask1, mask2), dim=0)
            return (mask1,)


class ReencodeLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "samples": ("LATENT", ),
                        "tile_mode": (["None", "Both", "Decode(input) only", "Encode(output) only"],),
                        "input_vae": ("VAE", ),
                        "output_vae": ("VAE", ),
                        "tile_size": ("INT", {"default": 512, "min": 320, "max": 4096, "step": 64}),
                    },
                "optional": {
                    "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32, "tooltip": "This setting applies when 'tile_mode' is enabled."}),
                    }
                }

    CATEGORY = "ImpactPack/Util"

    RETURN_TYPES = ("LATENT", )
    FUNCTION = "doit"

    def doit(self, samples, tile_mode, input_vae, output_vae, tile_size=512, overlap=64):
        if tile_mode in ["Both", "Decode(input) only"]:
            decoder = nodes.VAEDecodeTiled()
            if 'overlap' in inspect.signature(decoder.decode).parameters:
                pixels = decoder.decode(input_vae, samples, tile_size, overlap=overlap)[0]
            else:
                pixels = decoder.decode(input_vae, samples, tile_size, overlap=overlap)[0]
        else:
            pixels = nodes.VAEDecode().decode(input_vae, samples)[0]

        if tile_mode in ["Both", "Encode(output) only"]:
            encoder = nodes.VAEEncodeTiled()
            if 'overlap' in inspect.signature(encoder.encode).parameters:
                return encoder.encode(output_vae, pixels, tile_size, overlap=overlap)
            else:
                return encoder.encode(output_vae, pixels, tile_size)
        else:
            return nodes.VAEEncode().encode(output_vae, pixels)


class ReencodeLatentPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "samples": ("LATENT", ),
                        "tile_mode": (["None", "Both", "Decode(input) only", "Encode(output) only"],),
                        "input_basic_pipe": ("BASIC_PIPE", ),
                        "output_basic_pipe": ("BASIC_PIPE", ),
                    },
                }

    CATEGORY = "ImpactPack/Util"

    RETURN_TYPES = ("LATENT", )
    FUNCTION = "doit"

    def doit(self, samples, tile_mode, input_basic_pipe, output_basic_pipe):
        _, _, input_vae, _, _ = input_basic_pipe
        _, _, output_vae, _, _ = output_basic_pipe
        return ReencodeLatent().doit(samples, tile_mode, input_vae, output_vae)


class StringSelector:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "strings": ("STRING", {"multiline": True}),
            "multiline": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
            "select": ("INT", {"min": 0, "max": sys.maxsize, "step": 1, "default": 0}),
        }}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, strings, multiline, select):
        lines = strings.split('\n')

        if multiline:
            result = []
            current_string = ""

            for line in lines:
                if line.startswith("#"):
                    if current_string:
                        result.append(current_string.strip())
                        current_string = ""
                current_string += line + "\n"

            if current_string:
                result.append(current_string.strip())

            if len(result) == 0:
                selected = strings
            else:
                selected = result[select % len(result)]

            if selected.startswith('#'):
                selected = selected[1:]
        else:
            if len(lines) == 0:
                selected = strings
            else:
                selected = lines[select % len(lines)]

        return (selected, )


class StringListToString:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "join_with": ("STRING", {"default": "\\n"}),
                "string_list": ("STRING", {"forceInput": True}),
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, join_with, string_list):
        # convert \\n to newline character
        if join_with[0] == "\\n":
            join_with[0] = "\n"

        joined_text = join_with[0].join(string_list)

        return (joined_text,)


class WildcardPromptFromString:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "string": ("STRING", {"forceInput": True}),
                "delimiter": ("STRING", {"multiline": False, "default": "\\n" }),
                "prefix_all": ("STRING", {"multiline": False}),
                "postfix_all": ("STRING", {"multiline": False}),
                "restrict_to_tags": ("STRING", {"multiline": False}),
                "exclude_tags": ("STRING", {"multiline": False})
            },
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("wildcard", "segs_labels",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, string, delimiter, prefix_all, postfix_all, restrict_to_tags, exclude_tags):
        # convert \\n to newline character
        if delimiter == "\\n":
            delimiter = "\n"

        # some sanity checks and normalization for later processing
        if prefix_all is None:
            prefix_all = ""
        if postfix_all is None:
            postfix_all = ""
        if restrict_to_tags is None:
            restrict_to_tags = ""
        if exclude_tags is None:
            exclude_tags = ""

        restrict_to_tags = restrict_to_tags.split(", ")
        exclude_tags = exclude_tags.split(", ")

        # build the wildcard prompt per list entry
        output = ["[LAB]"]
        labels = []
        for x in string.split(delimiter):
            label = str(len(labels) + 1)
            labels.append(label)
            x = x.split(", ")
            # restrict to tags
            if restrict_to_tags != [""]:
                x = list(set(x) & set(restrict_to_tags))
            # remove tags
            if exclude_tags != [""]:
                x = list(set(x) - set(exclude_tags))
            # next row: <LABEL> <PREFIX> <TAGS> <POSTFIX>
            prompt_for_seg = f'[{label}] {prefix_all} {", ".join(x)} {postfix_all}'.strip()
            output.append(prompt_for_seg)
        output = "\n".join(output)

        # clean string: fixup double spaces, commas etc.
        output = re.sub(r' ,', ',', output)
        output = re.sub(r'  +', ' ', output)
        output = re.sub(r',,+', ',', output)
        output = re.sub(r'\n, ', '\n', output)

        return output, ", ".join(labels)
