import sys
from . import hooks
from . import defs


class SEGSOrderedFilterDetailerHookProvider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "target": (["area(=w*h)", "width", "height", "x1", "y1", "x2", "y2"],),
                        "order": ("BOOLEAN", {"default": True, "label_on": "descending", "label_off": "ascending"}),
                        "take_start": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
                        "take_count": ("INT", {"default": 1, "min": 0, "max": sys.maxsize, "step": 1}),
                     },
                }

    RETURN_TYPES = ("DETAILER_HOOK", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, target, order, take_start, take_count):
        hook = hooks.SEGSOrderedFilterDetailerHook(target, order, take_start, take_count)
        return (hook, )


class SEGSRangeFilterDetailerHookProvider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "target": (["area(=w*h)", "width", "height", "x1", "y1", "x2", "y2", "length_percent"],),
                        "mode": ("BOOLEAN", {"default": True, "label_on": "inside", "label_off": "outside"}),
                        "min_value": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
                        "max_value": ("INT", {"default": 67108864, "min": 0, "max": sys.maxsize, "step": 1}),
                     },
                }

    RETURN_TYPES = ("DETAILER_HOOK", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, target, mode, min_value, max_value):
        hook = hooks.SEGSRangeFilterDetailerHook(target, mode, min_value, max_value)
        return (hook, )


class SEGSLabelFilterDetailerHookProvider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "segs": ("SEGS", ),
                        "preset": (['all'] + defs.detection_labels,),
                        "labels": ("STRING", {"multiline": True, "placeholder": "List the types of segments to be allowed, separated by commas"}),
                     },
                }

    RETURN_TYPES = ("DETAILER_HOOK", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, preset, labels):
        hook = hooks.SEGSLabelFilterDetailerHook(labels)
        return (hook, )


class PreviewDetailerHookProvider:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"quality": ("INT", {"default": 95, "min": 20, "max": 100})},
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("DETAILER_HOOK", "UPSCALER_HOOK")
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    NOT_IDEMPOTENT = True

    def doit(self, quality, unique_id):
        hook = hooks.PreviewDetailerHook(unique_id, quality)
        return hook, hook


class LamaRemoverDetailerHookProvider:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask_threshold":("INT", {"default": 250, "min": 0, "max": 255, "step": 1, "display": "slider"}),
                "gaussblur_radius": ("INT", {"default": 8, "min": 0, "max": 20, "step": 1, "display": "slider"}),
                "skip_sampling": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("DETAILER_HOOK", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, mask_threshold, gaussblur_radius, skip_sampling):
        hook = hooks.LamaRemoverDetailerHook(mask_threshold, gaussblur_radius, skip_sampling)
        return (hook, )


class BlackPatchRetryHookProvider:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mean_thresh": ("INT", {"default": 10, "min": 0, "max": 255}),
                "var_thresh": ("INT", {"default": 5, "min": 0, "max": 255})
            },
        }

    RETURN_TYPES = ("DETAILER_HOOK", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    NOT_IDEMPOTENT = True

    def doit(self, mean_thresh, var_thresh):
        hook = hooks.BlackPatchRetryHook(mean_thresh, var_thresh)
        return hook, 
