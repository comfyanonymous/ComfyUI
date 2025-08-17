from nodes import MAX_RESOLUTION
import impact.core as core
from impact.core import SEG
from impact.segs_nodes import SEGSPaste
import comfy
from impact import utils
import torch
import nodes
import logging

try:
    from comfy_extras import nodes_differential_diffusion
except Exception:
    logging.warning("\n#############################################\n[Impact Pack] ComfyUI is an outdated version.\n#############################################\n")
    raise Exception("[Impact Pack] ComfyUI is an outdated version.")


class SEGSDetailerForAnimateDiff:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                     "image_frames": ("IMAGE", ),
                     "segs": ("SEGS", ),
                     "guide_size": ("FLOAT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                     "guide_size_for": ("BOOLEAN", {"default": True, "label_on": "bbox", "label_off": "crop_region"}),
                     "max_size": ("FLOAT", {"default": 768, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                     "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                     "scheduler": (core.SCHEDULERS,),
                     "denoise": ("FLOAT", {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01}),
                     "basic_pipe": ("BASIC_PIPE", {"tooltip": "If the `ImpactDummyInput` is connected to the model in the basic_pipe, the inference stage is skipped."}),
                     "refiner_ratio": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0}),
                     },
                "optional": {
                     "refiner_basic_pipe_opt": ("BASIC_PIPE",),
                     "noise_mask_feather": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1}),
                     "scheduler_func_opt": ("SCHEDULER_FUNC",),
                     }
                }

    RETURN_TYPES = ("SEGS", "IMAGE")
    RETURN_NAMES = ("segs", "cnet_images")
    OUTPUT_IS_LIST = (False, True)

    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Detailer"

    DESCRIPTION = "This node enhances details by inpainting each region within the detected area bundle (SEGS) after enlarging them based on the guide size.\nThis node is applied specifically to SEGS rather than the entire image. To apply it to the entire image, use the 'SEGS Paste' node.\nAs a specialized detailer node for improving video details, such as in AnimateDiff, this node can handle cases where the masks contained in SEGS serve as batch masks spanning multiple frames."

    @staticmethod
    def do_detail(image_frames, segs, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name, scheduler,
                  denoise, basic_pipe, refiner_ratio=None, refiner_basic_pipe_opt=None, noise_mask_feather=0, scheduler_func_opt=None):

        model, clip, vae, positive, negative = basic_pipe
        if refiner_basic_pipe_opt is None:
            refiner_model, refiner_clip, refiner_positive, refiner_negative = None, None, None, None
        else:
            refiner_model, refiner_clip, _, refiner_positive, refiner_negative = refiner_basic_pipe_opt

        segs = core.segs_scale_match(segs, image_frames.shape)

        new_segs = []
        cnet_image_list = []

        if not (isinstance(model, str) and model == "DUMMY") and noise_mask_feather > 0 and 'denoise_mask_function' not in model.model_options:
            model = nodes_differential_diffusion.DifferentialDiffusion().apply(model)[0]

        for seg in segs[1]:
            cropped_image_frames = None

            for image in image_frames:
                image = image.unsqueeze(0)
                cropped_image = seg.cropped_image if seg.cropped_image is not None else utils.crop_tensor4(image, seg.crop_region)
                cropped_image = utils.to_tensor(cropped_image)
                if cropped_image_frames is None:
                    cropped_image_frames = cropped_image
                else:
                    cropped_image_frames = torch.concat((cropped_image_frames, cropped_image), dim=0)

            cropped_image_frames = cropped_image_frames.cpu().numpy()

            # It is assumed that AnimateDiff does not support conditioning masks based on test results, but it will be added for future consideration.
            cropped_positive = [
                [condition, {
                    k: core.crop_condition_mask(v, cropped_image_frames, seg.crop_region) if k == "mask" else v
                    for k, v in details.items()
                }]
                for condition, details in positive
            ]

            cropped_negative = [
                [condition, {
                    k: core.crop_condition_mask(v, cropped_image_frames, seg.crop_region) if k == "mask" else v
                    for k, v in details.items()
                }]
                for condition, details in negative
            ]

            if not (isinstance(model, str) and model == "DUMMY"):
                enhanced_image_tensor, cnet_images = core.enhance_detail_for_animatediff(cropped_image_frames, model, clip, vae, guide_size, guide_size_for, max_size,
                                                                                         seg.bbox, seed, steps, cfg, sampler_name, scheduler,
                                                                                         cropped_positive, cropped_negative, denoise, seg.cropped_mask,
                                                                                         refiner_ratio=refiner_ratio, refiner_model=refiner_model,
                                                                                         refiner_clip=refiner_clip, refiner_positive=refiner_positive,
                                                                                         refiner_negative=refiner_negative, control_net_wrapper=seg.control_net_wrapper,
                                                                                         noise_mask_feather=noise_mask_feather, scheduler_func=scheduler_func_opt)
            else:
                enhanced_image_tensor = cropped_image_frames
                cnet_images = None

            if cnet_images is not None:
                cnet_image_list.extend(cnet_images)

            if enhanced_image_tensor is None:
                new_cropped_image = cropped_image_frames
            else:
                new_cropped_image = enhanced_image_tensor.cpu().numpy()

            new_seg = SEG(new_cropped_image, seg.cropped_mask, seg.confidence, seg.crop_region, seg.bbox, seg.label, None)
            new_segs.append(new_seg)

        return (segs[0], new_segs), cnet_image_list

    def doit(self, image_frames, segs, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name, scheduler,
             denoise, basic_pipe, refiner_ratio=None, refiner_basic_pipe_opt=None, inpaint_model=False, noise_mask_feather=0, scheduler_func_opt=None):

        segs, cnet_images = SEGSDetailerForAnimateDiff.do_detail(image_frames, segs, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name,
                                                                 scheduler, denoise, basic_pipe, refiner_ratio, refiner_basic_pipe_opt,
                                                                 noise_mask_feather=noise_mask_feather, scheduler_func_opt=scheduler_func_opt)

        if len(cnet_images) == 0:
            cnet_images = [utils.empty_pil_tensor()]

        return (segs, cnet_images)


class DetailerForEachPipeForAnimateDiff:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                      "image_frames": ("IMAGE", ),
                      "segs": ("SEGS", ),
                      "guide_size": ("FLOAT", {"default": 512, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 8}),
                      "guide_size_for": ("BOOLEAN", {"default": True, "label_on": "bbox", "label_off": "crop_region"}),
                      "max_size": ("FLOAT", {"default": 1024, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 8}),
                      "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                      "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                      "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                      "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                      "scheduler": (core.SCHEDULERS,),
                      "denoise": ("FLOAT", {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01}),
                      "feather": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
                      "basic_pipe": ("BASIC_PIPE", {"tooltip": "If the `ImpactDummyInput` is connected to the model in the basic_pipe, the inference stage is skipped."}),
                      "refiner_ratio": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0}),
                      },
                "optional": {
                      "detailer_hook": ("DETAILER_HOOK",),
                      "refiner_basic_pipe_opt": ("BASIC_PIPE",),
                      "noise_mask_feather": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1}),
                      "scheduler_func_opt": ("SCHEDULER_FUNC",),
                      }
                }

    RETURN_TYPES = ("IMAGE", "SEGS", "BASIC_PIPE", "IMAGE")
    RETURN_NAMES = ("image", "segs", "basic_pipe", "cnet_images")
    OUTPUT_IS_LIST = (False, False, False, True)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Detailer"

    DESCRIPTION = "This node enhances details by inpainting each region within the detected area bundle (SEGS) after enlarging them based on the guide size.\nThis node is a specialized detailer node for enhancing video details, such as in AnimateDiff. It can handle cases where the masks contained in SEGS serve as batch masks spanning multiple frames."

    @staticmethod
    def doit(image_frames, segs, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name, scheduler,
             denoise, feather, basic_pipe, refiner_ratio=None, detailer_hook=None, refiner_basic_pipe_opt=None,
             noise_mask_feather=0, scheduler_func_opt=None):

        enhanced_segs = []
        cnet_image_list = []

        for sub_seg in segs[1]:
            single_seg = segs[0], [sub_seg]
            enhanced_seg, cnet_images = SEGSDetailerForAnimateDiff().do_detail(image_frames, single_seg, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name, scheduler,
                                                                               denoise, basic_pipe, refiner_ratio, refiner_basic_pipe_opt, noise_mask_feather, scheduler_func_opt=scheduler_func_opt)

            image_frames = SEGSPaste.doit(image_frames, enhanced_seg, feather, alpha=255)[0]

            if cnet_images is not None:
                cnet_image_list.extend(cnet_images)

            if detailer_hook is not None:
                image_frames = detailer_hook.post_paste(image_frames)

            enhanced_segs += enhanced_seg[1]

        new_segs = segs[0], enhanced_segs
        return image_frames, new_segs, basic_pipe, cnet_image_list
