import folder_paths
import comfy.controlnet
import comfy.model_management
from nodes import NODE_CLASS_MAPPINGS

union_controlnet_types = {"auto": -1, "openpose": 0, "depth": 1, "hed/pidi/scribble/ted": 2, "canny/lineart/anime_lineart/mlsd": 3, "normal": 4, "segment": 5, "tile": 6, "repaint": 7}

class easyControlnet:
    def __init__(self):
        pass

    def apply(self, control_net_name, image, positive, negative, strength, start_percent=0, end_percent=1, control_net=None, scale_soft_weights=1, mask=None, union_type=None, easyCache=None, use_cache=True, model=None, vae=None):
        if strength == 0:
            return (positive, negative)

        # kolors controlnet patch
        from ..modules.kolors.loader import is_kolors_model, applyKolorsUnet
        if is_kolors_model(model):
            from ..modules.kolors.model_patch import patch_controlnet
            if control_net is None:
                with applyKolorsUnet():
                    control_net = easyCache.load_controlnet(control_net_name, scale_soft_weights, use_cache)
                control_net = patch_controlnet(model, control_net)
        else:
            if control_net is None:
                if easyCache is not None:
                    control_net = easyCache.load_controlnet(control_net_name, scale_soft_weights, use_cache)
                else:
                    controlnet_path = folder_paths.get_full_path("controlnet", control_net_name)
                    control_net = comfy.controlnet.load_controlnet(controlnet_path)

        # union controlnet
        if union_type is not None:
            control_net = control_net.copy()
            type_number = union_controlnet_types[union_type]
            if type_number >= 0:
                control_net.set_extra_arg("control_type", [type_number])
            else:
                control_net.set_extra_arg("control_type", [])

        if mask is not None:
            mask = mask.to(self.device)

        if mask is not None and len(mask.shape) < 3:
            mask = mask.unsqueeze(0)

        control_hint = image.movedim(-1, 1)

        is_cond = True
        if negative is None:
            p = []
            for t in positive:
                n = [t[0], t[1].copy()]
                c_net = control_net.copy().set_cond_hint(control_hint, strength, (start_percent, end_percent))
                if 'control' in t[1]:
                    c_net.set_previous_controlnet(t[1]['control'])
                n[1]['control'] = c_net
                n[1]['control_apply_to_uncond'] = True
                if mask is not None:
                    n[1]['mask'] = mask
                    n[1]['set_area_to_bounds'] = False
                p.append(n)
            positive = p
        else:
            cnets = {}
            out = []
            for conditioning in [positive, negative]:
                c = []
                for t in conditioning:
                    d = t[1].copy()

                    prev_cnet = d.get('control', None)
                    if prev_cnet in cnets:
                        c_net = cnets[prev_cnet]
                    else:
                        c_net = control_net.copy().set_cond_hint(control_hint, strength, (start_percent, end_percent), vae)
                        c_net.set_previous_controlnet(prev_cnet)
                        cnets[prev_cnet] = c_net

                    d['control'] = c_net
                    d['control_apply_to_uncond'] = False

                    if mask is not None:
                        d['mask'] = mask
                        d['set_area_to_bounds'] = False

                    n = [t[0], d]
                    c.append(n)
                out.append(c)
            positive = out[0]
            negative = out[1]

        return (positive, negative)