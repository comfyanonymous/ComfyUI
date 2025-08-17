import torch
from torch.nn import Linear
from types import MethodType
import comfy.model_management
import comfy.samplers
from comfy.cldm.cldm import ControlNet
from comfy.controlnet import ControlLora

def patch_controlnet(model, control_net):
    import comfy.controlnet
    if isinstance(control_net, ControlLora):
        del_keys = []
        for k in control_net.control_weights:
            if k.startswith("label_emb.0.0."):
                del_keys.append(k)

        for k in del_keys:
            control_net.control_weights.pop(k)

        super_pre_run = ControlLora.pre_run
        super_copy = ControlLora.copy

        super_forward = ControlNet.forward

        def KolorsControlNet_forward(self, x, hint, timesteps, context, **kwargs):
            with torch.cuda.amp.autocast(enabled=True):
                context = model.model.diffusion_model.encoder_hid_proj(context)
                return super_forward(self, x, hint, timesteps, context, **kwargs)

        def KolorsControlLora_pre_run(self, *args, **kwargs):
            result = super_pre_run(self, *args, **kwargs)

            if hasattr(self, "control_model"):
                self.control_model.forward = MethodType(
                    KolorsControlNet_forward, self.control_model)
            return result

        control_net.pre_run = MethodType(
            KolorsControlLora_pre_run, control_net)

        def KolorsControlLora_copy(self, *args, **kwargs):
            c = super_copy(self, *args, **kwargs)
            c.pre_run = MethodType(
                KolorsControlLora_pre_run, c)
            return c

        control_net.copy = MethodType(KolorsControlLora_copy, control_net)

    elif isinstance(control_net, comfy.controlnet.ControlNet):
        model_label_emb = model.model.diffusion_model.label_emb
        control_net.control_model.label_emb = model_label_emb
        control_net.control_model_wrapped.model.label_emb = model_label_emb
        super_forward = ControlNet.forward

        def KolorsControlNet_forward(self, x, hint, timesteps, context, **kwargs):
            with torch.cuda.amp.autocast(enabled=True):
                context = model.model.diffusion_model.encoder_hid_proj(context)
                return super_forward(self, x, hint, timesteps, context, **kwargs)

        control_net.control_model.forward = MethodType(
            KolorsControlNet_forward, control_net.control_model)

    else:
        raise NotImplementedError(f"Type {control_net} not supported for KolorsControlNetPatch")

    return control_net
