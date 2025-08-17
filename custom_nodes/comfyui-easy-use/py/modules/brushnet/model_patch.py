import torch
import comfy

# Check and add 'model_patch' to model.model_options['transformer_options']
def add_model_patch_option(model):
    if 'transformer_options' not in model.model_options:
        model.model_options['transformer_options'] = {}
    to = model.model_options['transformer_options']
    if "model_patch" not in to:
        to["model_patch"] = {}
    return to


# Patch model with model_function_wrapper
def patch_model_function_wrapper(model, forward_patch, remove=False):
    def brushnet_model_function_wrapper(apply_model_method, options_dict):
        to = options_dict['c']['transformer_options']

        control = None
        if 'control' in options_dict['c']:
            control = options_dict['c']['control']

        x = options_dict['input']
        timestep = options_dict['timestep']

        # check if there are patches to execute
        if 'model_patch' not in to or 'forward' not in to['model_patch']:
            return apply_model_method(x, timestep, **options_dict['c'])

        mp = to['model_patch']
        unet = mp['unet']

        all_sigmas = mp['all_sigmas']
        sigma = to['sigmas'][0].item()
        total_steps = all_sigmas.shape[0] - 1
        step = torch.argmin((all_sigmas - sigma).abs()).item()

        mp['step'] = step
        mp['total_steps'] = total_steps

        # comfy.model_base.apply_model
        xc = model.model.model_sampling.calculate_input(timestep, x)
        if 'c_concat' in options_dict['c'] and options_dict['c']['c_concat'] is not None:
            xc = torch.cat([xc] + [options_dict['c']['c_concat']], dim=1)
        t = model.model.model_sampling.timestep(timestep).float()
        # execute all patches
        for method in mp['forward']:
            method(unet, xc, t, to, control)

        return apply_model_method(x, timestep, **options_dict['c'])

    if "model_function_wrapper" in model.model_options and model.model_options["model_function_wrapper"]:
        print('BrushNet is going to replace existing model_function_wrapper:',
              model.model_options["model_function_wrapper"])
    model.set_model_unet_function_wrapper(brushnet_model_function_wrapper)

    to = add_model_patch_option(model)
    mp = to['model_patch']

    if isinstance(model.model.model_config, comfy.supported_models.SD15):
        mp['SDXL'] = False
    elif isinstance(model.model.model_config, comfy.supported_models.SDXL):
        mp['SDXL'] = True
    else:
        print('Base model type: ', type(model.model.model_config))
        raise Exception("Unsupported model type: ", type(model.model.model_config))

    if 'forward' not in mp:
        mp['forward'] = []

    if remove:
        if forward_patch in mp['forward']:
            mp['forward'].remove(forward_patch)
    else:
        mp['forward'].append(forward_patch)

    mp['unet'] = model.model.diffusion_model
    mp['step'] = 0
    mp['total_steps'] = 1

    # apply patches to code
    if comfy.samplers.sample.__doc__ is None or 'BrushNet' not in comfy.samplers.sample.__doc__:
        comfy.samplers.original_sample = comfy.samplers.sample
        comfy.samplers.sample = modified_sample

    if comfy.ldm.modules.diffusionmodules.openaimodel.apply_control.__doc__ is None or \
            'BrushNet' not in comfy.ldm.modules.diffusionmodules.openaimodel.apply_control.__doc__:
        comfy.ldm.modules.diffusionmodules.openaimodel.original_apply_control = comfy.ldm.modules.diffusionmodules.openaimodel.apply_control
        comfy.ldm.modules.diffusionmodules.openaimodel.apply_control = modified_apply_control


# Model needs current step number and cfg at inference step. It is possible to write a custom KSampler but I'd like to use ComfyUI's one.
# The first versions had modified_common_ksampler, but it broke custom KSampler nodes
def modified_sample(model, noise, positive, negative, cfg, device, sampler, sigmas, model_options={},
                    latent_image=None, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
    ''' Modified by BrushNet nodes'''
    cfg_guider = comfy.samplers.CFGGuider(model)
    cfg_guider.set_conds(positive, negative)
    cfg_guider.set_cfg(cfg)

    ### Modified part ######################################################################
    to = add_model_patch_option(model)
    to['model_patch']['all_sigmas'] = sigmas
    #######################################################################################

    return cfg_guider.sample(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)

# To use Controlnet with RAUNet it is much easier to modify apply_control a little
def modified_apply_control(h, control, name):
    '''Modified by BrushNet nodes'''
    if control is not None and name in control and len(control[name]) > 0:
        ctrl = control[name].pop()
        if ctrl is not None:
            if h.shape[2] != ctrl.shape[2] or h.shape[3] != ctrl.shape[3]:
                ctrl = torch.nn.functional.interpolate(ctrl, size=(h.shape[2], h.shape[3]), mode='bicubic').to(
                    h.dtype).to(h.device)
            try:
                h += ctrl
            except:
                print.warning("warning control could not be applied {} {}".format(h.shape, ctrl.shape))
    return h

def add_model_patch(model):
    to = add_model_patch_option(model)
    mp = to['model_patch']
    if "brushnet" in mp:
        if isinstance(model.model.model_config, comfy.supported_models.SD15):
            mp['SDXL'] = False
        elif isinstance(model.model.model_config, comfy.supported_models.SDXL):
            mp['SDXL'] = True
        else:
            print('Base model type: ', type(model.model.model_config))
            raise Exception("Unsupported model type: ", type(model.model.model_config))

        mp['unet'] = model.model.diffusion_model
        mp['step'] = 0
        mp['total_steps'] = 1