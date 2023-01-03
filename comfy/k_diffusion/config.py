from functools import partial
import json
import math
import warnings

from jsonmerge import merge

from . import augmentation, layers, models, utils


def load_config(file):
    defaults = {
        'model': {
            'sigma_data': 1.,
            'patch_size': 1,
            'dropout_rate': 0.,
            'augment_wrapper': True,
            'augment_prob': 0.,
            'mapping_cond_dim': 0,
            'unet_cond_dim': 0,
            'cross_cond_dim': 0,
            'cross_attn_depths': None,
            'skip_stages': 0,
            'has_variance': False,
        },
        'dataset': {
            'type': 'imagefolder',
        },
        'optimizer': {
            'type': 'adamw',
            'lr': 1e-4,
            'betas': [0.95, 0.999],
            'eps': 1e-6,
            'weight_decay': 1e-3,
        },
        'lr_sched': {
            'type': 'inverse',
            'inv_gamma': 20000.,
            'power': 1.,
            'warmup': 0.99,
        },
        'ema_sched': {
            'type': 'inverse',
            'power': 0.6667,
            'max_value': 0.9999
        },
    }
    config = json.load(file)
    return merge(defaults, config)


def make_model(config):
    config = config['model']
    assert config['type'] == 'image_v1'
    model = models.ImageDenoiserModelV1(
        config['input_channels'],
        config['mapping_out'],
        config['depths'],
        config['channels'],
        config['self_attn_depths'],
        config['cross_attn_depths'],
        patch_size=config['patch_size'],
        dropout_rate=config['dropout_rate'],
        mapping_cond_dim=config['mapping_cond_dim'] + (9 if config['augment_wrapper'] else 0),
        unet_cond_dim=config['unet_cond_dim'],
        cross_cond_dim=config['cross_cond_dim'],
        skip_stages=config['skip_stages'],
        has_variance=config['has_variance'],
    )
    if config['augment_wrapper']:
        model = augmentation.KarrasAugmentWrapper(model)
    return model


def make_denoiser_wrapper(config):
    config = config['model']
    sigma_data = config.get('sigma_data', 1.)
    has_variance = config.get('has_variance', False)
    if not has_variance:
        return partial(layers.Denoiser, sigma_data=sigma_data)
    return partial(layers.DenoiserWithVariance, sigma_data=sigma_data)


def make_sample_density(config):
    sd_config = config['sigma_sample_density']
    sigma_data = config['sigma_data']
    if sd_config['type'] == 'lognormal':
        loc = sd_config['mean'] if 'mean' in sd_config else sd_config['loc']
        scale = sd_config['std'] if 'std' in sd_config else sd_config['scale']
        return partial(utils.rand_log_normal, loc=loc, scale=scale)
    if sd_config['type'] == 'loglogistic':
        loc = sd_config['loc'] if 'loc' in sd_config else math.log(sigma_data)
        scale = sd_config['scale'] if 'scale' in sd_config else 0.5
        min_value = sd_config['min_value'] if 'min_value' in sd_config else 0.
        max_value = sd_config['max_value'] if 'max_value' in sd_config else float('inf')
        return partial(utils.rand_log_logistic, loc=loc, scale=scale, min_value=min_value, max_value=max_value)
    if sd_config['type'] == 'loguniform':
        min_value = sd_config['min_value'] if 'min_value' in sd_config else config['sigma_min']
        max_value = sd_config['max_value'] if 'max_value' in sd_config else config['sigma_max']
        return partial(utils.rand_log_uniform, min_value=min_value, max_value=max_value)
    if sd_config['type'] == 'v-diffusion':
        min_value = sd_config['min_value'] if 'min_value' in sd_config else 0.
        max_value = sd_config['max_value'] if 'max_value' in sd_config else float('inf')
        return partial(utils.rand_v_diffusion, sigma_data=sigma_data, min_value=min_value, max_value=max_value)
    if sd_config['type'] == 'split-lognormal':
        loc = sd_config['mean'] if 'mean' in sd_config else sd_config['loc']
        scale_1 = sd_config['std_1'] if 'std_1' in sd_config else sd_config['scale_1']
        scale_2 = sd_config['std_2'] if 'std_2' in sd_config else sd_config['scale_2']
        return partial(utils.rand_split_log_normal, loc=loc, scale_1=scale_1, scale_2=scale_2)
    raise ValueError('Unknown sample density type')
