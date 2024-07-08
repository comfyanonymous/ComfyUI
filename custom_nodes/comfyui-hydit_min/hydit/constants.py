import torch

# =======================================================

NOISE_SCHEDULES = {
    "linear",
    "scaled_linear",
    "squaredcos_cap_v2",
}

PREDICT_TYPE = {
    "epsilon",
    "sample",
    "v_prediction",
}

# =======================================================

NEGATIVE_PROMPT = '错误的眼睛，糟糕的人脸，毁容，糟糕的艺术，变形，多余的肢体，模糊的颜色，模糊，重复，病态，残缺，'

# =======================================================
TRT_MAX_BATCH_SIZE = 1
TRT_MAX_WIDTH = 1280
TRT_MAX_HEIGHT = 1280

# =======================================================
# Constants about models
# =======================================================

VAE_EMA_PATH = "ckpts/t2i/sdxl-vae-fp16-fix"
TOKENIZER = "ckpts/t2i/tokenizer"
TEXT_ENCODER = 'ckpts/t2i/clip_text_encoder'
T5_ENCODER = {
    'MT5': 'ckpts/t2i/mt5',
    'attention_mask': True,
    'layer_index': -1,
    'attention_pool': True,
    'torch_dtype': torch.float16,
    'learnable_replace': True
}

SAMPLER_FACTORY = {
    'ddpm': {
        'scheduler': 'DDPMScheduler',
        'name': 'DDPM',
        'kwargs': {
            'steps_offset': 1,
            'clip_sample': False,
            'clip_sample_range': 1.0,
            'beta_schedule': 'scaled_linear',
            'beta_start': 0.00085,
            'beta_end': 0.02,
            'prediction_type': 'v_prediction',
        }
    },
    'ddim': {
        'scheduler': 'DDIMScheduler',
        'name': 'DDIM',
        'kwargs': {
            'steps_offset': 1,
            'clip_sample': False,
            'clip_sample_range': 1.0,
            'beta_schedule': 'scaled_linear',
            'beta_start': 0.00085,
            'beta_end': 0.02,
            'prediction_type': 'v_prediction',
        }
    },
    'dpmms': {
        'scheduler': 'DPMSolverMultistepScheduler',
        'name': 'DPMMS',
        'kwargs': {
            'beta_schedule': 'scaled_linear',
            'beta_start': 0.00085,
            'beta_end': 0.02,
            'prediction_type': 'v_prediction',
            'trained_betas': None,
            'solver_order': 2,
            'algorithm_type': 'dpmsolver++',
        }
    },
}
