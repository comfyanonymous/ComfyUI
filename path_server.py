from singleton_decorator import singleton
import os

@singleton
class PathServer():

    def __init__(self):
        self.paths = {
            'checkpoints': os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models', 'checkpoints'),
            'clip': os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models', 'clip'),
            'clip_vision': os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models', 'clip_vision'),
            'configs': os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models', 'configs'),
            'controlnet': os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models', 'controlnet'),
            'embeddings': os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models', 'embeddings'),
            'loras': os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models', 'loras'),
            'style_models': os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models', 'style_models'),
            't2i_adapter': os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models', 't2i_adapter'),
            'upscale_models': os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models', 'upscale_models'),
            'vae': os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models', 'vae'),
        }

    def set_a1111_path(self, a1111_path):
        self.paths['checkpoints'] = os.path.join(a1111_path, 'models', 'Stable-diffusion')
        self.paths['clip'] = os.path.join(a1111_path, 'models', 'clip-interrogator')
        # self.paths['clip_vision'] = os.path.join(a1111_path, 'models', '')
        self.paths['configs'] = os.path.join(a1111_path, 'models', 'Stable-diffusion')
        self.paths['controlnet'] = os.path.join(a1111_path, 'models', 'ControlNet')
        self.paths['embeddings'] = os.path.join(a1111_path, 'embeddings')
        self.paths['loras'] = os.path.join(a1111_path, 'models', 'Lora')
        #self.paths['style_models'] = os.path.join(a1111_path, 'models', '')
        #self.paths['t2i_adapter'] = os.path.join(a1111_path, 'models', '')
        self.paths['upscale_models'] = os.path.join(a1111_path, 'models', 'ESRGAN')
        self.paths['vae'] = os.path.join(a1111_path, 'models', 'VAE')

    def get(self, key):
        return self.paths[key]

