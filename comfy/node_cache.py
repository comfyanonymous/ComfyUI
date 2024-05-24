from collections import OrderedDict, namedtuple
from functools import wraps
import os

import torch

MAX_CHECKPOINT_CACHE = int(os.environ.get("MAX_CHECKPOINT_CACHE", 1))
MAX_CONTROLNET_CACHE = int(os.environ.get("MAX_CONTROLNET_CACHE", 1))
MAX_VAE_CACHE = int(os.environ.get("MAX_VAE_CACHE", 1))


class Cache:
    def __init__(self, max_cache_size=0):
        self._max_size = max_cache_size
        self._cache = OrderedDict()

    @property
    def cache(self):
        return self._cache

    @property
    def size(self):
        return self._max_size

    def clear_cache(self):
        while len(self._cache) > self.size:
            k, _ = self._cache.popitem(last=False)
            print(f"[==Cache==]  Removed {k} from cache.")

    def add_cache(self, k, item):
        """model patch change the key name of tensor, cache the names here"""
        keys = list(item.keys())
        values = list(item.values())
        self._cache[k] = (keys, values)
        self._cache.move_to_end(k)

    def get_cache(self, k):
        data = self._cache.get(k)
        if data:
            keys, values = data
            return dict(zip(keys, values))
        return None


caches_mapping = {
    "checkpoints": Cache(max_cache_size=MAX_CHECKPOINT_CACHE),
    "controlnet": Cache(max_cache_size=MAX_CONTROLNET_CACHE),
    "vae": Cache(max_cache_size=MAX_VAE_CACHE),
}


def checkpoint_cache_decorator(func):
    @wraps(func)
    def wrapper(ckpt, *args, device=None, **kwargs):
        model_dir, model_name = os.path.split(ckpt)
        dir_name = os.path.basename(model_dir)
        # get matched ckpt, vae, controlnet queue
        cache_mapper = caches_mapping.get(dir_name, None)

        if cache_mapper is not None and device in (None, torch.device("cpu"), {}):
            if cache_mapper.get_cache(ckpt) is not None:
                return cache_mapper.get_cache(ckpt)

        sd = func(ckpt, *args, device=device, **kwargs)

        if cache_mapper is not None and device in (None, torch.device("cpu"), {}):
            cache_mapper.add_cache(ckpt, sd)
            print(f"[==Cache==] Add {dir_name}  {model_name} to cache")
            cache_mapper.clear_cache()
        return sd

    return wrapper
