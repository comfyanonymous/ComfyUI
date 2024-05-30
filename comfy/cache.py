import gc
from collections import OrderedDict
import os
import logging

from comfy.cli_args import args


class ModelCache:
    def __init__(self):
        self._cache = OrderedDict()
        # ignore gpu and highvram state, may cause OOM error
        self._cache_state = False if (args.highvram or args.gpu_only) else True

    def _get_item(self, key):
        if self._cache.get(key) is None:
            self._cache[key] = {}
        return self._cache[key]

    def cache_model(self, key, model):
        if not self._cache_state:
            return
        item = self._get_item(key)
        item['model'] = model

    def cache_clip_vision(self, key, clip_vision):
        if not self._cache_state:
            return
        item = self._get_item(key)
        item['clip_vision'] = clip_vision

    def cache_vae(self, key, vae):
        if not self._cache_state:
            return
        item = self._get_item(key)
        item['vae'] = vae

    def cache_sd(self, key, sd):
        if not self._cache_state:
            return
        assert isinstance(sd, dict)
        keys = list(sd.keys())
        values = list(sd.values())
        item = self._get_item(key)
        item['sd'] = (keys, values)

    def cache_clip(self, key, clip_key, clip):
        item = self._get_item(key)
        item[clip_key] = clip

    def refresh_cache(self, key):
        if key in self._cache:
            self._cache.move_to_end(key)

    def get_item(self, key, prop):
        item = self._cache.get(key)
        if item is None:
            return None
        if prop == "sd":
            if item.get('sd') is None:
                return None
            k, values = item.get("sd")
            return dict(zip(k, values))
        return item.get(prop)

    def __len__(self):
        return len(self._cache)

    @property
    def cache(self):
        return self._cache

    @staticmethod
    def unpatch_offload_model(model):
        model.model_patches_to(model.offload_device)

    def free_one_model_cache(self):
        if len(self) == 0:
            return
        cache_k, item = self._cache.popitem(last=False)
        item.pop("sd", None)

        for k in list(item.keys()):
            model = item.pop(k, None)
            if model is not None:
                if hasattr(model, "patcher"):
                    self.unpatch_offload_model(model.patcher)
                else:
                    self.unpatch_offload_model(model)

        item.clear()
        model_dir, model_name = os.path.split(cache_k)
        dir_name = os.path.basename(model_dir)
        gc.collect()
        logging.info(f"Drop model cache: {model_name} ({dir_name})")


model_cache = ModelCache()
