import hashlib
import logging
import threading
import time
from comfy.cli_args import args


class ModelHasher:

    def __init__(self):
        self._thread = None
        self._lock = threading.Lock()
        self._model_entity = None

    def start(self, model_entity):
        if args.disable_model_hashing:
            return

        self._model_entity = model_entity

        if self._thread is None:
            # Lock to prevent multiple threads from starting
            with self._lock:
                if self._thread is None:
                    self._thread = threading.Thread(target=self._hash_models)
                    self._thread.daemon = True
                    self._thread.start()

    def _get_models(self):
        models = self._model_entity.get("WHERE hash IS NULL")
        return models

    def _hash_model(self, model_path):
        h = hashlib.sha256()
        b = bytearray(128 * 1024)
        mv = memoryview(b)
        with open(model_path, "rb", buffering=0) as f:
            while n := f.readinto(mv):
                h.update(mv[:n])
        hash = h.hexdigest()
        return hash

    def _hash_models(self):
        while True:
            models = self._get_models()

            if len(models) == 0:
                break

            for model in models:
                time.sleep(0)
                now = time.time()
                logging.info(f"Hashing model {model['path']}")
                hash = self._hash_model(model["path"])
                logging.info(
                    f"Hashed model {model['path']} in {time.time() - now} seconds"
                )
                self._model_entity.update((model["id"],), {"hash": hash})

        self._thread = None


model_hasher = ModelHasher()
