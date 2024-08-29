import hashlib
import logging
import os
import json
from aiohttp import web
from pathlib import Path

import os
import time
import logging
import folder_paths

class ModelHash():
    def __init__(self):
        self.model_prefix_names = {"ckpt_name": "checkpoints",
                          "control_net_name": "controlnet",
                          "lora_name": "loras",
                          "unet_name":"unet",
                          "clip_name": "clip",
                          "style_model_name": "style_models",
                          "gligen_name": "gligen",
                          "vae_name": "vae" }
        pass


    def get_MD5_hash(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            return None

        file_digest = hashlib.sha256(open(path, 'rb').read()).hexdigest()
        return file_digest

    async def get_hash(self, request):

        json_data = await request.json()

        bPromtFlag = False
        prompt = {}
        resp = True

        try:
            if "prompt" in json_data:
                prompt = json_data["prompt"]

            hash_node = {}
            for (node_key, node_values) in prompt.items():
                if "class_type" in node_values and node_values["class_type"] =="SaveAllModelHashesNode" :
                    hash_node = node_values
                    if "digest" not in node_values:
                        hash_node['digest'] = {}
                    bPromtFlag = True
                    break

            if bPromtFlag:
                for (node_key, node_values) in prompt.items():
                    for (model_key, model_values) in self.model_prefix_names.items():
                        digest = None
                        if 'inputs' in node_values and model_key in node_values['inputs']:
                            ckpt_path = folder_paths.get_full_path(model_values, node_values['inputs'][model_key])
                            digest = self.get_MD5_hash(ckpt_path)

                            hash_node['digest'][node_values['inputs'][model_key]] = digest

        except Exception as e:
            print(f"Failed to parse the json data: {json_data} \n {e}")
            resp = False

        res = {"output": prompt, "res_status": resp}
        return res

    def add_routes(self, routes):
        @routes.post("/api/get_hash")
        async def post_gethash(request):
            res = await self.get_hash(request)
            return web.json_response(res)
