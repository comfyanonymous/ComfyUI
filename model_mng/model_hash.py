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
        logging.info("got hash")
        json_data = await request.json()
        #json_data = {'client_id': '092d9b6bc0a14093bba00189abd2a9f8', 'prompt': {'3': {'inputs': {'seed': 1001930077303734, 'steps': 24, 'cfg': 8, 'sampler_name': 'euler', 'scheduler': 'normal', 'denoise': 1, 'model': ['4', 0], 'positive': ['6', 0], 'negative': ['7', 0], 'latent_image': ['5', 0]}, 'class_type': 'KSampler'}, '4': {'inputs': {'ckpt_name': 'v1-5-pruned-emaonly.ckpt'}, 'class_type': 'CheckpointLoaderSimple'}, '5': {'inputs': {'width': 512, 'height': 512, 'batch_size': 1}, 'class_type': 'EmptyLatentImage'}, '6': {'inputs': {'text': 'beautiful scenery nature glass bottle landscape, , purple galaxy bottle,', 'clip': ['4', 1]}, 'class_type': 'CLIPTextEncode'}, '7': {'inputs': {'text': 'text, watermark', 'clip': ['4', 1]}, 'class_type': 'CLIPTextEncode'}, '8': {'inputs': {'samples': ['3', 0], 'vae': ['4', 2]}, 'class_type': 'VAEDecode'}, '9': {'inputs': {'filename_prefix': 'ComfyUI', 'images': ['8', 0]}, 'class_type': 'SaveImage'}}, 'workflow': {'last_node_id': 13, 'last_link_id': 9, 'nodes': [{'id': 9, 'type': 'SaveImage', 'pos': [1477, 389], 'size': [210, 270], 'flags': {}, 'order': 6, 'mode': 0, 'inputs': [{'name': 'images', 'type': 'IMAGE', 'link': 9}], 'properties': {}, 'widgets_values': ['ComfyUI']}, {'id': 8, 'type': 'VAEDecode', 'pos': [1287, 166], 'size': {'0': 210, '1': 46}, 'flags': {}, 'order': 5, 'mode': 0, 'inputs': [{'name': 'samples', 'type': 'LATENT', 'link': 7}, {'name': 'vae', 'type': 'VAE', 'link': 8}], 'outputs': [{'name': 'IMAGE', 'type': 'IMAGE', 'links': [9], 'slot_index': 0}], 'properties': {'Node name for S&R': 'VAEDecode'}}, {'id': 6, 'type': 'CLIPTextEncode', 'pos': [636, 62], 'size': {'0': 422.84503173828125, '1': 164.31304931640625}, 'flags': {}, 'order': 2, 'mode': 0, 'inputs': [{'name': 'clip', 'type': 'CLIP', 'link': 3}], 'outputs': [{'name': 'CONDITIONING', 'type': 'CONDITIONING', 'links': [4], 'slot_index': 0}], 'properties': {'Node name for S&R': 'CLIPTextEncode'}, 'widgets_values': ['beautiful scenery nature glass bottle landscape, , purple galaxy bottle,']}, {'id': 4, 'type': 'CheckpointLoaderSimple', 'pos': [143, 196], 'size': {'0': 315, '1': 98}, 'flags': {}, 'order': 0, 'mode': 0, 'outputs': [{'name': 'MODEL', 'type': 'MODEL', 'links': [1], 'slot_index': 0}, {'name': 'CLIP', 'type': 'CLIP', 'links': [3, 5], 'slot_index': 1}, {'name': 'VAE', 'type': 'VAE', 'links': [8], 'slot_index': 2}], 'properties': {'Node name for S&R': 'CheckpointLoaderSimple'}, 'widgets_values': ['v1-5-pruned-emaonly.ckpt']}, {'id': 3, 'type': 'KSampler', 'pos': [977, 379], 'size': {'0': 315, '1': 262}, 'flags': {}, 'order': 4, 'mode': 0, 'inputs': [{'name': 'model', 'type': 'MODEL', 'link': 1}, {'name': 'positive', 'type': 'CONDITIONING', 'link': 4}, {'name': 'negative', 'type': 'CONDITIONING', 'link': 6}, {'name': 'latent_image', 'type': 'LATENT', 'link': 2}], 'outputs': [{'name': 'LATENT', 'type': 'LATENT', 'links': [7], 'slot_index': 0}], 'properties': {'Node name for S&R': 'KSampler'}, 'widgets_values': [1001930077303734, 'randomize', 24, 8, 'euler', 'normal', 1]}, {'id': 7, 'type': 'CLIPTextEncode', 'pos': [479, 372], 'size': {'0': 425.27801513671875, '1': 180.6060791015625}, 'flags': {}, 'order': 3, 'mode': 0, 'inputs': [{'name': 'clip', 'type': 'CLIP', 'link': 5}], 'outputs': [{'name': 'CONDITIONING', 'type': 'CONDITIONING', 'links': [6], 'slot_index': 0}], 'properties': {'Node name for S&R': 'CLIPTextEncode'}, 'widgets_values': ['text, watermark']}, {'id': 5, 'type': 'EmptyLatentImage', 'pos': [465, 648], 'size': {'0': 315, '1': 106}, 'flags': {}, 'order': 1, 'mode': 0, 'outputs': [{'name': 'LATENT', 'type': 'LATENT', 'links': [2], 'slot_index': 0}], 'properties': {'Node name for S&R': 'EmptyLatentImage'}, 'widgets_values': [512, 512, 1]}], 'links': [[1, 4, 0, 3, 0, 'MODEL'], [2, 5, 0, 3, 3, 'LATENT'], [3, 4, 1, 6, 0, 'CLIP'], [4, 6, 0, 3, 1, 'CONDITIONING'], [5, 4, 1, 7, 0, 'CLIP'], [6, 7, 0, 3, 2, 'CONDITIONING'], [7, 3, 0, 8, 0, 'LATENT'], [8, 4, 2, 8, 1, 'VAE'], [9, 8, 0, 9, 0, 'IMAGE']], 'groups': [], 'config': {}, 'extra': {'ds': {'scale': 0.9090909090909091, 'offset': {'0': -168.21308950981313, '1': 0.3109240366721764}}}, 'version': 0.4}}
        # json_data = {}

        bPromtFlag = True
        prompt = {}
        workflow = {}
        nodes = {}
        resp = True

        try:
            if "prompt" in json_data:
                prompt = json_data["prompt"]

            if "workflow" in json_data and "nodes" in json_data["workflow"]:
                workflow = json_data["workflow"]
                nodes = workflow["nodes"]

            if bPromtFlag:
                for (node_key, node_values) in prompt.items():
                    for (model_key, model_values) in self.model_prefix_names.items():
                        digest = None
                        if 'inputs' in node_values and model_key in node_values['inputs']:
                            ckpt_path = folder_paths.get_full_path(model_values, node_values['inputs'][model_key])
                            digest = self.get_MD5_hash(ckpt_path)

                            if "digest" not in node_values:
                                node_values['digest'] = {}

                            node_values['digest'][node_values['inputs'][model_key]] = digest

        except Exception as e:
            print(f"Failed to parse the json data: {json_data} \n {e}")
            resp = False

        res = {"output": prompt, "workflow": workflow, "res_status": resp}

        return res

    def get_settings(self, request):
        return {}

    def add_routes(self, routes):
        @routes.post("/get_hash")
        async def post_gethash(request):
            res = await self.get_hash(request)
            return web.json_response(res)
