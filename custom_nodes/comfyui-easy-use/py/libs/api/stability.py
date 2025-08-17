import json
import os
import yaml
import requests
import pathlib
from aiohttp import web
from server import PromptServer
from ..image import tensor2pil, pil2tensor, image2base64, pil2byte
from ..log import log_node_error


root_path = pathlib.Path(__file__).parent.parent.parent.parent
config_path = os.path.join(root_path,'config.yaml')
default_key = [{'name':'Default', 'key':''}]


class StabilityAPI:
    def __init__(self):
        self.api_url = "https://api.stability.ai"
        self.api_keys = None
        self.api_current = 0
        self.user_info = {}

    def getErrors(self, code):
        errors = {
            400: "Bad Request",
            403: "ApiKey Forbidden",
            413: "Your request was larger than 10MiB.",
            429: "You have made more than 150 requests in 10 seconds.",
            500: "Internal Server Error",
        }
        return errors.get(code, "Unknown Error")

    def getAPIKeys(self):
        if os.path.isfile(config_path):
            with open(config_path, 'r') as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
                if not data:
                    data = {'STABILITY_API_KEY': default_key, 'STABILITY_API_DEFAULT':0}
                    with open(config_path, 'w') as f:
                        yaml.dump(data, f)
                if 'STABILITY_API_KEY' not in data:
                    data['STABILITY_API_KEY'] = default_key
                    data['STABILITY_API_DEFAULT'] = 0
                    with open(config_path, 'w') as f:
                        yaml.dump(data, f)
                api_keys = data['STABILITY_API_KEY']
                self.api_current = data['STABILITY_API_DEFAULT']
                self.api_keys = api_keys
                return api_keys
        else:
            # create a yaml file
            with open(config_path, 'w') as f:
                data = {'STABILITY_API_KEY': default_key, 'STABILITY_API_DEFAULT':0}
                yaml.dump(data, f)
                return data['STABILITY_API_KEY']
        pass

    def setAPIKeys(self, api_keys):
        if len(api_keys) > 0:
            self.api_keys = api_keys
            # load and save the yaml file
            with open(config_path, 'r') as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
                data['STABILITY_API_KEY'] = api_keys
                with open(config_path, 'w') as f:
                    yaml.dump(data, f)
        return True

    def setAPIDefault(self, current):
        if current is not None:
            self.api_current = current
            # load and save the yaml file
            with open(config_path, 'r') as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
                data['STABILITY_API_DEFAULT'] = current
                with open(config_path, 'w') as f:
                    yaml.dump(data, f)
        return True

    def generate_sd3_image(self, prompt, negative_prompt, aspect_ratio,  model, seed, mode='text-to-image', image=None, strength=1, output_format='png', node_name='easy stableDiffusion3API'):
        url = f"{self.api_url}/v2beta/stable-image/generate/sd3"
        api_key = self.api_keys[self.api_current]['key']
        files = None
        data = {
            "prompt": prompt,
            "mode": mode,
            "model": model,
            "seed": seed,
            "output_format": output_format,
        }
        if model == 'sd3':
            data['negative_prompt'] = negative_prompt

        if mode == 'text-to-image':
            files = {"none": ''}
            data['aspect_ratio'] = aspect_ratio
        elif mode == 'image-to-image':
            pil_image = tensor2pil(image)
            image_byte = pil2byte(pil_image)
            files = {"image": ("output.png", image_byte, 'image/png')}
            data['strength'] = strength

        response = requests.post(url,
            headers={"authorization": f"{api_key}", "accept": "application/json"},
            files=files,
            data=data,
        )
        if response.status_code == 200:
            PromptServer.instance.send_sync('stable-diffusion-api-generate-succeed',{"model":model})
            json_data = response.json()
            image_base64 = json_data['image']
            image_data = image2base64(image_base64)
            output_t = pil2tensor(image_data)
            return output_t
        else:
            if 'application/json' in response.headers['Content-Type']:
                error_info = response.json()
                log_node_error(node_name,  error_info.get('name', 'No name provided'))
                log_node_error(node_name, error_info.get('errors', ['No details provided']))
            error_status_text = self.getErrors(response.status_code)
            PromptServer.instance.send_sync('easyuse-toast',{"type": "error", "content": error_status_text})
            raise Exception(f"Failed to generate image: {error_status_text}")

    # get user account
    async def getUserAccount(self, cache=True):
        url = f"{self.api_url}/v1/user/account"
        api_key = self.api_keys[self.api_current]['key']
        name = self.api_keys[self.api_current]['name']
        if cache and name in self.user_info:
            return self.user_info[name]
        else:
            response = requests.get(url, headers={"Authorization": f"Bearer {api_key}"})
            if response.status_code == 200:
                user_info = response.json()
                self.user_info[name] = user_info
                return user_info
            else:
                PromptServer.instance.send_sync('easyuse-toast',{'type': 'error', 'content': self.getErrors(response.status_code)})
                return None

    # get user balance
    async def getUserBalance(self):
        url = f"{self.api_url}/v1/user/balance"
        api_key = self.api_keys[self.api_current]['key']
        response = requests.get(url, headers={
            "Authorization": f"Bearer {api_key}"
        })
        if response.status_code == 200:
            return response.json()
        else:
            PromptServer.instance.send_sync('easyuse-toast', {'type': 'error', 'content': self.getErrors(response.status_code)})
            return None

stableAPI = StabilityAPI()

@PromptServer.instance.routes.get("/easyuse/stability/api_keys")
async def get_stability_api_keys(request):
    stableAPI.getAPIKeys()
    return web.json_response({"keys": stableAPI.api_keys, "current": stableAPI.api_current})

@PromptServer.instance.routes.post("/easyuse/stability/set_api_keys")
async def set_stability_api_keys(request):
    post = await request.post()
    api_keys = post.get("api_keys")
    current = post.get('current')
    if api_keys is not None:
        api_keys = json.loads(api_keys)
        stableAPI.setAPIKeys(api_keys)
        if current is not None:
            print(current)
            stableAPI.setAPIDefault(int(current))
            account = await stableAPI.getUserAccount()
            balance = await stableAPI.getUserBalance()
            return web.json_response({'account': account, 'balance': balance})
        else:
            return web.json_response({'status': 'ok'})
    else:
        return web.Response(status=400)

@PromptServer.instance.routes.post("/easyuse/stability/set_apikey_default")
async def set_stability_api_default(request):
    post = await request.post()
    current = post.get("current")
    if current is not None and current < len(stableAPI.api_keys):
        stableAPI.api_current = current
        return web.json_response({'status': 'ok'})
    else:
        return web.Response(status=400)

@PromptServer.instance.routes.get("/easyuse/stability/user_info")
async def get_account_info(request):
    account = await stableAPI.getUserAccount()
    balance = await stableAPI.getUserBalance()
    return web.json_response({'account': account, 'balance': balance})

@PromptServer.instance.routes.get("/easyuse/stability/balance")
async def get_balance_info(request):
    balance = await stableAPI.getUserBalance()
    return web.json_response({'balance': balance})
