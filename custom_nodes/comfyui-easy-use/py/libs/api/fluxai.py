import json
import os
import yaml
import requests
import pathlib
from aiohttp import web

root_path = pathlib.Path(__file__).parent.parent.parent.parent
config_path = os.path.join(root_path,'config.yaml')
class FluxAIAPI:
    def __init__(self):
        self.api_url = "https://fluxaiimagegenerator.com/api"
        self.origin = "https://fluxaiimagegenerator.com"
        self.user_agent = None
        self.cookie = None

    def promptGenerate(self, text, cookies=None):
        cookie = self.cookie if cookies is None else cookies
        if cookie is None:
            if os.path.isfile(config_path):
                with open(config_path, 'r') as f:
                    data = yaml.load(f, Loader=yaml.FullLoader)
                    if 'FLUXAI_COOKIE' not in data:
                        raise Exception("Please add FLUXAI_COOKIE to config.yaml")
                    if "FLUXAI_USER_AGENT" in data:
                        self.user_agent = data["FLUXAI_USER_AGENT"]
                    self.cookie = cookie = data['FLUXAI_COOKIE']

        headers = {
            "Cookie": cookie,
            "Referer": "https://fluxaiimagegenerator.com/flux-prompt-generator",
            "Origin": self.origin,
            "Content-Type": "application/json",
        }
        if self.user_agent is not None:
            headers['User-Agent'] = self.user_agent

        url = self.api_url + '/prompt'
        json = {
            "prompt": text
        }

        response = requests.post(url, json=json, headers=headers)
        res = response.json()
        if "error" in res:
            return res['error']
        elif "data" in res and "prompt" in res['data']:
            return res['data']['prompt']

fluxaiAPI = FluxAIAPI()

