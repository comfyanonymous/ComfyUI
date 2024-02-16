import asyncio
import uuid
from asyncio import AbstractEventLoop
from collections import defaultdict
from pathlib import Path
from typing import Optional, List, Dict
from urllib.parse import urlparse, urljoin

import aiohttp
from aiohttp import WSMessage, ClientResponse

from ..api.components.schema.prompt import PromptDict
from ..api.api_client import JSONEncoder
from ..api.components.schema.prompt_request import PromptRequest
from ..api.paths.history.get.responses.response_200.content.application_json.schema import Schema as GetHistoryDict
from ..component_model.file_output_path import file_output_path


class AsyncRemoteComfyClient:
    __json_encoder = JSONEncoder()

    def __init__(self, server_address: str = "http://localhost:8188", client_id: str = str(uuid.uuid4()),
                 websocket_address: Optional[str] = None, loop: Optional[AbstractEventLoop] = None):
        self.client_id = client_id
        self.server_address = server_address
        server_address_url = urlparse(server_address)
        self.websocket_address = websocket_address if websocket_address is not None else urljoin(
            f"ws://{server_address_url.hostname}:{server_address_url.port}", f"/ws?clientId={client_id}")
        self.loop = loop or asyncio.get_event_loop()

    async def queue_prompt_uris(self, prompt: PromptDict) -> List[str]:
        """
        Calls the API to queue a prompt.
        :param prompt:
        :return: a list of URLs corresponding to the SaveImage nodes in the prompt.
        """
        prompt_json = AsyncRemoteComfyClient.__json_encoder.encode(prompt)
        async with aiohttp.ClientSession() as session:
            response: ClientResponse
            async with session.post(urljoin(self.server_address, "/api/v1/prompts"), data=prompt_json,
                                    headers={'Content-Type': 'application/json', 'Accept': 'application/json'}) as response:

                if response.status == 200:
                    return (await response.json())["urls"]
                else:
                    raise RuntimeError(f"could not prompt: {response.status}: {await response.text()}")

    async def queue_prompt(self, prompt: PromptDict) -> bytes:
        """
        Calls the API to queue a prompt. Returns the bytes of the first PNG returned by a SaveImage node.
        :param prompt:
        :return:
        """
        prompt_json = AsyncRemoteComfyClient.__json_encoder.encode(prompt)
        async with aiohttp.ClientSession() as session:
            response: ClientResponse
            async with session.post(urljoin(self.server_address, "/api/v1/prompts"), data=prompt_json,
                                    headers={'Content-Type': 'application/json', 'Accept': 'image/png'}) as response:

                if response.status == 200:
                    return await response.read()
                else:
                    raise RuntimeError(f"could not prompt: {response.status}: {await response.text()}")

    async def queue_prompt_ui(self, prompt: PromptDict) -> Dict[str, List[Path]]:
        """
        Uses the comfyui UI API calls to retrieve a list of paths of output files
        :param prompt:
        :return:
        """
        prompt_request = PromptRequest.validate({"prompt": prompt, "client_id": self.client_id})
        prompt_request_json = AsyncRemoteComfyClient.__json_encoder.encode(prompt_request)
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(self.websocket_address) as ws:
                async with session.post(urljoin(self.server_address, "/prompt"), data=prompt_request_json,
                                        headers={'Content-Type': 'application/json'}) as response:
                    if response.status == 200:
                        prompt_id = (await response.json())["prompt_id"]
                    else:
                        raise RuntimeError("could not prompt")
                msg: WSMessage
                async for msg in ws:
                    # Handle incoming messages
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        msg_json = msg.json()
                        if msg_json["type"] == "executing":
                            data = msg_json["data"]
                            if data['node'] is None and data['prompt_id'] == prompt_id:
                                break
                    elif msg.type == aiohttp.WSMsgType.CLOSED:
                        break
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        break
            async with session.get(urljoin(self.server_address, "/history")) as response:
                if response.status == 200:
                    history_json = GetHistoryDict.validate(await response.json())

            # images have filename, subfolder, type keys
            # todo: use the OpenAPI spec for this when I get around to updating it
            outputs_by_node_id = history_json[prompt_id].outputs
            res: Dict[str, List[Path]] = {}
            for node_id, output in outputs_by_node_id.items():
                if 'images' in output:
                    images = []
                    image_dicts: List[dict] = output['images']
                    for image_file_output_dict in image_dicts:
                        image_file_output_dict = defaultdict(None, image_file_output_dict)
                        filename = image_file_output_dict['filename']
                        subfolder = image_file_output_dict['subfolder']
                        type = image_file_output_dict['type']
                        images.append(Path(file_output_path(filename, subfolder=subfolder, type=type)))
                    res[node_id] = images
            return res
