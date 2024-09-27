import asyncio
import uuid
from asyncio import AbstractEventLoop
from collections import defaultdict
from pathlib import Path
from typing import Optional, List
from urllib.parse import urlparse, urljoin

import aiohttp
from aiohttp import WSMessage, ClientResponse, ClientTimeout

from .client_types import V1QueuePromptResponse
from ..api.api_client import JSONEncoder
from ..api.components.schema.prompt import PromptDict
from ..api.components.schema.prompt_request import PromptRequest
from ..api.paths.history.get.responses.response_200.content.application_json.schema import Schema as GetHistoryDict
from ..api.schemas import immutabledict
from ..component_model.file_output_path import file_output_path


class AsyncRemoteComfyClient:
    """
    An asynchronous client for remote servers
    """
    __json_encoder = JSONEncoder()

    def __init__(self, server_address: str = "http://localhost:8188", client_id: str = str(uuid.uuid4()),
                 websocket_address: Optional[str] = None, loop: Optional[AbstractEventLoop] = None):
        self.client_id = client_id
        self.server_address = server_address
        server_address_url = urlparse(server_address)
        self.websocket_address = websocket_address if websocket_address is not None else urljoin(
            f"ws://{server_address_url.hostname}:{server_address_url.port}", f"/ws?clientId={client_id}")
        self.loop = loop or asyncio.get_event_loop()
        self._session: aiohttp.ClientSession | None = None
        try:
            if asyncio.get_event_loop() is not None:
                self._ensure_session()
        except RuntimeError as no_running_event_loop:
            pass

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=ClientTimeout(total=10 * 60.0, connect=60.0))
        return self._session

    @property
    def session(self) -> aiohttp.ClientSession:
        return self._ensure_session()

    async def len_queue(self) -> int:
        async with self.session.get(urljoin(self.server_address, "/prompt"), headers={'Accept': 'application/json'}) as response:
            if response.status == 200:
                exec_info_dict = await response.json()
                return exec_info_dict["exec_info"]["queue_remaining"]
            else:
                raise RuntimeError(f"unexpected response: {response.status}: {await response.text()}")

    async def queue_and_forget_prompt_api(self, prompt: PromptDict) -> str:
        """
        Calls the API to queue a prompt, and forgets about it
        :param prompt:
        :return: the task ID
        """
        prompt_json = AsyncRemoteComfyClient.__json_encoder.encode(prompt)
        response: ClientResponse
        async with self.session.post(urljoin(self.server_address, "/api/v1/prompts"), data=prompt_json,
                                     headers={'Content-Type': 'application/json', 'Accept': 'application/json', 'Prefer': 'respond-async'}) as response:

            if 200 <= response.status < 400:
                response_json = await response.json()
                return response_json["prompt_id"]
            else:
                raise RuntimeError(f"could not prompt: {response.status}: {await response.text()}")

    async def queue_prompt_api(self, prompt: PromptDict) -> V1QueuePromptResponse:
        """
        Calls the API to queue a prompt.
        :param prompt:
        :return: the API response from the server containing URLs and the outputs for the UI (nodes with OUTPUT_NODE == true)
        """
        prompt_json = AsyncRemoteComfyClient.__json_encoder.encode(prompt)
        response: ClientResponse
        async with self.session.post(urljoin(self.server_address, "/api/v1/prompts"), data=prompt_json,
                                     headers={'Content-Type': 'application/json', 'Accept': 'application/json'}) as response:

            if 200 <= response.status < 400:
                return V1QueuePromptResponse(**(await response.json()))
            else:
                raise RuntimeError(f"could not prompt: {response.status}: {await response.text()}")

    async def queue_prompt_uris(self, prompt: PromptDict) -> List[str]:
        """
        Calls the API to queue a prompt.
        :param prompt:
        :return: a list of URLs corresponding to the SaveImage nodes in the prompt.
        """
        return (await self.queue_prompt_api(prompt)).urls

    async def queue_prompt(self, prompt: PromptDict) -> bytes | None:
        """
        Calls the API to queue a prompt. Returns the bytes of the first PNG returned by a SaveImage node.
        :param prompt:
        :return:
        """
        prompt_json = AsyncRemoteComfyClient.__json_encoder.encode(prompt)
        response: ClientResponse
        headers = {'Content-Type': 'application/json', 'Accept': 'image/png'}
        async with self.session.post(urljoin(self.server_address, "/api/v1/prompts"), data=prompt_json,
                                     headers=headers) as response:

            if 200 <= response.status < 400:
                return await response.read()
            else:
                raise RuntimeError(f"could not prompt: {response.status}: {await response.text()}")

    async def queue_prompt_ui(self, prompt: PromptDict) -> dict[str, List[Path]]:
        """
        Uses the comfyui UI API calls to retrieve a list of paths of output files
        :param prompt:
        :return:
        """
        prompt_request = PromptRequest.validate({"prompt": prompt, "client_id": self.client_id})
        prompt_request_json = AsyncRemoteComfyClient.__json_encoder.encode(prompt_request)
        async with self.session.ws_connect(self.websocket_address) as ws:
            async with self.session.post(urljoin(self.server_address, "/prompt"), data=prompt_request_json,
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
        async with self.session.get(urljoin(self.server_address, "/history")) as response:
            if response.status == 200:
                history_json = immutabledict(GetHistoryDict.validate(await response.json()))
            else:
                raise RuntimeError("Couldn't get history")

        # images have filename, subfolder, type keys
        # todo: use the OpenAPI spec for this when I get around to updating it
        outputs_by_node_id = history_json[prompt_id].outputs
        res: dict[str, List[Path]] = {}
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
