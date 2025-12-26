from __future__ import annotations
import asyncio
import uuid
from asyncio import AbstractEventLoop
from typing import Optional, List
from urllib.parse import urlparse, urljoin

import aiohttp
from aiohttp import WSMessage, ClientResponse, ClientTimeout
from opentelemetry import trace

from .client_types import V1QueuePromptResponse
from ..api.api_client import JSONEncoder
from ..api.components.schema.prompt import PromptDict
from ..api.components.schema.prompt_request import PromptRequest
from ..api.paths.history.get.responses.response_200.content.application_json.schema import Schema as GetHistoryDict
from ..api.schemas import immutabledict
from ..component_model.outputs_types import OutputsDict

tracer = trace.get_tracer(__name__)


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

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=ClientTimeout(total=10 * 60.0, connect=60.0))
        return self._session

    async def __aenter__(self):
        """Allows the client to be used in an 'async with' block."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Closes the session when exiting an 'async with' block."""
        await self.close()

    async def close(self):
        """Closes the underlying aiohttp.ClientSession."""
        if self._session and not self._session.closed:
            await self._session.close()

    @property
    def session(self) -> aiohttp.ClientSession:
        return self._ensure_session()

    def _build_headers(self, accept_header: str, prefer_header: Optional[str] = None, content_type: str = "application/json") -> dict:
        """Build HTTP headers for requests."""
        headers = {'Content-Type': content_type, 'Accept': accept_header}
        if prefer_header:
            headers['Prefer'] = prefer_header
        return headers

    @tracer.start_as_current_span("Post Prompt")
    async def _post_prompt(self, prompt: PromptDict | dict, endpoint: str, accept_header: str, prefer_header: Optional[str] = None) -> ClientResponse:
        """
        Common method to POST a prompt to a given endpoint.
        :param prompt: The prompt to send
        :param endpoint: The API endpoint (e.g., "/api/v1/prompts")
        :param accept_header: The Accept header value
        :param prefer_header: Optional Prefer header value
        :return: The response object
        """
        prompt_json = AsyncRemoteComfyClient.__json_encoder.encode(prompt)
        headers = self._build_headers(accept_header, prefer_header)
        return await self.session.post(urljoin(self.server_address, endpoint), data=prompt_json, headers=headers)

    async def len_queue(self) -> int:
        async with self.session.get(urljoin(self.server_address, "/prompt"), headers={'Accept': 'application/json'}) as response:
            if response.status == 200:
                exec_info_dict = await response.json()
                return exec_info_dict["exec_info"]["queue_remaining"]
            else:
                raise RuntimeError(f"unexpected response: {response.status}: {await response.text()}")

    async def queue_and_forget_prompt_api(self, prompt: PromptDict, prefer_header: Optional[str] = "respond-async", accept_header: str = "application/json") -> str:
        """
        Calls the API to queue a prompt, and forgets about it
        :param prompt:
        :param prefer_header: The Prefer header value (e.g., "respond-async" or None)
        :param accept_header: The Accept header value (e.g., "application/json", "application/json+respond-async")
        :return: the task ID
        """
        async with await self._post_prompt(prompt, "/api/v1/prompts", accept_header, prefer_header) as response:
            if 200 <= response.status < 400:
                response_json = await response.json()
                return response_json["prompt_id"]
            else:
                raise RuntimeError(f"could not prompt: {response.status}, reason={response.reason}: {await response.text()}")

    async def queue_prompt_api(self, prompt: PromptDict | dict, prefer_header: Optional[str] = None, accept_header: str = "application/json") -> V1QueuePromptResponse:
        """
        Calls the API to queue a prompt.
        :param prompt:
        :param prefer_header: The Prefer header value (e.g., "respond-async" or None)
        :param accept_header: The Accept header value (e.g., "application/json", "application/json+respond-async")
        :return: the API response from the server containing URLs and the outputs for the UI (nodes with OUTPUT_NODE == true)
        """
        async with await self._post_prompt(prompt, "/api/v1/prompts", accept_header, prefer_header) as response:
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
        async with await self._post_prompt(prompt, "/api/v1/prompts", "image/png") as response:
            if 200 <= response.status < 400:
                return await response.read()
            else:
                raise RuntimeError(f"could not prompt: {response.status}: {await response.text()}")

    @tracer.start_as_current_span("Post Prompt (UI)")
    async def queue_prompt_ui(self, prompt: PromptDict) -> OutputsDict:
        """
        Uses the comfyui UI API calls to retrieve the outputs dictionary
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
        return history_json[prompt_id].outputs

    async def get_prompt_status(self, prompt_id: str) -> ClientResponse:
        """
        Get the status of a prompt by ID using the API endpoint.
        :param prompt_id: The prompt ID to query
        :return: The ClientResponse object (caller should check status and read body)
        """
        return await self.session.get(urljoin(self.server_address, f"/api/v1/prompts/{prompt_id}"))

    @tracer.start_as_current_span("Poll Prompt Until Done")
    async def poll_prompt_until_done(self, prompt_id: str, max_attempts: int = 60, poll_interval: float = 1.0) -> tuple[int, dict | None]:
        """
        Poll a prompt until it's done (200), errors (500), or times out.
        :param prompt_id: The prompt ID to poll
        :param max_attempts: Maximum number of polling attempts
        :param poll_interval: Time to wait between polls in seconds
        :return: Tuple of (status_code, response_json or None)
        """
        span = trace.get_current_span()
        span.set_attribute("prompt_id", prompt_id)
        span.set_attribute("max_attempts", max_attempts)

        for _ in range(max_attempts):
            async with await self.get_prompt_status(prompt_id) as response:
                if response.status == 200:
                    return response.status, await response.json()
                elif response.status == 500:
                    return response.status, await response.json()
                elif response.status == 404:
                    return response.status, None
                elif response.status == 204:
                    # Still in progress
                    await asyncio.sleep(poll_interval)
                else:
                    # Unexpected status
                    return response.status, None
        # Timeout
        return 408, None

    async def get_jobs(self, status: Optional[str] = None, workflow_id: Optional[str] = None,
                       limit: Optional[int] = None, offset: Optional[int] = None,
                       sort_by: Optional[str] = None, sort_order: Optional[str] = None) -> dict:
        """
        List all jobs with filtering, sorting, and pagination.
        :param status: Filter by status (comma-separated): pending, in_progress, completed, failed
        :param workflow_id: Filter by workflow ID
        :param limit: Max items to return
        :param offset: Items to skip
        :param sort_by: Sort field: created_at (default), execution_duration
        :param sort_order: Sort direction: asc, desc (default)
        :return: Dictionary containing jobs list and pagination info
        """
        params = {}
        if status is not None:
            params["status"] = status
        if workflow_id is not None:
            params["workflow_id"] = workflow_id
        if limit is not None:
            params["limit"] = str(limit)
        if offset is not None:
            params["offset"] = str(offset)
        if sort_by is not None:
            params["sort_by"] = sort_by
        if sort_order is not None:
            params["sort_order"] = sort_order

        async with self.session.get(urljoin(self.server_address, "/api/jobs"), params=params) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise RuntimeError(f"could not get jobs: {response.status}: {await response.text()}")

    async def get_job(self, job_id: str) -> Optional[dict]:
        """
        Get a single job by ID.
        :param job_id: The job ID
        :return: Job dictionary or None if not found
        """
        async with self.session.get(urljoin(self.server_address, f"/api/jobs/{job_id}")) as response:
            if response.status == 200:
                return await response.json()
            elif response.status == 404:
                return None
            else:
                raise RuntimeError(f"could not get job: {response.status}: {await response.text()}")
