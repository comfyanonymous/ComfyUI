import logging
import uuid
from typing import Dict, Optional

from PIL import Image

from comfy.cli_args import default_configuration
from comfy.client.embedded_comfy_client import Comfy
from comfy.component_model.executor_types import SendSyncEvent, SendSyncData, DependencyCycleError, ExecutingMessage, ExecutionErrorMessage
from comfy.distributed.server_stub import ServerStub
from comfy.execution_context import context_add_custom_nodes
from comfy.nodes.package_typing import ExportedNodes

from comfy_execution.graph_utils import Node, GraphBuilder
from tests.conftest import current_test_name


class RunResult:
    def __init__(self, prompt_id: str):
        self.outputs: Dict[str, Dict] = {}
        self.runs: Dict[str, bool] = {}
        self.cached: Dict[str, bool] = {}
        self.prompt_id: str = prompt_id

    def get_output(self, node: Node):
        return self.outputs.get(node.id, None)

    def did_run(self, node: Node):
        return self.runs.get(node.id, False)

    def was_cached(self, node: Node):
        return self.cached.get(node.id, False)

    def was_executed(self, node: Node):
        """Returns True if node was either run or cached"""
        return self.did_run(node) or self.was_cached(node)

    def get_images(self, node: Node):
        output = self.get_output(node)
        if output is None:
            return []
        return output.get('image_objects', [])

    def get_prompt_id(self):
        return self.prompt_id


class _ProgressHandler(ServerStub):
    def __init__(self):
        super().__init__()
        self.tuples: list[tuple[SendSyncEvent, SendSyncData, str]] = []

    def send_sync(self,
                  event: SendSyncEvent,
                  data: SendSyncData,
                  sid: Optional[str] = None):
        self.tuples.append((event, data, sid))


class ComfyClient:
    def __init__(self, embedded_client: Comfy, progress_handler: _ProgressHandler, should_cache_results: bool = False):
        self.embedded_client = embedded_client
        self.progress_handler = progress_handler
        self.should_cache_results = should_cache_results

    async def run(self, graph: GraphBuilder, partial_execution_targets=None) -> RunResult:
        self.progress_handler.tuples = []
        # todo: what is a partial_execution_targets ???
        for node in graph.nodes.values():
            if node.class_type == 'SaveImage':
                node.inputs['filename_prefix'] = current_test_name.get()

        prompt_id = str(uuid.uuid4())
        try:
            outputs = await self.embedded_client.queue_prompt(graph.finalize(), prompt_id=prompt_id, partial_execution_targets=partial_execution_targets)
        except (RuntimeError, DependencyCycleError) as exc_info:
            logging.warning("error when queueing prompt", exc_info=exc_info)
            outputs = {}
        result = RunResult(prompt_id=prompt_id)
        result.outputs = outputs
        result.runs = {}
        send_sync_event: SendSyncEvent
        send_sync_data: SendSyncData
        for send_sync_event, send_sync_data, _ in self.progress_handler.tuples:
            if send_sync_event == "executing":
                send_sync_data: ExecutingMessage
                result.runs[send_sync_data["node"]] = True
            elif send_sync_event == "execution_error":
                send_sync_data: ExecutionErrorMessage
                raise Exception(send_sync_data)
            elif send_sync_event == 'execution_cached':
                if send_sync_data['prompt_id'] == prompt_id:
                    cached_nodes = send_sync_data.get('nodes', [])
                    for node_id in cached_nodes:
                        result.cached[node_id] = True

        for node in outputs.values():
            if "images" in node:
                image_objects = node["image_objects"] = []
                for image in node["images"]:
                    image_objects.append(Image.open(image["abs_path"]))
        return result

    def get_all_history(self, *args, **kwargs):
        return self.embedded_client.history.copy(*args, **kwargs)


async def client_fixture(self, request=None):
    from ..inference.testing_pack import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    configuration = default_configuration()
    if request is not None and "extra_args" in request.param:
        configuration.update(request.param["extra_args"])

    progress_handler = _ProgressHandler()
    with context_add_custom_nodes(ExportedNodes(NODE_CLASS_MAPPINGS=NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS=NODE_DISPLAY_NAME_MAPPINGS)):
        async with Comfy(configuration, progress_handler=progress_handler) as embedded_client:
            client = ComfyClient(embedded_client, progress_handler, should_cache_results=request.param["should_cache_results"] if request is not None and "should_cache_results" in request.param else True)
            yield client
