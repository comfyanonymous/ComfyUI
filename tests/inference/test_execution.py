import uuid
from contextvars import ContextVar
from typing import Dict, Optional

import numpy
import pytest
from PIL import Image
from pytest import fixture

from comfy.cli_args import default_configuration
from comfy.client.embedded_comfy_client import EmbeddedComfyClient
from comfy.component_model.executor_types import SendSyncEvent, SendSyncData, ExecutingMessage, ExecutionErrorMessage, DependencyCycleError
from comfy.distributed.server_stub import ServerStub
from comfy.graph_utils import GraphBuilder, Node
from comfy.nodes.package_typing import ExportedNodes

current_test_name = ContextVar('current_test_name', default=None)


@pytest.fixture(autouse=True)
def set_test_name(request):
    token = current_test_name.set(request.node.name)
    yield
    current_test_name.reset(token)


class RunResult:
    def __init__(self, prompt_id: str):
        self.outputs: Dict[str, Dict] = {}
        self.runs: Dict[str, bool] = {}
        self.prompt_id: str = prompt_id

    def get_output(self, node: Node):
        return self.outputs.get(node.id, None)

    def did_run(self, node: Node):
        return self.runs.get(node.id, False)

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


class Client:
    def __init__(self, embedded_client: EmbeddedComfyClient, progress_handler: _ProgressHandler):
        self.embedded_client = embedded_client
        self.progress_handler = progress_handler

    async def run(self, graph: GraphBuilder) -> RunResult:
        self.progress_handler.tuples = []
        for node in graph.nodes.values():
            if node.class_type == 'SaveImage':
                node.inputs['filename_prefix'] = current_test_name.get()

        prompt_id = str(uuid.uuid4())
        try:
            outputs = await self.embedded_client.queue_prompt(graph.finalize(), prompt_id=prompt_id)
        except (RuntimeError, DependencyCycleError):
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

        for node in outputs.values():
            if "images" in node:
                image_objects = node["image_objects"] = []
                for image in node["images"]:
                    image_objects.append(Image.open(image["abs_path"]))
        return result


# Loop through these variables
@pytest.mark.execution
class TestExecution:
    # Initialize server and client
    @fixture(scope="class", params=[
        # (lru_size)
        (0,),
        (100,),
    ])
    async def client(self, request) -> Client:
        from comfy.cmd.execution import nodes
        from .testing_pack import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

        nodes.update(ExportedNodes(NODE_CLASS_MAPPINGS=NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS=NODE_DISPLAY_NAME_MAPPINGS))
        lru_size, = request.param
        configuration = default_configuration()
        configuration.cache_lru = lru_size
        progress_handler = _ProgressHandler()
        async with EmbeddedComfyClient(configuration, progress_handler=progress_handler) as embedded_client:
            yield Client(embedded_client, progress_handler)

    @fixture
    def builder(self, request):
        yield GraphBuilder(prefix=request.node.name)

    async def test_lazy_input(self, client: Client, builder: GraphBuilder):
        g = builder
        input1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        input2 = g.node("StubImage", content="WHITE", height=512, width=512, batch_size=1)
        mask = g.node("StubMask", value=0.0, height=512, width=512, batch_size=1)

        lazy_mix = g.node("TestLazyMixImages", image1=input1.out(0), image2=input2.out(0), mask=mask.out(0))
        output = g.node("SaveImage", images=lazy_mix.out(0))
        result = await client.run(g)

        result_image = result.get_images(output)[0]
        assert numpy.array(result_image).any() == 0, "Image should be black"
        assert result.did_run(input1)
        assert not result.did_run(input2)
        assert result.did_run(mask)
        assert result.did_run(lazy_mix)

    async def test_full_cache(self, client: Client, builder: GraphBuilder):
        g = builder
        input1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        input2 = g.node("StubImage", content="NOISE", height=512, width=512, batch_size=1)
        mask = g.node("StubMask", value=0.5, height=512, width=512, batch_size=1)

        lazy_mix = g.node("TestLazyMixImages", image1=input1.out(0), image2=input2.out(0), mask=mask.out(0))
        g.node("SaveImage", images=lazy_mix.out(0))

        await client.run(g)
        result2 = await client.run(g)
        for node_id, node in g.nodes.items():
            assert not result2.did_run(node), f"Node {node_id} ran, but should have been cached"

    async def test_partial_cache(self, client: Client, builder: GraphBuilder):
        g = builder
        input1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        input2 = g.node("StubImage", content="NOISE", height=512, width=512, batch_size=1)
        mask = g.node("StubMask", value=0.5, height=512, width=512, batch_size=1)

        lazy_mix = g.node("TestLazyMixImages", image1=input1.out(0), image2=input2.out(0), mask=mask.out(0))
        g.node("SaveImage", images=lazy_mix.out(0))

        await client.run(g)
        mask.inputs['value'] = 0.4
        result2 = await client.run(g)
        assert not result2.did_run(input1), "Input1 should have been cached"
        assert not result2.did_run(input2), "Input2 should have been cached"

    async def test_error(self, client: Client, builder: GraphBuilder):
        g = builder
        input1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        # Different size of the two images
        input2 = g.node("StubImage", content="NOISE", height=256, width=256, batch_size=1)
        mask = g.node("StubMask", value=0.5, height=512, width=512, batch_size=1)

        lazy_mix = g.node("TestLazyMixImages", image1=input1.out(0), image2=input2.out(0), mask=mask.out(0))
        g.node("SaveImage", images=lazy_mix.out(0))

        try:
            await client.run(g)
            assert False, "Should have raised an error"
        except Exception as e:
            assert 'prompt_id' in e.args[0], f"Did not get back a proper error message: {e}"

    @pytest.mark.parametrize("test_value, expect_error", [
        (5, True),
        ("foo", True),
        (5.0, False),
    ])
    async def test_validation_error_literal(self, test_value, expect_error, client: Client, builder: GraphBuilder):
        g = builder
        validation1 = g.node("TestCustomValidation1", input1=test_value, input2=3.0)
        g.node("SaveImage", images=validation1.out(0))

        if expect_error:
            with pytest.raises(ValueError):
                await client.run(g)
        else:
            await client.run(g)

    @pytest.mark.parametrize("test_type, test_value", [
        ("StubInt", 5),
        ("StubFloat", 5.0)
    ])
    async def test_validation_error_edge1(self, test_type, test_value, client: Client, builder: GraphBuilder):
        g = builder
        stub = g.node(test_type, value=test_value)
        validation1 = g.node("TestCustomValidation1", input1=stub.out(0), input2=3.0)
        g.node("SaveImage", images=validation1.out(0))

        with pytest.raises(ValueError):
            await client.run(g)

    @pytest.mark.parametrize("test_type, test_value, expect_error", [
        ("StubInt", 5, True),
        ("StubFloat", 5.0, False)
    ])
    async def test_validation_error_edge2(self, test_type, test_value, expect_error, client: Client, builder: GraphBuilder):
        g = builder
        stub = g.node(test_type, value=test_value)
        validation2 = g.node("TestCustomValidation2", input1=stub.out(0), input2=3.0)
        g.node("SaveImage", images=validation2.out(0))

        if expect_error:
            with pytest.raises(ValueError):
                await client.run(g)
        else:
            await client.run(g)

    @pytest.mark.parametrize("test_type, test_value, expect_error", [
        ("StubInt", 5, True),
        ("StubFloat", 5.0, False)
    ])
    async def test_validation_error_edge3(self, test_type, test_value, expect_error, client: Client, builder: GraphBuilder):
        g = builder
        stub = g.node(test_type, value=test_value)
        validation3 = g.node("TestCustomValidation3", input1=stub.out(0), input2=3.0)
        g.node("SaveImage", images=validation3.out(0))

        if expect_error:
            with pytest.raises(ValueError):
                await client.run(g)
        else:
            await client.run(g)

    @pytest.mark.parametrize("test_type, test_value, expect_error", [
        ("StubInt", 5, True),
        ("StubFloat", 5.0, False)
    ])
    async def test_validation_error_edge4(self, test_type, test_value, expect_error, client: Client, builder: GraphBuilder):
        g = builder
        stub = g.node(test_type, value=test_value)
        validation4 = g.node("TestCustomValidation4", input1=stub.out(0), input2=3.0)
        g.node("SaveImage", images=validation4.out(0))

        if expect_error:
            with pytest.raises(ValueError):
                await client.run(g)
        else:
            await client.run(g)

    async def test_cycle_error(self, client: Client, builder: GraphBuilder):
        g = builder
        input1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        input2 = g.node("StubImage", content="WHITE", height=512, width=512, batch_size=1)
        mask = g.node("StubMask", value=0.5, height=512, width=512, batch_size=1)

        lazy_mix1 = g.node("TestLazyMixImages", image1=input1.out(0), mask=mask.out(0))
        lazy_mix2 = g.node("TestLazyMixImages", image1=lazy_mix1.out(0), image2=input2.out(0), mask=mask.out(0))
        g.node("SaveImage", images=lazy_mix2.out(0))

        # When the cycle exists on initial submission, it should raise a validation error
        with pytest.raises(ValueError):
            await client.run(g)

    async def test_dynamic_cycle_error(self, client: Client, builder: GraphBuilder):
        g = builder
        input1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        input2 = g.node("StubImage", content="WHITE", height=512, width=512, batch_size=1)
        generator = g.node("TestDynamicDependencyCycle", input1=input1.out(0), input2=input2.out(0))
        g.node("SaveImage", images=generator.out(0))

        # When the cycle is in a graph that is generated dynamically, it should raise a runtime error
        try:
            await client.run(g)
            assert False, "Should have raised an error"
        except Exception as e:
            assert 'prompt_id' in e.args[0], f"Did not get back a proper error message: {e}"
            assert e.args[0]['node_id'] == generator.id, "Error should have been on the generator node"

    async def test_custom_is_changed(self, client: Client, builder: GraphBuilder):
        g = builder
        # Creating the nodes in this specific order previously caused a bug
        save = g.node("SaveImage")
        is_changed = g.node("TestCustomIsChanged", should_change=False)
        input1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)

        save.set_input('images', is_changed.out(0))
        is_changed.set_input('image', input1.out(0))

        result1 = await client.run(g)
        result2 = await client.run(g)
        is_changed.set_input('should_change', True)
        result3 = await client.run(g)
        result4 = await client.run(g)
        assert result1.did_run(is_changed), "is_changed should have been run"
        assert not result2.did_run(is_changed), "is_changed should have been cached"
        assert result3.did_run(is_changed), "is_changed should have been re-run"
        assert result4.did_run(is_changed), "is_changed should not have been cached"

    async def test_undeclared_inputs(self, client: Client, builder: GraphBuilder):
        g = builder
        input1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        input2 = g.node("StubImage", content="WHITE", height=512, width=512, batch_size=1)
        input3 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        input4 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        average = g.node("TestVariadicAverage", input1=input1.out(0), input2=input2.out(0), input3=input3.out(0), input4=input4.out(0))
        output = g.node("SaveImage", images=average.out(0))

        result = await client.run(g)
        result_image = result.get_images(output)[0]
        expected = 255 // 4
        assert numpy.array(result_image).min() == expected and numpy.array(result_image).max() == expected, "Image should be grey"

    async def test_for_loop(self, client: Client, builder: GraphBuilder):
        g = builder
        iterations = 4
        input1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        input2 = g.node("StubImage", content="WHITE", height=512, width=512, batch_size=1)
        is_changed = g.node("TestCustomIsChanged", should_change=True, image=input2.out(0))
        for_open = g.node("TestForLoopOpen", remaining=iterations, initial_value1=is_changed.out(0))
        average = g.node("TestVariadicAverage", input1=input1.out(0), input2=for_open.out(2))
        for_close = g.node("TestForLoopClose", flow_control=for_open.out(0), initial_value1=average.out(0))
        output = g.node("SaveImage", images=for_close.out(0))

        for iterations in range(1, 5):
            for_open.set_input('remaining', iterations)
            result = await client.run(g)
            result_image = result.get_images(output)[0]
            expected = 255 // (2 ** iterations)
            assert numpy.array(result_image).min() == expected and numpy.array(result_image).max() == expected, "Image should be grey"
            assert result.did_run(is_changed)

    async def test_mixed_expansion_returns(self, client: Client, builder: GraphBuilder):
        g = builder
        val_list = g.node("TestMakeListNode", value1=0.1, value2=0.2, value3=0.3)
        mixed = g.node("TestMixedExpansionReturns", input1=val_list.out(0))
        output_dynamic = g.node("SaveImage", images=mixed.out(0))
        output_literal = g.node("SaveImage", images=mixed.out(1))

        result = await client.run(g)
        images_dynamic = result.get_images(output_dynamic)
        assert len(images_dynamic) == 3, "Should have 2 images"
        assert numpy.array(images_dynamic[0]).min() == 25 and numpy.array(images_dynamic[0]).max() == 25, "First image should be 0.1"
        assert numpy.array(images_dynamic[1]).min() == 51 and numpy.array(images_dynamic[1]).max() == 51, "Second image should be 0.2"
        assert numpy.array(images_dynamic[2]).min() == 76 and numpy.array(images_dynamic[2]).max() == 76, "Third image should be 0.3"

        images_literal = result.get_images(output_literal)
        assert len(images_literal) == 3, "Should have 2 images"
        for i in range(3):
            assert numpy.array(images_literal[i]).min() == 255 and numpy.array(images_literal[i]).max() == 255, "All images should be white"

    async def test_output_reuse(self, client: Client, builder: GraphBuilder):
        g = builder
        input1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)

        output1 = g.node("PreviewImage", images=input1.out(0))
        output2 = g.node("PreviewImage", images=input1.out(0))

        result = await client.run(g)
        images1 = result.get_images(output1)
        images2 = result.get_images(output2)
        assert len(images1) == 1, "Should have 1 image"
        assert len(images2) == 1, "Should have 1 image"
