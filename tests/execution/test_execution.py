import json
import logging
import time
import urllib.request
import uuid
from typing import Dict, Optional, AsyncGenerator

import numpy
import pytest
from PIL import Image
from pytest import fixture

from comfy.cli_args import default_configuration
from comfy.client.embedded_comfy_client import Comfy
from comfy.component_model.executor_types import SendSyncEvent, SendSyncData, ExecutingMessage, ExecutionErrorMessage, \
    DependencyCycleError
from comfy.distributed.server_stub import ServerStub
from comfy.execution_context import context_add_custom_nodes
from comfy.nodes.package_typing import ExportedNodes
from comfy_execution.graph_utils import GraphBuilder, Node
from ..conftest import current_test_name


async def run_warmup(client, prefix="warmup"):
    """Run a simple workflow to warm up the server."""
    warmup_g = GraphBuilder(prefix=prefix)
    warmup_image = warmup_g.node("StubImage", content="BLACK", height=32, width=32, batch_size=1)
    warmup_g.node("PreviewImage", images=warmup_image.out(0))
    await client.run(warmup_g)


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
    def __init__(self, embedded_client: Comfy, progress_handler: _ProgressHandler):
        self.embedded_client = embedded_client
        self.progress_handler = progress_handler

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


# Loop through these variables
@pytest.mark.execution
class TestExecution:
    # Initialize server and client
    @fixture(scope="class", params=[
        # ??? no cache lru, should cache results, etc. etc.
        # (lru_size, should_cache_results)
        (0, True),
        (100, True),
    ])
    async def client(self, request):
        from ..inference.testing_pack import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

        # ??? todo: we have to deal with this
        lru_size, should_cache_results = request.param
        configuration = default_configuration()
        configuration.cache_lru = lru_size

        progress_handler = _ProgressHandler()
        with context_add_custom_nodes(ExportedNodes(NODE_CLASS_MAPPINGS=NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS=NODE_DISPLAY_NAME_MAPPINGS)):
            async with Comfy(configuration, progress_handler=progress_handler) as embedded_client:
                yield ComfyClient(embedded_client, progress_handler)

    @fixture
    def builder(self, request):
        yield GraphBuilder(prefix=request.node.name)

    async def test_lazy_input(self, client: ComfyClient, builder: GraphBuilder):
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

    async def test_full_cache(self, client: ComfyClient, builder: GraphBuilder):
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

    async def test_partial_cache(self, client: ComfyClient, builder: GraphBuilder):
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

    async def test_error(self, client: ComfyClient, builder: GraphBuilder):
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
    async def test_validation_error_literal(self, test_value, expect_error, client: ComfyClient, builder: GraphBuilder):
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
        ("StubMask", 5.0)
    ])
    async def test_validation_error_edge1(self, test_type, test_value, client: ComfyClient, builder: GraphBuilder):
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
    async def test_validation_error_edge2(self, test_type, test_value, expect_error, client: ComfyClient, builder: GraphBuilder):
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
    async def test_validation_error_edge3(self, test_type, test_value, expect_error, client: ComfyClient, builder: GraphBuilder):
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
    async def test_validation_error_edge4(self, test_type, test_value, expect_error, client: ComfyClient, builder: GraphBuilder):
        g = builder
        stub = g.node(test_type, value=test_value)
        validation4 = g.node("TestCustomValidation4", input1=stub.out(0), input2=3.0)
        g.node("SaveImage", images=validation4.out(0))

        if expect_error:
            with pytest.raises(ValueError):
                await client.run(g)
        else:
            await client.run(g)

    async def test_cycle_error(self, client: ComfyClient, builder: GraphBuilder):
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

    async def test_dynamic_cycle_error(self, client: ComfyClient, builder: GraphBuilder):
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

    async def test_custom_is_changed(self, client: ComfyClient, builder: GraphBuilder):
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

    async def test_undeclared_inputs(self, client: ComfyClient, builder: GraphBuilder):
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

    async def test_for_loop(self, client: ComfyClient, builder: GraphBuilder):
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

    async def test_mixed_expansion_returns(self, client: ComfyClient, builder: GraphBuilder):
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

    async def test_mixed_lazy_results(self, client: ComfyClient, builder: GraphBuilder):
        g = builder
        val_list = g.node("TestMakeListNode", value1=0.0, value2=0.5, value3=1.0)
        mask = g.node("StubMask", value=val_list.out(0), height=512, width=512, batch_size=1)
        input1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        input2 = g.node("StubImage", content="WHITE", height=512, width=512, batch_size=1)
        mix = g.node("TestLazyMixImages", image1=input1.out(0), image2=input2.out(0), mask=mask.out(0))
        rebatch = g.node("RebatchImages", images=mix.out(0), batch_size=3)
        output = g.node("SaveImage", images=rebatch.out(0))

        result = await client.run(g)
        images = result.get_images(output)
        assert len(images) == 3, "Should have 3 image"
        assert numpy.array(images[0]).min() == 0 and numpy.array(images[0]).max() == 0, "First image should be 0.0"
        assert numpy.array(images[1]).min() == 127 and numpy.array(images[1]).max() == 127, "Second image should be 0.5"
        assert numpy.array(images[2]).min() == 255 and numpy.array(images[2]).max() == 255, "Third image should be 1.0"

    async def test_missing_node_error(self, client: ComfyClient, builder: GraphBuilder):
        g = builder
        input1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        input2 = g.node("StubImage", id="removeme", content="WHITE", height=512, width=512, batch_size=1)
        input3 = g.node("StubImage", content="WHITE", height=512, width=512, batch_size=1)
        mask = g.node("StubMask", value=0.5, height=512, width=512, batch_size=1)
        mix1 = g.node("TestLazyMixImages", image1=input1.out(0), image2=input2.out(0), mask=mask.out(0))
        mix2 = g.node("TestLazyMixImages", image1=input1.out(0), image2=input3.out(0), mask=mask.out(0))
        # We have multiple outputs. The first is invalid, but the second is valid
        g.node("SaveImage", images=mix1.out(0))
        g.node("SaveImage", images=mix2.out(0))
        g.remove_node("removeme")

        await client.run(g)

        # Add back in the missing node to make sure the error doesn't break the server
        input2 = g.node("StubImage", id="removeme", content="WHITE", height=512, width=512, batch_size=1)
        await client.run(g)

    async def test_output_reuse(self, client: ComfyClient, builder: GraphBuilder):
        g = builder
        input1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)

        output1 = g.node("SaveImage", images=input1.out(0))
        output2 = g.node("SaveImage", images=input1.out(0))

        result = await client.run(g)
        images1 = result.get_images(output1)
        images2 = result.get_images(output2)
        assert len(images1) == 1, "Should have 1 image"
        assert len(images2) == 1, "Should have 1 image"

    # This tests that only constant outputs are used in the call to `IS_CHANGED`
    async def test_is_changed_with_outputs(self, client: ComfyClient, builder: GraphBuilder):
        g = builder
        input1 = g.node("StubConstantImage", value=0.5, height=512, width=512, batch_size=1)
        test_node = g.node("TestIsChangedWithConstants", image=input1.out(0), value=0.5)

        output = g.node("PreviewImage", images=test_node.out(0))

        result = await client.run(g)
        images = result.get_images(output)
        assert len(images) == 1, "Should have 1 image"
        assert numpy.array(images[0]).min() == 63 and numpy.array(images[0]).max() == 63, "Image should have value 0.25"

        result = await client.run(g)
        images = result.get_images(output)
        assert len(images) == 1, "Should have 1 image"
        assert numpy.array(images[0]).min() == 63 and numpy.array(images[0]).max() == 63, "Image should have value 0.25"
        assert not result.did_run(test_node), "The execution should have been cached"

    async def test_parallel_sleep_nodes(self, client: ComfyClient, builder: GraphBuilder, skip_timing_checks):
        # Warmup execution to ensure server is fully initialized
        await run_warmup(client)
        g = builder
        image = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)

        # Create sleep nodes for each duration
        sleep_node1 = g.node("TestSleep", value=image.out(0), seconds=2.9)
        sleep_node2 = g.node("TestSleep", value=image.out(0), seconds=3.1)
        sleep_node3 = g.node("TestSleep", value=image.out(0), seconds=3.0)

        # Add outputs to verify the execution
        _output1 = g.node("PreviewImage", images=sleep_node1.out(0))
        _output2 = g.node("PreviewImage", images=sleep_node2.out(0))
        _output3 = g.node("PreviewImage", images=sleep_node3.out(0))

        start_time = time.time()
        result = await client.run(g)
        elapsed_time = time.time() - start_time

        # The test should take around 3.0 seconds (the longest sleep duration)
        # plus some overhead, but definitely less than the sum of all sleeps (9.0s)
        if not skip_timing_checks:
            assert elapsed_time < 8.9, f"Parallel execution took {elapsed_time}s, expected less than 8.9s"

        # Verify that all nodes executed
        assert result.did_run(sleep_node1), "Sleep node 1 should have run"
        assert result.did_run(sleep_node2), "Sleep node 2 should have run"
        assert result.did_run(sleep_node3), "Sleep node 3 should have run"

    async def test_parallel_sleep_expansion(self, client: ComfyClient, builder: GraphBuilder, skip_timing_checks):
        # Warmup execution to ensure server is fully initialized
        await run_warmup(client)
        g = builder
        # Create input images with different values
        image1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        image2 = g.node("StubImage", content="WHITE", height=512, width=512, batch_size=1)
        image3 = g.node("StubImage", content="WHITE", height=512, width=512, batch_size=1)

        # Create a TestParallelSleep node that expands into multiple TestSleep nodes
        parallel_sleep = g.node("TestParallelSleep",
                                image1=image1.out(0),
                                image2=image2.out(0),
                                image3=image3.out(0),
                                sleep1=4.8,
                                sleep2=4.9,
                                sleep3=5.0)
        output = g.node("SaveImage", images=parallel_sleep.out(0))

        start_time = time.time()
        result = await client.run(g)
        elapsed_time = time.time() - start_time

        # Similar to the previous test, expect parallel execution of the sleep nodes
        # which should complete in less than the sum of all sleeps
        # Lots of leeway here since Windows CI is slow
        if not skip_timing_checks:
            assert elapsed_time < 13.0, f"Expansion execution took {elapsed_time}s"

        # Verify the parallel sleep node executed
        assert result.did_run(parallel_sleep), "ParallelSleep node should have run"

        # Verify we get an image as output (blend of the three input images)
        result_images = result.get_images(output)
        assert len(result_images) == 1, "Should have 1 image"
        # Average pixel value should be around 170 (255 * 2 // 3)
        avg_value = numpy.array(result_images[0]).mean()
        assert avg_value == 170, f"Image average value {avg_value} should be 170"

    # This tests that nodes with OUTPUT_IS_LIST function correctly when they receive an ExecutionBlocker
    # as input. We also test that when that list (containing an ExecutionBlocker) is passed to a node,
    # only that one entry in the list is blocked.
    async def test_execution_block_list_output(self, client: ComfyClient, builder: GraphBuilder):
        g = builder
        image1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        image2 = g.node("StubImage", content="WHITE", height=512, width=512, batch_size=1)
        image3 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        image_list = g.node("TestMakeListNode", value1=image1.out(0), value2=image2.out(0), value3=image3.out(0))
        int1 = g.node("StubInt", value=1)
        int2 = g.node("StubInt", value=2)
        int3 = g.node("StubInt", value=3)
        int_list = g.node("TestMakeListNode", value1=int1.out(0), value2=int2.out(0), value3=int3.out(0))
        compare = g.node("TestIntConditions", a=int_list.out(0), b=2, operation="==")
        blocker = g.node("TestExecutionBlocker", input=image_list.out(0), block=compare.out(0), verbose=False)

        list_output = g.node("TestMakeListNode", value1=blocker.out(0))
        output = g.node("PreviewImage", images=list_output.out(0))

        result = await client.run(g)
        assert result.did_run(output), "The execution should have run"
        images = result.get_images(output)
        assert len(images) == 2, "Should have 2 images"
        assert numpy.array(images[0]).min() == 0 and numpy.array(images[0]).max() == 0, "First image should be black"
        assert numpy.array(images[1]).min() == 0 and numpy.array(images[1]).max() == 0, "Second image should also be black"

    # Output nodes included in the partial execution list are executed
    async def test_partial_execution_included_outputs(self, client: ComfyClient, builder: GraphBuilder):
        g = builder
        input1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        input2 = g.node("StubImage", content="WHITE", height=512, width=512, batch_size=1)

        # Create two separate output nodes
        output1 = g.node("SaveImage", images=input1.out(0))
        output2 = g.node("SaveImage", images=input2.out(0))

        # Run with partial execution targeting only output1
        result = await client.run(g, partial_execution_targets=[output1.id])

        assert result.was_executed(input1), "Input1 should have been executed (run or cached)"
        assert result.was_executed(output1), "Output1 should have been executed (run or cached)"
        assert not result.did_run(input2), "Input2 should not have run"
        assert not result.did_run(output2), "Output2 should not have run"

        # Verify only output1 produced results
        assert len(result.get_images(output1)) == 1, "Output1 should have produced an image"
        assert len(result.get_images(output2)) == 0, "Output2 should not have produced an image"

    # Output nodes NOT included in the partial execution list are NOT executed
    async def test_partial_execution_excluded_outputs(self, client: ComfyClient, builder: GraphBuilder):
        g = builder
        input1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        input2 = g.node("StubImage", content="WHITE", height=512, width=512, batch_size=1)
        input3 = g.node("StubImage", content="NOISE", height=512, width=512, batch_size=1)

        # Create three output nodes
        output1 = g.node("SaveImage", images=input1.out(0))
        output2 = g.node("SaveImage", images=input2.out(0))
        output3 = g.node("SaveImage", images=input3.out(0))

        # Run with partial execution targeting only output1 and output3
        result = await client.run(g, partial_execution_targets=[output1.id, output3.id])

        assert result.was_executed(input1), "Input1 should have been executed"
        assert result.was_executed(input3), "Input3 should have been executed"
        assert result.was_executed(output1), "Output1 should have been executed"
        assert result.was_executed(output3), "Output3 should have been executed"
        assert not result.did_run(input2), "Input2 should not have run"
        assert not result.did_run(output2), "Output2 should not have run"

    # Output nodes NOT in list ARE executed if necessary for nodes that are in the list
    async def test_partial_execution_dependencies(self, client: ComfyClient, builder: GraphBuilder):
        g = builder
        input1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)

        # Create a processing chain with an OUTPUT_NODE that has socket outputs
        output_with_socket = g.node("TestOutputNodeWithSocketOutput", image=input1.out(0), value=2.0)

        # Create another node that depends on the output_with_socket
        dependent_node = g.node("TestLazyMixImages",
                                image1=output_with_socket.out(0),
                                image2=input1.out(0),
                                mask=g.node("StubMask", value=0.5, height=512, width=512, batch_size=1).out(0))

        # Create the final output
        final_output = g.node("SaveImage", images=dependent_node.out(0))

        # Run with partial execution targeting only the final output
        result = await client.run(g, partial_execution_targets=[final_output.id])

        # All nodes should have been executed because they're dependencies
        assert result.was_executed(input1), "Input1 should have been executed"
        assert result.was_executed(output_with_socket), "Output with socket should have been executed (dependency)"
        assert result.was_executed(dependent_node), "Dependent node should have been executed"
        assert result.was_executed(final_output), "Final output should have been executed"

    # Lazy execution works with partial execution
    async def test_partial_execution_with_lazy_nodes(self, client: ComfyClient, builder: GraphBuilder):
        g = builder
        input1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        input2 = g.node("StubImage", content="WHITE", height=512, width=512, batch_size=1)
        input3 = g.node("StubImage", content="NOISE", height=512, width=512, batch_size=1)

        # Create masks that will trigger different lazy execution paths
        mask1 = g.node("StubMask", value=0.0, height=512, width=512, batch_size=1)  # Will only need image1
        mask2 = g.node("StubMask", value=0.5, height=512, width=512, batch_size=1)  # Will need both images

        # Create two lazy mix nodes
        lazy_mix1 = g.node("TestLazyMixImages", image1=input1.out(0), image2=input2.out(0), mask=mask1.out(0))
        lazy_mix2 = g.node("TestLazyMixImages", image1=input2.out(0), image2=input3.out(0), mask=mask2.out(0))

        output1 = g.node("SaveImage", images=lazy_mix1.out(0))
        output2 = g.node("SaveImage", images=lazy_mix2.out(0))

        # Run with partial execution targeting only output1
        result = await client.run(g, partial_execution_targets=[output1.id])

        # For output1 path - only input1 should run due to lazy evaluation (mask=0.0)
        assert result.was_executed(input1), "Input1 should have been executed"
        assert not result.did_run(input2), "Input2 should not have run (lazy evaluation)"
        assert result.was_executed(mask1), "Mask1 should have been executed"
        assert result.was_executed(lazy_mix1), "Lazy mix1 should have been executed"
        assert result.was_executed(output1), "Output1 should have been executed"

        # Nothing from output2 path should run
        assert not result.did_run(input3), "Input3 should not have run"
        assert not result.did_run(mask2), "Mask2 should not have run"
        assert not result.did_run(lazy_mix2), "Lazy mix2 should not have run"
        assert not result.did_run(output2), "Output2 should not have run"

    # Multiple OUTPUT_NODEs with dependencies
    async def test_partial_execution_multiple_output_nodes(self, client: ComfyClient, builder: GraphBuilder):
        g = builder
        input1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        input2 = g.node("StubImage", content="WHITE", height=512, width=512, batch_size=1)

        # Create a chain of OUTPUT_NODEs
        output_node1 = g.node("TestOutputNodeWithSocketOutput", image=input1.out(0), value=1.5)
        output_node2 = g.node("TestOutputNodeWithSocketOutput", image=output_node1.out(0), value=2.0)

        # Create regular output nodes
        save1 = g.node("SaveImage", images=output_node1.out(0))
        save2 = g.node("SaveImage", images=output_node2.out(0))
        save3 = g.node("SaveImage", images=input2.out(0))

        # Run targeting only save2
        result = await client.run(g, partial_execution_targets=[save2.id])

        # Should run: input1, output_node1, output_node2, save2
        assert result.was_executed(input1), "Input1 should have been executed"
        assert result.was_executed(output_node1), "Output node 1 should have been executed (dependency)"
        assert result.was_executed(output_node2), "Output node 2 should have been executed (dependency)"
        assert result.was_executed(save2), "Save2 should have been executed"

        # Should NOT run: input2, save1, save3
        assert not result.did_run(input2), "Input2 should not have run"
        assert not result.did_run(save1), "Save1 should not have run"
        assert not result.did_run(save3), "Save3 should not have run"

    # Empty partial execution list (should execute nothing)
    async def test_partial_execution_empty_list(self, client: ComfyClient, builder: GraphBuilder):
        g = builder
        input1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        _output1 = g.node("SaveImage", images=input1.out(0))

        # Run with empty partial execution list
        try:
            _result = await client.run(g, partial_execution_targets=[])
            # Should get an error because no outputs are selected
            assert False, "Should have raised an error for empty partial execution list"
        except Exception:
            pass  # Expected behavior

    async def _create_history_item(self, client, builder):
        g = GraphBuilder(prefix="offset_test")
        input_node = g.node(
            "StubImage", content="BLACK", height=32, width=32, batch_size=1
        )
        g.node("SaveImage", images=input_node.out(0))
        return await client.run(g)

    async def test_offset_returns_different_items_than_beginning_of_history(
            self, client: ComfyClient, builder: GraphBuilder
    ):
        """Test that offset skips items at the beginning"""
        for _ in range(5):
            await self._create_history_item(client, builder)

        first_two = client.get_all_history(max_items=2, offset=0)
        next_two = client.get_all_history(max_items=2, offset=2)

        assert set(first_two.keys()).isdisjoint(
            set(next_two.keys())
        ), "Offset should skip initial items"

    async def test_offset_beyond_history_length_returns_empty(
            self, client: ComfyClient, builder: GraphBuilder
    ):
        """Test offset larger than total history returns empty result"""
        await self._create_history_item(client, builder)

        result = client.get_all_history(offset=100)
        assert len(result) == 0, "Large offset should return no items"

    async def test_offset_at_exact_history_length_returns_empty(
            self, client: ComfyClient, builder: GraphBuilder
    ):
        """Test offset equal to history length returns empty"""
        for _ in range(3):
            await self._create_history_item(client, builder)

        all_history = client.get_all_history()
        result = client.get_all_history(offset=len(all_history))
        assert len(result) == 0, "Offset at history length should return empty"

    async def test_offset_zero_equals_no_offset_parameter(
            self, client: ComfyClient, builder: GraphBuilder
    ):
        """Test offset=0 behaves same as omitting offset"""
        await self._create_history_item(client, builder)

        with_zero = client.get_all_history(offset=0)
        without_offset = client.get_all_history()

        assert with_zero == without_offset, "offset=0 should equal no offset"

    async def test_offset_without_max_items_skips_from_beginning(
            self, client: ComfyClient, builder: GraphBuilder
    ):
        """Test offset alone (no max_items) returns remaining items"""
        for _ in range(4):
            await self._create_history_item(client, builder)

        all_items = client.get_all_history()
        offset_items = client.get_all_history(offset=2)

        assert (
                len(offset_items) == len(all_items) - 2
        ), "Offset should skip specified number of items"

    async def test_offset_with_max_items_returns_correct_window(
            self, client: ComfyClient, builder: GraphBuilder
    ):
        """Test offset + max_items returns correct slice of history"""
        for _ in range(6):
            await self._create_history_item(client, builder)

        window = client.get_all_history(max_items=2, offset=1)
        assert len(window) <= 2, "Should respect max_items limit"

    async def test_offset_near_end_returns_remaining_items_only(
            self, client: ComfyClient, builder: GraphBuilder
    ):
        """Test offset near end of history returns only remaining items"""
        for _ in range(3):
            await self._create_history_item(client, builder)

        all_history = client.get_all_history()
        # Offset to near the end
        result = client.get_all_history(max_items=5, offset=len(all_history) - 1)

        assert len(result) <= 1, "Should return at most 1 item when offset is near end"

    async def test_lazy_switch_true_branch(self, client: ComfyClient, builder: GraphBuilder):
        await client.embedded_client.clear_cache()
        g = builder
        # Create a "True" boolean value
        true_int = g.node("StubInt", value=1)
        true_bool = g.node("TestIntConditions", a=true_int.out(0), b=1, operation="==")  # 1 == 1 -> True

        # Create nodes for branches
        node_true = g.node("StubImage", content="WHITE", height=32, width=32, batch_size=1)
        node_false = g.node("StubImage", content="BLACK", height=32, width=32, batch_size=1)

        # Create lazy switch
        # Note: LazySwitch is imported at the top of the file
        switch = g.node("LazySwitch", switch=true_bool.out(0), on_true=node_true.out(0), on_false=node_false.out(0))
        output = g.node("SaveImage", images=switch.out(0))

        result = await client.run(g)

        # Check execution
        assert result.did_run(true_int), "True stub int should run"
        assert result.did_run(true_bool), "Boolean condition node should run"
        assert result.did_run(node_true), "on_true node should run"
        assert not result.did_run(node_false), "on_false node should NOT run"
        assert result.did_run(switch), "LazySwitch node should run"
        assert result.did_run(output), "SaveImage node should run"

        # Check output
        result_image = result.get_images(output)[0]
        assert numpy.array(result_image).mean() == 255, "Image should be white"

    async def test_lazy_switch_false_branch(self, client: ComfyClient, builder: GraphBuilder):
        await client.embedded_client.clear_cache()
        g = builder
        # Create a "False" boolean value
        false_int = g.node("StubInt", value=0)
        false_bool = g.node("TestIntConditions", a=false_int.out(0), b=1, operation="==")  # 0 == 1 -> False

        # Create nodes for branches
        node_true = g.node("StubImage", content="WHITE", height=32, width=32, batch_size=1)
        node_false = g.node("StubImage", content="BLACK", height=32, width=32, batch_size=1)

        # Create lazy switch
        switch = g.node("LazySwitch", switch=false_bool.out(0), on_true=node_true.out(0), on_false=node_false.out(0))
        output = g.node("SaveImage", images=switch.out(0))

        result = await client.run(g)

        # Check execution
        assert result.did_run(false_int), "False stub int should run"
        assert result.did_run(false_bool), "Boolean condition node should run"
        assert not result.did_run(node_true), "on_true node should NOT run"
        assert result.did_run(node_false), "on_false node should run"
        assert result.did_run(switch), "LazySwitch node should run"
        assert result.did_run(output), "SaveImage node should run"

        # Check output
        result_image = result.get_images(output)[0]
        assert numpy.array(result_image).mean() == 0, "Image should be black"

    async def test_lazy_binary_op_and_short_circuit(self, client: ComfyClient, builder: GraphBuilder):
        await client.embedded_client.clear_cache()
        g = builder
        # Create a "False" boolean value
        false_int = g.node("StubInt", value=0)
        lhs_bool = g.node("TestIntConditions", a=false_int.out(0), b=1, operation="==")  # 0 == 1 -> False

        # Create a "True" boolean value for RHS (this node should not run)
        true_int_rhs = g.node("StubInt", value=1)
        rhs_bool = g.node("TestIntConditions", a=true_int_rhs.out(0), b=1, operation="==")  # 1 == 1 -> True

        # Create binary op
        # Note: BinaryOperation is imported at the top of the file
        binary_op = g.node("BinaryOperation", lhs=lhs_bool.out(0), op="and", rhs=rhs_bool.out(0))

        # Create lazy switch to check result
        node_true = g.node("StubImage", content="WHITE", height=32, width=32, batch_size=1)
        node_false = g.node("StubImage", content="BLACK", height=32, width=32, batch_size=1)
        switch = g.node("LazySwitch", switch=binary_op.out(0), on_true=node_true.out(0), on_false=node_false.out(0))
        output = g.node("SaveImage", images=switch.out(0))

        result = await client.run(g)

        # Check execution
        assert result.did_run(false_int), "LHS int node should run"
        assert result.did_run(lhs_bool), "LHS bool node should run"
        assert not result.did_run(true_int_rhs), "RHS int node should NOT run (short-circuit)"
        assert not result.did_run(rhs_bool), "RHS bool node should NOT run (short-circuit)"
        assert result.did_run(binary_op), "BinaryOp should run"
        assert not result.did_run(node_true), "on_true node should NOT run"
        assert result.did_run(node_false), "on_false node should run"

        # Check output
        result_image = result.get_images(output)[0]
        assert numpy.array(result_image).mean() == 0, "Image should be black (result of 'and' was False)"

    async def test_lazy_binary_op_or_short_circuit(self, client: ComfyClient, builder: GraphBuilder):
        await client.embedded_client.clear_cache()
        g = builder
        # Create a "True" boolean value
        true_int = g.node("StubInt", value=1)
        lhs_bool = g.node("TestIntConditions", a=true_int.out(0), b=1, operation="==")  # 1 == 1 -> True

        # Create a "False" boolean value for RHS (this node should not run)
        false_int_rhs = g.node("StubInt", value=0)
        rhs_bool = g.node("TestIntConditions", a=false_int_rhs.out(0), b=1, operation="==")  # 0 == 1 -> False

        # Create binary op
        binary_op = g.node("BinaryOperation", lhs=lhs_bool.out(0), op="or", rhs=rhs_bool.out(0))

        # Create lazy switch to check result
        node_true = g.node("StubImage", content="WHITE", height=32, width=32, batch_size=1)
        node_false = g.node("StubImage", content="BLACK", height=32, width=32, batch_size=1)
        switch = g.node("LazySwitch", switch=binary_op.out(0), on_true=node_true.out(0), on_false=node_false.out(0))
        output = g.node("SaveImage", images=switch.out(0))

        result = await client.run(g)

        # Check execution
        assert result.did_run(true_int), "LHS int node should run"
        assert result.did_run(lhs_bool), "LHS bool node should run"
        assert not result.did_run(false_int_rhs), "RHS int node should NOT run (short-circuit)"
        assert not result.did_run(rhs_bool), "RHS bool node should NOT run (short-circuit)"
        assert result.did_run(binary_op), "BinaryOp should run"
        assert result.did_run(node_true), "on_true node should run"
        assert not result.did_run(node_false), "on_false node should NOT run"

        # Check output
        result_image = result.get_images(output)[0]
        assert numpy.array(result_image).mean() == 255, "Image should be white (result of 'or' was True)"

    async def test_lazy_switch_with_none_input(self, client: ComfyClient, builder: GraphBuilder):
        await client.embedded_client.clear_cache()
        g = builder
        # Create a "False" boolean value
        false_int = g.node("StubInt", value=0)
        false_bool = g.node("TestIntConditions", a=false_int.out(0), b=1, operation="==")  # 0 == 1 -> False

        # Create nodes for branches
        # This node will return None as its value is empty and default_if_empty is not set
        node_true_image = g.node("ImageRequestParameter", value="", description="1")
        node_false_image = g.node("StubImage", content="BLACK", height=32, width=32, batch_size=1)

        # Create lazy switch
        switch = g.node("LazySwitch", switch=false_bool.out(0), on_true=node_true_image.out(0), on_false=node_false_image.out(0))
        output = g.node("SaveImage", images=switch.out(0))

        result = await client.run(g)

        # Check execution
        assert result.did_run(false_int), "False stub int should run"
        assert result.did_run(false_bool), "Boolean condition node should run"
        assert not result.did_run(node_true_image), "on_true (ImageRequestParameter) node should NOT run"
        assert result.did_run(node_false_image), "on_false node should run"
        assert result.did_run(switch), "LazySwitch node should run"
        assert result.did_run(output), "SaveImage node should run"

        # Check output
        result_image = result.get_images(output)[0]
        assert numpy.array(result_image).mean() == 0, "Image should be black"
