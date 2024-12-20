from io import BytesIO
import numpy
from PIL import Image
import pytest
from pytest import fixture
import time
import torch
from typing import Union, Dict
import json
import subprocess
import websocket #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import urllib.request
import urllib.parse
import urllib.error
from comfy_execution.graph_utils import GraphBuilder, Node

class RunResult:
    def __init__(self, prompt_id: str):
        self.outputs: Dict[str,Dict] = {}
        self.runs: Dict[str,bool] = {}
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

class ComfyClient:
    def __init__(self):
        self.test_name = ""

    def connect(self, 
                    listen:str = '127.0.0.1', 
                    port:Union[str,int] = 8188,
                    client_id: str = str(uuid.uuid4())
                    ):
        self.client_id = client_id
        self.server_address = f"{listen}:{port}"
        ws = websocket.WebSocket()
        ws.connect("ws://{}/ws?clientId={}".format(self.server_address, self.client_id))
        self.ws = ws

    def queue_prompt(self, prompt):
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode('utf-8')
        req =  urllib.request.Request("http://{}/prompt".format(self.server_address), data=data)
        return json.loads(urllib.request.urlopen(req).read())

    def get_image(self, filename, subfolder, folder_type):
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        with urllib.request.urlopen("http://{}/view?{}".format(self.server_address, url_values)) as response:
            return response.read()

    def get_history(self, prompt_id):
        with urllib.request.urlopen("http://{}/history/{}".format(self.server_address, prompt_id)) as response:
            return json.loads(response.read())

    def set_test_name(self, name):
        self.test_name = name

    def run(self, graph):
        prompt = graph.finalize()
        for node in graph.nodes.values():
            if node.class_type == 'SaveImage':
                node.inputs['filename_prefix'] = self.test_name

        prompt_id = self.queue_prompt(prompt)['prompt_id']
        result = RunResult(prompt_id)
        while True:
            out = self.ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'executing':
                    data = message['data']
                    if data['prompt_id'] != prompt_id:
                        continue
                    if data['node'] is None:
                        break
                    result.runs[data['node']] = True
                elif message['type'] == 'execution_error':
                    raise Exception(message['data'])
                elif message['type'] == 'execution_cached':
                    pass # Probably want to store this off for testing

        history = self.get_history(prompt_id)[prompt_id]
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            result.outputs[node_id] = node_output
            images_output = []
            if 'images' in node_output:
                for image in node_output['images']:
                    image_data = self.get_image(image['filename'], image['subfolder'], image['type'])
                    image_obj = Image.open(BytesIO(image_data))
                    images_output.append(image_obj)
                node_output['image_objects'] = images_output

        return result

#
# Loop through these variables
#
@pytest.mark.execution
class TestExecution:
    #
    # Initialize server and client
    #
    @fixture(scope="class", autouse=True, params=[
        # (use_lru, lru_size)
        (False, 0),
        (True, 0),
        (True, 100),
    ])
    def _server(self, args_pytest, request):
        # Start server
        pargs = [
            'python','main.py', 
            '--output-directory', args_pytest["output_dir"],
            '--listen', args_pytest["listen"],
            '--port', str(args_pytest["port"]),
            '--extra-model-paths-config', 'tests/inference/extra_model_paths.yaml',
        ]
        use_lru, lru_size = request.param
        if use_lru:
            pargs += ['--cache-lru', str(lru_size)]
        print("Running server with args:", pargs)  # noqa: T201
        p = subprocess.Popen(pargs)
        yield
        p.kill()
        torch.cuda.empty_cache()

    def start_client(self, listen:str, port:int):
        # Start client
        comfy_client = ComfyClient()
        # Connect to server (with retries)
        n_tries = 5
        for i in range(n_tries):
            time.sleep(4)
            try:
                comfy_client.connect(listen=listen, port=port)
            except ConnectionRefusedError as e:
                print(e)  # noqa: T201
                print(f"({i+1}/{n_tries}) Retrying...")  # noqa: T201
            else:
                break
        return comfy_client

    @fixture(scope="class", autouse=True)
    def shared_client(self, args_pytest, _server):
        client = self.start_client(args_pytest["listen"], args_pytest["port"])
        yield client
        del client
        torch.cuda.empty_cache()

    @fixture
    def client(self, shared_client, request):
        shared_client.set_test_name(f"execution[{request.node.name}]")
        yield shared_client

    @fixture
    def builder(self, request):
        yield GraphBuilder(prefix=request.node.name)

    def test_lazy_input(self, client: ComfyClient, builder: GraphBuilder):
        g = builder
        input1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        input2 = g.node("StubImage", content="WHITE", height=512, width=512, batch_size=1)
        mask = g.node("StubMask", value=0.0, height=512, width=512, batch_size=1)

        lazy_mix = g.node("TestLazyMixImages", image1=input1.out(0), image2=input2.out(0), mask=mask.out(0))
        output = g.node("SaveImage", images=lazy_mix.out(0))
        result = client.run(g)

        result_image = result.get_images(output)[0]
        assert numpy.array(result_image).any() == 0, "Image should be black"
        assert result.did_run(input1)
        assert not result.did_run(input2)
        assert result.did_run(mask)
        assert result.did_run(lazy_mix)

    def test_full_cache(self, client: ComfyClient, builder: GraphBuilder):
        g = builder
        input1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        input2 = g.node("StubImage", content="NOISE", height=512, width=512, batch_size=1)
        mask = g.node("StubMask", value=0.5, height=512, width=512, batch_size=1)

        lazy_mix = g.node("TestLazyMixImages", image1=input1.out(0), image2=input2.out(0), mask=mask.out(0))
        g.node("SaveImage", images=lazy_mix.out(0))

        client.run(g)
        result2 = client.run(g)
        for node_id, node in g.nodes.items():
            assert not result2.did_run(node), f"Node {node_id} ran, but should have been cached"

    def test_partial_cache(self, client: ComfyClient, builder: GraphBuilder):
        g = builder
        input1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        input2 = g.node("StubImage", content="NOISE", height=512, width=512, batch_size=1)
        mask = g.node("StubMask", value=0.5, height=512, width=512, batch_size=1)

        lazy_mix = g.node("TestLazyMixImages", image1=input1.out(0), image2=input2.out(0), mask=mask.out(0))
        g.node("SaveImage", images=lazy_mix.out(0))

        client.run(g)
        mask.inputs['value'] = 0.4
        result2 = client.run(g)
        assert not result2.did_run(input1), "Input1 should have been cached"
        assert not result2.did_run(input2), "Input2 should have been cached"

    def test_error(self, client: ComfyClient, builder: GraphBuilder):
        g = builder
        input1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        # Different size of the two images
        input2 = g.node("StubImage", content="NOISE", height=256, width=256, batch_size=1)
        mask = g.node("StubMask", value=0.5, height=512, width=512, batch_size=1)

        lazy_mix = g.node("TestLazyMixImages", image1=input1.out(0), image2=input2.out(0), mask=mask.out(0))
        g.node("SaveImage", images=lazy_mix.out(0))

        try:
            client.run(g)
            assert False, "Should have raised an error"
        except Exception as e:
            assert 'prompt_id' in e.args[0], f"Did not get back a proper error message: {e}"

    @pytest.mark.parametrize("test_value, expect_error", [
        (5, True),
        ("foo", True),
        (5.0, False),
    ])
    def test_validation_error_literal(self, test_value, expect_error, client: ComfyClient, builder: GraphBuilder):
        g = builder
        validation1 = g.node("TestCustomValidation1", input1=test_value, input2=3.0)
        g.node("SaveImage", images=validation1.out(0))

        if expect_error:
            with pytest.raises(urllib.error.HTTPError):
                client.run(g)
        else:
            client.run(g)

    @pytest.mark.parametrize("test_type, test_value", [
        ("StubInt", 5),
        ("StubFloat", 5.0)
    ])
    def test_validation_error_edge1(self, test_type, test_value, client: ComfyClient, builder: GraphBuilder):
        g = builder
        stub = g.node(test_type, value=test_value)
        validation1 = g.node("TestCustomValidation1", input1=stub.out(0), input2=3.0)
        g.node("SaveImage", images=validation1.out(0))

        with pytest.raises(urllib.error.HTTPError):
            client.run(g)

    @pytest.mark.parametrize("test_type, test_value, expect_error", [
        ("StubInt", 5, True),
        ("StubFloat", 5.0, False)
    ])
    def test_validation_error_edge2(self, test_type, test_value, expect_error, client: ComfyClient, builder: GraphBuilder):
        g = builder
        stub = g.node(test_type, value=test_value)
        validation2 = g.node("TestCustomValidation2", input1=stub.out(0), input2=3.0)
        g.node("SaveImage", images=validation2.out(0))

        if expect_error:
            with pytest.raises(urllib.error.HTTPError):
                client.run(g)
        else:
            client.run(g)

    @pytest.mark.parametrize("test_type, test_value, expect_error", [
        ("StubInt", 5, True),
        ("StubFloat", 5.0, False)
    ])
    def test_validation_error_edge3(self, test_type, test_value, expect_error, client: ComfyClient, builder: GraphBuilder):
        g = builder
        stub = g.node(test_type, value=test_value)
        validation3 = g.node("TestCustomValidation3", input1=stub.out(0), input2=3.0)
        g.node("SaveImage", images=validation3.out(0))

        if expect_error:
            with pytest.raises(urllib.error.HTTPError):
                client.run(g)
        else:
            client.run(g)

    @pytest.mark.parametrize("test_type, test_value, expect_error", [
        ("StubInt", 5, True),
        ("StubFloat", 5.0, False)
    ])
    def test_validation_error_edge4(self, test_type, test_value, expect_error, client: ComfyClient, builder: GraphBuilder):
        g = builder
        stub = g.node(test_type, value=test_value)
        validation4 = g.node("TestCustomValidation4", input1=stub.out(0), input2=3.0)
        g.node("SaveImage", images=validation4.out(0))

        if expect_error:
            with pytest.raises(urllib.error.HTTPError):
                client.run(g)
        else:
            client.run(g)

    @pytest.mark.parametrize("test_value1, test_value2, expect_error", [
        (0.0, 0.5, False),
        (0.0, 5.0, False),
        (0.0, 7.0, True)
    ])
    def test_validation_error_kwargs(self, test_value1, test_value2, expect_error, client: ComfyClient, builder: GraphBuilder):
        g = builder
        validation5 = g.node("TestCustomValidation5", input1=test_value1, input2=test_value2)
        g.node("SaveImage", images=validation5.out(0))

        if expect_error:
            with pytest.raises(urllib.error.HTTPError):
                client.run(g)
        else:
            client.run(g)

    def test_cycle_error(self, client: ComfyClient, builder: GraphBuilder):
        g = builder
        input1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        input2 = g.node("StubImage", content="WHITE", height=512, width=512, batch_size=1)
        mask = g.node("StubMask", value=0.5, height=512, width=512, batch_size=1)

        lazy_mix1 = g.node("TestLazyMixImages", image1=input1.out(0), mask=mask.out(0))
        lazy_mix2 = g.node("TestLazyMixImages", image1=lazy_mix1.out(0), image2=input2.out(0), mask=mask.out(0))
        g.node("SaveImage", images=lazy_mix2.out(0))

        # When the cycle exists on initial submission, it should raise a validation error
        with pytest.raises(urllib.error.HTTPError):
            client.run(g)

    def test_dynamic_cycle_error(self, client: ComfyClient, builder: GraphBuilder):
        g = builder
        input1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        input2 = g.node("StubImage", content="WHITE", height=512, width=512, batch_size=1)
        generator = g.node("TestDynamicDependencyCycle", input1=input1.out(0), input2=input2.out(0))
        g.node("SaveImage", images=generator.out(0))

        # When the cycle is in a graph that is generated dynamically, it should raise a runtime error
        try:
            client.run(g)
            assert False, "Should have raised an error"
        except Exception as e:
            assert 'prompt_id' in e.args[0], f"Did not get back a proper error message: {e}"
            assert e.args[0]['node_id'] == generator.id, "Error should have been on the generator node"

    def test_missing_node_error(self, client: ComfyClient, builder: GraphBuilder):
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
        
        client.run(g)

        # Add back in the missing node to make sure the error doesn't break the server
        input2 = g.node("StubImage", id="removeme", content="WHITE", height=512, width=512, batch_size=1)
        client.run(g)

    def test_custom_is_changed(self, client: ComfyClient, builder: GraphBuilder):
        g = builder
        # Creating the nodes in this specific order previously caused a bug
        save = g.node("SaveImage")
        is_changed = g.node("TestCustomIsChanged", should_change=False)
        input1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)

        save.set_input('images', is_changed.out(0))
        is_changed.set_input('image', input1.out(0))

        result1 = client.run(g)
        result2 = client.run(g)
        is_changed.set_input('should_change', True)
        result3 = client.run(g)
        result4 = client.run(g)
        assert result1.did_run(is_changed), "is_changed should have been run"
        assert not result2.did_run(is_changed), "is_changed should have been cached"
        assert result3.did_run(is_changed), "is_changed should have been re-run"
        assert result4.did_run(is_changed), "is_changed should not have been cached"

    def test_undeclared_inputs(self, client: ComfyClient, builder: GraphBuilder):
        g = builder
        input1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        input2 = g.node("StubImage", content="WHITE", height=512, width=512, batch_size=1)
        input3 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        input4 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        average = g.node("TestVariadicAverage", input1=input1.out(0), input2=input2.out(0), input3=input3.out(0), input4=input4.out(0))
        output = g.node("SaveImage", images=average.out(0))

        result = client.run(g)
        result_image = result.get_images(output)[0]
        expected = 255 // 4
        assert numpy.array(result_image).min() == expected and numpy.array(result_image).max() == expected, "Image should be grey"

    def test_for_loop(self, client: ComfyClient, builder: GraphBuilder):
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
            result = client.run(g)
            result_image = result.get_images(output)[0]
            expected = 255 // (2 ** iterations)
            assert numpy.array(result_image).min() == expected and numpy.array(result_image).max() == expected, "Image should be grey"
            assert result.did_run(is_changed)

    def test_mixed_expansion_returns(self, client: ComfyClient, builder: GraphBuilder):
        g = builder
        val_list = g.node("TestMakeListNode", value1=0.1, value2=0.2, value3=0.3)
        mixed = g.node("TestMixedExpansionReturns", input1=val_list.out(0))
        output_dynamic = g.node("SaveImage", images=mixed.out(0))
        output_literal = g.node("SaveImage", images=mixed.out(1))

        result = client.run(g)
        images_dynamic = result.get_images(output_dynamic)
        assert len(images_dynamic) == 3, "Should have 2 images"
        assert numpy.array(images_dynamic[0]).min() == 25 and numpy.array(images_dynamic[0]).max() == 25, "First image should be 0.1"
        assert numpy.array(images_dynamic[1]).min() == 51 and numpy.array(images_dynamic[1]).max() == 51, "Second image should be 0.2"
        assert numpy.array(images_dynamic[2]).min() == 76 and numpy.array(images_dynamic[2]).max() == 76, "Third image should be 0.3"

        images_literal = result.get_images(output_literal)
        assert len(images_literal) == 3, "Should have 2 images"
        for i in range(3):
            assert numpy.array(images_literal[i]).min() == 255 and numpy.array(images_literal[i]).max() == 255, "All images should be white"

    def test_mixed_lazy_results(self, client: ComfyClient, builder: GraphBuilder):
        g = builder
        val_list = g.node("TestMakeListNode", value1=0.0, value2=0.5, value3=1.0)
        mask = g.node("StubMask", value=val_list.out(0), height=512, width=512, batch_size=1)
        input1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        input2 = g.node("StubImage", content="WHITE", height=512, width=512, batch_size=1)
        mix = g.node("TestLazyMixImages", image1=input1.out(0), image2=input2.out(0), mask=mask.out(0))
        rebatch = g.node("RebatchImages", images=mix.out(0), batch_size=3)
        output = g.node("SaveImage", images=rebatch.out(0))

        result = client.run(g)
        images = result.get_images(output)
        assert len(images) == 3, "Should have 3 image"
        assert numpy.array(images[0]).min() == 0 and numpy.array(images[0]).max() == 0, "First image should be 0.0"
        assert numpy.array(images[1]).min() == 127 and numpy.array(images[1]).max() == 127, "Second image should be 0.5"
        assert numpy.array(images[2]).min() == 255 and numpy.array(images[2]).max() == 255, "Third image should be 1.0"

    def test_output_reuse(self, client: ComfyClient, builder: GraphBuilder):
        g = builder
        input1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)

        output1 = g.node("SaveImage", images=input1.out(0))
        output2 = g.node("SaveImage", images=input1.out(0))

        result = client.run(g)
        images1 = result.get_images(output1)
        images2 = result.get_images(output2)
        assert len(images1) == 1, "Should have 1 image"
        assert len(images2) == 1, "Should have 1 image"


    # This tests that only constant outputs are used in the call to `IS_CHANGED`
    def test_is_changed_with_outputs(self, client: ComfyClient, builder: GraphBuilder):
        g = builder
        input1 = g.node("StubConstantImage", value=0.5, height=512, width=512, batch_size=1)
        test_node = g.node("TestIsChangedWithConstants", image=input1.out(0), value=0.5)

        output = g.node("PreviewImage", images=test_node.out(0))

        result = client.run(g)
        images = result.get_images(output)
        assert len(images) == 1, "Should have 1 image"
        assert numpy.array(images[0]).min() == 63 and numpy.array(images[0]).max() == 63, "Image should have value 0.25"

        result = client.run(g)
        images = result.get_images(output)
        assert len(images) == 1, "Should have 1 image"
        assert numpy.array(images[0]).min() == 63 and numpy.array(images[0]).max() == 63, "Image should have value 0.25"
        assert not result.did_run(test_node), "The execution should have been cached"

    # This tests that nodes with OUTPUT_IS_LIST function correctly when they receive an ExecutionBlocker
    # as input. We also test that when that list (containing an ExecutionBlocker) is passed to a node,
    # only that one entry in the list is blocked.
    def test_execution_block_list_output(self, client: ComfyClient, builder: GraphBuilder):
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

        result = client.run(g)
        assert result.did_run(output), "The execution should have run"
        images = result.get_images(output)
        assert len(images) == 2, "Should have 2 images"
        assert numpy.array(images[0]).min() == 0 and numpy.array(images[0]).max() == 0, "First image should be black"
        assert numpy.array(images[1]).min() == 0 and numpy.array(images[1]).max() == 0, "Second image should also be black"
