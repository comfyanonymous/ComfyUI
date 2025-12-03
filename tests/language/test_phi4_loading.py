import pytest
from comfy_execution.graph_utils import GraphBuilder
from comfy.client.embedded_comfy_client import Comfy
from comfy.api.components.schema.prompt import Prompt


class TestPhi4Loading:
    @pytest.mark.asyncio
    async def test_phi4_loading(self):
        graph = GraphBuilder()
        model_loader = graph.node("TransformersLoaderQuantized", ckpt_name="microsoft/phi-4", load_in_4bit=True, load_in_8bit=False)
        tokenizer = graph.node("OneShotInstructTokenize", model=model_loader.out(0), prompt="Hello", chat_template="default")
        generation = graph.node("TransformersGenerate", model=model_loader.out(0), tokens=tokenizer.out(0), max_new_tokens=1, seed=42)
        graph.node("SaveString", value=generation.out(0), filename_prefix="phi4_test")

        workflow = graph.finalize()
        prompt = Prompt.validate(workflow)

        async with Comfy() as client:
            await client.queue_prompt(prompt)
