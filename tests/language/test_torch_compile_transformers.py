import pytest
import torch
from comfy_execution.graph_utils import GraphBuilder
from comfy.client.embedded_comfy_client import Comfy
from comfy.api.components.schema.prompt import Prompt


class TestTorchCompileTransformers:
    @pytest.mark.asyncio
    async def test_torch_compile_transformers(self):
        graph = GraphBuilder()
        model_loader = graph.node("TransformersLoader1", ckpt_name="Qwen/Qwen2.5-0.5B")
        compiled_model = graph.node("TorchCompileModel", model=model_loader.out(0), backend="inductor", mode="max-autotune")
        tokenizer = graph.node("OneShotInstructTokenize", model=compiled_model.out(0), prompt="Hello, world!", chat_template="default")
        generation = graph.node("TransformersGenerate", model=compiled_model.out(0), tokens=tokenizer.out(0), max_new_tokens=10, seed=42)

        save_string = graph.node("SaveString", value=generation.out(0), filename_prefix="test_output")

        workflow = graph.finalize()
        prompt = Prompt.validate(workflow)

        from unittest.mock import patch
        with patch("torch.compile", side_effect=torch.compile) as mock_compile:
            async with Comfy() as client:
                outputs = await client.queue_prompt(prompt)
            
            assert mock_compile.called, "torch.compile should have been called"

        assert len(outputs) > 0
        assert save_string.id in outputs
        assert outputs[save_string.id]["string"][0] is not None
