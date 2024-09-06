import torch

from comfy_extras.nodes.nodes_language import TransformersLoader, OneShotInstructTokenize


def test_integration_transformers_loader_and_tokenize():
    loader = TransformersLoader()
    tokenize = OneShotInstructTokenize()

    model, = loader.execute("llava-hf/llava-v1.6-mistral-7b-hf", "")
    tokens, = tokenize.execute(model, "Describe this image:", torch.rand((1, 224, 224, 3)), "llava-v1.6-mistral-7b-hf", )

    assert isinstance(tokens, dict)
    assert "input_ids" in tokens or "inputs" in tokens
