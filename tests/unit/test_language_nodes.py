import io
import os
import tempfile
from unittest.mock import patch, Mock

import pytest
import torch
from PIL import Image

from comfy.language.language_types import LanguageModel, ProcessorResult
from comfy_extras.nodes.nodes_language import SaveString
from comfy_extras.nodes.nodes_language import TransformersLoader, OneShotInstructTokenize, TransformersGenerate, \
    PreviewString
from comfy_extras.nodes.nodes_openai import OpenAILanguageModelLoader, OpenAILanguageModelWrapper, DallEGenerate


@pytest.fixture
def save_string_node():
    return SaveString()


@pytest.fixture
def mock_get_save_path(save_string_node):
    with patch.object(save_string_node, 'get_save_path') as mock_method:
        mock_method.return_value = (tempfile.gettempdir(), "test", 0, "", "test")
        yield mock_method


def test_save_string_single(save_string_node, mock_get_save_path):
    test_string = "Test string content"
    result = save_string_node.execute(test_string, "test_prefix", ".txt")

    assert result == {"ui": {"string": [test_string]}}
    mock_get_save_path.assert_called_once_with("test_prefix")

    saved_file_path = os.path.join(tempfile.gettempdir(), "test_00000_.txt")
    assert os.path.exists(saved_file_path)
    with open(saved_file_path, "r") as f:
        assert f.read() == test_string


def test_save_string_list(save_string_node, mock_get_save_path):
    test_strings = ["First string", "Second string", "Third string"]
    result = save_string_node.execute(test_strings, "test_prefix", ".txt")

    assert result == {"ui": {"string": test_strings}}
    mock_get_save_path.assert_called_once_with("test_prefix")

    for i, test_string in enumerate(test_strings):
        saved_file_path = os.path.join(tempfile.gettempdir(), f"test_00000_{i:02d}_.txt")
        assert os.path.exists(saved_file_path)
        with open(saved_file_path, "r") as f:
            assert f.read() == test_string


def test_save_string_default_extension(save_string_node, mock_get_save_path):
    test_string = "Test string content"
    result = save_string_node.execute(test_string, "test_prefix")

    assert result == {"ui": {"string": [test_string]}}
    mock_get_save_path.assert_called_once_with("test_prefix")

    saved_file_path = os.path.join(tempfile.gettempdir(), "test_00000_.json")
    assert os.path.exists(saved_file_path)
    with open(saved_file_path, "r") as f:
        assert f.read() == test_string


@pytest.fixture
def mock_openai_client():
    with patch('comfy_extras.nodes.nodes_openai._Client') as mock_client:
        instance = mock_client.instance.return_value
        instance.chat.completions.create = Mock()
        instance.images.generate = Mock()
        yield instance

@pytest.mark.skip("broken transformers")
def test_transformers_loader(has_gpu):
    if not has_gpu:
        pytest.skip("requires GPU")
    loader = TransformersLoader()
    model, = loader.execute("microsoft/Phi-3-mini-4k-instruct", "")
    assert isinstance(model, LanguageModel)
    assert model.repo_id == "microsoft/Phi-3-mini-4k-instruct"


def test_one_shot_instruct_tokenize(mocker):
    tokenize = OneShotInstructTokenize()
    mock_model = mocker.Mock()
    mock_model.tokenize.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

    tokens, = tokenize.execute(mock_model, "What comes after apple?", [], "phi-3")
    mock_model.tokenize.assert_called_once_with("What comes after apple?", [], mocker.ANY)
    assert "input_ids" in tokens


def test_transformers_generate(mocker):
    generate = TransformersGenerate()
    mock_model = mocker.Mock()
    mock_model.generate.return_value = "The letter B comes after A in the alphabet."

    tokens: ProcessorResult = {"inputs": torch.tensor([[1, 2, 3]])}
    result, = generate.execute(mock_model, tokens, 512, 0, 42)
    mock_model.generate.assert_called_once()
    assert isinstance(result, str)
    assert "letter B" in result


def test_preview_string():
    preview = PreviewString()
    result = preview.execute("Test output")
    assert result == {"ui": {"string": ["Test output"]}}


def test_openai_language_model_loader():
    if not "OPENAI_API_KEY" in os.environ:
        pytest.skip("must set OPENAI_API_KEY")
    loader = OpenAILanguageModelLoader()
    model, = loader.execute("gpt-3.5-turbo")
    assert isinstance(model, OpenAILanguageModelWrapper)
    assert model.model == "gpt-3.5-turbo"


def test_openai_language_model_wrapper_generate(mock_openai_client):
    wrapper = OpenAILanguageModelWrapper("gpt-3.5-turbo")
    mock_stream = [
        Mock(choices=[Mock(delta=Mock(content="This "))]),
        Mock(choices=[Mock(delta=Mock(content="is "))]),
        Mock(choices=[Mock(delta=Mock(content="a "))]),
        Mock(choices=[Mock(delta=Mock(content="test "))]),
        Mock(choices=[Mock(delta=Mock(content="response."))]),
    ]

    mock_openai_client.chat.completions.create.return_value = mock_stream

    tokens = {"inputs": ["What is the capital of France?"]}
    result = wrapper.generate(tokens, max_new_tokens=50)

    mock_openai_client.chat.completions.create.assert_called_once_with(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": [{"type": "text", "text": "What is the capital of France?"}]}],
        max_tokens=50,
        temperature=1.0,
        top_p=1.0,
        seed=0,
        stream=True
    )
    assert result == "This is a test response."


def test_openai_language_model_wrapper_generate_with_image(mock_openai_client):
    wrapper = OpenAILanguageModelWrapper("gpt-4-vision-preview")
    mock_stream = [
        Mock(choices=[Mock(delta=Mock(content="This "))]),
        Mock(choices=[Mock(delta=Mock(content="image "))]),
        Mock(choices=[Mock(delta=Mock(content="shows "))]),
        Mock(choices=[Mock(delta=Mock(content="a "))]),
        Mock(choices=[Mock(delta=Mock(content="landscape."))]),
    ]
    mock_openai_client.chat.completions.create.return_value = mock_stream

    image_tensor = torch.rand((1, 224, 224, 3))
    tokens: ProcessorResult = {
        "inputs": ["Describe this image:"],
        "images": image_tensor
    }
    result = wrapper.generate(tokens, max_new_tokens=50)

    mock_openai_client.chat.completions.create.assert_called_once()
    assert result == "This image shows a landscape."


def test_dalle_generate(mock_openai_client):
    dalle = DallEGenerate()
    mock_openai_client.images.generate.return_value = Mock(
        data=[Mock(url="http://example.com/image.jpg", revised_prompt="A beautiful sunset")]
    )
    test_image = Image.new('RGB', (10, 10), color='red')
    img_byte_arr = io.BytesIO()
    test_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    with patch('requests.get') as mock_get:
        mock_get.return_value = Mock(content=img_byte_arr)
        image, revised_prompt = dalle.generate("dall-e-3", "Create a sunset image", "1024x1024", "standard")

    assert isinstance(image, torch.Tensor)
    assert image.shape == (1, 10, 10, 3)
    assert torch.allclose(image, torch.tensor([1.0, 0, 0]).view(1, 1, 1, 3).expand(1, 10, 10, 3))
    assert revised_prompt == "A beautiful sunset"
    mock_openai_client.images.generate.assert_called_once_with(
        model="dall-e-3",
        prompt="Create a sunset image",
        size="1024x1024",
        quality="standard",
        n=1,
    )


def test_integration_openai_loader_and_wrapper(mock_openai_client):
    loader = OpenAILanguageModelLoader()
    model, = loader.execute("gpt-4")

    mock_stream = [
        Mock(choices=[Mock(delta=Mock(content="Paris "))]),
        Mock(choices=[Mock(delta=Mock(content="is "))]),
        Mock(choices=[Mock(delta=Mock(content="the "))]),
        Mock(choices=[Mock(delta=Mock(content="capital "))]),
        Mock(choices=[Mock(delta=Mock(content="of France."))]),
    ]
    mock_openai_client.chat.completions.create.return_value = mock_stream

    tokens = {"inputs": ["What is the capital of France?"]}
    result = model.generate(tokens, max_new_tokens=50)

    assert result == "Paris is the capital of France."
