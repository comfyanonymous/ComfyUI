from comfy_api_nodes.apis.anthropic import (
    AnthropicMessage,
    AnthropicMessagesRequest,
    AnthropicRole,
    AnthropicTextContent,
    get_supported_temperature,
)


def test_claude_opus_47_omits_temperature():
    request = AnthropicMessagesRequest(
        model="claude-opus-4-7",
        max_tokens=128,
        messages=[AnthropicMessage(role=AnthropicRole.user, content=[AnthropicTextContent(text="Hello")])],
        temperature=get_supported_temperature("claude-opus-4-7", 0.25),
    )

    assert request.model == "claude-opus-4-7"
    assert request.temperature is None
    assert "temperature" not in request.model_dump(exclude_none=True)


def test_claude_models_that_support_sampling_keep_temperature():
    request = AnthropicMessagesRequest(
        model="claude-sonnet-4-6",
        max_tokens=128,
        messages=[AnthropicMessage(role=AnthropicRole.user, content=[AnthropicTextContent(text="Hello")])],
        temperature=get_supported_temperature("claude-sonnet-4-6", 0.25),
    )

    assert request.model == "claude-sonnet-4-6"
    assert request.temperature == 0.25
    assert request.model_dump(exclude_none=True)["temperature"] == 0.25
