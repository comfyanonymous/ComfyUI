import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy.text_encoders.gemma4 import Gemma4SDTokenizer  # noqa: E402


class _StubTokenizer:
    """Returns a canned decode so the marker translation can be tested without model files."""
    def __init__(self, text):
        self.text = text

    def decode(self, token_ids, skip_special_tokens=False):
        return self.text


def decode(text):
    tokenizer = Gemma4SDTokenizer.__new__(Gemma4SDTokenizer)
    tokenizer.tokenizer = _StubTokenizer(text)
    return tokenizer.decode([])


class TestGemma4Decode:
    def test_thought_channel_becomes_think_tags(self):
        assert decode("<|channel>thought\nreasoning<channel|>the answer") == "<think>\nreasoning</think>the answer"

    def test_primed_empty_thought_channel(self):
        assert decode("<|channel>thought\n<channel|>the answer") == "<think>\n</think>the answer"

    def test_unclosed_thought_channel(self):
        assert decode("<|channel>thought\nreasoning") == "<think>\nreasoning"

    def test_other_channel_close_is_not_reasoning(self):
        # Non-thinking LTX2 prompt enhancement primes a "final" channel, so only its close is
        # generated. Turning that into </think> made the whole answer look like reasoning.
        assert decode("the answer<channel|>") == "the answer<channel|>"

    def test_other_channel_kept_after_a_thought_channel(self):
        assert decode("<|channel>thought\nreasoning<channel|><|channel>final\nthe answer<channel|>") == "<think>\nreasoning</think><|channel>final\nthe answer<channel|>"

    def test_turn_and_eos_are_stripped(self):
        assert decode("the answer<turn|><eos>") == "the answer"
