from comfy.cli_args import args

args.cpu = True  # importing comfy.model_management probes the torch device at import time

from comfy.text_encoders.gemma4 import Gemma4_Tokenizer

PROMPT = "describe a cute anime girl with fennec ears"


class _CaptureTemplate:
    """Stands in for SDTokenizer.tokenize_with_weights so the built template is checked without model files."""
    def tokenize_with_weights(self, text, return_word_ids=False, **kwargs):
        self.llama_text = text
        return {}


class _Gemma4TemplateProbe(Gemma4_Tokenizer, _CaptureTemplate):
    pass


def build_template(**kwargs):
    probe = _Gemma4TemplateProbe()
    probe.tokenize_with_weights(PROMPT, **kwargs)
    return probe.llama_text


def test_thinking_disabled_does_not_prime_a_thought_channel():
    template = build_template(skip_template=False, thinking=False)
    assert template == f"<|turn>user\n{PROMPT}<turn|>\n<|turn>model\n"
    assert "channel" not in template
    assert "<|think|>" not in template


def test_thinking_enabled_asks_for_the_thought_channel():
    template = build_template(skip_template=False, thinking=True)
    assert template == f"<|turn>system\n<|think|>\n<turn|>\n<|turn>user\n{PROMPT}<turn|>\n<|turn>model\n"


def test_skip_template_passes_text_through_unchanged():
    assert build_template(skip_template=True, thinking=False) == PROMPT
    assert build_template(skip_template=True, thinking=True) == PROMPT
