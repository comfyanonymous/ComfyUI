"""Gemma4 tokenization regression tests."""

import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy.text_encoders.gemma4 import Gemma4SDTokenizer  # noqa: E402


class _SingleTokenTokenizer:
    def __call__(self, _text):
        return {"input_ids": [7]}


def _ltx_tokenizer():
    tokenizer = Gemma4SDTokenizer.__new__(Gemma4SDTokenizer)
    tokenizer.tokenizer = _SingleTokenTokenizer()
    tokenizer.max_length = 100
    tokenizer.min_length = 8
    tokenizer.min_padding = None
    tokenizer.pad_to_max_length = False
    tokenizer.pad_left = True
    tokenizer.pad_token = 0
    tokenizer.tokens_start = 0
    tokenizer.start_token = 2
    tokenizer.end_token = None
    tokenizer.tokenizer_adds_end_token = False
    tokenizer.max_word_length = 8
    tokenizer.embedding_identifier = "embedding:"
    tokenizer.embedding_directory = None
    tokenizer.embedding_key = "gemma4"
    tokenizer.disable_weights = True
    return tokenizer


def _token_ids(tokens):
    return [token for token, _weight in tokens[0]]


def test_text_generation_min_length_overrides_the_ltx_default():
    tokens = _ltx_tokenizer().tokenize_with_weights("<|turn>model\nanswer", min_length=1)
    assert _token_ids(tokens) == [2, 7]


def test_normal_tokenization_keeps_the_ltx_default_padding():
    tokens = _ltx_tokenizer().tokenize_with_weights("<|turn>model\nanswer")
    assert _token_ids(tokens) == [0, 0, 0, 0, 0, 0, 2, 7]
