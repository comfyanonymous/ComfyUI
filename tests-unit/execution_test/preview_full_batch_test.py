"""Unit tests for the --preview-full-batch CLI option."""
from comfy.cli_args import args, parser


class TestPreviewFullBatchArg:
    def test_default_is_false(self):
        assert args.preview_full_batch is False

    def test_parser_sets_true(self):
        ns = parser.parse_args(["--preview-full-batch"])
        assert ns.preview_full_batch is True

    def test_parser_without_flag_is_false(self):
        ns = parser.parse_args([])
        assert ns.preview_full_batch is False
