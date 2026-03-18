import pytest
from comfy_api_nodes.apis.minimax import MiniMaxChatModel


class TestMiniMaxChatModel:
    def test_m27_in_model_list(self):
        """MiniMax-M2.7 should be available in the chat model enum."""
        assert MiniMaxChatModel.M2_7.value == 'MiniMax-M2.7'

    def test_m27_highspeed_in_model_list(self):
        """MiniMax-M2.7-highspeed should be available in the chat model enum."""
        assert MiniMaxChatModel.M2_7_highspeed.value == 'MiniMax-M2.7-highspeed'

    def test_m27_is_first_in_enum(self):
        """M2.7 should appear before older models in the enum."""
        members = list(MiniMaxChatModel)
        assert members[0] == MiniMaxChatModel.M2_7
        assert members[1] == MiniMaxChatModel.M2_7_highspeed

    def test_legacy_models_still_available(self):
        """Previous M2.5 models should still be available."""
        assert MiniMaxChatModel.M2_5.value == 'MiniMax-M2.5'
        assert MiniMaxChatModel.M2_5_highspeed.value == 'MiniMax-M2.5-highspeed'

    def test_total_model_count(self):
        """Should have 4 chat models total (M2.7, M2.7-highspeed, M2.5, M2.5-highspeed)."""
        assert len(MiniMaxChatModel) == 4
