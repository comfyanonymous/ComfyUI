import pytest
from comfy_api_nodes.apis.minimax import MiniMaxChatModel


class TestMiniMaxChatModel:
    def test_m3_in_model_list(self):
        """MiniMax-M3 should be available in the chat model enum."""
        assert MiniMaxChatModel.M3.value == 'MiniMax-M3'

    def test_m27_in_model_list(self):
        """MiniMax-M2.7 should be available in the chat model enum."""
        assert MiniMaxChatModel.M2_7.value == 'MiniMax-M2.7'

    def test_m27_highspeed_in_model_list(self):
        """MiniMax-M2.7-highspeed should be available in the chat model enum."""
        assert MiniMaxChatModel.M2_7_highspeed.value == 'MiniMax-M2.7-highspeed'

    def test_m3_is_first_in_enum(self):
        """M3 should appear before older models in the enum."""
        members = list(MiniMaxChatModel)
        assert members[0] == MiniMaxChatModel.M3
        assert members[1] == MiniMaxChatModel.M2_7
        assert members[2] == MiniMaxChatModel.M2_7_highspeed

    def test_legacy_m25_models_removed(self):
        """Older M2.5 models should be removed."""
        values = [m.value for m in MiniMaxChatModel]
        assert 'MiniMax-M2.5' not in values
        assert 'MiniMax-M2.5-highspeed' not in values

    def test_total_model_count(self):
        """Should have 3 chat models total (M3, M2.7, M2.7-highspeed)."""
        assert len(MiniMaxChatModel) == 3
