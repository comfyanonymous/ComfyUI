from comfy_extras.nodes_video import AccumulateSaveVideo


def test_accumulate_save_video_passes_video_through():
    assert AccumulateSaveVideo.RETURN_TYPES == ["VIDEO"]
