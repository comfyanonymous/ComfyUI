import sd1_clip
import torch
import os

class SD2ClipModel(sd1_clip.SD1ClipModel):
    def __init__(self, arch="ViT-H-14", device="cpu", max_length=77, freeze=True, layer="penultimate", layer_idx=None):
        textmodel_json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "sd2_clip_config.json")
        super().__init__(device=device, freeze=freeze, textmodel_json_config=textmodel_json_config)
        self.empty_tokens = [[49406] + [49407] + [0] * 75]
        if layer == "last":
            layer_idx = -1
        elif layer == "penultimate":
            layer_idx = -2
        elif self.layer == "hidden":
            assert layer_idx is not None
            assert abs(layer_idx) < 24
        else:
            raise NotImplementedError()
        self.clip_layer(layer_idx)

class SD2Tokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, tokenizer_path=None):
        super().__init__(tokenizer_path, pad_with_end=False)


