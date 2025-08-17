from r_chainner.archs.face.gfpganv1_clean_arch import GFPGANv1Clean
from r_chainner.types import PyTorchModel


class UnsupportedModel(Exception):
    pass


def load_state_dict(state_dict) -> PyTorchModel:

    state_dict_keys = list(state_dict.keys())

    if "params_ema" in state_dict_keys:
        state_dict = state_dict["params_ema"]
    elif "params-ema" in state_dict_keys:
        state_dict = state_dict["params-ema"]
    elif "params" in state_dict_keys:
        state_dict = state_dict["params"]

    state_dict_keys = list(state_dict.keys())

    # GFPGAN
    if (
        "toRGB.0.weight" in state_dict_keys
        and "stylegan_decoder.style_mlp.1.weight" in state_dict_keys
    ):
        model = GFPGANv1Clean(state_dict)
    return model
