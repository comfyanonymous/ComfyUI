import torch
import comfy.ops

ops = comfy.ops.manual_cast

class ReduxImageEncoder(torch.nn.Module):
    def __init__(
        self,
        redux_dim: int = 1152,
        txt_in_features: int = 4096,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        self.redux_dim = redux_dim
        self.device = device
        self.dtype = dtype

        self.redux_up = ops.Linear(redux_dim, txt_in_features * 3, dtype=dtype)
        self.redux_down = ops.Linear(txt_in_features * 3, txt_in_features, dtype=dtype)

    def forward(self, sigclip_embeds) -> torch.Tensor:
        projected_x = self.redux_down(torch.nn.functional.silu(self.redux_up(sigclip_embeds)))
        return projected_x
