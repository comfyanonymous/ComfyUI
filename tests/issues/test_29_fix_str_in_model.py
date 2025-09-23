import torch.nn

from comfy.model_patcher import ModelPatcher


class HasOperationsNoName(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.operations = object()
        if hasattr(self.operations, "__name__"):
            delattr(self.operations, "__name__")


def test_str_model_patcher():
    model_patcher = ModelPatcher(HasOperationsNoName(), torch.device('cpu'), torch.device('cpu'))
    assert str(model_patcher) is not None
