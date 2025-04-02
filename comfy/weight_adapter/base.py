from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightAdapterBase:
    name: str
    loaded_keys: set[str]
    weights: list[torch.Tensor]

    @classmethod
    def load(cls, x: str, lora: dict[str, torch.Tensor]) -> "WeightAdapterBase" | None:
        raise NotImplementedError

    def to_train(self) -> "WeightAdapterTrainBase":
        raise NotImplementedError

    def calculate_weight(
        self,
        weight,
        key,
        strength,
        strength_model,
        offset,
        function,
        intermediate_dtype=torch.float32,
        original_weight=None,
    ):
        raise NotImplementedError


class WeightAdapterTrainBase(nn.Module):
    def __init__(self):
        super().__init__()

    # [TODO] Collaborate with LoRA training PR #7032
