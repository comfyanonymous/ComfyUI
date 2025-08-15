from .base import WeightAdapterBase, WeightAdapterTrainBase
from .lora import LoRAAdapter
from .loha import LoHaAdapter
from .lokr import LoKrAdapter
from .glora import GLoRAAdapter
from .oft import OFTAdapter
from .boft import BOFTAdapter


adapters: list[type[WeightAdapterBase]] = [
    LoRAAdapter,
    LoHaAdapter,
    LoKrAdapter,
    GLoRAAdapter,
    OFTAdapter,
    BOFTAdapter,
]
adapter_maps: dict[str, type[WeightAdapterBase]] = {
    "LoRA": LoRAAdapter,
    "LoHa": LoHaAdapter,
    "LoKr": LoKrAdapter,
    "OFT": OFTAdapter,
    ## We disable not implemented algo for now
    # "GLoRA": GLoRAAdapter,
    # "BOFT": BOFTAdapter,
}


__all__ = [
    "WeightAdapterBase",
    "WeightAdapterTrainBase",
    "adapters",
    "adapter_maps",
] + [a.__name__ for a in adapters]
