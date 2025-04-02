from .base import WeightAdapterBase
from .lora import LoRAAdapter
from .loha import LoHaAdapter
from .lokr import LoKrAdapter
from .glora import GLoRAAdapter


adapters: list[type[WeightAdapterBase]] = [
    LoRAAdapter,
    LoHaAdapter,
    LoKrAdapter,
    GLoRAAdapter,
]
