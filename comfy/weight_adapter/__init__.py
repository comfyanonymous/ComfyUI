from .base import WeightAdapterBase
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
