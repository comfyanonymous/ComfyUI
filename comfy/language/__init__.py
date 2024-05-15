from fastchat.model.model_adapter import register_model_adapter

from .fastchat_adapters import Phi3Adapter

register_model_adapter(Phi3Adapter)