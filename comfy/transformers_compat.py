try:
    from transformers import T5TokenizerFast
except (ImportError, ModuleNotFoundError):
    from transformers import T5Tokenizer as T5TokenizerFast

try:
    from transformers import LlamaTokenizerFast
except (ImportError, ModuleNotFoundError):
    from transformers import LlamaTokenizer as LlamaTokenizerFast

try:
    from transformers import CLIPTokenizerFast
except (ImportError, ModuleNotFoundError):
    from transformers import CLIPTokenizer as CLIPTokenizerFast

try:
    from transformers import GPT2TokenizerFast
except (ImportError, ModuleNotFoundError):
    from transformers import GPT2Tokenizer as GPT2TokenizerFast

try:
    from transformers import BertTokenizerFast
except (ImportError, ModuleNotFoundError):
    from transformers import BertTokenizer as BertTokenizerFast

try:
    from transformers import Qwen2TokenizerFast
except (ImportError, ModuleNotFoundError):
    try:
        from transformers import Qwen2Tokenizer as Qwen2TokenizerFast
    except (ImportError, ModuleNotFoundError):
        # Fallback if neither exists, primarily for earlier versions or specific environments
        Qwen2TokenizerFast = None

# Alias Qwen2Tokenizer to the "Fast" version we found/aliased, as we might use either name
Qwen2Tokenizer = Qwen2TokenizerFast

try:
    from transformers import ByT5TokenizerFast
except ImportError:
    try:
        from transformers import ByT5Tokenizer as ByT5TokenizerFast
    except (ImportError, ModuleNotFoundError):
        ByT5TokenizerFast = None

ByT5Tokenizer = ByT5TokenizerFast

__all__ = [
    "T5TokenizerFast",
    "LlamaTokenizerFast",
    "CLIPTokenizerFast",
    "GPT2TokenizerFast",
    "BertTokenizerFast",
    "Qwen2Tokenizer",
    "ByT5Tokenizer",
]
