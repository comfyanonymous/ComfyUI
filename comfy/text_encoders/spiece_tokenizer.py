import copy
from pathlib import Path

import sentencepiece
import torch


class SPieceTokenizer:
    @staticmethod
    def from_pretrained(path, **kwargs):
        return SPieceTokenizer(path, **kwargs)

    def __init__(self, tokenizer_path: bytes | str | Path, add_bos=False, add_eos=True, **kwargs):
        self.add_bos = add_bos
        self.add_eos = add_eos
        if torch.is_tensor(tokenizer_path):
            tokenizer_path = tokenizer_path.numpy().tobytes()

        construction_args = {
            'add_bos': self.add_bos,
            'add_eos': self.add_eos,
        }

        if isinstance(tokenizer_path, bytes):
            construction_args["model_proto"] = tokenizer_path
        else:
            if not Path(tokenizer_path).is_file():
                raise ValueError(f"invalid tokenizer {tokenizer_path}")
            construction_args["model_file"] = tokenizer_path

        self.tokenizer = sentencepiece.SentencePieceProcessor(**construction_args)  # pylint: disable=unexpected-keyword-arg

        self.end = self.tokenizer.eos_id()
        self.eos_token_id = self.end
        self.eos_token = self.tokenizer.id_to_piece(self.eos_token_id)
        self._vocab = {
            self.tokenizer.id_to_piece(i): i for i in range(self.tokenizer.get_piece_size())
        }

    def get_vocab(self):
        return self._vocab

    def __call__(self, string):
        out = self.tokenizer.encode(string)
        return {"input_ids": out}

    def serialize_model(self):
        return torch.ByteTensor(list(self.tokenizer.serialized_model_proto()))

    def clone(self):
        return copy.copy(self)
