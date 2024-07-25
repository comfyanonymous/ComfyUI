import os
import torch

class SPieceTokenizer:
    add_eos = True

    @staticmethod
    def from_pretrained(path):
        return SPieceTokenizer(path)

    def __init__(self, tokenizer_path):
        import sentencepiece
        if torch.is_tensor(tokenizer_path):
            tokenizer_path = tokenizer_path.numpy().tobytes()

        if isinstance(tokenizer_path, bytes):
            self.tokenizer = sentencepiece.SentencePieceProcessor(model_proto=tokenizer_path, add_eos=self.add_eos)
        else:
            self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=tokenizer_path, add_eos=self.add_eos)

    def get_vocab(self):
        out = {}
        for i in range(self.tokenizer.get_piece_size()):
            out[self.tokenizer.id_to_piece(i)] = i
        return out

    def __call__(self, string):
        out = self.tokenizer.encode(string)
        return {"input_ids": out}

    def serialize_model(self):
        return torch.ByteTensor(list(self.tokenizer.serialized_model_proto()))
