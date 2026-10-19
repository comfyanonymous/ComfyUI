import torch
import os

class SPieceTokenizer:
    @staticmethod
    def from_pretrained(path, **kwargs):
        return SPieceTokenizer(path, **kwargs)

    def __init__(self, tokenizer_path, add_bos=False, add_eos=True, special_tokens=None):
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.special_tokens = special_tokens
        import sentencepiece
        if torch.is_tensor(tokenizer_path):
            tokenizer_path = tokenizer_path.numpy().tobytes()

        if isinstance(tokenizer_path, bytes):
            self.tokenizer = sentencepiece.SentencePieceProcessor(model_proto=tokenizer_path, add_bos=self.add_bos, add_eos=self.add_eos)
        else:
            if not os.path.isfile(tokenizer_path):
                raise ValueError("invalid tokenizer")
            self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=tokenizer_path, add_bos=self.add_bos, add_eos=self.add_eos)

    def get_vocab(self):
        out = {}
        for i in range(self.tokenizer.get_piece_size()):
            out[self.tokenizer.id_to_piece(i)] = i
        return out

    def __call__(self, string):
        if self.special_tokens is not None:
            import re
            special_tokens_pattern = '|'.join(re.escape(token) for token in self.special_tokens.keys())
            if special_tokens_pattern and re.search(special_tokens_pattern, string):
                parts = re.split(f'({special_tokens_pattern})', string)
                result = []
                for part in parts:
                    if not part:
                        continue
                    if part in self.special_tokens:
                        result.append(self.special_tokens[part])
                    else:
                        encoded = self.tokenizer.encode(part, add_bos=False, add_eos=False)
                        result.extend(encoded)
                return {"input_ids": result}

        out = self.tokenizer.encode(string)
        return {"input_ids": out}

    def decode(self, token_ids, skip_special_tokens=False):

        if skip_special_tokens and self.special_tokens:
            special_token_ids = set(self.special_tokens.values())
            token_ids = [tid for tid in token_ids if tid not in special_token_ids]

        return self.tokenizer.decode(token_ids)

    def serialize_model(self):
        return torch.ByteTensor(list(self.tokenizer.serialized_model_proto()))
