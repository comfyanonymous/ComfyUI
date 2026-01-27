import torch
import os

class SPieceTokenizer:
    @staticmethod
    def from_pretrained(path, **kwargs):
        return SPieceTokenizer(path, **kwargs)

    def __init__(self, tokenizer_path, add_bos=False, add_eos=True, added_tokens=None):
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.added_tokens = added_tokens
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
        # Handle added_tokens by replacing them with their token IDs
        if self.added_tokens is not None:
            for token_str, token_id in self.added_tokens.items():
                if token_str in string:
                    # Split by the special token and encode parts separately
                    parts = string.split(token_str)
                    result = []
                    for i, part in enumerate(parts):
                        if part:
                            result.extend(self.tokenizer.encode(part, add_bos=False, add_eos=False))
                        if i < len(parts) - 1:
                            result.append(token_id)
                    return {"input_ids": result}

        out = self.tokenizer.encode(string)
        return {"input_ids": out}

    def decode(self, token_ids, skip_special_tokens=False):
        """Decode token IDs back to text"""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        if skip_special_tokens and self.added_tokens:
            special_token_ids = set(self.added_tokens.values())
            # Also filter common special tokens: BOS (2), EOS (1), PAD (0), and end_of_turn (106)
            special_token_ids.update([0, 1, 2, 106])
            token_ids = [tid for tid in token_ids if tid not in special_token_ids]

        return self.tokenizer.decode(token_ids)

    def serialize_model(self):
        return torch.ByteTensor(list(self.tokenizer.serialized_model_proto()))
