import sentencepiece
import torch


class SPieceTokenizer:
    add_eos = True

    @staticmethod
    def from_pretrained(path):
        return SPieceTokenizer(path)

    def __init__(self, tokenizer_path):
        if torch.is_tensor(tokenizer_path):
            tokenizer_path = tokenizer_path.numpy().tobytes()

        construction_args = {}
        if isinstance(tokenizer_path, bytes):
            construction_args["model_proto"] = tokenizer_path
        else:
            construction_args["model_file"] = tokenizer_path
        self.tokenizer = sentencepiece.SentencePieceProcessor(add_eos=SPieceTokenizer.add_eos, **construction_args)  # pylint: disable=unexpected-keyword-arg

        self.end = self.tokenizer.eos_id()
        self.eos_token_id = self.end
        self.eos_token = self.tokenizer.id_to_piece(self.eos_token_id)  # pylint: disable=no-member
        self._vocab = {
            self.tokenizer.id_to_piece(i): i for i in range(self.tokenizer.get_piece_size())  # pylint: disable=no-member
        }

    def get_vocab(self):
        return self._vocab

    def __call__(self, string):
        out = self.tokenizer.encode(string)  # pylint: disable=no-member
        return {"input_ids": out}

    def serialize_model(self):
        return torch.ByteTensor(list(self.tokenizer.serialized_model_proto()))
