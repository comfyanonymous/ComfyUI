import os

class LLAMATokenizer:
    @staticmethod
    def from_pretrained(path):
        return LLAMATokenizer(path)

    def __init__(self, tokenizer_path):
        import sentencepiece
        self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=tokenizer_path)
        self.end = self.tokenizer.eos_id()

    def get_vocab(self):
        out = {}
        for i in range(self.tokenizer.get_piece_size()):
            out[self.tokenizer.id_to_piece(i)] = i
        return out

    def __call__(self, string):
        out = self.tokenizer.encode(string)
        out += [self.end]
        return {"input_ids": out}
