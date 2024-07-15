class LLAMATokenizer:
    # todo: not sure why we're not using the tokenizer from transformers for this

    @staticmethod
    def from_pretrained(path):
        return LLAMATokenizer(path)

    def __init__(self, tokenizer_path):
        import sentencepiece
        self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=tokenizer_path)  # pylint: disable=unexpected-keyword-arg
        self.end = self.tokenizer.eos_id()
        self.eos_token_id = self.end
        self.eos_token = self.tokenizer.id_to_piece(self.eos_token_id)
        self._vocab = {
            self.tokenizer.id_to_piece(i): i for i in range(self.tokenizer.get_piece_size())
        }

    def get_vocab(self):
        return self._vocab

    def __call__(self, string):
        out = self.tokenizer.encode(string)  # pylint: disable=no-member
        out += [self.end]
        return {"input_ids": out}
