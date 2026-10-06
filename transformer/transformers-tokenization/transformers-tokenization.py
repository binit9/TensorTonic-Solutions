class SimpleTokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: list[str]) -> None:
        """
        Builds the vocabulary in place.
        """
        self.word_to_id = {"<PAD>":0, "<UNK>":1, "<BOS>":2, "<EOS>":3}
        unique_words = set()
        for text in texts:
            unique_words.update(text.lower().split())

        for i, word in enumerate(sorted(unique_words)):
            self.word_to_id[word] = i+4
        self.vocab_size = len(unique_words)+4
        self.id_to_word = {v:k for k,v in self.word_to_id.items()}

    def encode(self, text: str) -> list[int]:
        """
        Returns token IDs for the input text.
        """
        tokens = text.lower().split()
        return [self.word_to_id.get(token, self.word_to_id[self.unk_token]) for token in tokens]

    def decode(self, ids: list[int]) -> str:
        """
        Returns the decoded, space-separated text.
        """
        return " ".join([self.id_to_word.get(id, self.unk_token) for id in ids])