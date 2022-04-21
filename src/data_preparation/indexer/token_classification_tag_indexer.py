from .indexer import Indexer


class TokenClassificationTagIndexer(Indexer):
    def __init__(self, vocabs: list, pad_index: int = None, cls_index: int = None,
                 sep_index: int = None):
        super().__init__(vocabs)
        self.pad_index = pad_index
        self.cls_index = cls_index
        self.sep_index = sep_index
        self._remove_duplicate_special_token()

    def get_idx(self, tag):
        if not self._vocab2idx:
            self._empty_vocab_handler()
        if tag in self._vocab2idx.keys():
            return self._vocab2idx[tag]
        return Exception("input tag not in data")

    def get_word(self, idx):
        if not self._idx2vocab:
            self._empty_vocab_handler()
        if idx in self._idx2vocab.keys():
            return self._idx2vocab[idx]
        return Exception("idx not in range")

    def build_vocab2idx(self):
        """

        :return:
        """
        self._vocab2idx = dict()
        self._vocab2idx["[PAD]"] = self.pad_index
        if self.sep_index:
            self._vocab2idx["[SEP]"] = self.sep_index
        if self.cls_index:
            self._vocab2idx["[CLS]"] = self.cls_index
        for vocab in self.vocabs:
            self._vocab2idx[vocab] = len(self._vocab2idx)

    def build_idx2vocab(self):
        """

        :return:
        """
        self._idx2vocab = dict()
        self._idx2vocab[self.pad_index] = "[PAD]"
        if self.sep_index:
            self._idx2vocab[self.sep_index] = "[SEP]"
        if self.cls_index:
            self._idx2vocab[self.cls_index] = "[CLS]"
        for vocab in self.vocabs:
            self._idx2vocab[len(self._idx2vocab)] = vocab

    def _remove_duplicate_special_token(self):
        """

        :return:
        """
        special_tokens = ["[PAD]", "[CLS]", "[SEP]"]
        for token in special_tokens:
            if token in self.vocabs:
                self.vocabs.remove(token)
