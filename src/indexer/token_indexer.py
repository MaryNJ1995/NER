# -*- coding: utf-8 -*-
"""
    Complex NER Project:
        indexer:
            index
"""
# =============================== My packages ==============================
from .indexer import Indexer


class TokenIndexer(Indexer):
    def __init__(self, vocabs: list, pad_index: int = 0, unk_index: int = 1):
        super().__init__(vocabs)
        self.pad_index = pad_index
        self.unk_index = unk_index

    def get_idx(self, token: str) -> int:
        """
        method to get index of input token

        Args:
            token: input token

        Returns:
            index of input token

        """
        if not self._vocab2idx:
            self._empty_vocab_handler()
        if token in self._vocab2idx.keys():
            return self._vocab2idx[token]
        return self._vocab2idx["<UNK>"]

    def get_word(self, idx: int) -> str:
        """
        method to get word of input index

        Args:
            idx: input idx

        Returns:
            token of input index

        """
        if not self._idx2vocab:
            self._empty_vocab_handler()
        if idx in self._idx2vocab.keys():
            return self._idx2vocab[idx]
        return self._idx2vocab[self.unk_index]

    def build_vocab2idx(self) -> None:
        """
        method to build vocab2ix dictionary

        Returns:
            None

        """
        self._vocab2idx = dict()
        self._vocab2idx["<PAD>"] = self.pad_index
        self._vocab2idx["<UNK>"] = self.unk_index
        for vocab in self.vocabs:
            self._vocab2idx[vocab] = len(self._vocab2idx)

    def build_idx2vocab(self) -> None:
        """
        method to build idx2vocab dictionary

        Returns:
            None

        """
        self._idx2vocab = dict()
        self._idx2vocab[self.pad_index] = "<PAD>"
        self._idx2vocab[self.unk_index] = "<UNK>"
        for vocab in self.vocabs:
            self._idx2vocab[len(self._idx2vocab)] = vocab
