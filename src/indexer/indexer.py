# -*- coding: utf-8 -*-
"""
    Complex NER Project:
        indexer:
            indexer
"""

# ============================ Third Party libs ============================
import os
from typing import List
from abc import abstractmethod

# =============================== My packages ==============================
from data_loader import write_json, write_text


class Indexer:
    def __init__(self, vocabs: list):
        self.vocabs = vocabs
        self._vocab2idx = None
        self._idx2vocab = None
        self._unitize_vocabs()  # build unique vocab

    def get_vocab2idx(self) -> dict:
        """
        method to return vocab2idx

        Returns:
            vocab2idx

        """
        if not self._vocab2idx:
            self._empty_vocab_handler()
        return self._vocab2idx

    def get_idx2vocab(self) -> dict:
        """
        method to return idx2vocab
        Returns:
            idx2vocab

        """
        if not self._idx2vocab:
            self._empty_vocab_handler()
        return self._idx2vocab

    @abstractmethod
    def get_idx(self, token) -> int:
        """
        method to get index of input word
        Args:
            token: input token

        Returns:
            index of input token in vocab2idx
        """
        if not self._vocab2idx:
            self._empty_vocab_handler()
        if token in self._vocab2idx.keys():
            return self._vocab2idx[token]
        else:
            print("error handler")
            raise Exception("target is not available")

    @abstractmethod
    def get_word(self, idx) -> str:
        """
        method to get word with input index
        Args:
            idx: input index

        Returns:
            token in idx2vocab

        """
        if not self._idx2vocab:
            self._empty_vocab_handler()
        if idx in self._idx2vocab.keys():
            return self._idx2vocab[idx]
        else:
            print("error handler")
            raise Exception("target is not available")

    @abstractmethod
    def build_vocab2idx(self) -> None:
        """
        method to build vocab2ix dictionary
        Returns:
            None

        """
        self._vocab2idx = dict()
        for vocab in self.vocabs:
            self._vocab2idx[vocab] = len(self._vocab2idx)

    @abstractmethod
    def build_idx2vocab(self) -> None:
        """
        method to build idx2vocab dictionary

        Returns:
            None
        """
        self._idx2vocab = dict()
        for vocab in self.vocabs:
            self._idx2vocab[len(self._idx2vocab)] = vocab

    def _empty_vocab_handler(self) -> None:
        """
        method to handle the time that we have empty vocab

        Returns:
            None
        """
        self.build_vocab2idx()
        self.build_idx2vocab()

    def convert_samples_to_indexes(self, tokenized_samples: List[list]) -> List[list]:
        """
        Method to convert tokens to corresponding index

        Args:
            tokenized_samples: list of tokenized sentences

        Returns:
            list of samples which converted from token to index with vocab2idx
        """
        for index, tokenized_sample in enumerate(tokenized_samples):
            for token_index, token in enumerate(tokenized_sample):
                tokenized_samples[index][token_index] = self.get_idx(token)
        return tokenized_samples

    def convert_indexes_to_samples(self, indexed_samples: List[list]) -> List[list]:
        """
        Method to convert indexes to samples
        Args:
            indexed_samples: list of indexed samples

        Returns:
            list of samples which converted from index to token with idx2vocab
        """
        for index, indexed_sample in enumerate(indexed_samples):
            for token_index, token in enumerate(indexed_sample):
                indexed_samples[index][token_index] = self.get_word(token)
        return indexed_samples

    def _unitize_vocabs(self) -> None:
        """
        initialized created vocabs

        Returns:
            None
        """
        self.vocabs = list(set(self.vocabs))

    def save(self, path) -> None:
        """
        Method to save files
        Args:
            path:  path to save files

        Returns:
            None
        """
        write_text(data=self.vocabs, path=os.path.join(path, "vocabs.txt"))
        write_json(data=self.get_vocab2idx(),
                   path=os.path.join(path, "vocab2idx.json"))
        write_json(data=self.get_idx2vocab(),
                   path=os.path.join(path, "idx2vocab.json"))
