import os
from typing import List
from abc import abstractmethod

from .helper import save_json, save_text


class Indexer:
    def __init__(self, vocabs: list):
        self.vocabs = vocabs
        self._vocab2idx = None
        self._idx2vocab = None
        self._unitize_vocabs()  # build unique vocab

    def get_vocab2idx(self):
        if not self._vocab2idx:
            self._empty_vocab_handler()
        return self._vocab2idx

    def get_idx2vocab(self):
        if not self._idx2vocab:
            self._empty_vocab_handler()
        return self._idx2vocab

    @abstractmethod
    def get_idx(self, token):
        if not self._vocab2idx:
            self._empty_vocab_handler()
        if token in self._vocab2idx.keys():
            return self._vocab2idx[token]
        else:
            print("error handler")
            raise Exception("target is not available")

    @abstractmethod
    def get_word(self, idx):
        if not self._idx2vocab:
            self._empty_vocab_handler()
        if idx in self._idx2vocab.keys():
            return self._idx2vocab[idx]
        else:
            print("error handler")
            raise Exception("target is not available")

    @abstractmethod
    def build_vocab2idx(self):
        """

        :return:
        """
        # TODO: add custom pad_index and unk_index
        self._vocab2idx = dict()
        for vocab in self.vocabs:
            self._vocab2idx[vocab] = len(self._vocab2idx)

    @abstractmethod
    def build_idx2vocab(self) -> None:
        """

        :return:
        """
        # TODO: add custom pad_index and unk_index
        self._idx2vocab = dict()
        for vocab in self.vocabs:
            self._idx2vocab[len(self._idx2vocab)] = vocab

    def _empty_vocab_handler(self):
        self.build_vocab2idx()
        self.build_idx2vocab()

    def convert_samples_to_indexes(self, tokenized_samples: List[list]) -> List[list]:
        """
        :param tokenized_samples: [[target_1], ..., [target_n]]
        :return: [[target_1_index],...,[target_n_index]]
        """
        for index, tokenized_sample in enumerate(tokenized_samples):
            for token_index, token in enumerate(tokenized_sample):
                tokenized_samples[index][token_index] = self.get_idx(token)
        return tokenized_samples

    def convert_indexes_to_samples(self, indexed_samples: List[list]) -> List[list]:
        """

        :param indexed_samples:
        :return:
        """
        for index, indexed_sample in enumerate(indexed_samples):
            for token_index, token in enumerate(indexed_sample):
                indexed_samples[index][token_index] = self.get_word(token)
        return indexed_samples

    def _unitize_vocabs(self) -> None:
        self.vocabs = list(set(self.vocabs))

    def save(self, path) -> None:
        save_text(data=self.vocabs, path=os.path.join(path, "vocabs.txt"))
        save_json(data=self.get_vocab2idx(),
                  path=os.path.join(path, "vocab2idx.json"))
        save_json(data=self.get_idx2vocab(),
                  path=os.path.join(path, "idx2vocab.json"))

    def load(self):
        pass

# TODO : SAVE and LOAD _idx2vocab
