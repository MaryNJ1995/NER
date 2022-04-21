# -*- coding: utf-8 -*-
# ========================================================
"""data_preparation module is written for tokenizing texts"""
# ========================================================


# ========================================================
# Imports
# ========================================================

import torch
import numpy as np
from typing import List
import torch

__author__ = "Ehsan Tavan", "Ali Rahmati", "Maryam Najafi"
__project__ = "signal entity detection"
__version__ = "1.0.0"
__date__ = "2021/09/27"
__email__ = "tavan.ehsan@gmail.com"


def prepare_conll_data(data: list) -> [List[list], List[list]]:
    """
    prepare_conll_data function is written for loading data in conll format
    :param data:
    :return:
    """
    sentences, labels, tokens, tags = [], [], [], []
    for line in data:
        if not line.startswith("# id"):
            if line == "\n":
                if len(tokens) != 0:
                    sentences.append(tokens)
                    labels.append(tags)
                    tokens, tags = [], []
            else:
                line = line.strip().split()
                tokens.append(line[0].strip())
                tags.append(line[3].strip())
    return sentences, labels


def tokenize_and_keep_labels(sentences: List[list], labels: List[list], indexed_sentences: List[list],
                             tokenizer, bpemb, mode: str = "same") -> [List[list], List[list], List[list]]:
    """
    Function to tokenize and preserve labels
    :param sentences: [['تغییرات', 'قیمت', 'رمز', 'ارز',
                            'اتریوم', 'در', 'یک', 'هفته', 'قبل'], ... ]
    :param labels: [['O', 'O', 'B-ENT', 'I-ENT',
                        'I-ENT', 'O','B-TIM', 'I-TIM', 'I-TIM'], ...]
    :param indexed_sentences:
    :param tokenizer:
    :param bpemb:
    :param mode: ["same", "x_mode"]
    :return: [['تغییر', '##ات', 'قیمت', 'رمز', 'ا', '##رز', 'ا', '##تری', '##وم'
                , 'در', 'یک', 'هفت', '##ه', 'قبل'], ... ]
            [['O', 'O', 'O', 'B-ENT', 'I-ENT', 'I-ENT', 'I-ENT',
            'I-ENT', 'I-ENT', 'O', 'B-TIM', 'I-TIM', 'I-TIM', 'I-TIM'], ... ]
    """
    assert len(sentences) == len(labels), "Sentences and labels should have " \
                                          "the same number of samples"
    subtoken_checks = []
    bpemb_ids = []
    flair_tokens = []
    for idx, (sentence, label, indexed_sentence) in enumerate(zip(sentences, labels, indexed_sentences)):
        tokenized_sentence, labels_, indexed_sentences_, checks_, _bpemb, flair_ = [], [], [], [], [], []
        for word, tag, word_idx in zip(sentence, label, indexed_sentence):
            checks_.append("1")
            # Tokenize each word and count number of its subwords
            tokenized_word = tokenizer.tokenize(str(word))
            n_subwords = len(tokenized_word)
            flair_.extend([word] * n_subwords)

            # The tokenized word is added to the resulting tokenized word list
            tokenized_sentence.extend(tokenized_word)
            checks_.extend(["0"] * (n_subwords - 1))
            indexed_sentences_.extend([word_idx] * n_subwords)
            _bpemb.extend([np.max(bpemb.embed(word), axis=0)] * n_subwords)

            # The same label is added to the new list of labels `n_subwords` times
            if mode == "same":
                labels_.extend([tag] * n_subwords)
            elif mode == "x_mode":
                labels_.append(tag)
                labels_.extend(["X"] * (n_subwords - 1))
        sentences[idx], labels[idx], indexed_sentences[idx] = tokenized_sentence, labels_, indexed_sentences_
        subtoken_checks.append(checks_)
        flair_tokens.append(flair_)
        bpemb_ids.append(_bpemb)
    return sentences, labels, subtoken_checks, bpemb_ids, flair_tokens


def add_special_tokens(sentences: List[list], labels: List[list], cls_token: str,
                       sep_token: str) -> [List[list], List[list]]:
    """
    add_special_tokens function is written for add special tokens for samples
    :param sentences:
    :param labels:
    :param cls_token:
    :param sep_token:
    :return:
    """
    for idx, (sentence, label) in enumerate(zip(sentences, labels)):
        sentence.insert(0, cls_token)
        label.insert(0, cls_token)
        sentence.append(sep_token)
        label.append(sep_token)

        sentences[idx] = sentence
        labels[idx] = label
    return sentences, labels


def pad_sequence(texts: List[list], max_length: int, pad_item: str = "[PAD]") -> List[list]:
    """
    pad_sequence function is written for pad list of samples
    :param texts: [["item_1", "item_2", "item_3"], ["item_1", "item_2"]]
    :param max_length: 4
    :param pad_item: pad_item
    :return: [["item_1", "item_2", "item_3", pad_item],
                    ["item_1", "item_2", pad_item, pad_item]]
    """
    for idx, text in enumerate(texts):
        text_length = len(text)
        texts[idx].extend([pad_item] * (max_length - text_length))
    return texts


def pad_sequence_2(texts: List[list], max_length: int, pad_item: str = "[PAD]") -> List[list]:
    """
    pad_sequence function is written for pad list of samples
    :param texts: [["item_1", "item_2", "item_3"], ["item_1", "item_2"]]
    :param max_length: 4
    :param pad_item: pad_item
    :return: [["item_1", "item_2", "item_3", pad_item],
                    ["item_1", "item_2", pad_item, pad_item]]
    """
    for idx, text in enumerate(texts):
        text_length = len(text)
        texts[idx].extend([[pad_item] * 300] * (max_length - text_length))
    return texts


def truncate_sequence(texts: List[list], max_length: int) -> list:
    """
    truncate_sequence function is written for truncate list of samples
    :param texts: [["item_1", "item_2", "item_3"], ["item_1", "item_2"]]
    :param max_length: 2
    :return: [["item_1", "item_2"], ["item_1", "item_2"]]
    """
    for idx, text in enumerate(texts):
        if len(text) > max_length:
            texts[idx] = text[: max_length - 1]
            texts[idx].append("[SEP]")
    return texts


def create_attention_masks(data: List[list], pad_item: str = "[PAD]") -> List[list]:
    """
    create_attention_masks function is written for creating attention masks
    :param data: [["item_1", "item_2", "item_3", "pad_item"],
                ["item_1", "pad_item", "pad_item", "pad_item"]]
    :param pad_item: pad_item
    :return: [[1, 1, 1, 0], [1, 0, 0, 0]]
    """

    return [[1 if item != pad_item else 0 for item in sample] for sample in data]


def create_test_samples(data: List[list], tokenizer, bpemb) -> [List[list], List[list], List[list]]:
    """

    :param data:
    :param tokenizer:
    :param bpemb:
    :return:
    """
    subtoken_checks = []
    bpemb_ids = []
    flair_tokens = []
    for idx, item in enumerate(data):
        tokenized_item = []
        subtoken_checks_temp = []
        bpemb_tmp = []
        flair_ = []
        for tok in item:
            subtoken_checks_temp.append("1")
            tokenized_word = tokenizer.tokenize(tok)
            n_subwords = len(tokenized_word)
            tokenized_item.extend(tokenized_word)
            subtoken_checks_temp.extend(["0"] * (n_subwords - 1))
            flair_.extend([tok] * n_subwords)

            bpemb_tmp.extend([np.max(bpemb.embed(tok), axis=0)] * n_subwords)

        data[idx] = tokenized_item
        flair_tokens.append(flair_)
        subtoken_checks.append(subtoken_checks_temp)
        bpemb_ids.append(bpemb_tmp)
    return data, subtoken_checks, bpemb_ids, flair_tokens
