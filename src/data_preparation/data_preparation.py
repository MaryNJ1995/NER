# -*- coding: utf-8 -*-
"""
    Complex NER Project:
        data preparation:
            data_preparation.py

"""

# ============================ Third Party libs ============================
from typing import List


def prepare_conll_data(data: list) -> [List[list], List[list]]:
    """
    function to load data in conll format

    Args:
        data: NER data from txt file

    Returns:
        sentences: list of tokenized sentences
        labels: list of labels for each sentence

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


def tokenize_and_keep_labels(sentences: List[list], labels: List[list], tokenizer,
                             mode: str = "same") -> [List[list], List[list], List[list]]:
    """
    function to tokenize and preserve labels

    Args:
        sentences: list of tokenized sentences
        labels: list of labels for each sentence
        tokenizer: tokenizer object (ex: bert tokenizer)
        mode: use "same" or "x_mode". When using same, the same label is considered for all
        sub_tokens. And when x_mode is used, the first subtoken will be labeled ner tag and other
        sub_tokens will be labeled X tag.

    Returns:
        sentences: list of tokenized sentences
        labels: list of labels for each sentence
        subtoken_checks: list of subtoken check for each sentences

    """
    assert len(sentences) == len(labels), "Sentences and labels should have " \
                                          "the same number of samples"
    subtoken_checks = []
    for idx, (sentence, label) in enumerate(zip(sentences, labels)):
        sentence_, labels_, checks_ = [], [], []
        for word, tag in zip(sentence, label):
            checks_.append("1")
            # Tokenize each word and count number of its subwords
            tokenized_word = tokenizer.tokenize(str(word))
            n_subwords = len(tokenized_word)
            # The tokenized word is added to the resulting tokenized word list
            sentence_.extend(tokenized_word)
            checks_.extend(["0"] * (n_subwords - 1))

            # The same label is added to the new list of labels `n_subwords` times
            if mode == "same":
                labels_.extend([tag] * n_subwords)
            elif mode == "x_mode":
                labels_.append(tag)
                labels_.extend(["X"] * (n_subwords - 1))
        sentences[idx], labels[idx] = sentence_, labels_
        subtoken_checks.append(checks_)
    return sentences, labels, subtoken_checks


def add_special_tokens(sentences: List[list], labels: List[list], cls_token: str,
                       sep_token: str) -> [List[list], List[list]]:
    """
    function to add special tokens for samples

    Args:
        sentences: list of tokenized sentences
        labels: list of labels for each sentence
        cls_token: cls token
        sep_token: sep token

    Returns:
        sentences: list of tokenized sentences which special tokens have been added
        labels: list of labels for each sentence which special tokens have been added

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
    function to pad sentences

    Args:
        texts: list of tokenized sentences
        max_length: maximum length for sentences
        pad_item: pad token

    Returns:
        texts: padded sentences

    """
    for idx, text in enumerate(texts):
        text_length = len(text)
        texts[idx].extend([pad_item] * (max_length - text_length))
    return texts


def truncate_sequence(texts: List[list], max_length: int) -> list:
    """
    function to truncate sentences

    Args:
        texts: list of tokenized sentences
        max_length: maximum length for sentences

    Returns:
        truncated sentences

    """
    for idx, text in enumerate(texts):
        if len(text) > max_length:
            texts[idx] = text[: max_length - 1]
            texts[idx].append("[SEP]")
    return texts


def create_test_samples(data: List[list], tokenizer) -> [List[list], List[list], List[list]]:
    """
    function to prepare suitable examples for inference
    Args:
        data: list of sentences
        tokenizer: tokenizer object

    Returns:
        data: list of tokenized sentences
        subtoken_checks: list of subtoken check for each sentence

    """
    subtoken_checks = []
    for idx, item in enumerate(data):
        tokenized_item = []
        subtoken_checks_temp = []
        for tok in item:
            subtoken_checks_temp.append("1")
            tokenized_word = tokenizer.tokenize(tok)
            n_subwords = len(tokenized_word)
            tokenized_item.extend(tokenized_word)
            subtoken_checks_temp.extend(["0"] * (n_subwords - 1))

        data[idx] = tokenized_item
        subtoken_checks.append(subtoken_checks_temp)
    return data, subtoken_checks
