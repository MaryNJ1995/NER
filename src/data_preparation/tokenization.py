# -*- coding: utf-8 -*-
# ========================================================
"""tokenization module is written for tokenizing texts"""
# ========================================================


# ========================================================
# Imports
# ========================================================

__author__ = "Ehsan Tavan", "Ali Rahmati", "Maryam Najafi"
__project__ = "signal entity detection"
__version__ = "1.0.0"
__date__ = "2021/09/27"
__email__ = "tavan.ehsan@gmail.com"


def word_tokenizer(texts: list, tokenizer_obj) -> list:
    """
    word_tokenizer function is written for tokenizing sentence into words
    :param texts: list of sentences ex: ["first sent", "second sent"]
    :param tokenizer_obj:
    :return: list of tokenized words ex: [["first", "sent"], ["second", "sent"]]
    """
    return [tokenizer_obj(text) for text in texts]


def sent_tokenizer(texts: list, tokenizer_obj):
    """
    sent_tokenizer function is written for tokenizing texts into sentences
    :param texts: list of docs ex: ["first sent. second sent.", "first sent. second sent."]
    :param tokenizer_obj:
    :return: list of tokenized sents ex: [["first sent.", "second sent."],
                                        ["first sent.", "second sent."]]
    """
    return [tokenizer_obj(text) for text in texts]
