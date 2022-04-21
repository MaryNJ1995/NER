# -*- coding: utf-8 -*-
# ========================================================
"""This module is written for write useful function."""
# ========================================================


# ========================================================
# Imports
# ========================================================

import sys
from collections import Counter
from typing import List


def find_max_length_in_list(data: List[list]) -> int:
    """

    :param data: [["item_1", "item_2", "item_3", "item_4"], ["item_1", "item_2"]]
    :return: 4
    """
    return max(len(sample) for sample in data)


def ignore_pad_index(true_labels: List[list], pred_labels: List[list],
                     pad_token: str = "[PAD]") -> [List[list], List[list]]:
    """

    :param true_labels: ["item_1", "item_2", "pad", "pad"], ["item_3", "pad", "pad", "pad"]]
    :param pred_labels: [["item_1", "item_2", "item_3", "pad"], ["item_1", "pad", "item_2", "item_3"]]
    :param pad_token: pad
    :return: ([["item_1", "item_2"], ["item_3"]], [["item_1", "item_2"], ["item_1"]])
    """
    for idx, sample in enumerate(true_labels):
        true_ = list()
        pred_ = list()
        for index, item in enumerate(sample):
            if item != pad_token:
                true_.append(sample[index])
                pred_.append(pred_labels[idx][index])
        true_labels[idx] = true_
        pred_labels[idx] = pred_
    return true_labels, pred_labels


def convert_index_to_tag(data: List[list], idx2tag: dict) -> List[list]:
    """

    :param data: [["item_1", "item_2", "item_3"], ["item_1", "item_2"]]
    :param idx2tag: {"pad_item": 0, "item_1": 1, "item_2": 2, "item_3": 3}
    :return: [[1, 2, 3], [1, 2]]
    """
    return [[idx2tag[item] for item in sample] for sample in data]


def convert_subtoken_to_token(tokens: List[list], labels: List[list]) -> \
        [List[list], List[list]]:
    """

    :param tokens:
    :param labels:
    :return:
    """
    for idx, (tkns, lbls) in enumerate(zip(tokens, labels)):
        new_tokens, new_labels = list(), list()
        token_, label_ = None, None
        for token, label in zip(tkns, lbls):
            if token.startswith("##"):
                token_ = token_ + token[2:]
            elif label == "X":
                token_ = token_ + token
            else:
                if token_:
                    new_tokens.append(token_)
                    new_labels.append(label_)
                token_ = token
                label_ = label
        new_tokens.append(token_)
        new_labels.append(label_)

        tokens[idx] = new_tokens
        labels[idx] = new_labels

    return tokens, labels


def convert_predict_tag(tags: list, subtoken_check: list) -> list:
    """

    :param tags:
    :param subtoken_check:
    :return:
    """
    for idx, (tag, sbt_chk) in enumerate(zip(tags, subtoken_check)):
        processed_tags = []
        for index, (label, check) in enumerate(zip(tag, sbt_chk)):
            if check == 1:
                if label == "X":
                    tmp = tag[index - 1].replace('B-', 'I-')
                    processed_tags.append(tmp)
                else:
                    processed_tags.append(label)
        tags[idx] = processed_tags
    return tags


def handle_subtoken_labels(entities: list, subtoken_check: list) -> list:
    """

    :param entities:
    :param subtoken_check:
    :return:
    """
    # labels = []
    # tmp = []
    # for idx, (lbl, sub) in enumerate(zip(entities, subtoken_check)):
    #     if sub == "1":
    #         if tmp == []:
    #             tmp.append(lbl)
    #         elif tmp != []:
    #             labels.append(handle_subtoken_prediction(tmp))
    #             tmp = []
    #             tmp.append(lbl)
    #     else:
    #         tmp.append(lbl)
    # if tmp != []:
    #     labels.append(handle_subtoken_prediction(tmp))
    # return labels
    return [entities[idx] for idx, item in enumerate(subtoken_check) if item == "1"]


def convert_x_label_to_true_label(predicted_tags: list, unexpected_entity: str) -> list:
    """

    :param predicted_tags:
    :param unexpected_entity:
    :return:
    """
    for tag_index, tag in enumerate(predicted_tags):
        if tag is unexpected_entity:
            predicted_tags[tag_index] = predicted_tags[tag_index - 1].replace("B-", "I-")
    return predicted_tags


def progress_bar(index, max, postText):
    """

    """
    n_bar = 50  # size of progress bar
    j = index / max
    sys.stdout.write('\r')
    sys.stdout.write(f"[{'=' * int(n_bar * j):{n_bar}s}] {int(100 * j)}%  {postText}")
    sys.stdout.flush()


def label_correction(tags: list) -> list:
    """

    :param tags:
    :return:
    """
    for idx, tag in enumerate(tags):
        if (idx == 0) and (tag.startswith("I-")):
            tags[idx] = tag.replace("I-", "B-")
        if (tag.startswith("I-")) and (tags[idx - 1] == "O"):
            tags[idx] = tag.replace("I-", "B-")
        if (tag.startswith("B-")) and (tags[idx - 1].startswith("B-")):
            tags[idx] = tag.replace("B-", "I-")
        if (tag.startswith("B-")) and (tags[idx - 1].startswith("I-")):
            tags[idx] = tag.replace("B-", "I-")
    return tags


def handle_subtoken_prediction(data: list) -> str:
    """

    :param data:
    :return:
    """
    c = Counter()
    while "X" in data:
        data.remove("X")
    while "O" in data:
        data.remove("O")
    c.update(data)
    if data:
        return c.most_common(1)[0][0]
    else:
        return "O"
