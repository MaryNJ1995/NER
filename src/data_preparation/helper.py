from typing import List


def tokenize_and_keep_labels(sentences: List[list], labels: List[list],
                             tokenizer, mode: str = "same") -> [List[list], List[list]]:
    """
    Function to tokenize and preserve labels
    :param sentences: [['تغییرات', 'قیمت', 'رمز', 'ارز',
                            'اتریوم', 'در', 'یک', 'هفته', 'قبل'], ... ]
    :param labels: [['O', 'O', 'B-ENT', 'I-ENT',
                        'I-ENT', 'O','B-TIM', 'I-TIM', 'I-TIM'], ...]
    :param tokenizer:
    :param mode: ["same", "x_mode"]
    :return: [['تغییر', '##ات', 'قیمت', 'رمز', 'ا', '##رز', 'ا', '##تری', '##وم'
                , 'در', 'یک', 'هفت', '##ه', 'قبل'], ... ]
            [['O', 'O', 'O', 'B-ENT', 'I-ENT', 'I-ENT', 'I-ENT',
            'I-ENT', 'I-ENT', 'O', 'B-TIM', 'I-TIM', 'I-TIM', 'I-TIM'], ... ]
    """
    assert len(sentences) == len(labels), "Sentences and labels should have " \
                                          "the same number of samples"

    for idx, (sentence, label) in enumerate(zip(sentences, labels)):
        tokenized_sentence = list()
        labels_ = list()
        for word, tag in zip(sentence, label):
            # Tokenize each word and count number of its subwords
            tokenized_word = tokenizer.tokenize(str(word))
            n_subwords = len(tokenized_word)

            # The tokenized word is added to the resulting tokenized word list
            tokenized_sentence.extend(tokenized_word)

            # The same label is added to the new list of labels `n_subwords` times
            if mode == "same":
                labels_.extend([tag] * n_subwords)
            elif mode == "x_mode":
                labels_.append(tag)
                labels_.extend(["X"] * (n_subwords - 1))
        sentences[idx], labels[idx] = tokenized_sentence, labels_

    return sentences, labels


def add_special_tokens(sentences: List[list], labels: List[list], cls_token: str,
                       sep_token: str) -> [List[list], List[list]]:
    """

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


def truncate_sequence(texts: List[list], max_length: int) -> list:
    """

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

    :param data: [["item_1", "item_2", "item_3", "pad_item"],
                ["item_1", "pad_item", "pad_item", "pad_item"]]
    :param pad_item: pad_item
    :return: [[1, 1, 1, 0], [1, 0, 0, 0]]
    """

    return [[1 if item != pad_item else 0 for item in sample] for sample in data]
