from torch.utils.data import Dataset
from typing import List
import torch
from data_preparation import pad_sequence_2, pad_sequence


class InferenceDataset(Dataset):
    def __init__(self, texts: List[list], subtoken_checks: List[list], bpemb_ids: List[list],
                 tokens: List[list], tokenizer, max_length: int):
        self.texts = texts
        self.subtoken_checks = subtoken_checks
        self.bpemb_ids = bpemb_ids
        self.tokens = tokens
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.position = [str(p) for p in range(self.max_length)]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item_index):
        inputs = self.tokenizer.encode_plus(
            text=self.texts[item_index],
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True
        )

        subtoken_check = self.tokenizer.encode_plus(
            text=self.subtoken_checks[item_index],
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True
        ).input_ids

        position = self.tokenizer.encode_plus(text=self.position,
                                              add_special_tokens=True,
                                              max_length=self.max_length,
                                              return_tensors="pt",
                                              padding="max_length",
                                              truncation=True).input_ids

        bpemb_ids = self.bpemb_ids[item_index]

        bpemb_ids = pad_sequence_2([bpemb_ids], max_length=self.max_length, pad_item=0)[0]
        tokens = self.tokens[item_index]
        str_tokens = []
        tokens = pad_sequence([tokens], max_length=self.max_length, pad_item="<PAD>")[0]
        str_tokens.append(" ".join(tokens))

        assert len(bpemb_ids) == len(inputs["input_ids"].flatten())

        return {"input_ids": inputs["input_ids"].flatten(),
                "attention_mask": inputs["attention_mask"].flatten(),
                "subtoken_check": subtoken_check.flatten(),
                "bpemb_ids": torch.tensor(bpemb_ids),
                "position": position}
