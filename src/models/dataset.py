import torch
import pytorch_lightning as pl
from typing import List
from torch.utils.data import Dataset, DataLoader
from data_preparation import create_attention_masks, pad_sequence, truncate_sequence, pad_sequence_2
from utils import find_max_length_in_list


class CustomDataset(Dataset):
    def __init__(self, texts: List[list], targets: List[list], indexed_sentences: List[list],
                 subtoken_checks: List[list], bpemb_ids: List[list], tokens: List[list],
                 max_length: int, tokenizer, target_indexer):
        self.texts = texts
        self.targets = targets
        self.indexed_sentences = indexed_sentences
        self.subtoken_checks = subtoken_checks
        self.bpemb_ids = bpemb_ids
        self.tokens = tokens
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.target_indexer = target_indexer
        self.position = [str(p) for p in range(self.max_length)]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item_index):
        data = self.tokenizer.encode_plus(text=self.texts[item_index],
                                          add_special_tokens=True,
                                          max_length=self.max_length,
                                          return_tensors="pt",
                                          padding="max_length",
                                          truncation=True)
        subtoken_check = self.tokenizer.encode_plus(text=self.subtoken_checks[item_index],
                                                    add_special_tokens=True,
                                                    max_length=self.max_length,
                                                    return_tensors="pt",
                                                    padding="max_length",
                                                    truncation=True).input_ids

        # position = self.tokenizer.encode_plus(text=self.position,
        #                                       add_special_tokens=True,
        #                                       max_length=self.max_length,
        #                                       return_tensors="pt",
        #                                       padding="max_length",
        #                                       truncation=True).input_ids

        target = self.target_indexer.convert_samples_to_indexes([self.targets[item_index]])[0]

        # indexed_sentences = self.indexed_sentences[item_index]
        bpemb_ids = self.bpemb_ids[item_index]
        # tokens = self.tokens[item_index]

        # indexed_sentences = pad_sequence([indexed_sentences], max_length=self.max_length, pad_item=0)[0]
        # tokens = pad_sequence([tokens], max_length=self.max_length, pad_item="<PAD>")[0]

        bpemb_ids = pad_sequence_2([bpemb_ids], max_length=self.max_length, pad_item=0)[0]

        # print(type(bpemb_ids))

        # indexed_sentences = truncate_sequence(indexed_sentences, self.max_length)[0]

        # assert len(indexed_sentences) == len(subtoken_check.flatten())

        return {"input_ids": data["input_ids"].flatten(), "target": torch.LongTensor(target),
                "attention_mask": data["attention_mask"].flatten(),
                "subtoken_check": subtoken_check.flatten(),
                "bpemb_ids": torch.tensor(bpemb_ids)}
                #"position": position}  # ,
        # "tokens": tokens}#,
        # "indexed_sentences": torch.LongTensor(indexed_sentences)}  # ,
        # "bpemb_ids": torch.tensor(bpemb_ids),
        # "tokens": tokens}


class LstmDataset(Dataset):
    def __init__(self, texts: List[list], targets: List[list], max_length: int,
                 tokenizer, target_indexer):
        self.texts = texts
        self.targets = targets
        self.max_length = max_length
        self.token_indexer = tokenizer
        self.target_indexer = target_indexer

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item_index):
        attention_masks = create_attention_masks(data=[self.texts[item_index]],
                                                 pad_item="[PAD]")[0]

        text = self.token_indexer.convert_samples_to_indexes([self.texts[item_index]])[0]

        target = self.target_indexer.convert_samples_to_indexes([self.targets[item_index]])[0]

        return {"input_ids": torch.LongTensor(text), "target": torch.LongTensor(target),
                "attention_mask": torch.LongTensor(attention_masks)}


class DataModule(pl.LightningDataModule):

    def __init__(self, data: dict, batch_size, max_length, tokenizer, target_indexer):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.target_indexer = target_indexer
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None

    def setup(self):
        self.train_dataset = CustomDataset(texts=self.data["train_data"][0],
                                           targets=self.data["train_data"][1],
                                           indexed_sentences=self.data["train_data"][2],
                                           subtoken_checks=self.data["train_data"][3],
                                           bpemb_ids=self.data["train_data"][4],
                                           tokens=self.data["train_data"][5],
                                           max_length=self.max_length,
                                           tokenizer=self.tokenizer,
                                           target_indexer=self.target_indexer)

        self.val_dataset = CustomDataset(texts=self.data["val_data"][0],
                                         targets=self.data["val_data"][1],
                                         indexed_sentences=self.data["val_data"][2],
                                         subtoken_checks=self.data["val_data"][3],
                                         bpemb_ids=self.data["val_data"][4],
                                         tokens=self.data["val_data"][5],
                                         max_length=self.max_length,
                                         tokenizer=self.tokenizer,
                                         target_indexer=self.target_indexer)

        self.test_dataset = CustomDataset(texts=self.data["val_data"][0],
                                          targets=self.data["val_data"][1],
                                          indexed_sentences=self.data["val_data"][2],
                                          subtoken_checks=self.data["val_data"][3],
                                          bpemb_ids=self.data["val_data"][4],
                                          tokens=self.data["val_data"][5],
                                          max_length=self.max_length,
                                          tokenizer=self.tokenizer,
                                          target_indexer=self.target_indexer)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=10)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=10)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=10)
