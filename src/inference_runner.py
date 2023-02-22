# -*- coding: utf-8 -*-
"""
    Complex NER Project:
"""

# ============================ Third Party libs ============================
import os
import copy
import pickle as pkl
from torch.utils.data import DataLoader
import transformers
import logging

# ============================ My packages ============================
from configuration import BaseConfig
from data_loader import read_text
from models.complex_ner_model import Classifier
from dataset import InferenceDataset
from data_preparation import prepare_conll_data, create_test_samples
from utils import find_max_length_in_list, handle_subtoken_labels, convert_x_label_to_true_label, \
    progress_bar
from inference import Inference

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    CONFIG = CONFIG_CLASS.get_config()

    logging.debug("test file : {}".format(CONFIG.dev_data))

    MODEL_PATH = "../assets/saved_models/english/version_1/checkpoints/" \
                 "QTag-epoch=00-val_loss=1.85.ckpt"

    MODEL = Classifier.load_from_checkpoint(MODEL_PATH, map_location="cuda:1")
    MODEL.eval().to("cuda:1")

    TOKENIZER = transformers.MT5Tokenizer.from_pretrained(CONFIG.language_model_tokenizer_path)

    # load raw data
    RAW_DATA = read_text(path=os.path.join(CONFIG.processed_data_dir, CONFIG.dev_data))

    TOKENS, LABELS = prepare_conll_data(RAW_DATA)
    TOKENS = TOKENS[:10]
    LABELS = LABELS[:10]
    TOKEN_ = copy.copy(TOKENS)

    SENTENCES, SUBTOKEN_CHECKS = create_test_samples(TOKEN_, TOKENIZER)
    SEN_MAX_LENGTH = find_max_length_in_list(SENTENCES)

    INFER = Inference(MODEL, TOKENIZER)

    DATASET = InferenceDataset(texts=SENTENCES, subtoken_checks=SUBTOKEN_CHECKS,
                               tokenizer=TOKENIZER, max_length=SEN_MAX_LENGTH)

    DATALOADER = DataLoader(DATASET, batch_size=1,
                            shuffle=False, num_workers=4)

    FILE = open("tr.pred.conll", "w")

    PREDICTED_LABELS = []
    for i_batch, sample_batched in enumerate(DATALOADER):
        sample_batched["input_ids"] = sample_batched["input_ids"].to("cuda:1")
        sample_batched["attention_mask"] = sample_batched["attention_mask"].to("cuda:1")
        OUTPUT = INFER.predict(sample_batched)

        ENTITIES = INFER.convert_ids_to_entities(OUTPUT)

        ENTITIES = handle_subtoken_labels(ENTITIES, SUBTOKEN_CHECKS[i_batch])

        ENTITIES = convert_x_label_to_true_label(ENTITIES, "X")
        assert len(ENTITIES) == len(LABELS[i_batch]), f"{len(LABELS[i_batch])}, {len(ENTITIES)}"

        PREDICTED_LABELS.append(ENTITIES)

        for entity in ENTITIES:
            FILE.write(entity.strip())
            FILE.write("\n")
        FILE.write("\n")

        progress_bar(i_batch, len(DATALOADER), "testing ....")

    DATA = [TOKENS, LABELS, PREDICTED_LABELS]
    with open("preds.pkl", "wb") as file:
        pkl.dump(DATA, file)
