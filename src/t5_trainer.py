# -*- coding: utf-8 -*-
"""
    Complex NER Project:
"""

# ============================ Third Party libs ============================

import os
import copy
import logging
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping
from transformers import T5Tokenizer
import itertools

# ============================ My packages ============================
from configuration import BaseConfig
from data_loader import read_text, write_json
from data_preparation import prepare_conll_data, tokenize_and_keep_labels, \
    pad_sequence, truncate_sequence
from indexer import Indexer
from utils import find_max_length_in_list
from models import build_checkpoint_callback
from dataset import DataModule
from models.complex_ner_model import Classifier

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    # create config instance
    CONFIG_CLASS = BaseConfig()
    CONFIG = CONFIG_CLASS.get_config()

    # create CSVLogger instance
    LOGGER = CSVLogger(save_dir=CONFIG.saved_model_path, name=CONFIG.model_name)

    # create BertTokenizer instance
    TOKENIZER = T5Tokenizer.from_pretrained(CONFIG.language_model_tokenizer_path)

    # load raw data
    RAW_TRAIN_DATA = read_text(path=os.path.join(CONFIG.processed_data_dir, CONFIG.train_data))
    RAW_VAL_DATA = read_text(path=os.path.join(CONFIG.processed_data_dir, CONFIG.dev_data))

    TRAIN_SENTENCES, TRAIN_LABELS = prepare_conll_data(RAW_TRAIN_DATA)
    TRAIN_SENTENCES = TRAIN_SENTENCES[:100]
    TRAIN_LABELS = TRAIN_LABELS[:100]
    logging.debug("We have {} train samples.".format(len(TRAIN_LABELS)))

    VAL_SENTENCES, VAL_LABELS = prepare_conll_data(RAW_VAL_DATA)
    logging.debug("We have {} val samples.".format(len(VAL_LABELS)))
    VAL_SENTENCES = VAL_SENTENCES
    VAL_LABELS = VAL_LABELS

    # Create token indexer
    TOKENS = list(itertools.chain(*TRAIN_SENTENCES))

    TRAIN_SENTENCES_COPY = copy.deepcopy(TRAIN_SENTENCES)
    VAL_SENTENCES_COPY = copy.deepcopy(VAL_SENTENCES)

    # prepare data for training
    TRAIN_SENTENCES, TRAIN_LABELS, TRAIN_SUBTOKEN_CHECKS = tokenize_and_keep_labels(
        sentences=TRAIN_SENTENCES,
        labels=TRAIN_LABELS,
        tokenizer=TOKENIZER,
        mode="x_mode")

    logging.debug("Create Train Samples")

    VAL_SENTENCES, VAL_LABELS, VAL_SUBTOKEN_CHECKS = tokenize_and_keep_labels(
        sentences=VAL_SENTENCES,
        labels=VAL_LABELS,
        tokenizer=TOKENIZER,
        mode="x_mode")
    logging.debug("Create Valid Samples")

    # label padding
    SENTENCE_MAX_LENGTH = find_max_length_in_list(TRAIN_SENTENCES)
    CONFIG.SENTENCE_MAX_LENGTH = SENTENCE_MAX_LENGTH

    TRAIN_LABELS = pad_sequence(TRAIN_LABELS, max_length=SENTENCE_MAX_LENGTH,
                                pad_item=TOKENIZER.pad_token)
    VAL_LABELS = pad_sequence(VAL_LABELS, max_length=SENTENCE_MAX_LENGTH,
                              pad_item=TOKENIZER.pad_token)

    # label truncating
    TRAIN_LABELS = truncate_sequence(TRAIN_LABELS, SENTENCE_MAX_LENGTH)
    VAL_LABELS = truncate_sequence(VAL_LABELS, SENTENCE_MAX_LENGTH)

    # Create target indexer
    TAGS = list(itertools.chain(*TRAIN_LABELS))
    TAGS.append(TOKENIZER.pad_token)

    TARGET_INDEXER = Indexer(vocabs=TAGS)
    TARGET_INDEXER.build_vocab2idx()
    TARGET_INDEXER.build_idx2vocab()
    TARGET_INDEXER.save(CONFIG.assets_dir)

    NUM_BATCHES = len(TRAIN_SENTENCES) // CONFIG.batch_size
    TOTAL_STEPS = 50 * NUM_BATCHES
    WARMUP_STEPS = int(TOTAL_STEPS * 0.01)
    STEPS_PER_EPOCH = len(TRAIN_SENTENCES) // CONFIG.batch_size

    CONFIG.steps_per_epoch = STEPS_PER_EPOCH
    CONFIG.warmup_steps = WARMUP_STEPS

    DATA_MODULE = DataModule(data={"train_data": [TRAIN_SENTENCES, TRAIN_LABELS,
                                                  TRAIN_SUBTOKEN_CHECKS],
                                   "val_data": [VAL_SENTENCES, VAL_LABELS,
                                                VAL_SUBTOKEN_CHECKS],
                                   "test_data": [VAL_SENTENCES, VAL_LABELS,
                                                 VAL_SUBTOKEN_CHECKS]},
                             batch_size=CONFIG.batch_size,
                             max_length=SENTENCE_MAX_LENGTH,
                             tokenizer=TOKENIZER,
                             target_indexer=TARGET_INDEXER)

    DATA_MODULE.setup()

    CHECKPOINT_CALLBACK = build_checkpoint_callback(CONFIG.save_top_k)
    CHECKPOINT_CALLBACK_F1 = build_checkpoint_callback(CONFIG.save_top_k, monitor="val_f1",
                                                       mode="max")
    EARLY_STOPPING_CALLBACK = EarlyStopping(monitor="val_loss", patience=30, mode="min")

    # Instantiate the Model Trainer
    TRAINER = pl.Trainer(max_epochs=CONFIG.n_epochs, gpus=[1],  # CONFIG.num_of_gpu,
                         callbacks=[CHECKPOINT_CALLBACK, CHECKPOINT_CALLBACK_F1,
                                    EARLY_STOPPING_CALLBACK],
                         progress_bar_refresh_rate=60, logger=LOGGER, auto_scale_batch_size=True)

    # Train the Classifier Model
    MODEL = Classifier(tag2idx=TARGET_INDEXER.get_vocab2idx(),
                       idx2tag=TARGET_INDEXER.get_idx2vocab(),
                       pad_token=TOKENIZER.pad_token, config=CONFIG)

    TRAINER.fit(MODEL, DATA_MODULE)
    TRAINER.test(ckpt_path="best", datamodule=DATA_MODULE)

    # save best model path
    write_json(path=os.path.join(CONFIG.saved_model_path, CONFIG.model_name,
                                 "b_model_path.json"),
               data={"best_model_path": CHECKPOINT_CALLBACK.best_model_path})
