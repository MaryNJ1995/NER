# -*- coding: utf-8 -*-
# ========================================================
"""trainer module is written for train model"""
# ========================================================


# ========================================================
# Imports
# ========================================================

import os
import logging
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping
from transformers import BertTokenizer
import itertools

from configuration import BaseConfig
from data_loader import read_text, write_json
from data_preparation import prepare_conll_data, \
    add_special_tokens, tokenize_and_keep_labels, pad_sequence, truncate_sequence
from indexer import Indexer
from utils import find_max_length_in_list
from models import DataModule, build_checkpoint_callback
from models.Bert_LSTM_CRF import Classifier
from evaluation import create_best_result_log

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    # create config instance
    CONFIG_CLASS = BaseConfig()
    CONFIG = CONFIG_CLASS.get_config()

    # create CSVLogger instance
    LOGGER = CSVLogger(save_dir=CONFIG.saved_model_path, name=CONFIG.model_name)

    # create BertTokenizer instance
    TOKENIZER = BertTokenizer.from_pretrained(CONFIG.language_model_tokenizer_path)

    # load raw data
    RAW_TRAIN_DATA = read_text(path=os.path.join(CONFIG.processed_data_dir, CONFIG.train_data))
    RAW_VAL_DATA = read_text(path=os.path.join(CONFIG.processed_data_dir, CONFIG.dev_data))

    TRAIN_SENTENCES, TRAIN_LABELS = prepare_conll_data(RAW_TRAIN_DATA)
    logging.debug("We have {} train samples.".format(len(TRAIN_LABELS)))

    VAL_SENTENCES, VAL_LABELS = prepare_conll_data(RAW_VAL_DATA)
    logging.debug("We have {} val samples.".format(len(VAL_LABELS)))

    # prepare data for training
    TRAIN_SENTENCES, TRAIN_LABELS = tokenize_and_keep_labels(sentences=TRAIN_SENTENCES,
                                                             labels=TRAIN_LABELS,
                                                             tokenizer=TOKENIZER,
                                                             mode="x_mode")
    TRAIN_SENTENCES, TRAIN_LABELS = add_special_tokens(TRAIN_SENTENCES, TRAIN_LABELS,
                                                       TOKENIZER.cls_token, TOKENIZER.sep_token)

    VAL_SENTENCES, VAL_LABELS = tokenize_and_keep_labels(sentences=VAL_SENTENCES,
                                                         labels=VAL_LABELS,
                                                         tokenizer=TOKENIZER,
                                                         mode="x_mode")
    VAL_SENTENCES, VAL_LABELS = add_special_tokens(VAL_SENTENCES, VAL_LABELS,
                                                   TOKENIZER.cls_token, TOKENIZER.sep_token)

    SENTENCE_MAX_LENGTH = int(find_max_length_in_list(TRAIN_SENTENCES) * 0.75)

    TRAIN_SENTENCES = truncate_sequence(TRAIN_SENTENCES, SENTENCE_MAX_LENGTH)
    TRAIN_LABELS = truncate_sequence(TRAIN_LABELS, SENTENCE_MAX_LENGTH)

    VAL_SENTENCES = truncate_sequence(VAL_SENTENCES, SENTENCE_MAX_LENGTH)
    VAL_LABELS = truncate_sequence(VAL_LABELS, SENTENCE_MAX_LENGTH)

    TRAIN_SENTENCES = pad_sequence(TRAIN_SENTENCES, SENTENCE_MAX_LENGTH)
    TRAIN_LABELS = pad_sequence(TRAIN_LABELS, SENTENCE_MAX_LENGTH)

    VAL_SENTENCES = pad_sequence(VAL_SENTENCES, SENTENCE_MAX_LENGTH)
    VAL_LABELS = pad_sequence(VAL_LABELS, SENTENCE_MAX_LENGTH)

    TARGET_INDEXER = Indexer(vocabs=list(itertools.chain(*TRAIN_LABELS)))
    TARGET_INDEXER.build_vocab2idx()
    TARGET_INDEXER.build_idx2vocab()

    NUM_BATCHES = len(TRAIN_SENTENCES) // CONFIG.batch_size
    TOTAL_STEPS = 50 * NUM_BATCHES
    WARMUP_STEPS = int(TOTAL_STEPS * 0.01)

    CONFIG.total_steps = TOTAL_STEPS
    CONFIG.warmup_steps = WARMUP_STEPS

    DATA_MODULE = DataModule(data={"train_data": [TRAIN_SENTENCES, TRAIN_LABELS],
                                   "val_data": [VAL_SENTENCES, VAL_LABELS],
                                   "test_data": [VAL_SENTENCES, VAL_LABELS]},
                             batch_size=CONFIG.batch_size,
                             max_length=SENTENCE_MAX_LENGTH,
                             tokenizer=TOKENIZER,
                             target_indexer=TARGET_INDEXER)

    DATA_MODULE.setup()

    CHECKPOINT_CALLBACK = build_checkpoint_callback(CONFIG.save_top_k)
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=5)

    # Instantiate the Model Trainer
    TRAINER = pl.Trainer(max_epochs=CONFIG.n_epochs, gpus=[1],  # CONFIG.num_of_gpu,
                         callbacks=[CHECKPOINT_CALLBACK, early_stopping_callback],
                         progress_bar_refresh_rate=60, logger=LOGGER)

    # Train the Classifier Model
    MODEL = Classifier(tag2idx=TARGET_INDEXER.get_vocab2idx(),
                       idx2tag=TARGET_INDEXER.get_idx2vocab(), pad_token="[PAD]",
                       config=CONFIG)

    TRAINER.fit(MODEL, DATA_MODULE)
    TRAINER.test(ckpt_path="best", datamodule=DATA_MODULE)

    TRAIN_LOGS = TRAINER.test(ckpt_path="best", test_dataloaders=DATA_MODULE.train_dataloader())[0]
    VALID_LOGS = TRAINER.test(ckpt_path="best", test_dataloaders=DATA_MODULE.val_dataloader())[0]
    TEST_LOGS = TRAINER.test(ckpt_path="best", test_dataloaders=DATA_MODULE.test_dataloader())[0]

    BEST_RESULT_LOG = create_best_result_log(train_logs=TRAIN_LOGS, valid_logs=VALID_LOGS, test_logs=TEST_LOGS)

    BEST_RESULT_LOG.to_csv(LOGGER.log_dir + "/best_result_log.csv", index=False)

    # save best model path
    write_json(path=os.path.join(CONFIG.saved_model_path, CONFIG.model_name,
                                 "b_model_path.json"),
               data={"best_model_path": CHECKPOINT_CALLBACK.best_model_path})
