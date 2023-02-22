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
from indexer import Indexer, TokenIndexer
from utils import find_max_length_in_list
from models import DataModule, build_checkpoint_callback
from models.lstm_model import Classifier
from evaluation import create_best_result_log

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    # create config instance
    CONFIG_CLASS = BaseConfig()
    CONFIG = CONFIG_CLASS.get_config()

    # create CSVLogger instance
    LOGGER = CSVLogger(save_dir=CONFIG.saved_model_path, name=CONFIG.model_name)

    # create BertTokenizer instance
    # TOKENIZER = BertTokenizer.from_pretrained(CONFIG.language_model_tokenizer_path)

    # load raw data
    RAW_TRAIN_DATA = read_text(path=os.path.join(CONFIG.processed_data_dir, CONFIG.train_data))
    RAW_VAL_DATA = read_text(path=os.path.join(CONFIG.processed_data_dir, CONFIG.dev_data))

    TRAIN_SENTENCES, TRAIN_LABELS = prepare_conll_data(RAW_TRAIN_DATA)
    logging.debug("We have {} train samples.".format(len(TRAIN_LABELS)))

    VAL_SENTENCES, VAL_LABELS = prepare_conll_data(RAW_VAL_DATA)
    logging.debug("We have {} val samples.".format(len(VAL_LABELS)))

    TRAIN_SENTENCES, TRAIN_LABELS = add_special_tokens(TRAIN_SENTENCES, TRAIN_LABELS,
                                                       "[CLS]", "[SEP]")

    VAL_SENTENCES, VAL_LABELS = add_special_tokens(VAL_SENTENCES, VAL_LABELS,
                                                   "[CLS]", "[SEP]")

    SENTENCE_MAX_LENGTH = find_max_length_in_list(TRAIN_SENTENCES)

    TRAIN_SENTENCES = pad_sequence(TRAIN_SENTENCES, SENTENCE_MAX_LENGTH)
    TRAIN_LABELS = pad_sequence(TRAIN_LABELS, SENTENCE_MAX_LENGTH)

    VAL_SENTENCES = pad_sequence(VAL_SENTENCES, SENTENCE_MAX_LENGTH)
    VAL_LABELS = pad_sequence(VAL_LABELS, SENTENCE_MAX_LENGTH)

    TOKEN_INDEXER = TokenIndexer(vocabs=list(itertools.chain(*TRAIN_SENTENCES)))
    TOKEN_INDEXER.build_vocab2idx()
    TOKEN_INDEXER.build_idx2vocab()

    TARGET_INDEXER = Indexer(vocabs=list(itertools.chain(*TRAIN_LABELS)))
    TARGET_INDEXER.build_vocab2idx()
    TARGET_INDEXER.build_idx2vocab()

    DATA_MODULE = DataModule(data={"train_data": [TRAIN_SENTENCES, TRAIN_LABELS],
                                   "val_data": [VAL_SENTENCES, VAL_LABELS],
                                   "test_data": [VAL_SENTENCES, VAL_LABELS]},
                             batch_size=CONFIG.batch_size,
                             max_length=SENTENCE_MAX_LENGTH,
                             tokenizer=TOKEN_INDEXER,
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
                       config=CONFIG, num_vocab=len(TOKEN_INDEXER.get_idx2vocab()))

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
