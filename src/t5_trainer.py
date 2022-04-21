# -*- coding: utf-8 -*-
# ========================================================
"""trainer module is written for train model"""
# ========================================================


# ========================================================
# Imports
# ========================================================

import os
import copy
import logging
from bpemb import BPEmb
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.tuner.tuning import Tuner
from transformers import BertTokenizer, T5Tokenizer, MT5Tokenizer
import itertools

from configuration import BaseConfig
from data_loader import read_text, write_json
from data_preparation import prepare_conll_data, \
    add_special_tokens, tokenize_and_keep_labels, pad_sequence, truncate_sequence
from indexer import Indexer, TokenIndexer
from utils import find_max_length_in_list
from models import DataModule, build_checkpoint_callback
from models.mt5_transformer import Classifier

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    # create config instance
    CONFIG_CLASS = BaseConfig()
    CONFIG = CONFIG_CLASS.get_config()

    # load BPEmb
    BPEMB = BPEmb(model_file=CONFIG.bpemb_model_path,
                  emb_file=CONFIG.bpemb_vocab_path, dim=300)

    # create CSVLogger instance
    LOGGER = CSVLogger(save_dir=CONFIG.saved_model_path, name=CONFIG.model_name)

    # create BertTokenizer instance
    TOKENIZER = T5Tokenizer.from_pretrained(CONFIG.language_model_tokenizer_path)

    # load raw data
    RAW_TRAIN_DATA = read_text(path=os.path.join(CONFIG.processed_data_dir, CONFIG.train_data))
    RAW_VAL_DATA = read_text(path=os.path.join(CONFIG.processed_data_dir, CONFIG.dev_data))

    TRAIN_SENTENCES, TRAIN_LABELS = prepare_conll_data(RAW_TRAIN_DATA)
    logging.debug("We have {} train samples.".format(len(TRAIN_LABELS)))

    VAL_SENTENCES, VAL_LABELS = prepare_conll_data(RAW_VAL_DATA)
    logging.debug("We have {} val samples.".format(len(VAL_LABELS)))

    # Create token indexer
    TOKENS = list(itertools.chain(*TRAIN_SENTENCES))

    TOKEN_INDEXER = TokenIndexer(vocabs=TOKENS)
    TOKEN_INDEXER.build_vocab2idx()
    TOKEN_INDEXER.build_idx2vocab()

    TRAIN_SENTENCES_COPY = copy.deepcopy(TRAIN_SENTENCES)
    VAL_SENTENCES_COPY = copy.deepcopy(VAL_SENTENCES)

    # PADDED_TRAIN_SENTENCES = pad_sequence(TRAIN_SENTENCES_COPY, max_length=SENTENCE_MAX_LENGTH, pad_item="<PAD>")
    # PADDED_VAL_SENTENCES = pad_sequence(VAL_SENTENCES_COPY, max_length=SENTENCE_MAX_LENGTH, pad_item="<PAD>")
    #
    # # sentence truncating
    # TRUNCATED_TRAIN_SENTENCES = truncate_sequence(PADDED_TRAIN_SENTENCES, SENTENCE_MAX_LENGTH)
    # TRUNCATED_VAL_SENTENCES = truncate_sequence(PADDED_VAL_SENTENCES, SENTENCE_MAX_LENGTH)

    INDEXED_TRAIN_SENTENCES = TOKEN_INDEXER.convert_samples_to_indexes(tokenized_samples=TRAIN_SENTENCES_COPY)
    INDEXED_VAL_SENTENCES = TOKEN_INDEXER.convert_samples_to_indexes(tokenized_samples=VAL_SENTENCES_COPY)

    # prepare data for training
    TRAIN_SENTENCES, TRAIN_LABELS, TRAIN_SUBTOKEN_CHECKS, TRAIN_BPEMB_IDS, TRAIN_FLAIR_TOKENS = tokenize_and_keep_labels(
        sentences=TRAIN_SENTENCES,
        labels=TRAIN_LABELS,
        indexed_sentences=INDEXED_TRAIN_SENTENCES,
        tokenizer=TOKENIZER,
        bpemb=BPEMB,
        mode="x_mode")

    VAL_SENTENCES, VAL_LABELS, VAL_SUBTOKEN_CHECKS, VAL_BPEMB_IDS, VAL_FLAIR_TOKENS = tokenize_and_keep_labels(
        sentences=VAL_SENTENCES,
        labels=VAL_LABELS,
        indexed_sentences=INDEXED_VAL_SENTENCES,
        tokenizer=TOKENIZER,
        bpemb=BPEMB,
        mode="x_mode")

    # label padding
    SENTENCE_MAX_LENGTH = find_max_length_in_list(TRAIN_SENTENCES)
    CONFIG.SENTENCE_MAX_LENGTH = SENTENCE_MAX_LENGTH

    TRAIN_LABELS = pad_sequence(TRAIN_LABELS, max_length=SENTENCE_MAX_LENGTH, pad_item=TOKENIZER.pad_token)
    VAL_LABELS = pad_sequence(VAL_LABELS, max_length=SENTENCE_MAX_LENGTH, pad_item=TOKENIZER.pad_token)

    # label truncating
    TRAIN_LABELS = truncate_sequence(TRAIN_LABELS, SENTENCE_MAX_LENGTH)
    VAL_LABELS = truncate_sequence(VAL_LABELS, SENTENCE_MAX_LENGTH)

    # Create target indexer
    TAGS = list(itertools.chain(*TRAIN_LABELS))
    TAGS.append(TOKENIZER.pad_token)

    TARGET_INDEXER = Indexer(vocabs=TAGS)
    TARGET_INDEXER.build_vocab2idx()
    TARGET_INDEXER.build_idx2vocab()

    NUM_BATCHES = len(TRAIN_SENTENCES) // CONFIG.batch_size
    TOTAL_STEPS = 50 * NUM_BATCHES
    WARMUP_STEPS = int(TOTAL_STEPS * 0.01)
    STEPS_PER_EPOCH = len(TRAIN_SENTENCES) // CONFIG.batch_size

    CONFIG.steps_per_epoch = STEPS_PER_EPOCH
    CONFIG.warmup_steps = WARMUP_STEPS

    DATA_MODULE = DataModule(data={"train_data": [TRAIN_SENTENCES, TRAIN_LABELS,
                                                  INDEXED_TRAIN_SENTENCES, TRAIN_SUBTOKEN_CHECKS,
                                                  TRAIN_BPEMB_IDS, TRAIN_FLAIR_TOKENS],
                                   "val_data": [VAL_SENTENCES, VAL_LABELS,
                                                INDEXED_VAL_SENTENCES, VAL_SUBTOKEN_CHECKS,
                                                VAL_BPEMB_IDS, VAL_FLAIR_TOKENS],
                                   "test_data": [VAL_SENTENCES, VAL_LABELS,
                                                 INDEXED_VAL_SENTENCES, VAL_SUBTOKEN_CHECKS,
                                                 VAL_BPEMB_IDS, VAL_FLAIR_TOKENS]},
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
                         callbacks=[CHECKPOINT_CALLBACK, CHECKPOINT_CALLBACK_F1, EARLY_STOPPING_CALLBACK],
                         progress_bar_refresh_rate=60, logger=LOGGER, auto_scale_batch_size=True)

    # Train the Classifier Model
    MODEL = Classifier(tag2idx=TARGET_INDEXER.get_vocab2idx(),
                       idx2tag=TARGET_INDEXER.get_idx2vocab(), token2idx=TOKEN_INDEXER.get_vocab2idx(),
                       pad_token=TOKENIZER.pad_token, config=CONFIG)

    TRAINER.fit(MODEL, DATA_MODULE)
    TRAINER.test(ckpt_path="best", datamodule=DATA_MODULE)

    # TRAIN_LOGS = TRAINER.test(ckpt_path="best", test_dataloaders=DATA_MODULE.train_dataloader())[0]
    # VALID_LOGS = TRAINER.test(ckpt_path="best", test_dataloaders=DATA_MODULE.val_dataloader())[0]
    # TEST_LOGS = TRAINER.test(ckpt_path="best", test_dataloaders=DATA_MODULE.test_dataloader())[0]
    #
    # BEST_RESULT_LOG = create_best_result_log(train_logs=TRAIN_LOGS, valid_logs=VALID_LOGS, test_logs=TEST_LOGS)
    #
    # BEST_RESULT_LOG.to_csv(LOGGER.log_dir + "/best_result_log.csv", index=False)
    #
    # save best model path
    write_json(path=os.path.join(CONFIG.saved_model_path, CONFIG.model_name,
                                 "b_model_path.json"),
               data={"best_model_path": CHECKPOINT_CALLBACK.best_model_path})
