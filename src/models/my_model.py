# -*- coding: utf-8 -*-
# pylint: disable-msg=import-error
# pylint: disable-msg=too-many-ancestors
# pylint: disable-msg=arguments-differ
# ========================================================
"""This module is written for write MY MODEL classifier."""
# ========================================================


# ========================================================
# Imports
# ========================================================

import torch
import pytorch_lightning as pl
import transformers
import numpy as np
from typing import List
from seqeval.metrics import f1_score, accuracy_score, classification_report

from utils import ignore_pad_index, convert_index_to_tag
from evaluation import Evaluator
from .helper import add_metric_to_log_dic
from models.transformers import EncoderLayer, MultiHeadAttentionLayer


class Classifier(pl.LightningModule):
    def __init__(self, idx2tag: dict, tag2idx: dict, token2idx: dict, pad_token: str, config):
        super().__init__()
        self.config = config
        self.idx2tag = idx2tag
        self.tag2idx = tag2idx
        self.token2idx = token2idx
        self.pad_token = pad_token

        self.tags = self.extract_tags()

        self.t5_model = transformers.T5EncoderModel.from_pretrained(
            config.language_model_path)
        transformer_input_dim = (2 * self.t5_model.config.hidden_size)  # + self.config.embedding_dim

        self.attn = MultiHeadAttentionLayer(hid_dim=transformer_input_dim,
                                            n_heads=16,
                                            dropout=config.dropout)

        self.enc_layer = EncoderLayer(hid_dim=transformer_input_dim,
                                      n_heads=8, pf_dim=transformer_input_dim * 2,
                                      dropout=config.dropout)

        # self.position_embedding = torch.nn.Embedding(self.config.SENTENCE_MAX_LENGTH, 2048)

        # self.fc_layer = torch.nn.Linear(in_features=300,
        #                                 out_features=256)

        self.lstm_layer = torch.nn.LSTM(input_size=transformer_input_dim,
                                        hidden_size=256,
                                        num_layers=2,
                                        bidirectional=True,
                                        batch_first=True)

        self.output_layer = torch.nn.Linear(in_features=transformer_input_dim + 512,
                                            out_features=len(self.idx2tag))

        self.dropout = torch.nn.Dropout(config.dropout)

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.tag2idx[self.pad_token])
        self.save_hyperparameters()

    def forward(self, batch, labels=None):
        """
        forward method to run model
        :param batch:
        :param labels:
        :return:
        """
        attn_masks = batch["attention_mask"].type(torch.uint8)
        # positions_embedding = self.position_embedding(positions)

        mt5_tokens = self.t5_model(input_ids=batch["input_ids"]).last_hidden_state

        mt5_subtoken_check = self.t5_model(input_ids=batch["subtoken_check"]).last_hidden_state
        # print(batch["position"])
        # position = batch["position"].squeeze(1)
        # position = self.t5_model(input_ids=position).last_hidden_state

        token_features = torch.cat((mt5_tokens, mt5_subtoken_check), dim=2)

        token_features = self.dropout(token_features)

        attn_context, _ = self.attn(token_features, token_features, token_features)

        # token_features = token_features + positions_embedding
        # token_features.size() = [batch_size, sen_len, embedding_dim+mt5_model.config.hidden_size]

        enc_out = self.enc_layer(attn_context, src_mask=attn_masks)
        # token_features.size() = [batch_size, sen_len, 2048]

        lstm_output, (_, _) = self.lstm_layer(attn_context)
        # lstm_output.size() = [batch_size, sen_len, 512]

        token_features = torch.cat((enc_out, lstm_output), dim=2)

        token_features = self.dropout(token_features)

        # token_features = torch.cat((enc_out, batch["bpemb_ids"]), dim=2)

        return self.output_layer(token_features)

    def extract_tags(self):
        """

        :return:
        """
        tags = set(dict.fromkeys([item.replace("B-", "").replace("I-", "")
                                  for item in self.idx2tag.values()]))
        tags = tags - {self.tag2idx[self.pad_token], "O", "X"}
        return list(tags)

    def _convert_pred_indexes_to_entities(self, true_indexes: torch.Tensor,
                                          pred_indexes: torch.Tensor) -> [List[list], List[list]]:
        """

        :param true_indexes:
        :param pred_indexes:
        :return:
        """
        # Transfer logits and labels to CPU
        true_indexes = true_indexes.detach().cpu().numpy()
        pred_indexes = pred_indexes.detach().cpu().numpy()
        pred_indexes = [list(p) for p in np.argmax(pred_indexes, axis=2)]

        # convert predicted and true index to their tags
        true_entities = convert_index_to_tag(data=true_indexes, idx2tag=self.idx2tag)
        pred_entities = convert_index_to_tag(data=pred_indexes, idx2tag=self.idx2tag)

        # ignore predicted and true pad item
        true_entities, pred_entities = ignore_pad_index(true_labels=true_entities,
                                                        pred_labels=pred_entities,
                                                        pad_token=self.pad_token)
        return true_entities, pred_entities

    def training_step(self, batch: dict, _):
        """
        training_step method for train model
        :param batch:
        :param _:
        :return:
        """
        outputs = self.forward(batch)
        loss = self.criterion(outputs.view(-1, outputs.shape[-1]), batch["target"].view(-1))

        true_targets, pred_targets = self._convert_pred_indexes_to_entities(
            true_indexes=batch["target"], pred_indexes=outputs)

        metrics2value = {"train_loss": loss,
                         "train_accuracy": accuracy_score(true_targets, pred_targets),
                         "train_f1": f1_score(true_targets, pred_targets)}

        metric_names, metric_value = self.classification_f1_report(pred_targets,
                                                                   true_targets,
                                                                   data_name="train")
        for name, value in zip(metric_names, metric_value):
            metrics2value.update({name: value})

        self.log_dict(metrics2value, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {"loss": loss, "predictions": outputs, "labels": batch["target"]}

    def validation_step(self, batch: dict, _):
        """
        validation_step method for evaluate model
        :param batch:
        :param _:
        :return:
        """
        outputs = self.forward(batch)
        loss = self.criterion(outputs.view(-1, outputs.shape[-1]), batch["target"].view(-1))

        true_targets, pred_targets = self._convert_pred_indexes_to_entities(
            true_indexes=batch["target"], pred_indexes=outputs)

        metrics2value = {"val_loss": loss,
                         "val_accuracy": accuracy_score(true_targets, pred_targets),
                         "val_f1": f1_score(true_targets, pred_targets)}

        metric_names, metric_value = self.classification_f1_report(pred_targets,
                                                                   true_targets,
                                                                   data_name="val")
        for name, value in zip(metric_names, metric_value):
            metrics2value.update({name: value})

        self.log_dict(metrics2value, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch: dict, _):
        """
        test_step method for test model
        :param batch:
        :param _:
        :return:
        """
        outputs = self.forward(batch)
        loss = self.criterion(outputs.view(-1, outputs.shape[-1]), batch["target"].view(-1))

        true_targets, pred_targets = self._convert_pred_indexes_to_entities(
            true_indexes=batch["target"], pred_indexes=outputs)

        metrics2value = {"test_loss": loss,
                         "test_accuracy": accuracy_score(true_targets, pred_targets),
                         "test_f1": f1_score(true_targets, pred_targets)}

        evaluator = Evaluator(true=true_targets, pred=pred_targets, tags=self.tags)
        _, results_agg = evaluator.evaluate()
        metrics2value = add_metric_to_log_dic(log_dict=metrics2value, results_agg=results_agg,
                                              tags=self.tags)

        self.log_dict(metrics2value, on_step=False, on_epoch=True, prog_bar=True, logger=False)

        return loss

    def classification_f1_report(self, pred_targets_value: List[list],
                                 true_targets_value: List[list], data_name: str) -> [list, list]:
        """

        :param pred_targets_value:
        :param true_targets_value:
        :param data_name:
        :return:
        """
        report = classification_report(y_true=true_targets_value, y_pred=pred_targets_value)
        metric_names, metric_values = [], []
        for line in report.split("\n"):
            if line != "" and (line.split()[0] in self.tags):
                metric_names.append(data_name + "_" + line.split()[0] + "_f1")
                metric_values.append(float(line.split()[3]))
            elif line != "" and (line.split()[0] == "avg"):
                metric_names.append(data_name + "_" + line.split()[0] + "_entities_f1")
                metric_values.append(float(line.split()[5]))
        return metric_names, metric_values

    def configure_optimizers(self):
        """
        configure_optimizers method to config optimizer
        :return
        """
        optimizer = transformers.AdamW(self.parameters(), lr=self.config.lr)
        # warmup_steps = self.config.steps_per_epoch // 3
        # total_steps = self.config.steps_per_epoch * self.config.n_epochs - warmup_steps
        # scheduler = transformers.get_linear_schedule_with_warmup(
        #     optimizer, warmup_steps, total_steps
        # )
        return [optimizer]  # , [scheduler]
