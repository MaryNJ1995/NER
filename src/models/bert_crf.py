# -*- coding: utf-8 -*-
# pylint: disable-msg=import-error
# pylint: disable-msg=too-many-ancestors
# pylint: disable-msg=arguments-differ
# ========================================================
"""This module is written for write BERT_CRF classifier."""
# ========================================================


# ========================================================
# Imports
# ========================================================


from typing import List
import torch
from torchcrf import CRF
import pytorch_lightning as pl
import transformers
import numpy as np
from seqeval.metrics import f1_score, accuracy_score, classification_report

from utils import ignore_pad_index, convert_index_to_tag
from evaluation import Evaluator
from models import add_metric_to_log_dic

__author__ = "Ehsan Tavan"
__project__ = "NER_SemEval"
__version__ = "1.0.0"
__date__ = "2021/11/18"
__email__ = "tavan.ehsan@gmail.com"


class Classifier(pl.LightningModule):
    """
    Classifier class to write BERT_CRF classifier
    """

    def __init__(self, idx2tag: dict, tag2idx: dict, pad_token: str, config):
        super().__init__()
        self.config = config
        self.idx2tag = idx2tag
        self.tag2idx = tag2idx
        self.pad_token = pad_token

        self.tags = self.extract_tags()

        bert_model = transformers.BertModel.from_pretrained(
            config.language_model_path, return_dict=True)

        self.network = torch.nn.ModuleDict({
            "bert_model": bert_model,
            "output_layer": torch.nn.Linear(in_features=bert_model.config.hidden_size,
                                            out_features=len(self.idx2tag)),
            "crf_layer": CRF(len(self.idx2tag), batch_first=True)
        })

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

        bert_output = self.network["bert_model"](input_ids=batch["input_ids"],
                                                 attention_mask=batch["attention_mask"])
        emission = self.network["output_layer"](bert_output.last_hidden_state)
        if labels is not None:
            loss = -self.network["crf_layer"](
                torch.nn.functional.log_softmax(emission, 2),
                labels, mask=attn_masks, reduction="mean")
            prediction = self.network["crf_layer"].decode(
                emission, mask=attn_masks
            )
            return prediction, loss
        else:
            prediction = self.network["crf_layer"].decode(
                emission, mask=attn_masks
            )
        return prediction

    def extract_tags(self) -> list:
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

        true_indexes = true_indexes.to("cpu").numpy()
        pred_indexes = list(pred_indexes)

        # true_indexes = true_indexes.to("cpu").numpy()
        # pred_indexes = pred_indexes.detach().cpu().numpy()
        # pred_indexes = [list(p) for p in np.argmax(pred_indexes, axis=2)]

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
        outputs, loss = self.forward(batch, batch["target"])

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
        outputs, loss = self.forward(batch, batch["target"])

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
        outputs, loss = self.forward(batch, batch["target"])

        true_targets, pred_targets = self._convert_pred_indexes_to_entities(
            true_indexes=batch["target"], pred_indexes=outputs)

        metrics2value = {"val_loss": loss,
                         "val_accuracy": accuracy_score(true_targets, pred_targets),
                         "val_f1": f1_score(true_targets, pred_targets)}

        evaluator = Evaluator(true=true_targets, pred=pred_targets, tags=self.tags)
        results, results_agg = evaluator.evaluate()
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
        return [optimizer]
