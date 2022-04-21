# -*- coding: utf-8 -*-
# pylint: disable-msg=import-error
# pylint: disable-msg=too-many-ancestors
# pylint: disable-msg=arguments-differ
# ========================================================
"""This module is written for write multi model classifier."""
# ========================================================


# ========================================================
# Imports
# ========================================================

import torch
import pytorch_lightning as pl
import transformers
import numpy as np
from typing import List
import torch.nn.functional as F
from seqeval.metrics import f1_score, accuracy_score, classification_report

from utils import ignore_pad_index, convert_index_to_tag
from evaluation import Evaluator
from .helper import add_metric_to_log_dic
from models.transformers import EncoderLayer


def kl_div(p, q):
    out = (p * ((p + 1e-5).log() - (q + 1e-5).log())).sum(-1)
    return out


class Model(torch.nn.Module):
    def __init__(self, idx2tag: dict, tag2idx: dict, token2idx: dict, pad_token: str, config):
        super().__init__()
        self.config = config
        self.idx2tag = idx2tag
        self.tag2idx = tag2idx
        self.token2idx = token2idx
        self.pad_token = pad_token

        self.mt5_model = transformers.MT5EncoderModel.from_pretrained(
            config.language_model_path)
        self.output_layer = torch.nn.Linear(in_features=self.mt5_model.config.hidden_size,
                                            out_features=len(self.idx2tag))
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.tag2idx[self.pad_token])

    def forward(self, batch, labels=None):
        """
        forward method to run model
        :param batch:
        :param labels:
        :return:
        """
        mt5_tokens = self.mt5_model(input_ids=batch["input_ids"]).last_hidden_state
        outputs = self.output_layer(mt5_tokens)
        if labels is not None:
            loss = self.criterion(
                outputs.view(-1, outputs.shape[-1]), batch["target"].view(-1))
            return loss, outputs
        return outputs


class Classifier(pl.LightningModule):
    def __init__(self, idx2tag: dict, tag2idx: dict, token2idx: dict, pad_token: str, config):
        super().__init__()
        self.config = config
        self.idx2tag = idx2tag
        self.tag2idx = tag2idx
        self.token2idx = token2idx
        self.pad_token = pad_token

        self.tags = self.extract_tags()

        self.models = torch.nn.ModuleList()
        self.criterion = torch.nn.CrossEntropyLoss()
        for i in range(2):
            model = Model(idx2tag, tag2idx, token2idx, pad_token, config)
            self.models.append(model)

        self.save_hyperparameters()

    def forward(self, batch, labels=None):
        """
        forward method to run model
        :param batch:
        :param labels:
        :return:
        """
        num_models = len(self.models)
        outputs = []
        if labels is not None:
            loss = 0
            for i in range(num_models):
                loss, output = self.models[i](
                    batch,
                    labels=labels,  # .to("cuda:0"),
                )
                outputs.append(output)
                loss += loss
            loss = loss / num_models
            probs = [F.softmax(output, dim=-1) for output in outputs]
            avg_prob = torch.stack(probs, dim=0).mean(0)
            mask = (labels.view(-1) != -1).to(outputs[0])
            reg_loss = sum([kl_div(avg_prob, prob) * labels for prob in probs]) / num_models
            reg_loss = reg_loss.sum() / (mask.sum() + 1e-3)
            loss = loss + self.config.alpha_t * reg_loss
            return loss, avg_prob
        else:
            for i in range(num_models):
                output = self.models[i](batch)
                outputs.append(output)
            probs = [F.softmax(output, dim=-1) for output in outputs]
            avg_prob = torch.stack(probs, dim=0).mean(0)
            return avg_prob

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

    def training_step(self, batch: dict, batch_idx: int):
        """
        training_step method for train model
        :param batch:
        :param batch_idx:
        :return:
        """
        if batch_idx < self.config.alpha_warmup_ratio * self.config.steps_per_epoch:
            self.config.alpha_t = 0.0
        else:
            self.config.alpha_t = self.config.alpha

        loss, outputs = self.forward(batch, labels=batch["target"])

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
        return [optimizer]# , [scheduler]
