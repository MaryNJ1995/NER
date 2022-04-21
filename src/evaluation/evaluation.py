import logging
from collections import namedtuple
from copy import deepcopy
from typing import List
from seqeval.metrics import f1_score, accuracy_score, classification_report

from .helper import compute_metrics, collect_named_entities, compute_precision_recall_f1_wrapper

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level="DEBUG",
)

Entity = namedtuple("Entity", "e_type start_offset end_offset")


def evaluate_with_seqeval(true_labels: List[list], predicted_labels: List[list]) -> dict:
    """

    :param true_labels:
    :param predicted_labels:
    :return:
    """
    tags = ["PER", "LOC", "CW", "GRP", "PROD", "CORP"]

    metrics2value = {"acc": accuracy_score(true_labels, predicted_labels),
                     "f1_score": f1_score(true_labels, predicted_labels)}

    metric_names, metric_value = classification_f1_report(
        true_labels, predicted_labels, tags
    )

    for name, value in zip(metric_names, metric_value):
        metrics2value.update({name: value})
    return metrics2value


def classification_f1_report(true_targets: List[list],
                             pred_targets: List[list], tags: list) -> [list, list]:
    """

    :param pred_targets:
    :param true_targets:
    :param tags:
    :return:
    """
    report = classification_report(y_true=true_targets, y_pred=pred_targets)
    metric_names, metric_values = [], []
    for line in report.split("\n"):
        if line != "" and (line.split()[0] in tags):
            metric_names.append(line.split()[0] + "_f1")
            metric_values.append(float(line.split()[3]))
        elif line != "" and (line.split()[0] == "avg"):
            metric_names.append(line.split()[0] + "_entities_f1")
            metric_values.append(float(line.split()[5]))
    return metric_names, metric_values


class Evaluator():

    def __init__(self, true, pred, tags):
        """
        """

        if len(true) != len(pred):
            raise ValueError("Number of predicted documents does not equal true")

        self.true = true
        self.pred = pred
        self.tags = tags

        # Setup dict into which metrics will be stored.

        self.metrics_results = {
            'correct': 0,
            'incorrect': 0,
            'partial': 0,
            'missed': 0,
            'spurious': 0,
            'possible': 0,
            'actual': 0,
            'precision': 0,
            'recall': 0,
            'f_score': 0
        }

        # Copy results dict to cover the four schemes.

        self.results = {
            'strict': deepcopy(self.metrics_results),
            'ent_type': deepcopy(self.metrics_results),
            'partial': deepcopy(self.metrics_results),
            'exact': deepcopy(self.metrics_results),
        }

        # Create an accumulator to store results

        self.evaluation_agg_entities_type = {e: deepcopy(self.results) for e in tags}

    def evaluate(self):

        # logging.info(
        #     "Imported %s predictions for %s true examples",
        #     len(self.pred), len(self.true)
        # )

        for true_ents, pred_ents in zip(self.true, self.pred):

            # Check that the length of the true and predicted examples are the
            # same. This must be checked here, because another error may not
            # be thrown if the lengths do not match.

            if len(true_ents) != len(pred_ents):
                raise ValueError("Prediction length does not match true example length")

            # Compute results for one message

            tmp_results, tmp_agg_results = compute_metrics(
                collect_named_entities(true_ents),
                collect_named_entities(pred_ents),
                self.tags
            )

            # Cycle through each result and accumulate

            # TODO: Combine these loops below:

            for eval_schema in self.results:

                for metric in self.results[eval_schema]:
                    self.results[eval_schema][metric] += tmp_results[eval_schema][metric]

            # Calculate global precision and recall

            self.results = compute_precision_recall_f1_wrapper(self.results)

            # Aggregate results by entity type

            for e_type in self.tags:

                for eval_schema in tmp_agg_results[e_type]:

                    for metric in tmp_agg_results[e_type][eval_schema]:
                        self.evaluation_agg_entities_type[e_type][eval_schema][metric] += \
                            tmp_agg_results[e_type][eval_schema][metric]

                # Calculate precision recall at the individual entity level

                self.evaluation_agg_entities_type[e_type] = compute_precision_recall_f1_wrapper(
                    self.evaluation_agg_entities_type[e_type])

        return self.results, self.evaluation_agg_entities_type
