# -*- coding: utf-8 -*-
"""
    Complex NER Project:
        models:
            helper.py
"""

# ============================ Third Party libs ============================
from pytorch_lightning.callbacks import ModelCheckpoint


def build_checkpoint_callback(save_top_k: int, filename="QTag-{epoch:02d}-{val_loss:.2f}",
                              monitor="val_loss", mode="min"):
    """
    function to build checkpoint callback
    Args:
        save_top_k: save top k best model
        filename: the name that checkpoint is saved.
        monitor: how to monitor val loss
        mode: mode of the monitored quantity for optimization

    Returns:
        callback of checkpoint

    """
    # saves a file like: input/QTag-epoch=02-val_loss=0.32.ckpt
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,  # monitored quantity
        filename=filename,
        save_top_k=save_top_k,  # save the top k models
        mode=mode,  # mode of the monitored quantity for optimization
    )
    return checkpoint_callback


def add_metric_to_log_dic(log_dict: dict, results_agg: dict, tags: list) -> dict:
    """

    Args:
        log_dict:
        results_agg:
        tags:

    Returns:

    """
    for item in tags:
        for key, value in results_agg[item]['ent_type'].items():
            log_dict[item + '_' + key] = value

    return log_dict
