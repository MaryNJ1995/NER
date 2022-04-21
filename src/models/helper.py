from pytorch_lightning.callbacks import ModelCheckpoint


def build_checkpoint_callback(save_top_k, filename="QTag-{epoch:02d}-{val_loss:.2f}",
                              monitor="val_loss", mode="min"):
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

    :param log_dict:
    :param results_agg:
    :param tags:
    :return:
    """
    for item in tags:
        for key, value in results_agg[item]['ent_type'].items():
            log_dict[item + '_' + key] = value

    return log_dict
