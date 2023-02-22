# -*- coding: utf-8 -*-
# ==========================================================================

"""
This module is written to write function for read different file types.
"""

# ============================ Third Party libs ============================
import json
import pickle


# ==========================================================================


def write_json(data: dict, path: str) -> None:
    """
    write_json function is written for write in json files

    Args:
        data: data to save in json
        path: json path

    Returns:
        None

    """

    with open(path, "w", encoding="utf8") as outfile:
        json.dump(data, outfile, separators=(",", ":"), indent=4)


def write_pickle(data: list, path: str) -> None:
    """
    write_pickle function is written for write data in pickle file

    Args:
        data: data to save in pickle file
        path: pickle path

    Returns:
        None

    """

    with open(path, "wb") as outfile:
        pickle.dump(data, outfile)


def write_text(data: list, path: str) -> None:
    """
    save_text function is written for write in text files
    :param data:
    :param path:
    :return:
    """
    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(data))
