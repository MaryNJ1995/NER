# -*- coding: utf-8 -*-
# ========================================================
"""data_reader module is written for read files"""
# ========================================================


# ========================================================
# Imports
# ========================================================

import json5
import pandas as pd

__author__ = "Ehsan Tavan", "Ali Rahmati", "Maryam Najafi"
__project__ = "signal entity detection"
__version__ = "1.0.0"
__date__ = "2021/09/27"
__email__ = "tavan.ehsan@gmail.com"


def read_csv(path: str, columns: list = None, names: list = None) -> pd.DataFrame:
    """
    read_csv function for reading csv files
    :param path:
    :param columns:
    :param names:
    :return:
    """
    dataframe = pd.read_csv(path, usecols=columns) if columns else pd.read_csv(path)
    return dataframe.rename(columns=dict(zip(columns, names))) if names else dataframe


def read_json5(path: str):
    """
    read_json5 function for  reading json file
    :param path:
    :return:
    """
    with open(path, "r", encoding="utf-8") as file:
        data = json5.load(file)
    return data


def read_text(path: str) -> list:
    """
    read_text function for  reading text file
    :param path:
    :return:
    """
    with open(path, "r", encoding="utf-8") as file:
        data = file.readlines()
    return data
