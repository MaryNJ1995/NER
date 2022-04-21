# -*- coding: utf-8 -*-
# ========================================================
"""data_writer module is written for write data in files"""
# ========================================================


# ========================================================
# Imports
# ========================================================

import json

__author__ = "Ehsan Tavan", "Ali Rahmati", "Maryam Najafi"
__project__ = "signal entity detection"
__version__ = "1.0.0"
__date__ = "2021/09/27"
__email__ = "tavan.ehsan@gmail.com"


def write_json(path: str, data: dict) -> None:
    """
    write_json function is written for write in json files
    :param path:
    :param data:
    :return:
    """
    with open(path, "w", encoding="utf-8") as outfile:
        json.dump(data, outfile, separators=(",", ":"), indent=4)
