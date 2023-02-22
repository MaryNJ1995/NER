# -*- coding: utf-8 -*-
# ========================================================
"""
This module is written to read function for read different file types.
"""

# ============================ Third Party libs ============================
import pickle


# ==========================================================================

def read_text(path: str) -> list:
    """
    function to read .txt file

    Args:
        path: path of .txt file

    Returns:
        .txt data in list

    """

    with open(path, "r", encoding="utf8") as file:
        data = file.readlines()
    return data
