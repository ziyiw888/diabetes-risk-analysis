"""
Ziyi Wang
CSE 163 AC
NHANES demographics loader.
"""

import pandas as pd


def load_nhanes(path: str) -> pd.DataFrame:
    """
    Loads an NHANES demographics CSV.
    """
    return pd.read_csv(path)