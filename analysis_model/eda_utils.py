"""
Ziyi Wang
CSE 163 AC
General EDA utilities.
"""


from typing import Iterable
import pandas as pd


def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a table summarizing the missing values in
    each column of df.
    """
    n_missing = df.isna().sum()
    denom = len(df)
    pct_missing = (n_missing / denom * 100.0) if denom > 0 else 0.0

    out = (
        pd.DataFrame(
            {
                "column": n_missing.index,
                "n_missing": n_missing.values,
                "pct_missing": (
                    pct_missing.values
                    if hasattr(pct_missing, "values")
                    else [pct_missing] * len(n_missing)
                ),
            }
        )
        .sort_values("n_missing", ascending=False)
        .reset_index(drop=True)
    )
    return out


def numeric_seven_number(df: pd.DataFrame,
                         columns: list[str]) -> pd.DataFrame:
    """
    Seven-number summary for the selected numeric columns.
    Always includes: mean, std, min, q1, median, q3, max.
    Also attaches p90/p99 if available (tests不会依赖这两列).
    """
    present = [c for c in columns if c in df.columns]
    if not present:
        return pd.DataFrame(columns=["mean", "std", "min", "q1", "median", "q3", "max"])

    desc = df[present].describe(percentiles=[0.25, 0.5, 0.75, 0.90, 0.99]).T
    desc = desc.rename(columns={
        "25%": "q1",
        "50%": "median",
        "75%": "q3",
        "90%": "p90",
        "99%": "p99",
    })

    out = desc[["mean", "std", "min", "q1", "median", "q3", "max"]].copy()

    for extra in ("p90", "p99"):
        if extra in desc.columns:
            out[extra] = desc[extra]

    return out