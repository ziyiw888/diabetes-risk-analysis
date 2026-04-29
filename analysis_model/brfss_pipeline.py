"""
Ziyi Wang
CSE 163 AC
BRFSS data loading and cleaning.
"""

from typing import Final
import pandas as pd

AGE_LABELS: Final[dict[int, str]] = {
    1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44",
    6: "45-49", 7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69",
    11: "70-74", 12: "75-79", 13: "80+",
}
INCOME_LABELS: Final[dict[int, str]] = {
    1: "<$10k", 2: "$10k-$15k", 3: "$15k-$20k", 4: "$20k-$25k",
    5: "$25k-$35k", 6: "$35k-$50k", 7: "$50k-$75k", 8: "$75k+",
}
SEX_LABELS: Final[dict[int, str]] = {0: "Female", 1: "Male"}

MODEL_FEATURES: Final[list[str]] = [
    "BMI",
    "HighBP",
    "GenHlth",
    "PhysHlth",
    "Age",
    "Income"
]

FEATURES: Final[list[str]] = MODEL_FEATURES


def available_features(df: pd.DataFrame) -> list[str]:
    """
    Returns the subset of model features that are
    actually present in df.
    """
    return [c for c in FEATURES if c in df.columns]


def load_brfss(path: str) -> pd.DataFrame:
    """
    Loads Kaggle BRFSS indicators CSV.
    """
    df = pd.read_csv(path)

    if "Diabetes_012" in df.columns and "Diabetes_binary" not in df.columns:
        df["Diabetes_binary"] = (df["Diabetes_012"] == 2).astype(int)

    if "Age" in df.columns:
        df["Age_grp"] = df["Age"].map(AGE_LABELS)
    if "Income" in df.columns:
        df["Income_grp"] = df["Income"].map(INCOME_LABELS)
    if "Sex" in df.columns:
        df["Sex_label"] = df["Sex"].map(SEX_LABELS)
    return df