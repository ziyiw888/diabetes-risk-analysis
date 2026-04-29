"""
Ziyi Wang
CSE 163 AC
Plotting figures.
"""


import os
from typing import Final, Iterable, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_FIG_DIR: Final[str] = "figs"


def plot_brfss_income_prevalence(df: pd.DataFrame,
                                 fig_dir: str = DEFAULT_FIG_DIR) -> None:
    """
    Save the bar chart: Diabetes Prevalence by Income_grp.
    """
    if "Diabetes_binary" not in df.columns or "Income_grp" not in df.columns:
        return

    group = (df.groupby("Income_grp")["Diabetes_binary"]
               .mean()
               .reset_index())

    plt.figure(figsize=(8, 4.5))
    plt.bar(group["Income_grp"], group["Diabetes_binary"])
    plt.title("BRFSS: Diabetes Prevalence by Income Group")
    plt.xlabel("Income group")
    plt.ylabel("Prevalence (mean Diabetes_binary)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, "brfss_prev_by_income.png")
    plt.savefig(path, dpi=160, bbox_inches="tight")


def plot_brfss_age_prevalence(df: pd.DataFrame,
                              fig_dir: str = DEFAULT_FIG_DIR) -> None:
    """
    Save the bar chart of Diabetes Prevalence by Age_grp.
    """
    if "Diabetes_binary" not in df.columns or "Age_grp" not in df.columns:
        return

    group = (df.groupby("Age_grp")["Diabetes_binary"]
               .mean()
               .reset_index())

    plt.figure(figsize=(8, 4.5))
    plt.bar(group["Age_grp"], group["Diabetes_binary"])
    plt.title("BRFSS: Diabetes Prevalence by Age Group")
    plt.xlabel("Age group")
    plt.ylabel("Prevalence (mean Diabetes_binary)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, "brfss_prev_by_age.png")
    plt.savefig(path, dpi=160, bbox_inches="tight")


def plot_nhanes_age_hist(df: pd.DataFrame,
                         fig_dir: str = DEFAULT_FIG_DIR) -> None:
    """
    Save the NHANES age histogram.
    """
    if "RIDAGEYR" not in df.columns:
        return

    ages = df["RIDAGEYR"].dropna()
    os.makedirs(fig_dir, exist_ok=True)

    plt.figure(figsize=(8, 4.5))
    plt.hist(ages, bins=30, rwidth=0.92, edgecolor="white", linewidth=0.7)
    plt.title("NHANES: Age Distribution (RIDAGEYR)")
    plt.xlabel("Age (years)")
    plt.ylabel("Count")
    plt.tight_layout()

    path = os.path.join(fig_dir, "nhanes_age_hist.png")
    plt.savefig(path, dpi=160, bbox_inches="tight")


def plot_nhanes_pir_by_gender(df: pd.DataFrame,
                              fig_dir: str = DEFAULT_FIG_DIR) -> None:
    """
    Save boxplot INDFMPIR by RIAGENDR (1=Male, 2=Female).
    """
    if "INDFMPIR" not in df.columns or "RIAGENDR" not in df.columns:
        return

    gender = df["RIAGENDR"].map({1: "Male", 2: "Female"})
    data_m = df.loc[gender == "Male", "INDFMPIR"].dropna()
    data_f = df.loc[gender == "Female", "INDFMPIR"].dropna()
    if len(data_m) == 0 or len(data_f) == 0:
        return

    os.makedirs(fig_dir, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    plt.boxplot([data_m, data_f], labels=["Male", "Female"], showmeans=True)
    plt.title("NHANES: Income-Poverty Ratio by Gender (INDFMPIR)")
    plt.xlabel("Gender")
    plt.ylabel("INDFMPIR")
    plt.tight_layout()

    path = os.path.join(fig_dir, "nhanes_pir_by_gender.png")
    plt.savefig(path, dpi=160, bbox_inches="tight")


def plot_feature_importances(importances: Iterable[float],
                             feature_names: Sequence[str],
                             out_path: str,
                             top_k: int = 12) -> None:
    """
    Plotting decision tree feature importances.
    """
    pairs = sorted(zip(feature_names, importances),
                   key=lambda x: x[1],
                   reverse=True)[:top_k]
    names = [p[0] for p in pairs][::-1]
    vals = [p[1] for p in pairs][::-1]

    plt.figure(figsize=(7, 6))
    plt.barh(range(len(vals)), vals)
    plt.yticks(range(len(vals)), names)
    plt.xlabel("Importance")
    plt.title(f"Decision Tree Feature Importances (Top {top_k})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)

def plot_group_rate(df: pd.DataFrame,
                    flag_col: str,
                    by: str,
                    out_path: str,
                    order: Optional[Sequence] = None,
                    title: Optional[str] = None) -> None:
    """
    Plot the mean of the binary column flag_col grouped by.
    """
    prev = df.groupby(by)[flag_col].mean()
    if order is not None:
        prev = prev.reindex(order)

    labels = [str(x) for x in prev.index]
    vals = prev.values * 100.0

    plt.figure(figsize=(8, 4.8))
    plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(vals)), labels, rotation=30, ha="right")
    plt.ylabel(f"{flag_col} rate (%)")
    plt.title(title if title else f"{flag_col} by {by}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)


def plot_prev_by_bmi_bins(df: pd.DataFrame,
                          target_col: str,
                          out_path: str) -> None:
    """
    Plot the prevalence of target_col by binning BMI
    into <25/25–30/≥30.
    """
    if "BMI" not in df.columns or target_col not in df.columns:
        return

    bins = [0, 25, 30, 100]
    labels = ["<25", "25–30", ">=30"]
    bmi_bin = pd.cut(df["BMI"], bins, labels=labels, include_lowest=True)
    tmp = df.assign(BMI_bin=bmi_bin).dropna(subset=["BMI_bin"])

    prev = (tmp.groupby("BMI_bin", observed=False)[target_col]
              .mean()
              .reindex(labels))

    vals = prev.values * 100.0

    plt.figure(figsize=(6.6, 4.2))
    plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(vals)), labels)
    plt.ylabel("Prevalence (%)")
    plt.title("Diabetes Prevalence by BMI bins")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)


def plot_risk_ratio_binary(df: pd.DataFrame,
                           binary_cols: Sequence[str],
                           target_col: str,
                           out_path: str) -> None:
    """
    Compute and plot the hazard ratio for each binary factor.
    """
    records: list[Tuple[str, float]] = []

    for col in binary_cols:
        if col not in df.columns:
            continue

        grp = df.groupby(col, dropna=False)[target_col].mean()

        has0 = (0 in grp.index) or (False in grp.index)
        has1 = (1 in grp.index) or (True in grp.index)
        if not (has0 and has1):
            continue

        p0 = float(grp.get(0, grp.get(False, float("nan"))))
        p1 = float(grp.get(1, grp.get(True, float("nan"))))

        if p0 > 0 and not (pd.isna(p0) or pd.isna(p1)):
            rr = p1 / p0
            records.append((col, rr))

    if not records:
        return

    records.sort(key=lambda x: x[1], reverse=True)
    names = [r[0] for r in records][::-1]
    vals = [r[1] for r in records][::-1]

    plt.figure(figsize=(6.4, 4.8))
    plt.barh(range(len(vals)), vals)
    plt.yticks(range(len(vals)), names)
    plt.xlabel("Risk ratio (prev=1 / prev=0)")
    plt.title("Risk Ratio of Binary Factors")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)

def _rate_table(df: pd.DataFrame,
                rate_col: str,
                group_col: str,
                order: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    Generate a table of group participation rates.
    """
    tbl = (df.groupby(group_col, dropna=False)[rate_col]
             .agg(["mean", "size"])
             .rename(columns={"mean": "rate", "size": "n"})
             .reset_index())

    if order is not None:
        tbl[group_col] = pd.Categorical(tbl[group_col], order, True)
        tbl = tbl.sort_values(group_col)

    tbl["rate_pct"] = tbl["rate"] * 100.0
    return tbl


def _normal_ci_for_proportion(p: pd.Series,
                              n: pd.Series,
                              z: float = 1.96
                              ) -> Tuple[pd.Series, pd.Series]:
    """
    A 95% confidence interval for the proportion based on the
    simple normal approximation, returning (lo_pct, hi_pct)
    (percentages).
    """
    se = np.sqrt(p * (1 - p) / n.clip(lower=1))
    lo = (p - z * se).clip(lower=0, upper=1) * 100.0
    hi = (p + z * se).clip(lower=0, upper=1) * 100.0
    return lo, hi


def plot_group_rate_zoom(df: pd.DataFrame,
                         rate_col: str,
                         group_col: str,
                         out_path: str,
                         order: Optional[Sequence[str]] = None,
                         title: Optional[str] = None,
                         y_min: Optional[float] = None,
                         annotate: bool = True,
                         with_ci: bool = True) -> None:
    """
    Enlarged version of the group participation rate bar chart
    to make the differences more obvious.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tbl = _rate_table(df, rate_col, group_col, order=order)

    if y_min is None:
        y_min = max(0.0, float(tbl["rate_pct"].min()) - 3.0)

    yerr = None
    if with_ci:
        lo, hi = _normal_ci_for_proportion(tbl["rate"], tbl["n"])
        yerr = np.vstack((tbl["rate_pct"] - lo,
                          hi - tbl["rate_pct"]))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(tbl[group_col].astype(str), tbl["rate_pct"], yerr=yerr,
           capsize=3)
    ax.set_ylim(y_min, 100)
    ax.set_ylabel(f"{rate_col} rate (%)")
    ax.set_title(title or f"{rate_col} by {group_col}")

    if annotate:
        for i, v in enumerate(tbl["rate_pct"]):
            ax.text(i, v + 0.2, f"{v:.1f}%", ha="center",
                    va="bottom", fontsize=9)

    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)


def plot_group_rate_delta(df: pd.DataFrame,
                          rate_col: str,
                          group_col: str,
                          out_path: str,
                          order: Optional[Sequence[str]] = None,
                          baseline: Optional[str] = None,
                          title: Optional[str] = None) -> None:
    """
    A bar chart showing relative improvement or decrease.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tbl = _rate_table(df, rate_col, group_col, order=order)

    if baseline is None:
        base = float(tbl["rate_pct"].mean())
        base_name = "overall mean"
    else:
        base = float(tbl.loc[tbl[group_col] == baseline, "rate_pct"].iloc[0])
        base_name = baseline

    tbl["delta"] = tbl["rate_pct"] - base

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(tbl[group_col].astype(str), tbl["delta"])
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel(f"Δ {rate_col} vs {base_name} (pp)")
    ax.set_title(title or f"{rate_col} Δ vs {base_name}")

    for i, v in enumerate(tbl["delta"]):
        ax.text(i, v + (0.2 if v >= 0 else -0.8),
                f"{v:+.1f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=9)

    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)