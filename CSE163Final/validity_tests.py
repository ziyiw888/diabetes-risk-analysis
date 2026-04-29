"""
Ziyi Wang
CSE 163
Result validity Chi-square tests and logistic trend test.
"""


from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import statsmodels.api as sm


def chi2_test(df: pd.DataFrame, group_col: str, target_col: str) -> Dict[str, Any]:
    """
    Pearson chi-square test of independence for (group_col x target_col).

    Returns:
        {
            "table": pd.DataFrame (contingency table),
            "chi2": float,
            "dof": int,
            "p": float
        }
    """
    if group_col not in df.columns or target_col not in df.columns:
        raise KeyError(f"Columns not found: {group_col}, {target_col}")

    # Build contingency table (rows=groups, cols=0/1 outcome)
    table = pd.crosstab(df[group_col], df[target_col])

    # SciPy returns (chi2, p, dof, expected)
    chi2, p, dof, _ = chi2_contingency(table)
    return {"table": table, "chi2": float(chi2), "dof": int(dof), "p": float(p)}


def chi_square_test(df: pd.DataFrame, group_col: str, target_col: str) -> Dict[str, Any]:
    """
    Alias for chi2_test to match calls from different modules.
    """
    return chi2_test(df, group_col, target_col)


def logit_trend(df: pd.DataFrame,
                target_col: str,
                predictors: Sequence[str]) -> pd.DataFrame:
    """
    Fits a logistic regression:
        target_col ~ predictors (add constant)

    Predictors should be numeric (e.g., Income 1-8, Age 1-13, Sex 0/1).
    Returns a table with coefficients, p-values, and odds ratios (95% CI).

    Columns:
        param, coef, se, z, pval, OR, OR_lo95, OR_hi95
    """
    cols_needed = [target_col] + list(predictors)
    for c in cols_needed:
        if c not in df.columns:
            raise KeyError(f"Column not found: {c}")

    d = df[cols_needed].dropna()
    # Ensure numeric types for statsmodels
    y = d[target_col].astype(float)
    X = d[list(predictors)].astype(float)

    # Add intercept
    X = sm.add_constant(X, has_constant="add")

    model = sm.Logit(y, X)
    fit = model.fit(disp=False)

    params = fit.params
    bse = fit.bse
    pvals = fit.pvalues
    zvals = params / bse

    # Odds ratios and 95% CI
    OR = np.exp(params)
    OR_lo = np.exp(params - 1.96 * bse)
    OR_hi = np.exp(params + 1.96 * bse)

    out = pd.DataFrame({
        "param": params.index,
        "coef": params.values,
        "se": bse.values,
        "z": zvals.values,
        "pval": pvals.values,
        "OR": OR.values,
        "OR_lo95": OR_lo.values,
        "OR_hi95": OR_hi.values,
    })
    return out


__all__ = ["chi2_test", "chi_square_test", "logit_trend"]