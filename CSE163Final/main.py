"""
Ziyi Wang
CSE 163 AC
This program Loads BRFSS & NHANES.
Prints missing tables and seven-number summaries.
Produce EDA figures. Trains a Decision Tree and
saves feature-importance plot.
"""
import os
from typing import Final, Sequence

import pandas as pd

from brfss_pipeline import (
    load_brfss,
    AGE_LABELS,
    INCOME_LABELS,
    available_features,
)
from nhanes_utils import load_nhanes
from eda_utils import missing_table, numeric_seven_number
from modeling_dt import split_data, train_decision_tree, evaluate_model
from plotting import (
    plot_brfss_income_prevalence,
    plot_brfss_age_prevalence,
    plot_nhanes_age_hist,
    plot_nhanes_pir_by_gender,
    plot_feature_importances,
    plot_prev_by_bmi_bins,
    plot_risk_ratio_binary,
    plot_group_rate_zoom,
    plot_group_rate_delta,
)

import validity_tests as vt

DATA_DIR: Final[str] = "."
BRFSS_CSV: Final[str] = os.path.join(
    DATA_DIR, "diabetes_012_health_indicators_BRFSS2015.csv"
)
NHANES_CSV: Final[str] = os.path.join(DATA_DIR, "nhanes_demo.csv")
FIG_DIR: Final[str] = os.path.join(DATA_DIR, "figs")


def _categorical_counts(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Returns counts per category for a column if present. Else empty table.
    """
    if col not in df.columns:
        return pd.DataFrame(columns=["value", "count"])
    out = (df[col].value_counts(dropna=False)
           .rename_axis("value")
           .reset_index(name="count"))
    return out


def _age_order() -> list[str]:
    """
    Return age labels in numeric order according to AGE_LABELS.
    """
    return [AGE_LABELS[k] for k in sorted(AGE_LABELS.keys())]


def _income_order() -> list[str]:
    """
    Return income labels in numeric order according to INCOME_LABELS.
    """
    return [INCOME_LABELS[k] for k in sorted(INCOME_LABELS.keys())]


def run_eda(brfss: pd.DataFrame, nhanes: pd.DataFrame) -> None:
    """
    Basic EDA prints + core figures saved to FIG_DIR.
    """
    print("BRFSS shape:", brfss.shape)
    print(missing_table(brfss).head())

    print("NHANES shape:", nhanes.shape)
    print(missing_table(nhanes).head())

    numeric_wishlist: list[str] = [
        "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
        "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
        "HvyAlcoholConsump", "AnyHealthcare", "NoDocbc", "GenHlth",
        "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education",
        "Income",
    ]
    numeric_cols = [c for c in numeric_wishlist if c in brfss.columns]
    seven = numeric_seven_number(brfss, numeric_cols).round(3)
    print(seven.head())

    if "Income_grp" in brfss.columns:
        counts = _categorical_counts(brfss, "Income_grp")
        order = _income_order()
        counts["value"] = pd.Categorical(counts["value"], order, True)
        print(counts.sort_values("value"))

    plot_brfss_income_prevalence(brfss, FIG_DIR)
    plot_brfss_age_prevalence(brfss, FIG_DIR)
    plot_nhanes_age_hist(nhanes, FIG_DIR)
    plot_nhanes_pir_by_gender(nhanes, FIG_DIR)


def run_rq3_feature_importance(brfss: pd.DataFrame) -> None:
    """
    Trains a Decision Tree and saves the feature-importance bar chart.
    """
    feat: list[str] = available_features(brfss)
    X_train, X_test, y_train, y_test = split_data(
        brfss[feat], brfss["Diabetes_binary"]
    )
    model = train_decision_tree(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    out_path = os.path.join(FIG_DIR, "rq3_feature_importance.png")
    plot_feature_importances(model.feature_importances_, feat, out_path, 12)


def run_rq4_risk_visuals(brfss: pd.DataFrame) -> None:
    """
    Saves iabetes prevalence by BMI bins and
    risk ratios for selected binary factors.
    """
    out_prev = os.path.join(FIG_DIR, "rq4_prev_by_BMIbins.png")
    plot_prev_by_bmi_bins(brfss, "Diabetes_binary", out_prev)

    binary_candidates: Sequence[str] = ["HighBP", "HighChol", "Smoker"]
    binary_candidates = [c for c in binary_candidates if c in brfss.columns]
    out_rr = os.path.join(FIG_DIR, "rq4_risk_ratio_binary.png")
    plot_risk_ratio_binary(brfss, binary_candidates, "Diabetes_binary", out_rr)


def run_rq5_participation(brfss: pd.DataFrame) -> None:
    """
    Saves participation-rate plots (CholCheck / AnyHealthcare) by
    income and age groups, with zoomed and delta variants.
    """
    inc_order = _income_order()
    age_order = _age_order()

    out_inc_zoom = os.path.join(FIG_DIR, "rq5_cholcheck_by_income_zoom.png")
    plot_group_rate_zoom(
        brfss, "CholCheck", "Income_grp", out_inc_zoom, inc_order,
        title="Cholesterol Check Rate by Income (zoom)"
    )
    out_inc_delta = os.path.join(FIG_DIR, "rq5_cholcheck_income_delta.png")
    plot_group_rate_delta(
        brfss, "CholCheck", "Income_grp", out_inc_delta, inc_order,
        baseline=inc_order[-1],
        title="Cholesterol Check Δ vs $75k+ (pp)"
    )

    out_age_zoom = os.path.join(FIG_DIR, "rq5_cholcheck_by_age_zoom.png")
    plot_group_rate_zoom(
        brfss, "CholCheck", "Age_grp", out_age_zoom, age_order,
        title="Cholesterol Check Rate by Age (zoom)"
    )
    out_age_delta = os.path.join(FIG_DIR, "rq5_cholcheck_age_delta.png")
    plot_group_rate_delta(
        brfss, "CholCheck", "Age_grp", out_age_delta, age_order,
        baseline=None,
        title="Cholesterol Check Δ vs overall mean (pp)"
    )

    if "AnyHealthcare" in brfss.columns:
        out_i_zoom = os.path.join(FIG_DIR, "rq5_insurance_by_income_zoom.png")
        plot_group_rate_zoom(
            brfss, "AnyHealthcare", "Income_grp", out_i_zoom, inc_order,
            title="Has Any Healthcare by Income (zoom)"
        )
        out_i_delta = os.path.join(FIG_DIR, "rq5_insurance_income_delta.png")
        plot_group_rate_delta(
            brfss, "AnyHealthcare", "Income_grp", out_i_delta, inc_order,
            baseline=inc_order[-1],
            title="Any Healthcare Δ vs $75k+ (pp)"
        )

        out_a_zoom = os.path.join(FIG_DIR, "rq5_insurance_by_age_zoom.png")
        plot_group_rate_zoom(
            brfss, "AnyHealthcare", "Age_grp", out_a_zoom, age_order,
            title="Has Any Healthcare by Age (zoom)"
        )
        out_a_delta = os.path.join(FIG_DIR, "rq5_insurance_age_delta.png")
        plot_group_rate_delta(
            brfss, "AnyHealthcare", "Age_grp", out_a_delta, age_order,
            baseline=None,
            title="Any Healthcare Δ vs overall mean (pp)"
        )


def run_result_validity(brfss: pd.DataFrame) -> None:
    """
    Prints chi-square tests (Income/Age vs Diabetes_binary) and
    saves logistic-trend coefficients/ORs to CSV.
    """
    os.makedirs(FIG_DIR, exist_ok=True)

    if {"Income_grp", "Diabetes_binary"}.issubset(brfss.columns):
        res_inc = vt.chi_square_test(brfss, "Income_grp", "Diabetes_binary")
        print("\n[Chi-square] Income_grp × Diabetes_binary")
        print("Contingency table:")
        print(res_inc["table"])
        print(f"chi2={res_inc['chi2']:.3f}, "
              f"dof={res_inc['dof']}, p={res_inc['p']:.3g}")
        res_inc["table"].to_csv(os.path.join(FIG_DIR, "chi2_income_table.csv"))

    if {"Age_grp", "Diabetes_binary"}.issubset(brfss.columns):
        res_age = vt.chi_square_test(brfss, "Age_grp", "Diabetes_binary")
        print("\n[Chi-square] Age_grp × Diabetes_binary")
        print("Contingency table:")
        print(res_age["table"])
        print(f"chi2={res_age['chi2']:.3f}, "
              f"dof={res_age['dof']}, p={res_age['p']:.3g}")
        res_age["table"].to_csv(os.path.join(FIG_DIR, "chi2_age_table.csv"))

    needed = {"Income", "Age", "Sex", "Diabetes_binary"}
    if needed.issubset(brfss.columns):
        coef_tbl = vt.logit_trend(
            brfss, "Diabetes_binary", ["Income", "Age", "Sex"]
        )
        path = os.path.join(FIG_DIR, "logit_trend.csv")
        coef_tbl.to_csv(path, index=False)
        print("\n[Logit Trend] Diabetes_binary ~ Income + Age + Sex")
        print(coef_tbl.to_string(index=False,
                                 float_format=lambda x: f"{x:.4g}"))
        print(f"(Saved coefficients to {path})")


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)

    brfss = load_brfss(BRFSS_CSV)
    nhanes = load_nhanes(NHANES_CSV)

    run_eda(brfss, nhanes)
    run_rq3_feature_importance(brfss)
    run_rq4_risk_visuals(brfss)
    run_rq5_participation(brfss)
    run_result_validity(brfss)

    print(f"Figures/results saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()